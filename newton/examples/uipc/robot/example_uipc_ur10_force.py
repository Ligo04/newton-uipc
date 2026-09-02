# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Example UIPC UR10 Force — Joint-Space PD + Gravity Compensation

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode
from newton.actuators import Actuator as _NewtonActuator
from newton.actuators import ClampingMaxEffort, ControllerPD, ControllerStablePD
from newton.selection import ArticulationView


@wp.kernel
def _sum_bias_forces_to_worlds(
    gravity_force: wp.array[float],
    coriolis_force: wp.array[float],
    dofs_per_world: int,
    # outputs
    bias_forces: wp.array2d[float],
):
    """Fold the flat inverse-dynamics bias ``g(q) + C(q,q̇)q̇`` into the controller's
    per-world ``(world, dof)`` layout. Valid when every world owns a contiguous,
    equal-width DOF block (one articulation per world, no padding)."""
    w, j = wp.tid()
    idx = w * dofs_per_world + j
    bias_forces[w, j] = gravity_force[idx] + coriolis_force[idx]


class Example:
    # Use a bent UR10 home pose.
    HOME_POSE = np.array(
        [0.0, -np.pi / 3, np.pi / 2, -np.pi / 6, np.pi / 2, 0.0],
        dtype=np.float32,
    )
    # Share the trajectory with the aim-drive example.
    TRAJ_AMP = 0.4  # [rad]
    TRAJ_OMEGA = 1.2  # [rad/s]
    TRAJ_PHASE = 0.8  # [rad] per DOF index

    def __init__(self, viewer, args):
        # Run physics at 240 Hz with four substeps per frame.
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.solver_name = args.solver
        self.stable_pd = bool(args.stable_pd)
        self.hold = bool(args.hold)
        self.viewer = viewer

        # Joint-space PD gains
        self.kp = np.array([500.0] * 6, dtype=np.float32)
        self.kd = np.array([50.0] * 6, dtype=np.float32)
        # Torque clamps sized roughly to UR10's real effort limits.
        self.max_torque = np.array([330.0, 330.0, 150.0, 54.0, 54.0, 54.0], dtype=np.float32)

        ur10 = newton.ModelBuilder()

        # Register MuJoCo-specific USD attributes before add_usd.
        if self.solver_name == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(ur10)

        asset_path = newton.utils.download_asset("universal_robots_ur10")
        asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")
        height = 1.2
        # Weld the UR10 base with an explicit FIXED joint.
        ur10.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0.0, 0.0, height)),
            floating=False,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )
        ur10.add_shape_cylinder(
            -1,
            xform=wp.transform(wp.vec3(0, 0, height / 2)),
            half_height=height / 2,
            radius=0.08,
        )

        # Set controlled DOFs to EFFORT mode.
        for i in range(len(ur10.joint_target_ke)):
            ur10.joint_target_ke[i] = 0.0
            ur10.joint_target_kd[i] = 0.0
            ur10.joint_target_mode[i] = int(JointTargetMode.EFFORT)

        # Register one PD actuator per UR10 DOF.
        controller_cls = ControllerStablePD if self.stable_pd else ControllerPD
        for dof_idx in range(len(self.kp)):
            # Set the stable-PD world count for the replicated builder.
            extra_kwargs = {"num_worlds": 1} if self.stable_pd else {}
            ur10.add_actuator(
                controller_cls,
                index=dof_idx,
                kp=float(self.kp[dof_idx]),
                kd=float(self.kd[dof_idx]),
                clamping=[
                    (ClampingMaxEffort, {"max_effort": float(self.max_torque[dof_idx])}),
                ],
                **extra_kwargs,
            )

        if self.world_count > 1:
            builder = newton.ModelBuilder()
            builder.replicate(ur10, self.world_count, spacing=(2.0, 2.0, 0.0))
        else:
            builder = ur10

        # Start all worlds from the same home pose.
        joint_q_all = np.tile(self.HOME_POSE, self.world_count).astype(np.float32)
        builder.joint_q = joint_q_all.tolist()

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.state_0 = self.model.state()

        self.solver, self._uses_contacts = self._build_solver(self.solver_name)

        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.CollisionPipeline(self.model).contacts() if self._uses_contacts else None

        # Select revolute UR10 DOFs and exclude FREE/DISTANCE joints.
        self.ur10s = ArticulationView(
            self.model,
            "*ur10*",
            exclude_joint_types=[newton.JointType.FREE, newton.JointType.DISTANCE],
        )
        assert self.ur10s.count == self.world_count, (
            f"expected one UR10 per world, got {self.ur10s.count} for {self.world_count} worlds"
        )

        # Cache per-world DOF and body strides.
        self.dofs_per_world = self.ur10s.joint_dof_count
        self.bodies_per_world = self.model.body_count // self.world_count
        assert self.dofs_per_world == 6, f"expected 6 UR10 DOFs per world, got {self.dofs_per_world}"

        # Cache replicated joint targets in the ArticulationView layout.
        self.q_target = (
            np.broadcast_to(self.HOME_POSE, (self.world_count, 1, self.dofs_per_world)).astype(np.float32).copy()
        )
        self.qd_target = np.zeros((self.world_count, 1, self.dofs_per_world), dtype=np.float32)

        # Populate joint_target_q/qd — ActuatorPD reads them every step.
        self.ur10s.set_attribute("joint_target_q", self.control, self.q_target)
        self.ur10s.set_attribute("joint_target_qd", self.control, self.qd_target)

        # Cache the composed actuator and its controller state.
        expected_controller_cls = ControllerStablePD if self.stable_pd else ControllerPD
        pd_actuator = next(
            a
            for a in self.model.actuators
            if isinstance(a, _NewtonActuator) and isinstance(a.controller, expected_controller_cls)
        )
        self.pd_actuator = pd_actuator
        expected = self.world_count * self.dofs_per_world
        kp_len = len(pd_actuator.controller.kp)  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        assert kp_len == expected, (
            f"PD actuator kp length {kp_len} != {expected}; "
            "builder/replicate did not merge the per-DOF actuators as expected"
        )

        # Provide mass matrix and bias forces to stable-PD.
        self._act_state: _NewtonActuator.State | None = None
        if self.stable_pd:
            self._act_state = self.pd_actuator.state()

            # Reuse scratch buffers for mass and Jacobian evaluation.
            W = self.world_count
            max_dofs = self.model.max_dofs_per_articulation
            max_links = self.model.max_joints_per_articulation
            dev = self.model.device
            self._mm_J = wp.zeros((W, max_links * 6, max_dofs), dtype=float, device=dev)
            self._mm_body_I_s = wp.zeros(self.model.body_count, dtype=wp.spatial_matrix, device=dev)
            self._mm_joint_S_s = wp.zeros(self.model.joint_dof_count, dtype=wp.spatial_vector, device=dev)
            self._H_buf = wp.empty((W, max_dofs, max_dofs), dtype=float, device=dev)
            # Allocate reusable gravity and Coriolis buffers.
            self._id_gravity_force = wp.zeros(self.model.joint_dof_count, dtype=wp.float32, device=dev)
            self._id_coriolis_force = wp.zeros(self.model.joint_dof_count, dtype=wp.float32, device=dev)

        # Track CUDA-graph state for plain-PD execution.
        self._actuator_graphs: list | None = None
        self._state_parity = 0

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(5.0, 5.0, 3.0),
            pitch=-20.0,
            yaw=-135.0,
        )
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        self.viewer._paused = True

        self._capture_actuator_graphs()

    # solver
    def _build_solver(self, name: str) -> tuple[object, bool]:
        """Construct the requested Newton solver.

        Returns the solver and a flag telling the caller whether a
        ``Contacts`` object must be passed to ``solver.step()``.
        """
        if name == "uipc":
            import uipc  # noqa: PLC0415  # deferred so non-uipc solvers run without the dep

            solver = newton.solvers.SolverUIPC(
                self.model,
                workspace="/tmp/newton_uipc",
                dt=self.sim_dt,
                logger_level=uipc.Logger.Warn,
                dump_enable=True,
            )
            solver.set_contact(True, 0.001)
            solver.sync_uipc_inertia_with_model()
            solver.initialize()
            return solver, True
        if name == "mujoco":
            return newton.solvers.SolverMuJoCo(self.model), False
        if name == "featherstone":
            return newton.solvers.SolverFeatherstone(self.model), False
        raise ValueError(f"unsupported --solver: {name!r}")

    def _apply_feedback(self, state):
        """Run every registered actuator -> ``control.joint_f``.

        ``ActuatorPD`` already handles the PD math and the ``max_force`` clamp
        in its Warp kernel, and it reads position / velocity targets from
        ``control.joint_target_q`` / ``joint_target_qd`` (written once in
        ``__init__``). The only per-substep bookkeeping we need is:

        1. Clear ``control.joint_f`` — the PD kernel accumulates with ``+=``,
           so an un-cleared buffer would blow past ``max_force`` each frame.
        2. If ``--gravity-comp`` is on, refresh the ``constant_force`` array
           with the current Jacobian-transpose gravity compensation (it
           depends on the current pose, so it must be recomputed every
           substep).
        3. Run each actuator.

        Args:
            state: Simulation state the actuator reads ``joint_q`` / ``joint_qd``
                from. Passed explicitly (rather than ``self.state_0``) so the
                feedback pass can be captured into a CUDA graph per physical
                state buffer — see :meth:`_capture_actuator_graphs`.
        """
        self.control.joint_f.zero_()  # pyright: ignore[reportOptionalMemberAccess]  # ty:ignore[unresolved-attribute]

        if self.stable_pd:
            if self._act_state is None:
                raise ValueError("ControllerStablePD state was not initialized")
            ctrl_state = self._act_state.controller_state

            # Evaluate the mass matrix at the current pose.
            newton.eval_jacobian(self.model, state, self._mm_J, joint_S_s=self._mm_joint_S_s)
            newton.eval_mass_matrix(self.model, state, H=self._H_buf, J=self._mm_J, body_I_s=self._mm_body_I_s)
            # Match the controller mass-matrix layout.
            ctrl_state.mass_matrix.assign(self._H_buf)  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

            # Sum gravity and Coriolis into the stable-PD bias force.
            newton.eval_inverse_dynamics_passive(
                self.model, state, gravity_force=self._id_gravity_force, coriolis_force=self._id_coriolis_force
            )
            n = ctrl_state.bias_forces.shape[1]  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
            wp.launch(
                _sum_bias_forces_to_worlds,
                dim=(self.world_count, n),
                inputs=[self._id_gravity_force, self._id_coriolis_force, n],
                outputs=[ctrl_state.bias_forces],  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
                device=self.model.device,
            )

        # Stable-PD state uses per-step scratch only.
        next_state = self._act_state if self.stable_pd else None
        self.pd_actuator.step(
            sim_state=state,
            sim_control=self.control,
            current_act_state=self._act_state,
            next_act_state=next_state,
            dt=self.sim_dt,
        )

    def _capture_actuator_graphs(self):
        """Capture the actuator step into a CUDA graph per state buffer.

        Both actuator paths are pure ``wp.launch`` sequences with no
        host<->device transfers: plain PD (zero ``joint_f`` ->
        ``ControllerPD`` -> ``ClampingMaxEffort`` -> scatter-add), and
        stable PD, whose bias assembly (:func:`newton.eval_jacobian`,
        :func:`newton.eval_mass_matrix`, :func:`newton.eval_inverse_dynamics_passive`
        RNEA passes, and the blocked-LLT solve inside ``ControllerStablePD``)
        reuses buffers preallocated at init — nothing allocates, reads back,
        or synchronizes during capture.

        ``simulate`` ping-pongs ``state_0``/``state_1`` every substep, and a
        captured graph bakes in the array pointers it was recorded against.
        We therefore capture one graph per *physical* state buffer and replay
        whichever one currently holds the live state (tracked by
        ``_state_parity``). Falls back to eager execution on non-CUDA devices.
        """
        if not wp.get_device().is_cuda:
            self._actuator_graphs = None
            return
        # Warm up kernels before CUDA-graph capture.
        self._apply_feedback(self.state_0)
        graphs = []
        for state in (self.state_0, self.state_1):
            with wp.ScopedCapture() as capture:
                self._apply_feedback(state)
            graphs.append(capture.graph)
        self._actuator_graphs = graphs
        # state_0 is the live buffer at capture time -> graphs[0].
        self._state_parity = 0

    # runtime
    def simulate(self):
        for _ in range(self.sim_substeps):
            if self._actuator_graphs is not None:
                wp.capture_launch(self._actuator_graphs[self._state_parity])
            else:
                self._apply_feedback(self.state_0)
            self.state_0.clear_forces()
            self.solver.step(  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            if self._actuator_graphs is not None:
                # Track the active state buffer for graph replay.
                self._state_parity ^= 1

    def _update_targets(self):
        """Apply the shared home-centered sinusoidal trajectory to all worlds.

        Writes into the existing ``control.joint_target_q`` buffer, so the
        captured plain-PD CUDA graphs keep reading fresh targets.
        """
        if self.hold:
            target = self.HOME_POSE
        else:
            phases = self.TRAJ_PHASE * np.arange(self.dofs_per_world, dtype=np.float32)
            target = self.HOME_POSE + self.TRAJ_AMP * np.sin(self.TRAJ_OMEGA * self.sim_time + phases)
        self.q_target[:] = target.astype(np.float32)
        self.ur10s.set_attribute("joint_target_q", self.control, self.q_target)

    def step(self):
        self._update_targets()
        self.simulate()
        self._log_tracking()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def _log_tracking(self):
        """Print world-0 joint positions, velocities, tracking error, and torques."""
        q = self.ur10s.get_attribute("joint_q", self.state_0).numpy()[0, 0]
        qd = self.ur10s.get_attribute("joint_qd", self.state_0).numpy()[0, 0]
        f = self.ur10s.get_attribute("joint_f", self.control).numpy()[0, 0]
        err = self.q_target[0, 0] - q
        q_str = " ".join(f"{p:+.4f}" for p in q)
        qd_str = " ".join(f"{v:+.4f}" for v in qd)
        err_str = " ".join(f"{e:+.3f}" for e in err)
        tau_str = " ".join(f"{t:+7.2f}" for t in f)
        print(
            f"[t={self.sim_time:6.3f}s] q(rad)=[{q_str}]  qd(rad/s)=[{qd_str}]\n"
            f"              err(rad)=[{err_str}]  tau(N·m)=[{tau_str}]"
        )

    # test
    def test_final(self):
        """Smoke-check: all joint values must be finite.

        We deliberately do NOT assert tight tracking — the conservative PD
        used here does not fully cancel gravity and different solvers
        reach different steady states.  The purpose of this test is just
        to verify the pipeline runs end-to-end without producing NaNs or
        exploding to astronomical values, for every advertised ``--solver``.
        """
        # Shape: (world_count, 1, dofs_per_arti)
        q = self.ur10s.get_attribute("joint_q", self.state_0).numpy()
        qd = self.ur10s.get_attribute("joint_qd", self.state_0).numpy()
        assert np.all(np.isfinite(q)), "joint_q went non-finite (divergence)"
        assert np.all(np.isfinite(qd)), "joint_qd went non-finite (divergence)"
        # Sanity bound: angles must be within a few full revolutions.
        assert np.all(np.abs(q) < 20.0), f"joint_q blew up: {q.tolist()}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.add_argument(
            "--solver",
            choices=("uipc", "mujoco", "featherstone", "semi_implicit"),
            default="uipc",
            help="Newton solver backend driving the force controller.",
        )
        parser.add_argument(
            "--stable-pd",
            action="store_true",
            help=(
                "Use ControllerStablePD (Tan et al. 2011) instead of ControllerPD. "
                "The controller runs an on-device (M + diag(Kd)·Δt)·qddot = b "
                "solve each substep, so the example populates its State with "
                "M = newton.eval_mass_matrix and bias_forces = RNEA gravity + "
                "Coriolis via newton.eval_inverse_dynamics_passive every substep. "
                "Mutually exclusive with --gravity-comp (bias_forces "
                "already contains gravity)."
            ),
        )
        parser.add_argument(
            "--hold",
            action="store_true",
            help="Hold the home pose instead of tracking the shared sinusoid (settling / steady-state comparison).",
        )
        parser.set_defaults(world_count=4)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
