# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverUIPC mimic joint coupling (follower = coef0 + coef1 * leader)."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
from newton import JointTargetMode
from newton.tests.unittest_utils import get_selected_cuda_test_devices

_HAS_UIPC = importlib.util.find_spec("uipc") is not None
_CUDA_TEST_DEVICES = get_selected_cuda_test_devices(mode="basic")

if _HAS_UIPC:
    import uipc

_Q_ID = wp.quat_identity(dtype=wp.float64)  # pyright: ignore[reportArgumentType]


def _add_joint_pair(builder: newton.ModelBuilder, y: float, *, prismatic=False, balanced_leader=False):
    """Build two independent scalar joints attached to a fixed anchor.

    Returns ``(leader_joint, follower_joint)`` — two scalar joints that
    share no kinematic coupling in the model, so any tracking between them
    must come from a mimic constraint.
    """
    hx = hy = 0.1
    upper_hz = 0.2
    link_hz = 0.5
    drop_z = 2.0

    anchor = builder.add_link(xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=_Q_ID))
    builder.add_shape_box(anchor, hx=hx, hy=hy, hz=upper_hz)

    leader_offset = 0.0 if balanced_leader else link_hz
    leader_link = builder.add_link(xform=wp.transform(p=wp.vec3(0.0, y, drop_z - leader_offset), q=_Q_ID))
    builder.add_shape_box(leader_link, hx=hx, hy=hy, hz=link_hz)

    follower_link = builder.add_link(xform=wp.transform(p=wp.vec3(0.5, y, drop_z - link_hz), q=_Q_ID))
    builder.add_shape_box(follower_link, hx=hx, hy=hy, hz=link_hz)

    j_fixed = builder.add_joint_fixed(
        parent=-1,
        child=anchor,
        parent_xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=_Q_ID),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=_Q_ID),
        label="anchor",
    )
    add_joint = builder.add_joint_prismatic if prismatic else builder.add_joint_revolute
    j_leader = add_joint(
        parent=anchor,
        child=leader_link,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -upper_hz), q=_Q_ID),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, leader_offset), q=_Q_ID),
        label="leader",
    )
    j_follower = add_joint(
        parent=anchor,
        child=follower_link,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(0.5, 0.0, -upper_hz), q=_Q_ID),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +link_hz), q=_Q_ID),
        label="follower",
    )
    builder.add_articulation([j_fixed, j_leader, j_follower], label="mimic_articulation")
    return j_leader, j_follower


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCMimicResolution(unittest.TestCase):
    """Mimic resolution — runs with ``backend="none"`` (no GPU required)."""

    def test_mimic_resolved_into_constraints(self):
        """Resolve the scalar relation and disable the follower's independent drive."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        j_leader, j_follower = _add_joint_pair(builder, y=0.0)
        builder.add_constraint_mimic(joint0=j_follower, joint1=j_leader, coef0=0.1, coef1=-1.0)

        model = builder.finalize()
        self.assertEqual(model.constraint_mimic_count, 1)

        solver = newton.solvers.SolverUIPC(model, backend="none", dt=1.0 / 60.0)
        solver.initialize(model.state())

        constraints = solver._articulation_builder._mimic_constraints
        self.assertEqual(len(constraints), 1)
        follower_art, follower_local, leader_art, leader_local, coef0, coef1, _ = constraints[0]
        self.assertAlmostEqual(coef0, 0.1, places=6)
        self.assertAlmostEqual(coef1, -1.0, places=6)
        # Both joints are active revolute joints in the same articulation.
        self.assertIs(follower_art, leader_art)
        self.assertNotEqual(follower_local, leader_local)

    def test_disabled_mimic_skipped(self):
        """Preserve ordinary control when the mimic relation is disabled."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        j_leader, j_follower = _add_joint_pair(builder, y=0.0)
        builder.add_constraint_mimic(joint0=j_follower, joint1=j_leader, enabled=False)

        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(model, backend="none", dt=1.0 / 60.0)
        solver.initialize(model.state())

        self.assertEqual(len(solver._articulation_builder._mimic_constraints), 0)

    def test_mimic_with_inactive_joint_skipped_with_warning(self):
        """Warn and skip relations referencing unsupported joint coordinates."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        _, j_follower = _add_joint_pair(builder, y=0.0)
        # The fixed anchor joint (index 0) is not an active UIPC joint.
        builder.add_constraint_mimic(joint0=j_follower, joint1=0, coef1=1.0)

        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(model, backend="none", dt=1.0 / 60.0)
        with self.assertWarns(UserWarning):
            solver.initialize(model.state())

        self.assertEqual(len(solver._articulation_builder._mimic_constraints), 0)


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCMimicTracking(unittest.TestCase):
    """Follower physically tracks ``coef0 + coef1 * leader`` after stepping."""

    def _run(
        self,
        device,
        coef0: float,
        coef1: float,
        *,
        prismatic=False,
        joint_owned=False,
        implicit_pd=False,
        follower_target=None,
    ):
        """Drive the leader while its undriven follower maintains the authored relation."""
        wp.set_device(device)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        j_leader, j_follower = _add_joint_pair(builder, y=0.0, prismatic=prismatic)
        if joint_owned:
            builder.set_joint_mimic(j_follower, j_leader, coeffs=(coef0, coef1))
        else:
            builder.add_constraint_mimic(joint0=j_follower, joint1=j_leader, coef0=coef0, coef1=coef1)

        leader_dof = builder.joint_qd_start[j_leader]
        builder.joint_target_ke[leader_dof] = 1.0e4
        builder.joint_target_kd[leader_dof] = 1.0e2
        builder.joint_target_mode[leader_dof] = int(JointTargetMode.POSITION)
        follower_dof = builder.joint_qd_start[j_follower]
        if follower_target is not None:
            builder.joint_target_ke[follower_dof] = 1.0e4
            builder.joint_target_kd[follower_dof] = 1.0e2
            builder.joint_target_mode[follower_dof] = int(JointTargetMode.POSITION)

        model = builder.finalize(device=device)
        solver = newton.solvers.SolverUIPC(model, backend="cuda", dt=1.0 / 60.0, implicit_pd=implicit_pd)
        state_0, state_1 = model.state(), model.state()
        control = model.control()

        leader_qd_start = int(model.joint_qd_start.numpy()[j_leader])
        leader_target = 0.4
        target_q = control.joint_target_q.numpy()
        target_q[leader_qd_start] = leader_target
        if follower_target is not None:
            target_q[follower_dof] = follower_target
        control.joint_target_q.assign(target_q)

        for step in range(120):
            if implicit_pd and step == 20:
                model.joint_target_ke.assign(model.joint_target_ke.numpy() * 1.5)
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
            solver.step(state_0, state_1, control, None, dt=1.0 / 60.0)
            state_0, state_1 = state_1, state_0

        q = state_0.joint_q.numpy()
        leader_q_start = int(model.joint_q_start.numpy()[j_leader])
        follower_q_start = int(model.joint_q_start.numpy()[j_follower])
        q_leader = float(q[leader_q_start])
        q_follower = float(q[follower_q_start])

        # Leader reached its commanded target, follower tracks the coupling.
        self.assertAlmostEqual(q_leader, leader_target, delta=0.05)
        self.assertAlmostEqual(q_follower, coef0 + coef1 * q_leader, delta=0.01)

    def test_mimic_tracking_inverse(self):
        """Track an inverse revolute relation without a follower position drive."""
        for device in _CUDA_TEST_DEVICES:
            with self.subTest(device=str(device)):
                self._run(device, coef0=0.0, coef1=-1.0)

    def test_mimic_tracking_scaled_offset(self):
        """Track a scaled revolute relation with a nonzero offset."""
        for device in _CUDA_TEST_DEVICES:
            with self.subTest(device=str(device)):
                self._run(device, coef0=0.1, coef1=0.5)

    def test_prismatic_mimic_tracking(self):
        """Track a negative prismatic ratio with a nonzero offset."""
        for device in _CUDA_TEST_DEVICES:
            with self.subTest(device=str(device)):
                self._run(device, coef0=0.05, coef1=-0.5, prismatic=True)

    def test_joint_owned_mimic_tracking(self):
        """Honor the current public set_joint_mimic API during physical stepping."""
        for device in _CUDA_TEST_DEVICES:
            with self.subTest(device=str(device)):
                self._run(device, coef0=0.1, coef1=0.5, joint_owned=True)

    def test_mimic_ignores_follower_position_drive(self):
        """Keep follower position commands inactive, including after implicit-PD gain refresh."""
        for device in _CUDA_TEST_DEVICES:
            for implicit_pd in (False, True):
                with self.subTest(device=str(device), implicit_pd=implicit_pd):
                    self._run(
                        device,
                        coef0=0.1,
                        coef1=0.5,
                        joint_owned=True,
                        implicit_pd=implicit_pd,
                        follower_target=-0.3,
                    )

    def test_zero_multiplier_holds_follower_offset(self):
        """Hold the follower at its offset independently of leader motion when the multiplier is zero."""
        for device in _CUDA_TEST_DEVICES:
            with self.subTest(device=str(device)):
                self._run(device, coef0=0.15, coef1=0.0, joint_owned=True)

    def _run_follower_load(self, device, *, enabled: bool, prismatic: bool, direct_torque=False):
        """Load only the follower, using force or a gravity-loaded unbalanced hinge."""
        wp.set_device(device)
        gravity = (0.0, 0.0, 0.0) if prismatic or direct_torque else (0.0, 9.81, 0.0)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=gravity)
        j_leader, j_follower = _add_joint_pair(
            builder, y=0.0, prismatic=prismatic, balanced_leader=not (prismatic or direct_torque)
        )
        builder.add_constraint_mimic(joint0=j_follower, joint1=j_leader, coef1=-1.0, enabled=enabled)
        for dof in range(len(builder.joint_target_mode)):
            builder.joint_target_mode[dof] = int(JointTargetMode.NONE)
        follower_dof = builder.joint_qd_start[j_follower]
        if prismatic or direct_torque:
            builder.joint_target_mode[follower_dof] = int(JointTargetMode.EFFORT)

        model = builder.finalize(device=device)
        dt = 1.0 / 120.0
        solver = newton.solvers.SolverUIPC(model, backend="cuda", dt=dt)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        effort = control.joint_f.numpy()
        effort[follower_dof] = 100.0 if prismatic else (20.0 if direct_torque else 0.0)
        control.joint_f.assign(effort)
        for _ in range(30):
            solver.step(state_0, state_1, control, None, dt=dt)
            state_0, state_1 = state_1, state_0

        q = state_0.joint_q.numpy()
        return float(q[builder.joint_q_start[j_leader]]), float(q[builder.joint_q_start[j_follower]])

    @unittest.skipUnless(_CUDA_TEST_DEVICES, "CUDA is not available")
    def test_follower_load_moves_leader(self):
        """Transmit a follower load to an undriven leader only when mimic is enabled."""
        for device in _CUDA_TEST_DEVICES:
            for prismatic, direct_torque in ((False, False), (True, False), (False, True)):
                with self.subTest(device=str(device), prismatic=prismatic, direct_torque=direct_torque):
                    coupled_leader, coupled_follower = self._run_follower_load(
                        device, enabled=True, prismatic=prismatic, direct_torque=direct_torque
                    )
                    free_leader, free_follower = self._run_follower_load(
                        device, enabled=False, prismatic=prismatic, direct_torque=direct_torque
                    )
                    self.assertLess(coupled_leader, -0.01)
                    self.assertGreater(coupled_follower, 0.01)
                    self.assertAlmostEqual(coupled_follower, -coupled_leader, delta=0.005)
                    self.assertAlmostEqual(free_leader, 0.0, delta=0.001)
                    self.assertGreater(free_follower, coupled_follower)

    def test_replicated_mimics_preserve_world_indices(self):
        """Keep passive follower geometries aligned with later worlds' driven joints."""
        for device in _CUDA_TEST_DEVICES:
            with self.subTest(device=str(device)):
                wp.set_device(device)
                world = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                leader, follower = _add_joint_pair(world, y=0.0, prismatic=True)
                world.set_joint_mimic(follower, leader, coeffs=(0.1, -1.0))
                world.joint_target_mode[world.joint_qd_start[leader]] = int(JointTargetMode.POSITION)
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                builder.replicate(world, world_count=2, spacing=(0.0, 3.0, 0.0))
                model = builder.finalize(device=device)
                leaders = [leader + i * world.joint_count for i in range(2)]
                followers = [follower + i * world.joint_count for i in range(2)]
                q_start = model.joint_q_start.numpy()
                qd_start = model.joint_qd_start.numpy()
                control = model.control()
                goals = np.array([0.3, -0.2])
                targets = control.joint_target_q.numpy()
                targets[qd_start[leaders]] = goals
                control.joint_target_q.assign(targets)
                solver = newton.solvers.SolverUIPC(model, backend="cuda", dt=1.0 / 60.0)
                state, next_state = model.state(), model.state()
                for _ in range(90):
                    solver.step(state, next_state, control, dt=1.0 / 60.0)
                    state, next_state = next_state, state
                q = state.joint_q.numpy()
                self.assertTrue(np.isfinite(q).all())
                np.testing.assert_allclose(q[q_start[leaders]], goals, atol=0.01)
                np.testing.assert_allclose(q[q_start[followers]], [-0.2, 0.3], atol=0.01)


@unittest.skipUnless(_HAS_UIPC and _CUDA_TEST_DEVICES, "uipc and CUDA are required")
class TestUIPCMimicShapelessMotor(unittest.TestCase):
    def test_shapeless_motor_stays_rigid_during_repeated_open_close(self):
        """A geometry-free motor must remain rigid under torque and mixed-joint mimic loads."""
        for device in _CUDA_TEST_DEVICES:
            with self.subTest(device=str(device)), tempfile.TemporaryDirectory() as workspace:
                wp.set_device(device)
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                anchor = builder.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
                builder.add_shape_box(anchor, hx=0.1, hy=0.1, hz=0.1)
                root = builder.add_joint_revolute(
                    parent=-1, child=anchor, parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity())
                )
                # EX001-like virtual motor: no shape, tiny spatial inertia, reflected joint inertia.
                motor = builder.add_link(
                    xform=wp.transform((0.0, 0.2, 1.0), wp.quat_identity()),
                    mass=0.1,
                    inertia=wp.mat33(np.eye(3) * 1.6667e-10),
                )
                leader = builder.add_joint_revolute(
                    parent=anchor,
                    child=motor,
                    axis=(0.0, 1.0, 0.0),
                    parent_xform=wp.transform((0.0, 0.2, 0.0), wp.quat_identity()),
                )
                followers = []
                scales = (-0.00015 * 180.0 / np.pi, 0.00015 * 180.0 / np.pi)
                for side, scale in zip((-1, 1), scales, strict=True):
                    offset = (0.2, side * 0.1, 0.0)
                    finger = builder.add_link(xform=wp.transform((offset[0], offset[1], 1.0), wp.quat_identity()))
                    builder.add_shape_box(finger, hx=0.01, hy=0.01, hz=0.05)
                    follower = builder.add_joint_prismatic(
                        parent=anchor,
                        child=finger,
                        axis=(0.0, 1.0, 0.0),
                        parent_xform=wp.transform(offset, wp.quat_identity()),
                    )
                    builder.set_joint_mimic(follower, leader, coeffs=(0.0, scale))
                    followers.append(follower)
                builder.add_articulation([root, leader, *followers])
                leader_dof = builder.joint_qd_start[leader]
                builder.joint_target_mode[leader_dof] = int(JointTargetMode.EFFORT)
                builder.joint_armature[leader_dof] = 1.0
                model = builder.finalize(device=device)
                dt = 1.0 / 120.0
                solver = newton.solvers.SolverUIPC(model, dt=dt, workspace=workspace)
                state, next_state = model.state(), model.state()
                control = model.control()
                q_index = builder.joint_q_start[leader]
                follower_indices = [builder.joint_q_start[j] for j in followers]
                for step in range(1200):
                    target = 5.717175 if (step // 150) % 2 else 0.0
                    q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
                    force = control.joint_f.numpy()
                    force[leader_dof] = np.clip(4000.0 * (target - q[q_index]) - 50.0 * qd[leader_dof], -500.0, 500.0)
                    control.joint_f.assign(force)
                    solver.step(state, next_state, control, dt=dt)
                    state, next_state = next_state, state
                    self.assertTrue(np.isfinite(state.body_q.numpy()).all(), f"step {step}")
                    transform = np.asarray(uipc.view(solver.mapping.body_geo_slots[motor].geometry().transforms()))[0]
                    np.testing.assert_allclose(
                        np.linalg.svd(transform[:3, :3], compute_uv=False), 1.0, atol=0.01, err_msg=f"step {step}"
                    )
                    if step % 150 == 149:
                        q = state.joint_q.numpy()
                        self.assertAlmostEqual(float(q[q_index]), target, delta=0.05)
                        np.testing.assert_allclose(q[follower_indices], np.asarray(scales) * target, atol=5e-4)


if __name__ == "__main__":
    unittest.main()
