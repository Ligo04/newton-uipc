# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Grasp a deformable Lulu toy with a Franka Panda using UIPC and GPU IK.

Load the bundled Lulu USD tetrahedral mesh generated with PyTetWild 0.4.2.
It contains 7,221 vertices, 29,381 tetrahedra, and 8,044 boundary triangles.
Scale the toy to fit the gripper and rest it on its back for a stable grasp.
UIPC simulates the articulated robot and the Stable Neo-Hookean soft body,
including their frictional contact. The generated boundary has one vertex
joining two surface fans. Some local tetrahedra invert under contact.

Success criteria:
- A pink Lulu toy rests on the table in front of the Franka arm.
- The gripper approaches Lulu, grasps its head, lifts it, and releases it
  back onto the table.
- Lulu deforms under contact and remains finite and within the workspace.

Runs:
- `python -m newton.examples uipc_softbody_franka_lulu`
"""

from __future__ import annotations

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
from newton.examples.uipc.softbody.example_uipc_softbody_franka import Example as FrankaExample

LULU_SCALE = 0.3


class Example(FrankaExample):
    workspace = "/tmp/newton_uipc/uipc_softbody_franka_lulu"

    def __init__(self, viewer, args=None):
        super().__init__(viewer, args)
        self.viewer._paused = bool(getattr(args, "paused", False))
        self.viewer.set_camera(wp.vec3(0.28, -0.85, 0.40), -6.0, 127.0)

    def configure_solver(self):
        """Give the hand and fingers higher friction against Lulu."""
        super().configure_solver()
        self.solver.configure_scene(
            {
                "contact": {"eps_velocity": 1.0e-3, "eps_velocity_relative": 0.0},
                "newton": {"semi_implicit": {"enable": 0}},
            }
        )
        gripper_bodies = (
            self.endeffector_id,
            next(i for i, label in enumerate(self.model.body_label) if label.endswith("/fr3_leftfinger")),
            next(i for i, label in enumerate(self.model.body_label) if label.endswith("/fr3_rightfinger")),
        )

        def setup_contacts(tabular, world_index, ground, environment, robot, actor):
            gripper = tabular.create(f"gripper_{world_index}")
            tabular.insert(gripper, actor, 0.5, -1.0, True)
            tabular.insert(gripper, ground, 0.5, -1.0, False)
            tabular.insert(gripper, environment, 0.5, -1.0, False)
            tabular.insert(gripper, robot, 0.5, -1.0, False)
            tabular.insert(gripper, gripper, 0.5, -1.0, False)
            return dict.fromkeys(gripper_bodies, gripper)

        self.solver.configure_contact_tabular(setup_contacts)

    def create_soft_body(self):
        """Load Lulu from USD and rest its back above the tabletop."""
        stage = Usd.Stage.Open(newton.examples.get_asset("toy_lulu_tet.usd"))
        tetmesh = newton.TetMesh.create_from_usd(stage.GetPrimAtPath("/World/Lulu"), compat_namespaces=())
        rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * wp.pi)
        rotation_matrix = np.asarray(wp.quat_to_matrix(rotation)).reshape(3, 3)
        rotated_vertices = tetmesh.vertices @ rotation_matrix.T
        bottom = float(rotated_vertices[:, 2].min()) * LULU_SCALE
        self.particle_radius = 0.002
        youngs_modulus = 30_000.0
        poisson_ratio = 0.3
        self.scene.add_soft_mesh(
            pos=wp.vec3(0.0, -0.5, 0.2 - bottom + self.particle_radius),
            rot=rotation,
            scale=LULU_SCALE,
            vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=tetmesh,
            density=225.0,
            k_mu=youngs_modulus / (2.0 * (1.0 + poisson_ratio)),
            k_lambda=youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)),
            k_damp=0,
            particle_radius=self.particle_radius,
            color=wp.vec3(0.9, 0.5, 0.58),
            opacity=0.85,
            label="lulu",
        )

    def create_articulation(self, builder):
        """Align the jaws with Lulu's head for the pinch and placement poses."""
        super().create_articulation(builder)
        self.robot_key_poses[:, 1:3] = (0.0, -0.48)
        # Cancel the hand's mounting rotation so the jaws close along world X.
        self.robot_key_poses[:, 4:8] = wp.quat_rpy(wp.pi, 0.0, 0.25 * wp.pi)

    def set_up_ik(self):
        """Start above Lulu with open fingers to avoid sweeping through the toy."""
        super().set_up_ik()
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=64)
        joint_q = self.ik_joint_q.numpy()
        joint_q[0, self.finger_idx0] = 0.04
        joint_q[0, self.finger_idx1] = 0.04
        self.ik_joint_q.assign(joint_q)
        self.model.joint_q.assign(joint_q.ravel())
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        for state in (self.state_0, self.state_1):
            state.joint_q.assign(joint_q.ravel())
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)

    def test_post_step(self):
        """Check that Lulu clears the table at both ends of the hold phase."""
        hold_start = float(self.robot_key_poses_time[3])
        hold_end = float(self.robot_key_poses_time[4])
        if (
            hold_start <= self.sim_time < hold_start + self.frame_dt
            or hold_end - self.frame_dt <= self.sim_time < hold_end
        ):
            newton.examples.test_particle_state(
                self.state_0,
                "Lulu is held above the table",
                lambda q, qd: q[2] > 0.25,
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=1000)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
