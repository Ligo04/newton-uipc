# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Port libuipc-samples example 6 with its original contact-coupled rigid chain."""

import json

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.uipc._sample_scenes import asset_directory, create_solver, read_mesh


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = self.sim_dt = 0.02
        self.sim_substeps = 1
        assets = asset_directory(args.asset_dir)
        descriptions = json.loads((assets / "wrecking_ball.json").read_text())
        meshes = {}
        for name in ("cube.msh", "ball.msh", "link.msh"):
            vertices, faces = read_mesh(assets / "tetmesh" / name, surface=True)
            meshes[name] = newton.Mesh(vertices=vertices, indices=faces.reshape(-1))

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        self.fixed_bodies = []
        self.ball_body = -1
        colors = {"cube.msh": (0.55, 0.7, 0.9), "link.msh": (0.8, 0.7, 0.4), "ball.msh": (0.85, 0.25, 0.15)}
        for index, desc in enumerate(descriptions):
            rotation = np.deg2rad(desc.get("rotation", (0.0, 0.0, 0.0)))
            q = (
                wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(rotation[2]))
                * wp.quat_from_axis_angle(wp.vec3(0, 1, 0), float(rotation[1]))
                * wp.quat_from_axis_angle(wp.vec3(1, 0, 0), float(rotation[0]))
            )
            fixed = bool(desc.get("is_dof_fixed", False))
            body = builder.add_body(
                xform=wp.transform(wp.vec3(*desc["position"]), q),
                is_kinematic=fixed,
                label=f"{desc['mesh'].removesuffix('.msh')}_{index}",
            )
            cfg = newton.ModelBuilder.ShapeConfig(density=float(desc.get("density", 1000.0)))
            builder.add_shape_mesh(body, mesh=meshes[desc["mesh"]], cfg=cfg, color=colors[desc["mesh"]])
            if fixed:
                self.fixed_bodies.append(body)
            if desc["mesh"] == "ball.msh":
                self.ball_body = body
        builder.add_ground_plane(height=-1.0)
        self.model = builder.finalize()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.initial_body_q = self.state_0.body_q.numpy().copy()
        self.control = self.model.control()
        self.solver = create_solver(self.model, dt=self.sim_dt, workspace="/tmp/newton_uipc/rigid_wrecking_balls")
        self.solver.initialize(self.state_0)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(34.0, 24.0, 30.0), pitch=-25.0, yaw=-130.0)
        self.viewer._paused = True

    def step(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, dt=self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        poses = self.state_0.body_q.numpy()
        assert np.isfinite(poses).all() and np.isfinite(self.state_0.body_qd.numpy()).all(), "Non-finite rigid state"
        assert float(np.abs(poses[:, :3]).max()) < 100.0, "Rigid scene escaped its expected bounds"
        np.testing.assert_allclose(poses[self.fixed_bodies], self.initial_body_q[self.fixed_bodies], atol=2.0e-5)

    def test_final(self):
        self.test_post_step()
        if self.sim_time >= 0.1:
            displacement = self.state_0.body_q.numpy()[self.ball_body, :3] - self.initial_body_q[self.ball_body, :3]
            assert float(np.linalg.norm(displacement)) > 0.01, "Wrecking ball did not move"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=120)
        parser.add_argument("--asset-dir", help="Sample sim_data directory; otherwise download the pinned assets.")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
