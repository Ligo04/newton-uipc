# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop the 1920-body cube wall onto pinned cloth from libuipc sample 93.

Preserve the source cube mesh, 100 MPa ABD rigidity, density, wall layout,
cloth materials, strain_rate=10000, fixed cloth sides, and 0.01 s time step.
The cube volume meshes are represented by closed rigid collision surfaces
through Newton's AffineBody interface.

Success criteria:
- The wall falls under gravity and interacts with the cloth.
- Body and particle states remain finite and inside the scene bounds.
- The cloth side vertices stay within 1 mm of their authored positions,
  and cloth stays above the floor.

Runs:
- `python -m newton.examples cube-wall-cloth`
"""

import numpy as np
import uipc
import warp as wp
from uipc.constitution import DiscreteShellBending, ElasticModuli2D, StrainLimitingBaraffWitkinShell
from uipc.geometry import label_surface
from uipc.geometry import trimesh as uipc_trimesh

import newton
import newton.examples
from newton.examples.uipc._sample_scenes import asset_directory, create_solver, read_mesh
from newton.solvers import SolverUIPC


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = self.sim_dt = 0.01
        self.sim_substeps = 1
        assets = asset_directory(args.asset_dir)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=wp.vec3(0.0, -9.8, 0.0))
        SolverUIPC.register_custom_attributes(builder)
        vertices, faces = read_mesh(assets / "tetmesh/stiff_cube.msh", surface=True)
        self.cube_vertices = np.ascontiguousarray(vertices * 0.3, dtype=np.float32)
        self.cube_faces = faces
        mesh = newton.Mesh(vertices=self.cube_vertices, indices=faces.reshape(-1))
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
        colors = ((0.3, 0.6, 0.9), (0.95, 0.55, 0.2))
        for layer in range(15):
            for i in range(8):
                for j in range(8):
                    x, z = (i - 3.5) * 0.15, (j - 3.5) * 0.15
                    for group, base in enumerate((1.05, 1.2)):
                        body = builder.add_body(
                            xform=wp.transform(wp.vec3(x, base + 0.3 * layer, z), wp.quat_identity()),
                            label=f"cube_{layer}_{i}_{j}_{group}",
                        )
                        builder.add_shape_mesh(body, mesh=mesh, cfg=shape_cfg, color=colors[group])

        cloth_vertices, cloth_faces = read_mesh(assets / "trimesh/cloth_high.obj")
        self.cloth_faces = cloth_faces
        reference = uipc_trimesh(
            np.ascontiguousarray(cloth_vertices, dtype=np.float64),
            np.ascontiguousarray(cloth_faces, dtype=np.int32),
        )
        label_surface(reference)
        poisson, radius = 0.49, 0.001
        StrainLimitingBaraffWitkinShell().apply_to(
            reference,
            stretch_moduli=ElasticModuli2D.youngs_poisson(5.0e4, poisson),
            shear_moduli=ElasticModuli2D.youngs_poisson(10.0, poisson),
            mass_density=200.0,
            thickness=radius,
            strain_rate=10000.0,
        )
        DiscreteShellBending().apply_to(reference, 3.0e4, poisson)
        shear = float(uipc.view(reference.triangles().find("mu"))[0])
        stretch = float(uipc.view(reference.triangles().find("lambda"))[0])
        bending = float(uipc.view(reference.edges().find("bending_stiffness"))[0])
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.525, 0.0),
            rot=wp.quat_identity(),
            scale=1.5,
            vel=wp.vec3(0.0),
            vertices=cloth_vertices,
            indices=cloth_faces.reshape(-1),
            density=200.0 * 2.0 * radius,
            tri_ke=shear,
            tri_ka=stretch / (2.0 * radius) - 2.0 * shear,
            tri_kd=0.0,
            edge_ke=bending,
            edge_kd=0.0,
            particle_radius=radius,
            color=(0.35, 0.8, 0.55),
            label="cloth",
            custom_attributes_triangles={"uipc:tri_strain_rate": [10000.0] * len(cloth_faces)},
        )
        points = np.asarray(builder.particle_q)
        self.fixed_particles = np.flatnonzero(np.abs(points[:, 0]) > 1.5 - 1.0e-4)
        for index in self.fixed_particles:
            builder.particle_mass[index] = 0.0
        builder.add_ground_plane(height=-1.0)
        self.model = builder.finalize()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.initial_particle_q = self.model.particle_q.numpy()
        self.initial_body_q = self.model.body_q.numpy()
        self.control = self.model.control()
        self.solver = create_solver(
            self.model,
            dt=self.sim_dt,
            workspace=args.workspace,
            fem_preconditioner=args.fem_preconditioner,
        )
        self.solver.configure_scene({"linear_system": {"use_cuda_graph": args.use_graph}})
        self.solver.initialize(self.state_0)
        self.solver.set_animator_substep(1)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(8.0, 4.0, 8.0), pitch=-10.0, yaw=-135.0)

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
        particles = self.state_0.particle_q.numpy()
        poses = self.state_0.body_q.numpy()
        assert np.isfinite(particles).all() and np.isfinite(poses).all(), "Non-finite scene positions"
        assert np.isfinite(self.state_0.particle_qd.numpy()).all(), "Non-finite cloth velocities"
        assert np.isfinite(self.state_0.body_qd.numpy()).all(), "Non-finite body velocities"
        assert float(np.abs(particles).max()) < 20.0 and float(np.abs(poses[:, :3]).max()) < 20.0
        np.testing.assert_allclose(
            particles[self.fixed_particles], self.initial_particle_q[self.fixed_particles], atol=1.0e-3, rtol=0.0
        )
        assert float(particles[:, 1].min()) > -1.05, "Cloth penetrated the floor"

    def test_final(self):
        self.test_post_step()
        if self.sim_time >= 0.05:
            displacement = self.state_0.body_q.numpy()[:, 1] - self.initial_body_q[:, 1]
            assert float(displacement.mean()) < -0.001, "Cube wall did not fall"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=100)
        parser.add_argument("--asset-dir", help="Sample sim_data directory; otherwise download the pinned assets.")
        parser.add_argument("--fem-preconditioner", choices=("mas", "diag"), default="mas")
        parser.add_argument("--use-graph", type=int, choices=(0, 1, 2), default=2, help="UIPC PCG CUDA Graph mode.")
        parser.add_argument("--workspace", default="/tmp/newton_uipc/cube_wall_cloth")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
