# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Port libuipc-samples example 88: two FEM bunnies falling onto free cloth.

The source script executes E=1e4 Pa for the bunnies, despite its E=1e7
module description. This port preserves the executable parameters.

Success criteria:
- Two FEM bunnies fall onto the free cloth and deform through contact.
- Particle positions and velocities remain finite, and the FEM mesh stays above the floor.

Runs:
- `python -m newton.examples uipc_stiff_gipc_benchmark`
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
        vertices, tets = read_mesh(assets / "tetmesh/bunny2.msh")
        young, poisson = 1.0e4, 0.49
        mu = young / (2.0 * (1.0 + poisson))
        lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        for height, color, label in (
            (0.5, (0.9, 0.45, 0.2), "upper_bunny"),
            (-0.65, (0.3, 0.75, 0.45), "lower_bunny"),
        ):
            builder.add_soft_mesh(
                pos=wp.vec3(0.0, height, 0.0),
                rot=wp.quat_identity(),
                scale=0.2,
                vel=wp.vec3(0.0),
                vertices=vertices,
                indices=tets.reshape(-1),
                density=1000.0,
                k_mu=mu,
                k_lambda=lam,
                k_damp=0.0,
                particle_radius=0.0,
                color=color,
                label=label,
            )

        cloth_vertices, faces = read_mesh(assets / "trimesh/cloth_high.obj")
        radius = 0.001
        reference_cloth = uipc_trimesh(
            np.ascontiguousarray(cloth_vertices, dtype=np.float64),
            np.ascontiguousarray(faces, dtype=np.int32),
        )
        label_surface(reference_cloth)
        StrainLimitingBaraffWitkinShell().apply_to(
            reference_cloth,
            stretch_moduli=ElasticModuli2D.youngs_poisson(5.0e4, poisson),
            shear_moduli=ElasticModuli2D.youngs_poisson(10.0, poisson),
            mass_density=200.0,
            thickness=radius,
            strain_rate=100.0,
        )
        DiscreteShellBending().apply_to(reference_cloth, 3.0e4, poisson)
        shear_mu = float(uipc.view(reference_cloth.triangles().find("mu"))[0])
        stretch = float(uipc.view(reference_cloth.triangles().find("lambda"))[0])
        # Invert Newton's cloth mapping using this UIPC build's coefficients.
        stretch_lame = stretch / (2.0 * radius) - 2.0 * shear_mu
        bending = float(uipc.view(reference_cloth.edges().find("bending_stiffness"))[0])
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=cloth_vertices,
            indices=faces.reshape(-1),
            density=200.0 * (2.0 * radius),
            tri_ke=shear_mu,
            tri_ka=stretch_lame,
            tri_kd=0.0,
            edge_ke=bending,
            edge_kd=0.0,
            particle_radius=radius,
            color=(0.3, 0.55, 0.9),
            label="cloth",
        )
        builder.add_ground_plane(height=-1.0)
        self.model = builder.finalize()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.solver = create_solver(
            self.model,
            dt=self.sim_dt,
            workspace="/tmp/newton_uipc/stiff_gipc_benchmark",
            fem_preconditioner=args.fem_preconditioner,
        )
        self.solver.initialize(self.state_0)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.8, 0.7, 2.8), pitch=-10.0, yaw=-135.0)
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
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        assert np.isfinite(positions).all() and np.isfinite(velocities).all(), "Non-finite FEM state"
        assert float(np.abs(positions).max()) < 10.0, "FEM scene escaped its expected bounds"

    def test_final(self):
        self.test_post_step()
        assert float(self.state_0.particle_q.numpy()[:, 1].min()) > -1.05, "FEM mesh penetrated the floor"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=250)
        parser.add_argument("--asset-dir", help="Sample sim_data directory; otherwise download the pinned assets.")
        parser.add_argument("--fem-preconditioner", choices=("mas", "diag"), default="mas")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
