# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smash a stepped ring of bricks with a rolling sphere using VBD or UIPC.

The 28k and 110k presets follow Steven Bobyn's three-avbd Brick Ring scenes:
https://github.com/sbobyn/three-avbd/blob/afc1442e64e4b36f2d49be70bf35ea0c9b85cf6c/src/avbd3d/bench-scenes.ts
The source's MIT notice is in newton/licenses/three-avbd-LICENSE.txt.
The presets contain 28,112 and 110,440 bricks respectively, plus one sphere.
Bricks are 1 x 0.5 x 0.5 m, with density 1 kg/m^3.
Each frame advances 1/60 s, using one UIPC step or four VBD substeps by default.
UIPC's --substeps option controls animation target interpolation.
Friction is disabled for both solvers; UIPC uses semi-implicit Newton updates.
Both solvers start with 2 mm of vertical clearance between courses and above
the floor so that UIPC initializes without touching surfaces. Solver settings
and collision representations differ from three-avbd; this is a scene port.
UIPC initialization sanity checking is opt-in because older libuipc wheels use
quadratic host memory when expanding shared instances. Enable --sanity-check
with a build containing the fix from spiriMirror/libuipc#508.
Runtime collision detection remains active with either setting.
The GL view uses deterministic pastel colors, a warm checkerboard floor, and soft lighting.

Success criteria:
- The selected preset forms a complete ring with a stepped outer wall.
- The sphere rolls into the near wall and displaces bricks through contact.
  UIPC can lift the sphere up the stepped exterior; the solvers need not produce the same breach.
- Rigid transforms and velocities stay finite, and bodies remain above the floor.

Runs:
- `python -m newton.examples brick_ring --size 28k --solver vbd`
- `python -m newton.examples brick_ring --size 110k --solver vbd`
- `python -m newton.examples brick_ring --size 28k --solver uipc`
- `python -m newton.examples brick_ring --size 110k --solver uipc`
- `python -m newton.examples brick_ring --size 28k --solver uipc --sanity-check`
"""

import math
import tempfile

import numpy as np
import warp as wp

import newton
import newton.examples

_PRESETS = {"28k": (40.0, 20, 2, 10), "110k": (80.0, 40, 4, 10)}
_CLEARANCE = 0.002
_FLOOR_HEIGHT = 0.5
_BALL_SPEED = 30.0
_BRICK_COLORS = (
    (0.80, 0.59, 0.64),
    (0.86, 0.73, 0.49),
    (0.72, 0.79, 0.52),
    (0.58, 0.75, 0.65),
    (0.57, 0.69, 0.81),
    (0.72, 0.61, 0.81),
    (0.82, 0.70, 0.78),
)


class Example:
    def __init__(self, viewer, args):
        substeps = args.substeps
        if substeps is None:
            substeps = 1 if args.solver == "uipc" else 4
        if substeps < 1 or args.vbd_iterations < 1:
            raise ValueError("Substeps and VBD iterations must be positive.")

        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = 1 if args.solver == "uipc" else substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.solver_type = args.solver
        self.radius, courses, tier, rows = _PRESETS[args.size]
        self.ball_radius = courses / 5.0

        builder = newton.ModelBuilder()
        # The 8 m sphere weighs over 20 tonnes even at the source density of 10 kg/m^3.
        cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.0, ke=1.0e7, kd=1.0, gap=0.05)
        builder.add_ground_plane(height=_FLOOR_HEIGHT, cfg=cfg, color=(0.80, 0.78, 0.72))
        color_groups = [[] for _ in range(12)]
        color_rng = np.random.default_rng(42)

        # Running bond, with one fewer radial row every tier of courses.
        for course in range(courses):
            depth = max(1, rows - course // tier)
            for row in range(depth):
                ring_radius = self.radius + 0.25 + 0.52 * row
                count = math.floor(2.0 * math.pi * (ring_radius - 0.25) / 1.02)
                for index in range(count):
                    angle = (index + (course % 2) * 0.5) * 2.0 * math.pi / count
                    body = builder.add_body(
                        xform=wp.transform(
                            wp.vec3(
                                ring_radius * math.cos(angle),
                                ring_radius * math.sin(angle),
                                _FLOOR_HEIGHT + 0.25 + _CLEARANCE + course * (0.5 + _CLEARANCE),
                            ),
                            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle + math.pi / 2.0),
                        ),
                    )
                    color = _BRICK_COLORS[int(color_rng.integers(len(_BRICK_COLORS)))]
                    builder.add_shape_box(body, hx=0.5, hy=0.25, hz=0.25, cfg=cfg, color=color)
                    # Odd cycles need a third tangential color at the seam.
                    tangent_color = 2 if count % 2 and index == count - 1 else index % 2
                    group = ((course % 2) * 2 + row % 2) * 3 + tangent_color
                    color_groups[group].append(body)

        self.brick_count = builder.body_count
        self.ball_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(
                    0.0,
                    -(self.radius + 0.52 * rows + self.ball_radius + 2.0),
                    _FLOOR_HEIGHT + self.ball_radius + _CLEARANCE,
                ),
                wp.quat_identity(),
            ),
            label="wrecking_ball",
        )
        ball_cfg = cfg.copy()
        ball_cfg.density = 10.0
        builder.add_shape_sphere(self.ball_body, radius=self.ball_radius, cfg=ball_cfg, color=(0.60, 0.67, 0.78))
        builder.body_qd[self.ball_body] = wp.spatial_vector(
            0.0, _BALL_SPEED, 0.0, -_BALL_SPEED / self.ball_radius, 0.0, 0.0
        )
        if self.solver_type == "vbd":
            # Joint-only automatic coloring puts this entire contact network in one group.
            builder.body_color_groups = [np.asarray(group, dtype=np.int32) for group in color_groups if group]
            builder.body_color_groups.append(np.asarray([self.ball_body], dtype=np.int32))
        self.model = builder.finalize()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.initial_positions = self.state_0.body_q.numpy()[:, :3].copy()

        self.collision_pipeline = None
        self.contacts = None
        if self.solver_type == "vbd":
            # Size storage linearly; the default all-pairs broad phase is unsuitable here.
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                broad_phase="sap",
                shape_pairs_max=16 * self.model.shape_count,
                rigid_contact_max=64 * self.model.shape_count,
            )
            self.contacts = self.collision_pipeline.contacts()
            self.solver = newton.solvers.SolverVBD(
                self.model,
                iterations=args.vbd_iterations,
                rigid_body_contact_buffer_size=1024,
            )
        else:
            if not self.model.device.is_cuda:
                raise ValueError("The UIPC solver requires a CUDA device.")
            self.solver = newton.solvers.SolverUIPC(
                self.model,
                dt=self.sim_dt,
                workspace=tempfile.mkdtemp(prefix=f"newton-brick-ring-{args.size}-"),
                enable_soft_position_constraint=False,
            )
            self.solver.set_contact(True, d_hat=0.001)
            # Older wheels need the shared-instance memory fix before enabling this scan.
            self.solver.configure_scene(
                {
                    "sanity_check": {"enable": int(args.sanity_check)},
                    "newton": {"semi_implicit": {"enable": 1}},
                    "contact": {"friction": {"enable": 0}},
                    "linear_system": {"tol_rate": 1.0e-4},
                    "line_search": {"max_iter": 24},
                }
            )
            self.solver.initialize(self.state_0)
            self.solver.set_animator_substep(substeps)

        print(f"Brick Ring ({args.size}): {self.brick_count:,} bricks + 1 sphere, solver={self.solver_type}")
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.0, -3.0 * self.radius, 1.55 * self.radius),
            pitch=-25.0,
            yaw=102.0,
        )
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is not None:
            renderer.sky_upper = (0.94, 0.94, 0.92)
            renderer.sky_lower = (0.87, 0.86, 0.82)
            renderer.ambient_sky = (0.80, 0.80, 0.78)
            renderer.ambient_ground = (0.60, 0.59, 0.56)
            renderer.exposure = 1.3
            renderer.diffuse_scale = 0.75
            renderer.specular_scale = 0.12
            renderer.spotlight_enabled = False
            renderer.shadow_radius = 3.0
            renderer.shadow_extents = 4.0 * self.radius
            renderer.fog_range = (2.8 * self.radius, 10.0 * self.radius)
            renderer.checker_scale = 1.0 / (0.075 * self.radius)
            # Keep the horizontal framing consistent for square and landscape captures.
            self.viewer.camera.fov = math.degrees(
                2.0 * math.atan(math.tan(math.radians(27.5)) * self.viewer.camera.height / self.viewer.camera.width)
            )
            self.viewer.camera.look_at((0.0, -0.6 * self.radius, 0.5 * courses * (0.5 + _CLEARANCE)))
        self.graph = None
        if self.solver_type == "vbd" and self.model.device.is_cuda and not wp.config.verify_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.collision_pipeline is not None:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            # Keep graph input/output arrays stable with odd substep counts too.
            if self.sim_substeps % 2 == 1 and substep == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Check floor support and keep the distant wall stable before impact reaches it."""
        poses = self.state_0.body_q.numpy()
        assert np.isfinite(poses).all(), "Non-finite body transforms"
        assert np.isfinite(self.state_0.body_qd.numpy()).all(), "Non-finite body velocities"
        assert float(poses[: self.brick_count, 2].min()) > _FLOOR_HEIGHT - 0.1, "A brick fell through the floor"
        assert poses[self.ball_body, 2] > _FLOOR_HEIGHT + self.ball_radius - 0.1, "The sphere fell through the floor"
        if self.sim_time < 1.0:
            distant = self.initial_positions[: self.brick_count, 1] > 0.5 * self.radius
            delta = poses[: self.brick_count, :2][distant] - self.initial_positions[: self.brick_count, :2][distant]
            assert float(np.linalg.norm(delta, axis=1).max()) < 0.5, (
                "The distant wall moved before the sphere reached it"
            )

    def test_final(self):
        """Check the rolling sphere transfers motion into the brick ring through contact."""
        self.test_post_step()
        poses = self.state_0.body_q.numpy()
        if self.sim_time >= 0.1:
            assert poses[self.ball_body, 1] > self.initial_positions[self.ball_body, 1] + 0.1, "The sphere did not roll"
        if self.sim_time >= 0.5:
            displacement = poses[: self.brick_count, :2] - self.initial_positions[: self.brick_count, :2]
            assert float(np.linalg.norm(displacement, axis=1).max()) > 0.05, "The sphere did not displace any bricks"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--size", choices=tuple(_PRESETS), default="28k", help="Brick-ring scale preset.")
        parser.add_argument("--solver", choices=("vbd", "uipc"), default="vbd", help="Rigid-body solver.")
        parser.add_argument(
            "--sanity-check",
            action="store_true",
            help="Enable UIPC initialization sanity checks; requires libuipc's shared-instance memory fix.",
        )
        parser.add_argument(
            "--substeps",
            type=int,
            default=None,
            help="UIPC animation substeps (default 1) or VBD physics substeps (default 4) per 60 Hz frame.",
        )
        parser.add_argument("--vbd-iterations", type=int, default=10, help="VBD iterations per physics substep.")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
