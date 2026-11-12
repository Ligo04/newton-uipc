# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the UIPC centerline approximation of Newton rod joints."""

from __future__ import annotations

import importlib.util
import unittest
from contextlib import nullcontext

import numpy as np
import warp as wp

import newton
from newton._src.exceptions import NewtonDeprecationWarning, NewtonWarning
from newton.sensors import SensorContact
from newton.tests._usd_deformable_test_utils import _add_cable_curve, _deformable_stage
from newton.tests.unittest_utils import USD_AVAILABLE, get_selected_cuda_test_devices

_HAS_UIPC = importlib.util.find_spec("uipc") is not None
_CUDA_TEST_DEVICES = get_selected_cuda_test_devices(mode="basic")

if _HAS_UIPC:
    from uipc import view

if _HAS_UIPC and USD_AVAILABLE:
    import newton_usd_schemas  # noqa: F401
    from pxr import Sdf


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCRod(unittest.TestCase):
    def _model(self, device="cpu", *, origin="start", fixed=False, colliders=False, ground_plane=False):
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        if colliders:
            if ground_plane:
                builder.add_ground_plane()
            else:
                builder.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.05)), hx=3.0, hy=3.0, hz=0.05)
            body = builder.add_link(xform=wp.transform(wp.vec3(2.0, 0.0, 1.0)))
            builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
            builder.add_articulation([builder.add_joint_free(child=body)])
        with (
            self.assertWarnsRegex(NewtonDeprecationWarning, 'body_frame_origin="start"')
            if origin == "start"
            else nullcontext()
        ):
            bodies, joints = builder.add_rod(
                rod=newton.Rod(
                    [(0.0, 0.0, 1.0), (0.2, 0.0, 1.0), (0.4, 0.0, 1.0), (0.6, 0.0, 1.0)],
                    radius=0.01,
                    youngs_modulus=1.0e4,
                    poissons_ratio=0.3,
                ),
                body_frame_origin=origin,
            )
        if fixed:
            builder.body_flags[bodies[0]] = newton.BodyFlags.KINEMATIC
        return builder.finalize(device), bodies, joints

    def _initialize(self, model, state=None, *, backend="none", contact=False):
        solver = newton.solvers.SolverUIPC(model, backend=backend, dt=0.002)
        if contact:
            solver.set_contact(True)
        with self.assertWarnsRegex(NewtonWarning, "ROD.*centerline"):
            solver.initialize(model.state() if state is None else state)
        return solver

    @unittest.skipUnless(_CUDA_TEST_DEVICES, "CUDA device unavailable")
    def test_cuda_masked_reset_preserves_global_rod_nodes(self):
        builder = newton.ModelBuilder()
        points = [(0, 0, 1), (0.2, 0, 1), (0.4, 0, 1)]
        builder.add_rod(rod=newton.Rod(points, radius=0.01), label="global")
        builder.begin_world()
        builder.add_rod(rod=newton.Rod(points, radius=0.01), label="local")
        builder.end_world()
        model = builder.finalize(_CUDA_TEST_DEVICES[0])
        state = model.state()
        solver = self._initialize(model, state, backend="cuda")
        nodes = solver.get_rod_state()
        before = nodes.positions.numpy().copy()
        poses = state.body_q.numpy()
        poses[:, 2] += 0.5
        state.body_q.assign(poses)
        solver.reset(
            state,
            world_mask=wp.array([True, False], dtype=wp.bool, device=model.device),
            flags=newton.StateFlags.BODY_Q,
        )
        starts = nodes.curve_vertex_start.numpy()
        after = nodes.positions.numpy()
        for index, world in enumerate(nodes.world_indices.numpy()):
            bounds = slice(starts[index], starts[index + 1])
            if world == -1:
                np.testing.assert_array_equal(after[bounds], before[bounds])
            else:
                np.testing.assert_allclose(after[bounds], before[bounds] + [0, 0, 0.5], atol=1e-6)

    def test_rod_uses_fem_constitutions_and_preserves_mass(self):
        model, bodies, _ = self._model()
        solver = self._initialize(model)
        self.assertFalse(set(bodies).intersection(solver.mapping.body_geo_slots))
        self.assertEqual(len(solver.mapping.rod_geo_slots), 1)
        geo = solver.mapping.rod_geo_slots[0].geometry()
        self.assertEqual(int(np.asarray(view(geo.meta().find("constitution_uid"))).reshape(-1)[0]), 12)
        extra_uids = geo.meta().to_json()["extra_constitution_uids"]["attribute"]["values"]
        self.assertIn(15, np.asarray(extra_uids).reshape(-1))
        np.testing.assert_allclose(view(geo.edges().find("kappa")), 1.0e4, rtol=1.0e-5)
        bending = np.asarray(view(geo.vertices().find("bending_stiffness"))).reshape(-1)
        np.testing.assert_allclose(bending[1:-1], 1.0e4, rtol=1.0e-5)
        masses = np.asarray(view(geo.vertices().find("mass"))).reshape(-1)
        self.assertAlmostEqual(float(masses.sum()), float(model.body_mass.numpy()[bodies].sum()), places=7)
        np.testing.assert_allclose(
            np.asarray(view(geo.positions())).reshape(-1, 3),
            [(0.0, 0.0, 1.0), (0.2, 0.0, 1.0), (0.4, 0.0, 1.0), (0.6, 0.0, 1.0)],
            atol=1.0e-7,
        )

    def test_fixed_segment_fixes_its_two_nodes(self):
        model, _, _ = self._model(fixed=True)
        solver = self._initialize(model)
        fixed = np.asarray(view(solver.mapping.rod_geo_slots[0].geometry().vertices().find("is_fixed"))).reshape(-1)
        np.testing.assert_array_equal(fixed, [1, 1, 0, 0])

    def test_invalid_rod_constitution_is_rejected(self):
        model, _, _ = self._model()
        model.uipc.rod_model[0] = "neo_hookean"
        solver = newton.solvers.SolverUIPC(model, backend="none")
        with self.assertRaisesRegex(ValueError, "rod_model"):
            solver.initialize(model.state())

    def test_branching_rod_is_rejected(self):
        builder = newton.ModelBuilder()
        builder.add_rod(
            rod=newton.Rod(
                [(0.0, 0.0, 1.0), (0.2, 0.0, 1.0), (0.0, 0.2, 1.0), (0.0, 0.0, 1.2)],
                edges=[(0, 1), (0, 2), (0, 3)],
                radius=0.01,
            )
        )
        model = builder.finalize("cpu")
        solver = newton.solvers.SolverUIPC(model, backend="none")
        with self.assertRaisesRegex(ValueError, "nonbranching"):
            solver.initialize(model.state())

    def test_backtracking_bending_rod_is_rejected(self):
        builder = newton.ModelBuilder()
        builder.add_rod(
            rod=newton.Rod([(0.0, 0.0, 1.0), (0.2, 0.0, 1.0), (0.0, 0.0, 1.0)], radius=0.01),
            bend_stiffness=1.0,
        )
        solver = newton.solvers.SolverUIPC(builder.finalize("cpu"), backend="none")
        with self.assertRaisesRegex(ValueError, "backtracking"):
            solver.initialize(solver.model.state())

    def test_closed_rod_has_one_bending_hinge_per_node(self):
        builder = newton.ModelBuilder()
        builder.add_rod(
            rod=newton.Rod(
                [(0.0, 0.0, 1.0), (0.2, 0.0, 1.0), (0.2, 0.2, 1.0), (0.0, 0.2, 1.0), (0.0, 0.0, 1.0)],
                closed=True,
                radius=0.01,
            ),
            bend_stiffness=10.0,
        )
        solver = self._initialize(builder.finalize("cpu"))
        geo = solver.mapping.rod_geo_slots[0].geometry()
        self.assertEqual(geo.vertices().size(), 4)
        self.assertEqual(geo.edges().size(), 4)
        self.assertTrue(np.all(np.asarray(view(geo.vertices().find("bending_stiffness"))) > 0.0))
        nodes = solver.get_rod_state()
        np.testing.assert_array_equal(nodes.curve_vertex_start.numpy(), [0, 4])
        np.testing.assert_array_equal(nodes.curve_edge_start.numpy(), [0, 4])
        self.assertEqual(nodes.positions.size, 4)
        np.testing.assert_array_equal(nodes.edges.numpy(), np.asarray(view(geo.edges().topo())).reshape(-1, 2))

    @unittest.skipUnless(_CUDA_TEST_DEVICES, "CUDA device is unavailable")
    def test_cuda_native_nodes_drive_visual_mesh_and_masked_reset(self):
        """Drive a higher-resolution visual mesh from stretched nodes across a partial reset."""
        prototype = newton.ModelBuilder(gravity=wp.vec3(0.0))
        bodies, _ = prototype.add_rod(
            rod=newton.Rod([(0, 0, 1), (0.2, 0, 1), (0.4, 0, 1)], radius=0.01, youngs_modulus=1.0e4, poissons_ratio=0.3)
        )
        vertices = [(0, 0, 1), (0.2, 0, 1), (0, 0.01, 1), (0.2, 0, 1), (0.4, 0, 1), (0.2, 0.01, 1)]
        prototype.add_deformable_visual_mesh(
            vertices,
            [0, 1, 2, 3, 4, 5],
            kind="body",
            parent=[bodies[0]] * 3 + [bodies[1]] * 3,
            uvs=[(0, 0), (1, 0), (0, 1)] * 2,
            texture=np.array([[[255, 0, 0]]], dtype=np.uint8),
        )
        prototype.shape_flags = [flags & ~newton.ShapeFlags.VISIBLE for flags in prototype.shape_flags]
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.replicate(prototype, copy_count=2, xforms=[wp.transform(), wp.transform(wp.vec3(2.0, 0.0, 0.0))])
        model = builder.finalize(_CUDA_TEST_DEVICES[0])
        state, result = model.state(), model.state()
        solver = newton.solvers.SolverUIPC(model, dt=0.002)
        with self.assertRaisesRegex(RuntimeError, "initialize"):
            solver.get_rod_state()
        with self.assertWarnsRegex(NewtonWarning, "ROD.*centerline"):
            solver.initialize(state)
        nodes = solver.get_rod_state()
        np.testing.assert_array_equal(nodes.curve_vertex_start.numpy(), [0, 3, 6])
        np.testing.assert_array_equal(nodes.curve_edge_start.numpy(), [0, 2, 4])
        np.testing.assert_array_equal(nodes.world_indices.numpy(), [0, 1])
        np.testing.assert_array_equal(nodes.body_indices.numpy(), [0, 1, 2, 3])

        # Author real FEM strain through the backend state accessor.
        native = solver._fem_accessor.create_geometry()
        native.vertices().create("position", np.zeros(3))
        native.vertices().create("velocity", np.zeros(3))
        solver._fem_accessor.copy_to(native)
        for slot in solver.mapping.rod_geo_slots:
            offset = int(np.asarray(view(slot.geometry().meta().find("backend_fem_vertex_offset"))).flat[0])
            points = view(native.vertices().find("position")).reshape(-1, 3)[offset : offset + 3]
            points[:, 0] = points[0, 0] + 1.5 * (points[:, 0] - points[0, 0])
        solver._fem_accessor.copy_from(native)
        solver.step(state, result, None, dt=0.002)
        actual = np.concatenate(
            [np.asarray(view(slot.geometry().positions())).reshape(-1, 3) for slot in solver.mapping.rod_geo_slots]
        )
        np.testing.assert_allclose(nodes.positions.numpy(), actual, atol=1.0e-12)
        self.assertGreater(np.linalg.norm(actual[1] - actual[0]), 0.21)
        self.assertIs(nodes, solver.get_rod_state())
        affine, valid = solver.get_body_affine_transforms()
        np.testing.assert_array_equal(valid.numpy(), [True, True, True, True])
        visuals = newton.geometry.DeformableVisuals(model)
        visuals.update(result, body_affine=affine, body_affine_valid=valid)
        for world in range(2):
            points = visuals.get_points(world).numpy()
            np.testing.assert_allclose(points[[0, 1, 3, 4]], actual[3 * world + np.array([0, 1, 1, 2])], atol=2e-6)
            self.assertAlmostEqual(float(np.linalg.norm(points[2] - points[0])), 0.01, places=6)

        # This pixel lies beyond the rest-length cable but inside its stretched skin.
        camera = newton.sensors.SensorCamera(
            model, default_render_config=newton.sensors.SensorCamera.RenderConfig(enable_textures=True)
        )
        rays = camera.compute_camera_rays_pinhole(1, 1, camera_fov=1.0, device=model.device)
        camera_pose = wp.array([wp.transform(wp.vec3(0.5, 0.001, 2.0))], dtype=wp.transform, device=model.device)
        depth = camera.allocate_image_depth(1, 1, 1)
        albedo = camera.allocate_image_albedo(1, 1, 1)
        model.bvh_refit_shapes(result)
        camera.update(result, camera_pose, rays, depth_image=depth, albedo_image=albedo, deformable_visuals=visuals)
        self.assertAlmostEqual(float(depth.numpy()[0, 0, 0]), 1.0, places=4)
        np.testing.assert_allclose(albedo.numpy().view(np.uint8).reshape(-1, 4)[0, :3], [255, 0, 0], atol=1)

        before = nodes.positions.numpy().copy()
        before_affine = affine.numpy().copy()
        reset = model.state()
        pose = reset.body_q.numpy()
        pose[model.body_world.numpy() == 0, :3] += [0.3, 0.2, 0.1]
        reset.body_q.assign(pose)
        solver.reset(
            reset,
            world_mask=wp.array([True, False], dtype=wp.bool, device=model.device),
            flags=newton.StateFlags.BODY_Q | newton.StateFlags.BODY_QD,
        )
        np.testing.assert_allclose(
            nodes.positions.numpy()[:3], [[0.3, 0.2, 1.1], [0.5, 0.2, 1.1], [0.7, 0.2, 1.1]], atol=2e-6
        )
        np.testing.assert_array_equal(nodes.positions.numpy()[3:], before[3:])
        np.testing.assert_array_equal(affine.numpy()[2:], before_affine[2:])
        visuals.update(reset, body_affine=affine, body_affine_valid=valid)
        np.testing.assert_allclose(
            visuals.get_points(0).numpy()[[0, 1, 3, 4]], nodes.positions.numpy()[[0, 1, 1, 2]], atol=2e-6
        )

    @unittest.skipUnless(USD_AVAILABLE, "usd-core is not installed")
    def test_usd_selector_follows_curve_rows_across_imports(self):
        builder = newton.ModelBuilder()
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        builder.add_rod(rod=newton.Rod([(0.0, 0.0, 1.0), (0.2, 0.0, 1.0), (0.4, 0.0, 1.0)], radius=0.01))
        stage = _deformable_stage()
        points = [(0.0, 1.0, 1.0), (0.2, 1.0, 1.0), (0.4, 1.0, 1.0)]
        prim = _add_cable_curve(stage, "/World/Schema", points).GetPrim()
        prim.ApplyAPI("NewtonUIPCRodAPI")
        ignored = _add_cable_curve(stage, "/World/Ignored", points).GetPrim()
        ignored.CreateAttribute("uipc:constitution", Sdf.ValueTypeNames.Token).Set("ignored")
        builder.add_usd(stage, ignore_paths=["/World/Ignored"])
        second_stage = _deformable_stage()
        raw = _add_cable_curve(second_stage, "/World/Raw", points).GetPrim()
        raw.CreateAttribute("uipc:constitution", Sdf.ValueTypeNames.Token).Set("unsupported")
        builder.add_usd(second_stage)
        model = builder.finalize("cpu")
        self.assertEqual(model.uipc.rod_model, ["hookean_spring", "hookean_spring", "unsupported"])
        solver = newton.solvers.SolverUIPC(model, backend="none")
        with self.assertRaisesRegex(ValueError, "rod_model\\[2\\]"):
            solver.initialize(model.state())

    @unittest.skipUnless(_CUDA_TEST_DEVICES, "CUDA device is unavailable")
    def test_cuda_inertial_translation_and_joint_readback(self):
        model, bodies, _ = self._model(_CUDA_TEST_DEVICES[0], origin="com")
        state, result = model.state(), model.state()
        velocity = np.zeros((model.body_count, 6), dtype=np.float32)
        velocity[bodies, 1] = 0.2
        state.body_qd.assign(velocity)
        initial = state.body_q.numpy().copy()
        solver = self._initialize(model, state, backend="cuda")
        for _ in range(10):
            solver.step(state, result, None, dt=0.002)
            state, result = result, state
        expected = initial.copy()
        expected[bodies, 1] += 0.004
        np.testing.assert_allclose(state.body_q.numpy(), expected, atol=2.0e-5)
        np.testing.assert_allclose(state.body_qd.numpy(), velocity, atol=2.0e-4)
        reference_q, reference_qd = wp.clone(state.joint_q), wp.clone(state.joint_qd)
        newton.eval_ik(model, state, reference_q, reference_qd)
        np.testing.assert_allclose(state.joint_q.numpy(), reference_q.numpy(), atol=1.0e-6)
        np.testing.assert_allclose(state.joint_qd.numpy(), reference_qd.numpy(), atol=1.0e-6)

        reset_state = model.state()
        reset_pose = reset_state.body_q.numpy()
        reset_pose[bodies, 0] += 0.5
        reset_state.body_q.assign(reset_pose)
        reset_velocity = np.zeros_like(velocity)
        reset_velocity[bodies, 1] = -0.3
        reset_state.body_qd.assign(reset_velocity)
        solver.reset(reset_state, flags=newton.StateFlags.BODY_Q | newton.StateFlags.BODY_QD)
        solver.step(reset_state, result, None, dt=0.002)
        reset_pose[bodies, 1] -= 0.0006
        np.testing.assert_allclose(result.body_q.numpy(), reset_pose, atol=2.0e-5)
        np.testing.assert_allclose(result.body_qd.numpy(), reset_velocity, atol=2.0e-4)

    @unittest.skipUnless(_CUDA_TEST_DEVICES, "CUDA device is unavailable")
    def test_cuda_body_force_accelerates_the_centerline(self):
        model, bodies, _ = self._model(_CUDA_TEST_DEVICES[0])
        state, result = model.state(), model.state()
        solver = self._initialize(model, state, backend="cuda")
        forces = np.zeros((model.body_count, 6), dtype=np.float32)
        forces[bodies, 1] = 100.0 * model.body_mass.numpy()[bodies]
        state.body_f.assign(forces)
        solver.step(state, result, None, dt=0.002)
        np.testing.assert_allclose(result.body_qd.numpy()[bodies, 1], 0.2, atol=2.0e-4)
        np.testing.assert_allclose(
            result.body_q.numpy()[bodies, 1] - state.body_q.numpy()[bodies, 1], 0.0004, atol=2.0e-6
        )

    @unittest.skipUnless(_CUDA_TEST_DEVICES, "CUDA device is unavailable")
    def test_cuda_fixed_segment_stays_fixed_while_tip_falls(self):
        model, bodies, _ = self._model(_CUDA_TEST_DEVICES[0], fixed=True)
        model.gravity.assign(np.tile([0.0, 0.0, -9.81], (model.gravity.size, 1)).astype(np.float32))
        state, result = model.state(), model.state()
        initial = state.body_q.numpy().copy()
        solver = self._initialize(model, state, backend="cuda")
        for _ in range(20):
            solver.step(state, result, None, dt=0.002)
            state, result = result, state
        np.testing.assert_allclose(state.body_q.numpy()[bodies[0]], initial[bodies[0]], atol=2.0e-6)
        self.assertLess(float(state.body_q.numpy()[bodies[-1], 2]), float(initial[bodies[-1], 2]) - 1.0e-4)

        target = state.body_q.numpy()
        target[bodies[0], 2] += 0.002
        state.body_q.assign(target)
        solver.step(state, result, None, dt=0.002)
        np.testing.assert_allclose(result.body_q.numpy()[bodies[0]], target[bodies[0]], atol=2.0e-6)

    @unittest.skipUnless(_CUDA_TEST_DEVICES, "CUDA device is unavailable")
    def test_cuda_transverse_body_torque_preserves_the_applied_wrench(self):
        model, bodies, _ = self._model(_CUDA_TEST_DEVICES[0])
        state, result = model.state(), model.state()
        solver = self._initialize(model, state, backend="cuda")
        geo = solver.mapping.rod_geo_slots[0].geometry()
        initial_points = np.asarray(view(geo.positions())).reshape(-1, 3).copy()
        forces = np.zeros((model.body_count, 6), dtype=np.float32)
        forces[bodies[1], 5] = 0.1
        state.body_f.assign(forces)
        solver.step(state, result, None, dt=0.002)
        node_forces = np.asarray(view(geo.vertices().find("external_force"))).reshape(-1, 3)
        np.testing.assert_allclose(node_forces.sum(axis=0), 0.0, atol=1.0e-8)
        np.testing.assert_allclose(np.cross(initial_points, node_forces).sum(axis=0), [0.0, 0.0, 0.1], atol=1.0e-7)
        self.assertGreater(float(result.body_qd.numpy()[bodies[1], 5]), 0.0)

    @unittest.skipUnless(_CUDA_TEST_DEVICES, "CUDA device is unavailable")
    def test_cuda_masked_velocity_reset_preserves_other_world_and_positions(self):
        prototype = newton.ModelBuilder(gravity=wp.vec3(0.0))
        prototype.add_rod(rod=newton.Rod([(0.0, 0.0, 1.0), (0.2, 0.0, 1.0), (0.4, 0.0, 1.0)], radius=0.01))
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.add_world(prototype)
        builder.add_world(prototype, xform=wp.transform(wp.vec3(2.0, 0.0, 0.0)))
        model = builder.finalize(_CUDA_TEST_DEVICES[0])
        state, result = model.state(), model.state()
        velocity = np.zeros((model.body_count, 6), dtype=np.float32)
        velocity[:, 1] = 0.2
        state.body_qd.assign(velocity)
        solver = self._initialize(model, state, backend="cuda")
        for _ in range(5):
            solver.step(state, result, None, dt=0.002)
            state, result = result, state
        previous = state.body_q.numpy().copy()
        reset = model.state()
        reset_pose = reset.body_q.numpy()
        reset_pose[:, 0] += 100.0
        reset.body_q.assign(reset_pose)
        velocity[:, 1] = -0.3
        reset.body_qd.assign(velocity)
        solver.reset(reset, world_mask=np.asarray([True, False]), flags=newton.StateFlags.BODY_QD)
        solver.step(reset, result, None, dt=0.002)
        world = model.body_world.numpy()
        expected_velocity = np.where(world == 0, -0.3, 0.2)
        np.testing.assert_allclose(result.body_qd.numpy()[:, 1], expected_velocity, atol=2.0e-4)
        previous[:, 1] += expected_velocity * 0.002
        np.testing.assert_allclose(result.body_q.numpy(), previous, atol=2.0e-5)

    @unittest.skipUnless(_CUDA_TEST_DEVICES, "CUDA device is unavailable")
    def test_cuda_rod_and_abd_settle_on_empty_static_collider(self):
        """Report ground reactions on rod segments and an ABD body after settling."""
        self._assert_ground_reactions(ground_plane=False)

    @unittest.skipUnless(_CUDA_TEST_DEVICES, "CUDA device is unavailable")
    def test_cuda_rod_and_abd_settle_on_halfplane(self):
        """Report rod and ABD support forces and contact points on an implicit halfplane."""
        self._assert_ground_reactions(ground_plane=True)

    def _assert_ground_reactions(self, *, ground_plane):
        model, bodies, _ = self._model(_CUDA_TEST_DEVICES[0], colliders=True, ground_plane=ground_plane)
        model.gravity.assign(np.tile([0.0, 0.0, -9.81], (model.gravity.size, 1)).astype(np.float32))
        state, result = model.state(), model.state()
        solver = self._initialize(model, state, backend="cuda", contact=True)
        self.assertEqual(set(solver.mapping.body_geo_slots), {0})
        sensor = SensorContact(
            model, sensing_bodies=[0, *bodies], counterpart_shapes=[0], request_contact_attributes=False
        )
        contacts = newton.CollisionPipeline(model, rigid_contact_max=solver.get_max_contact_count()).contacts()
        observables = solver.observables(sensor.solver_observable_kinds, contacts=contacts)
        reactions = []
        for step in range(300):
            solver.step(state, result, None, contacts, 0.002, observables=observables)
            state, result = result, state
            if step >= 280:
                sensor.update(state, contacts, observables=observables)
                reactions.append(sensor.total_force.numpy()[:, 2])
        rod_points = np.asarray(view(solver.mapping.rod_geo_slots[0].geometry().positions())).reshape(-1, 3)
        self.assertTrue(np.all(np.isfinite(state.body_q.numpy())))
        self.assertGreater(float(np.min(rod_points[:, 2])), 0.009)
        self.assertLess(float(np.max(rod_points[:, 2])), 0.03)
        self.assertGreater(float(state.body_q.numpy()[0, 2]), 0.099)
        self.assertLess(float(state.body_q.numpy()[0, 2]), 0.12)
        support = np.mean(reactions, axis=0)
        self.assertTrue(np.all(np.isfinite(support)))
        self.assertTrue(np.all(support > 0.0))
        expected = model.body_mass.numpy()[[0, *bodies]] * 9.81
        np.testing.assert_allclose(support, expected, rtol=0.3, atol=1.0e-4)
        positions = sensor.position_matrix.numpy()[:, 0]
        self.assertTrue(np.all(np.isfinite(positions)))
        np.testing.assert_allclose(positions[:, 2], 0.0, atol=0.003)

    @unittest.skipUnless(_CUDA_TEST_DEVICES, "CUDA device is unavailable")
    def test_cuda_crossed_rods_report_segment_contact_pairs(self):
        """Resolve edge contacts to capsule segments and report equal opposite reactions."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        first, _ = builder.add_rod(rod=newton.Rod([(-0.3, 0.0, 0.2), (0.1, 0.0, 0.2), (0.4, 0.0, 0.2)], radius=0.01))
        second, _ = builder.add_rod(
            rod=newton.Rod([(0.0, -0.3, 0.2205), (0.0, 0.1, 0.2205), (0.0, 0.4, 0.2205)], radius=0.01)
        )
        model = builder.finalize(_CUDA_TEST_DEVICES[0])
        state, result = model.state(), model.state()
        solver = self._initialize(model, state, backend="cuda", contact=True)
        sensor = SensorContact(
            model,
            sensing_bodies=[*first, *second],
            counterpart_bodies=[*first, *second],
            request_contact_attributes=False,
        )
        contacts = newton.CollisionPipeline(model, rigid_contact_max=solver.get_max_contact_count()).contacts()
        observables = solver.observables(sensor.solver_observable_kinds, contacts=contacts)
        solver.step(state, result, None, contacts, 0.002, observables=observables)
        sensor.update(result, contacts, observables=observables)
        forces = sensor.total_force.numpy()
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertLess(float(forces[0, 2]), -1.0e-6)
        self.assertGreater(float(forces[2, 2]), 1.0e-6)
        np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1.0e-6)
        np.testing.assert_allclose(forces[[1, 3]], 0.0, atol=1.0e-6)
        np.testing.assert_allclose(sensor.position_matrix.numpy()[0, 2, :2], 0.0, atol=1.0e-3)

    def test_rod_contact_gradient_packing_preserves_force_channels(self):
        """Preserve normal and friction loads for shared points and mixed ABD/rod edges."""
        from newton._src.solvers.uipc.contact_forces import _pack_rod_contacts, _RodContactVertex  # noqa: PLC0415

        vertices = {
            0: _RodContactVertex(np.array([0, 0]), np.array([1.0, 0.0]), np.array([0.0, 0.0, 0.0]), 0.1),
            1: _RodContactVertex(np.array([0, 1]), np.array([0.25, 0.75]), np.array([1.0, 0.0, 0.0]), 0.1),
            2: _RodContactVertex(np.array([1, 1]), np.array([1.0, 0.0]), np.array([2.0, 0.0, 0.0]), 0.1),
        }
        edges = {(0, 1): 0, (1, 2): 1}
        vertex_to_body = np.array([-1, -1, -1, 2])
        data = _pack_rod_contacts(
            "PH",
            np.array([[0], [1]]),
            np.array([[[0.0, 0.0, 4.0]], [[0.0, 0.0, 8.0]]]),
            np.array([[[2.0, 0.0, 0.0]]]),
            vertices,
            edges,
            vertex_to_body,
            friction_topology=np.array([[1]]),
        )
        np.testing.assert_array_equal(data.body_a, [0, 0, 1])
        np.testing.assert_allclose(data.forces_n, [[0.0, 0.0, 4.0], [0.0, 0.0, 2.0], [0.0, 0.0, 6.0]])
        np.testing.assert_allclose(data.forces_f, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
        np.testing.assert_allclose(data.point_a, [[0.0, 0.0, -0.1], [1.0, 0.0, -0.1], [1.0, 0.0, -0.1]])
        data = _pack_rod_contacts(
            "PE",
            np.array([[3, 0, 1]]),
            np.array([[[0.0, 0.0, 8.0], [0.0, 0.0, -2.0], [0.0, 0.0, -6.0]]]),
            np.array([[[2.0, 0.0, 0.0], [-0.5, 0.0, 0.0], [-1.5, 0.0, 0.0]]]),
            vertices,
            edges,
            vertex_to_body,
        )
        np.testing.assert_array_equal(data.vert_a, [3])
        np.testing.assert_array_equal(data.body_b, [0])
        np.testing.assert_allclose(data.forces_n, [[0.0, 0.0, 8.0]])
        np.testing.assert_allclose(data.forces_f, [[2.0, 0.0, 0.0]])
        np.testing.assert_allclose(data.point_b, [[0.75, 0.0, 0.1]])

    def test_direct_rod_joints_accept_world_fixed_and_ball_roots(self):
        """Accept direct rod articulations with fixed or ball roots and preserve their fixed vertices."""
        for root_type, expected_fixed in ((newton.JointType.FIXED, [1, 1, 0]), (newton.JointType.BALL, [1, 0, 0])):
            with self.subTest(root_type=root_type):
                builder = newton.ModelBuilder()
                first = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
                second = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.2)))
                for body in (first, second):
                    builder.add_shape_capsule(body=body, radius=0.01, half_height=0.1)
                if root_type == newton.JointType.FIXED:
                    root = builder.add_joint_fixed(
                        parent=-1, child=first, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0))
                    )
                else:
                    root = builder.add_joint_ball(
                        parent=-1,
                        child=first,
                        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.9)),
                        child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1)),
                    )
                joint = builder.add_joint_rod(
                    parent=first,
                    child=second,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1)),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1)),
                )
                builder.add_articulation([root, joint])
                solver = self._initialize(builder.finalize("cpu"))
                fixed = np.asarray(
                    view(solver.mapping.rod_geo_slots[0].geometry().vertices().find("is_fixed"))
                ).reshape(-1)
                np.testing.assert_array_equal(fixed, expected_fixed)


if __name__ == "__main__":
    unittest.main()
