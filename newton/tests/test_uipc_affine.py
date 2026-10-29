# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for full affine body export from UIPC."""

import importlib.util
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import get_selected_cuda_test_devices

_HAS_UIPC = importlib.util.find_spec("uipc") is not None
if _HAS_UIPC:
    from uipc import view

    from newton._src.solvers.uipc.converter import (
        UIpcMappingInfo,
        _gather_body_affine_kernel,
        populate_backend_offsets,
    )


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCAffine(unittest.TestCase):
    def test_cuda_body_visuals_use_native_affine_and_rigid_fallback(self):
        """Skin from native ABD strain without replacing unmapped bodies' rigid poses."""
        devices = get_selected_cuda_test_devices(mode="basic")
        if not devices:
            self.skipTest("CUDA is unavailable")
        for device in devices:
            with self.subTest(device=str(device)), tempfile.TemporaryDirectory() as workspace:
                builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
                abd = builder.add_body()
                builder.add_shape_box(abd, hx=0.1, hy=0.1, hz=0.1)
                rod_bodies, _ = builder.add_rod(rod=newton.Rod([(2, 3, 4), (2.2, 3, 4), (2.4, 3, 4)], radius=0.01))
                rigid = rod_bodies[0]
                vertices = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0.05]], dtype=np.float32)
                rigid_vertices = vertices + np.array([2, 3, 4], dtype=np.float32)
                builder.add_deformable_visual_mesh(vertices, [0, 1, 2], kind="body", parent=[abd] * 3)
                builder.add_deformable_visual_mesh(rigid_vertices, [0, 1, 2], kind="body", parent=[rigid] * 3)
                model = builder.finalize(device)
                state, result = model.state(), model.state()
                solver = newton.solvers.SolverUIPC(model, workspace=workspace, dt=0.002)
                with self.assertWarnsRegex(UserWarning, "ROD.*centerline"):
                    solver.initialize(state)
                affine, valid = solver.get_body_affine_transforms()
                np.testing.assert_array_equal(valid.numpy(), [True, False, False])

                # Set native strain directly; converting it through a quaternion
                # would hide the stretch/shear this rendering interface preserves.
                matrix = np.array(
                    [[1.2, 0.3, 0, 0.1], [0, 0.8, 0.2, -0.2], [0, 0, 1.1, 0.3], [0, 0, 0, 1]],
                    dtype=np.float64,
                )
                native = solver._abd_accessor.create_geometry()
                native.instances().create("transform", np.eye(4))
                solver._abd_accessor.copy_to(native)
                offset = int(solver.mapping.backend_offsets_wp.numpy()[0])
                view(native.instances().find("transform"))[offset] = matrix
                solver._abd_accessor.copy_from(native)
                solver._update_body_affine_transforms()
                np.testing.assert_allclose(affine.numpy()[abd], matrix, atol=1.0e-12)

                visuals = newton.geometry.DeformableVisuals(model)
                visuals.update(state, body_affine=affine, body_affine_valid=valid)
                np.testing.assert_allclose(
                    visuals.get_points(0).numpy(), vertices @ matrix[:3, :3].T + matrix[:3, 3], atol=1.0e-6
                )
                np.testing.assert_allclose(visuals.get_points(1).numpy(), rigid_vertices, atol=1.0e-6)

                solver.step(state, result, None, dt=0.002)
                self.assertIs(solver.get_body_affine_transforms()[0], affine)
                visuals.update(result, body_affine=affine, body_affine_valid=valid)
                updated = affine.numpy()[abd]
                np.testing.assert_allclose(
                    visuals.get_points(0).numpy(), vertices @ updated[:3, :3].T + updated[:3, 3], atol=1.0e-6
                )
                np.testing.assert_allclose(visuals.get_points(1).numpy(), rigid_vertices, atol=1.0e-6)

    def test_missing_backend_offsets_do_not_alias_body_zero(self):
        """Leave bodies with missing or negative backend offsets unmapped."""
        mapping = UIpcMappingInfo()
        for body, offset in [(9, 5), (2, 0), (4, None), (8, -1)]:
            slot = Mock()
            attribute = None if offset is None else Mock(view=lambda offset=offset: np.array([offset]))
            slot.geometry.return_value.meta.return_value.find.return_value = attribute
            mapping.body_geo_slots[body] = slot
        mapping.body_instance_ids[9] = 1
        with self.assertWarnsRegex(UserWarning, "Body 4"):
            populate_backend_offsets(mapping, wp.get_device("cpu"))
        np.testing.assert_array_equal(mapping.body_indices_wp.numpy(), [2, 9])
        np.testing.assert_array_equal(mapping.backend_offsets_wp.numpy(), [0, 6])
        self.assertEqual(mapping.num_mapped_bodies, 2)
        self.assertEqual(mapping.max_backend_count, 7)

    def test_noncontiguous_mapping_preserves_shear_and_invalid_bodies(self):
        """Gather reordered backend matrices without losing stretch or shear."""
        for device in [wp.get_device("cpu"), *get_selected_cuda_test_devices(mode="basic")]:
            with self.subTest(device=str(device)):
                matrices = np.tile(np.eye(4), (6, 1, 1))
                matrices[4] = [[0.8, 0.3, 0, 1], [0, 1.2, 0.1, 2], [0, 0, 0.9, 3], [0, 0, 0, 1]]
                matrices[1] = [[0, -1.1, 0.2, -2], [0.7, 0, 0, 4], [0, 0, 1.3, 6], [0, 0, 0, 1]]
                output = wp.full(4, wp.mat44d(np.eye(4)), device=device)
                valid = wp.zeros(4, dtype=wp.bool, device=device)
                wp.launch(
                    _gather_body_affine_kernel,
                    dim=2,
                    inputs=[
                        wp.array([4, 1], dtype=wp.uint32, device=device),
                        wp.array([3, 0], dtype=wp.int32, device=device),
                        wp.array(matrices.transpose(0, 2, 1).copy(), dtype=wp.mat44d, device=device),
                        output,
                        valid,
                    ],
                    device=device,
                )
                np.testing.assert_array_equal(valid.numpy(), [True, False, False, True])
                np.testing.assert_allclose(output.numpy()[[3, 0]], matrices[[4, 1]], atol=0)
                np.testing.assert_array_equal(output.numpy()[1:3], np.tile(np.eye(4), (2, 1, 1)))

    def test_cuda_export_updates_after_step_and_masked_reset(self):
        """Keep exported matrices current across stepping and a single-world reset."""
        devices = get_selected_cuda_test_devices(mode="basic")
        if not devices:
            self.skipTest("CUDA is unavailable")
        for device in devices:
            with self.subTest(device=str(device)), tempfile.TemporaryDirectory() as workspace:
                world = newton.ModelBuilder()
                for x in (0.0, 0.4):
                    body = world.add_body(xform=wp.transform(wp.vec3(x, 0, 1), wp.quat_identity()))
                    world.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
                builder = newton.ModelBuilder()
                builder.replicate(world, world_count=2, spacing=(2, 0, 0))
                model = builder.finalize(device=device)
                state, output = model.state(), model.state()
                solver = newton.solvers.SolverUIPC(model, workspace=workspace, auto_sync_inertia=False)
                with self.assertRaisesRegex(RuntimeError, "initialize"):
                    solver.get_body_affine_transforms()
                solver.initialize(state)
                affine, valid = solver.get_body_affine_transforms()
                np.testing.assert_array_equal(valid.numpy(), np.ones(model.body_count, dtype=bool))
                np.testing.assert_allclose(affine.numpy()[:, :3, 3], state.body_q.numpy()[:, :3])
                self.assertLess(len({id(v) for v in solver.mapping.body_geo_slots.values()}), model.body_count)

                solver.step(state, output, model.control(), dt=1 / 60)
                self.assertIs(affine, solver.get_body_affine_transforms()[0])
                backend = solver._abd_transform_buf.warp().numpy().transpose(0, 2, 1)
                np.testing.assert_array_equal(
                    affine.numpy()[solver.mapping.body_indices_wp.numpy()],
                    backend[solver.mapping.backend_offsets_wp.numpy()],
                )
                np.testing.assert_allclose(affine.numpy()[:, :3, 3], output.body_q.numpy()[:, :3], atol=1e-7)

                before = affine.numpy().copy()
                body_world = model.body_world.numpy()
                target = output.body_q.numpy().copy()
                target[body_world == 0, :3] += [1, 2, 3]
                output.body_q.assign(target)
                solver.reset(
                    output,
                    world_mask=wp.array([True, False], dtype=wp.bool, device=device),
                    flags=newton.StateFlags.BODY_Q,
                )
                after = affine.numpy()
                np.testing.assert_allclose(after[body_world == 0, :3, 3], target[body_world == 0, :3], atol=1e-7)
                np.testing.assert_array_equal(after[body_world == 1], before[body_world == 1])


if __name__ == "__main__":
    unittest.main()
