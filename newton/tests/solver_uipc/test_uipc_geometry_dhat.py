# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""USD-to-native geometry contact distances for every UIPC geometry family."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests._usd_deformable_test_utils import (
    _add_cable_curve,
    _add_cloth_mesh,
    _author_deformable_element_array,
    _deformable_stage,
)
from newton.tests.unittest_utils import get_selected_cuda_test_devices, has_module, requires

_CUDA_TEST_DEVICES = get_selected_cuda_test_devices(mode="basic")

if has_module("uipc") and has_module("pxr.Usd"):
    import newton_usd_schemas  # noqa: F401
    import uipc
    from pxr import Gf, UsdGeom, UsdPhysics


@requires("uipc", "pxr.Usd")
class TestUIPCGeometryDhat(unittest.TestCase):
    def _stage(self):
        stage = _deformable_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        rigid = UsdGeom.Cube.Define(stage, "/World/Rigid")
        rigid.CreateSizeAttr(0.1)
        rigid.AddTranslateOp().Set(Gf.Vec3d(-1.0, 0.0, 1.0))
        UsdPhysics.RigidBodyAPI.Apply(rigid.GetPrim())
        UsdPhysics.CollisionAPI.Apply(rigid.GetPrim())
        collider = UsdGeom.Cube.Define(stage, "/World/StaticCollider")
        collider.CreateSizeAttr(0.1)
        collider.AddTranslateOp().Set(Gf.Vec3d(-2.0, 0.0, 1.0))
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        _author_deformable_element_array(cloth.GetPrim(), "thicknesses", [0.001], "constant")
        tet = UsdGeom.TetMesh.Define(stage, "/World/Tet")
        tet.CreatePointsAttr([(2, 0, 1), (2.1, 0, 1), (2, 0.1, 1), (2, 0, 1.1)])
        tet.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
        rod = _add_cable_curve(stage, "/World/Rod", [(3, 0, 1), (3.1, 0, 1), (3.2, 0, 1)])
        for prim, schema, distance in (
            (rigid.GetPrim(), "NewtonUIPCRigidBodyAPI", 2e-5),
            (cloth.GetPrim(), "NewtonUIPCClothAPI", 3e-5),
            (tet.GetPrim(), "NewtonUIPCDeformableAPI", 4e-5),
            (rod.GetPrim(), "NewtonUIPCRodAPI", 5e-5),
            (collider.GetPrim(), "NewtonUIPCRigidBodyAPI", 6e-5),
        ):
            self.assertTrue(prim.ApplyAPI(schema))
            attr = prim.GetAttribute("uipc:dhat")
            self.assertTrue(attr, f"{schema} must declare uipc:dhat")
            self.assertEqual(attr.Get(), -1.0)
            self.assertTrue(attr.Set(distance))
        return stage

    def _model(self, device="cpu", *, stage=None, ground=False):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        imported = builder.add_usd(self._stage() if stage is None else stage)
        if ground:
            builder.add_ground_plane()
        return builder.finalize(device), imported

    def _assert_geometry_distances(self, solver, rigid):
        geometries = (
            solver.mapping.body_geo_slots[rigid].geometry(),
            solver.mapping.cloth_geo_slots[0].geometry(),
            solver.mapping.deformable_geo_slots[0].geometry(),
            solver.mapping.rod_geo_slots[0].geometry(),
        )
        for geometry, expected in zip(geometries, (2e-5, 3e-5, 4e-5, 5e-5), strict=True):
            actual = uipc.view(geometry.meta().find("d_hat"))
            np.testing.assert_allclose(actual, [expected], rtol=1e-6)
        static_geometry = solver.mapping.static_geo_slots[0].geometry()
        np.testing.assert_allclose(uipc.view(static_geometry.meta().find("d_hat")), [6e-5], rtol=1e-6)

    def test_usd_dhat_reaches_each_native_geometry_family(self):
        model, imported = self._model()
        solver = newton.solvers.SolverUIPC(model, backend="none", logger_level=uipc.Logger.Error)
        solver.initialize(model.state())
        self._assert_geometry_distances(solver, imported["path_body_map"]["/World/Rigid"])
        for slots, expected in (
            (solver.mapping.cloth_rest_geo_slots, 3e-5),
            (solver.mapping.deformable_rest_geo_slots, 4e-5),
            (solver.mapping.rod_rest_geo_slots, 5e-5),
        ):
            np.testing.assert_allclose(uipc.view(slots[0].geometry().meta().find("d_hat")), [expected], rtol=1e-6)
        cloth = solver.mapping.cloth_geo_slots[0].geometry()
        np.testing.assert_allclose(uipc.view(cloth.vertices().find("thickness")), 0.0005)
        rod = solver.mapping.rod_geo_slots[0].geometry()
        np.testing.assert_allclose(uipc.view(rod.vertices().find("thickness")), 0.01)

    def test_geometry_distances_preserve_rigid_instancing_and_split_static_meshes(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        for x, dhat in enumerate((-1.0, -2.0, 2e-5, 2e-5, 4e-5)):
            body = builder.add_body(
                xform=wp.transform(wp.vec3(float(x), 0.0, 1.0)), custom_attributes={"uipc:dhat": dhat}
            )
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        for x, dhat in enumerate((-1.0, -2.0, 2e-5, 4e-5)):
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(float(x), 1.0, 1.0)),
                hx=0.05,
                hy=0.05,
                hz=0.05,
                custom_attributes={"uipc:shape_dhat": dhat},
            )
        model = builder.finalize("cpu")
        solver = newton.solvers.SolverUIPC(model, backend="none", logger_level=uipc.Logger.Error)
        solver.set_contact(True, d_hat=1e-4)
        solver.initialize(model.state())
        slots = solver.mapping.body_geo_slots
        self.assertIs(slots[0], slots[1])
        self.assertIs(slots[2], slots[3])
        self.assertIsNot(slots[2], slots[4])
        self.assertIsNone(slots[0].geometry().meta().find("d_hat"))
        np.testing.assert_allclose(uipc.view(slots[2].geometry().meta().find("d_hat")), [2e-5])
        self.assertEqual(len(solver.mapping.static_geo_slots), 3)
        for slot, shapes in zip(solver.mapping.static_geo_slots, solver.mapping.static_shape_indices, strict=True):
            expected = model.uipc.shape_dhat.numpy()[shapes[0]]
            attribute = slot.geometry().meta().find("d_hat")
            if expected < 0.0:
                self.assertIsNone(attribute)
                self.assertEqual(len(shapes), 2)
            else:
                np.testing.assert_allclose(uipc.view(attribute), [expected])
        self.assertAlmostEqual(float(uipc.view(solver.scene.config().find("contact/d_hat"))[0]), 1e-4)

    def test_replicated_models_keep_cloth_fem_and_curve_distances(self):
        source = newton.ModelBuilder()
        newton.solvers.SolverUIPC.register_custom_attributes(source)
        source.add_usd(self._stage())
        combined = newton.ModelBuilder()
        combined.replicate(source, 2)
        model = combined.finalize("cpu")
        for name, distance in (("cloth_dhat", 3e-5), ("deformable_dhat", 4e-5), ("rod_dhat", 5e-5)):
            attribute = getattr(model.uipc, name)
            np.testing.assert_allclose(attribute.numpy(), [distance, distance])
            attribute.assign(np.array([distance, 2.0 * distance], dtype=np.float32))
        solver = newton.solvers.SolverUIPC(model, backend="none", logger_level=uipc.Logger.Error)
        solver.initialize(model.state())
        for slots, expected in (
            (solver.mapping.cloth_geo_slots, 3e-5),
            (solver.mapping.deformable_geo_slots, 4e-5),
            (solver.mapping.rod_geo_slots, 5e-5),
        ):
            self.assertEqual(len(slots), 2)
            for world, slot in enumerate(slots):
                actual = uipc.view(slot.geometry().meta().find("d_hat"))
                np.testing.assert_allclose(actual, [(world + 1) * expected], rtol=1e-6)

    def test_legacy_usd_spelling_and_canonical_precedence_use_meters(self):
        from pxr import Sdf

        stage = self._stage()
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        for name in ("Rigid", "Cloth", "Tet", "Rod"):
            prim = stage.GetPrimAtPath("/World/" + name)
            prim.RemoveProperty("uipc:dhat")
            prim.CreateAttribute("uipc:d_hat", Sdf.ValueTypeNames.Float).Set(7e-5)
        stage.GetPrimAtPath("/World/Cloth").GetAttribute("uipc:dhat").Set(9e-5)
        builder = newton.ModelBuilder()
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize("cpu")
        for name, expected in (("dhat", 7e-5), ("cloth_dhat", 9e-5), ("deformable_dhat", 7e-5), ("rod_dhat", 7e-5)):
            with self.subTest(attribute=name):
                self.assertAlmostEqual(float(getattr(model.uipc, name).numpy()[0]), expected)

    def test_invalid_distances_and_conflicting_geometry_overrides_are_rejected(self):
        model, imported = self._model()
        for name in ("dhat", "shape_dhat", "cloth_dhat", "deformable_dhat", "rod_dhat"):
            attribute = getattr(model.uipc, name)
            original = attribute.numpy().copy()
            for invalid in (0.0, float("nan"), float("inf")):
                values = original.copy()
                values[0] = invalid
                attribute.assign(values)
                with (
                    self.subTest(attribute=name, value=invalid),
                    self.assertRaisesRegex(ValueError, "finite and nonzero"),
                ):
                    newton.solvers.SolverUIPC(model, backend="none", logger_level=uipc.Logger.Error).initialize(
                        model.state()
                    )
            attribute.assign(original)

        values = model.uipc.shape_dhat.numpy()
        shape = imported["path_shape_map"]["/World/Rigid"]
        values[shape] = 6e-5
        model.uipc.shape_dhat.assign(values)
        with self.assertRaisesRegex(ValueError, "uniform uipc:dhat"):
            newton.solvers.SolverUIPC(model, backend="none", logger_level=uipc.Logger.Error).initialize(model.state())

    @unittest.skipUnless(_CUDA_TEST_DEVICES, "CUDA device unavailable")
    def test_cuda_cloth_distance_controls_contact_in_a_mixed_scene(self):
        stage = self._stage()
        narrow = UsdGeom.Mesh.Get(stage, "/World/Cloth")
        wide = _add_cloth_mesh(stage, "/World/WideCloth")
        self.assertTrue(wide.GetPrim().ApplyAPI("NewtonUIPCClothAPI"))
        wide.GetPrim().GetAttribute("uipc:dhat").Set(1e-4)
        _author_deformable_element_array(wide.GetPrim(), "thicknesses", [0.001], "constant")
        # Both cloth surfaces start 60 micrometers above the halfplane.
        for mesh, x in ((narrow, 0.0), (wide, 5.0)):
            mesh.GetPrim().GetAttribute("uipc:constitution").Set("neo_hookean")
            mesh.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, -1.0 + 0.00056))
        model, imported = self._model(_CUDA_TEST_DEVICES[0], stage=stage, ground=True)
        state = model.state()
        target = model.state()
        solver = newton.solvers.SolverUIPC(model, backend="cuda", dt=1 / 120, logger_level=uipc.Logger.Error)
        solver.set_contact(True, d_hat=1e-4)
        solver.initialize(state)
        self._assert_geometry_distances(solver, imported["path_body_map"]["/World/Rigid"])
        initial = state.particle_q.numpy()
        narrow_indices, wide_indices = solver.mapping.cloth_particle_indices
        control = model.control()
        for _ in range(3):
            solver.step(state, target, control, None, 1 / 120)
            state, target = target, state
        wp.synchronize()
        self.assertTrue(solver.world.is_valid())
        self.assertEqual(solver.world.frame(), 3)
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())
        final = state.particle_q.numpy()
        self.assertTrue(np.isfinite(final).all())
        np.testing.assert_allclose(final[narrow_indices], initial[narrow_indices], atol=1e-7, rtol=0.0)
        self.assertGreater(float(np.min(final[wide_indices, 2] - initial[wide_indices, 2])), 1e-7)


if __name__ == "__main__":
    unittest.main()
