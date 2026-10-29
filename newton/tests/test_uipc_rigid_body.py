# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for UIPC rigid-body construction."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import USD_AVAILABLE

_HAS_UIPC = importlib.util.find_spec("uipc") is not None
if _HAS_UIPC:
    import uipc

if USD_AVAILABLE and _HAS_UIPC:
    import newton_usd_schemas  # noqa: F401
    from pxr import Usd, UsdGeom, UsdPhysics


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCRigidBodyBuilder(unittest.TestCase):
    @staticmethod
    def _add_box_body(builder: newton.ModelBuilder, x: float, kappa: float | None = None) -> int:
        custom_attributes = {"uipc:abd_kappa": kappa} if kappa is not None else None
        body = builder.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, 0.2), wp.quat_identity()),
            custom_attributes=custom_attributes,
            label=f"box_{x}",
        )
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        return body

    def test_register_custom_attributes_adds_body_abd_kappa(self):
        """Register rigid-body constitution selectors alongside ABD stiffness."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        body_a = self._add_box_body(builder, 0.0, 1.0e8)
        body_b = self._add_box_body(builder, 0.3)

        model = builder.finalize()

        self.assertTrue(hasattr(model, "uipc"))
        self.assertTrue(hasattr(model.uipc, "abd_kappa"))
        values = model.uipc.abd_kappa.numpy()
        self.assertEqual(values.shape[0], model.body_count)
        self.assertAlmostEqual(float(values[body_a]), 1.0e8)
        self.assertAlmostEqual(float(values[body_b]), -1.0)
        self.assertEqual(model.uipc.rigid_model, ["abd", "abd"])
        self.assertEqual(model.uipc.shape_constitution, ["", ""])

    def test_body_abd_kappa_splits_instancing_and_sets_uipc_kappa(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0, 0.0, 0.0))
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        body_soft = self._add_box_body(builder, 0.0, 1.0e8)
        body_stiff = self._add_box_body(builder, 0.3, 1.0e10)
        model = builder.finalize()

        solver = newton.solvers.SolverUIPC(
            model,
            backend="none",
            logger_level=uipc.Logger.Error,
            auto_sync_inertia=False,
        )
        solver.initialize(model.state())

        soft_slot = solver.mapping.body_geo_slots[body_soft]
        stiff_slot = solver.mapping.body_geo_slots[body_stiff]
        self.assertIsNot(soft_slot, stiff_slot)
        self.assertEqual(solver.mapping.body_instance_ids[body_soft], 0)
        self.assertEqual(solver.mapping.body_instance_ids[body_stiff], 0)

        np.testing.assert_allclose(
            uipc.view(soft_slot.geometry().instances().find("kappa")),
            np.array([1.0e8], dtype=np.float32),
        )
        np.testing.assert_allclose(
            uipc.view(stiff_slot.geometry().instances().find("kappa")),
            np.array([1.0e10], dtype=np.float32),
        )

    def test_negative_body_abd_kappa_uses_solver_default_and_preserves_instancing(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0, 0.0, 0.0))
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        body_a = self._add_box_body(builder, 0.0)
        body_b = self._add_box_body(builder, 0.3)
        model = builder.finalize()

        solver = newton.solvers.SolverUIPC(
            model,
            backend="none",
            kappa=3.0e8,
            logger_level=uipc.Logger.Error,
            auto_sync_inertia=False,
        )
        solver.initialize(model.state())

        shared_slot = solver.mapping.body_geo_slots[body_a]
        self.assertIs(shared_slot, solver.mapping.body_geo_slots[body_b])
        np.testing.assert_allclose(
            uipc.view(shared_slot.geometry().instances().find("kappa")),
            np.array([3.0e8, 3.0e8], dtype=np.float32),
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_usd_rigid_constitutions_preserve_static_and_dynamic_roles(self):
        """Import ABD bodies and split static ABD and Empty collision geometry."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/PhysicsScene").CreateGravityMagnitudeAttr(0.0)
        body_prim = UsdGeom.Xform.Define(stage, "/World/Rigid").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        self.assertTrue(body_prim.ApplyAPI("NewtonUIPCRigidBodyAPI"))
        body_prim.GetAttribute("uipc:constitution").Set("abd")
        for path, x, constitution in (
            ("/World/Rigid/Collider", 0.0, None),
            ("/World/StaticABD", 1.0, "abd"),
            ("/World/StaticEmpty", 2.0, "empty"),
            ("/World/StaticDefault", 3.0, None),
        ):
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(0.1)
            cube.AddTranslateOp().Set((x, 0.0, 0.5))
            prim = cube.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            if constitution is not None:
                self.assertTrue(prim.ApplyAPI("NewtonUIPCRigidBodyAPI"))
                prim.GetAttribute("uipc:constitution").Set(constitution)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        result = builder.add_usd(stage)
        model = builder.finalize(device="cpu")
        self.assertEqual(model.uipc.rigid_model, ["abd"])
        self.assertEqual(model.uipc.shape_constitution[builder.shape_label.index("/World/StaticEmpty")], "empty")
        self.assertEqual(model.uipc.shape_constitution[builder.shape_label.index("/World/StaticABD")], "abd")
        solver = newton.solvers.SolverUIPC(model, backend="none", logger_level=uipc.Logger.Error)
        solver.initialize(model.state())
        body = result["path_body_map"]["/World/Rigid"]
        dynamic_geo = solver.mapping.body_geo_slots[body].geometry()
        self.assertEqual(int(uipc.view(dynamic_geo.meta().find("constitution_uid"))[0]), 1)

        for name, uid, x_min, x_max in (
            ("static_colliders", 0, 1.95, 3.05),
            ("static_colliders_abd", 1, 0.95, 1.05),
        ):
            objects = solver.scene.objects().find(name)
            self.assertEqual(len(objects), 1)
            slot, _ = solver.scene.geometries().find(objects[0].geometries().ids()[0])
            geo = slot.geometry()
            self.assertEqual(int(uipc.view(geo.meta().find("constitution_uid"))[0]), uid)
            fixed_attrs = geo.vertices() if uid == 0 else geo.instances()
            np.testing.assert_array_equal(uipc.view(fixed_attrs.find("is_fixed")), 1)
            vertices = np.asarray(uipc.view(geo.positions())).reshape(-1, 3)
            self.assertAlmostEqual(float(vertices[:, 0].min()), x_min, places=5)
            self.assertAlmostEqual(float(vertices[:, 0].max()), x_max, places=5)

    def test_empty_constitution_rejects_body_backed_geometry(self):
        """Reject Empty on dynamic or kinematic bodies and their owned colliders."""
        for is_kinematic in (False, True):
            for attribute in ("rigid_model", "shape_constitution"):
                with self.subTest(is_kinematic=is_kinematic, attribute=attribute):
                    builder = newton.ModelBuilder()
                    newton.solvers.SolverUIPC.register_custom_attributes(builder)
                    body = builder.add_body(is_kinematic=is_kinematic)
                    shape = builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
                    model = builder.finalize(device="cpu")
                    getattr(model.uipc, attribute)[body if attribute == "rigid_model" else shape] = "empty"
                    solver = newton.solvers.SolverUIPC(model, backend="none", logger_level=uipc.Logger.Error)
                    with self.assertRaisesRegex(ValueError, "empty.*(static|world)"):
                        solver.initialize(model.state())

    def test_static_constitutions_work_without_rigid_bodies(self):
        """Build fixed ABD and Empty geometry in a world containing no rigid bodies."""
        builder = newton.ModelBuilder()
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        for x, constitution in ((0.0, "abd"), (0.5, "empty")):
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(x, 0.0, 0.5), wp.quat_identity()),
                hx=0.05,
                hy=0.05,
                hz=0.05,
                custom_attributes={"uipc:shape_constitution": constitution},
            )
        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverUIPC(model, backend="none", logger_level=uipc.Logger.Error)
        solver.initialize(model.state())
        self.assertEqual(model.body_count, 0)
        self.assertEqual(len(solver.scene.objects().find("static_colliders")), 1)
        self.assertEqual(len(solver.scene.objects().find("static_colliders_abd")), 1)


if __name__ == "__main__":
    unittest.main()
