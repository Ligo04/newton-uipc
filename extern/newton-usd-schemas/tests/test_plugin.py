# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import pathlib
import unittest

from pxr import Plug, Usd, UsdGeom

import newton_usd_schemas


class TestNewtonPlugin(unittest.TestCase):
    def test_newton_plugin_registered(self):
        plugin = Plug.Registry().GetPluginWithName("newton")
        self.assertIsInstance(plugin, Plug.Plugin)
        self.assertEqual(plugin.resourcePath, pathlib.Path(newton_usd_schemas.__file__).parent.as_posix())

    def test_uipc_deformable_schema_defines_constitution(self):
        """Apply the UIPC schema and inspect its declared constitution attribute."""
        stage = Usd.Stage.CreateInMemory()
        prim = UsdGeom.TetMesh.Define(stage, "/Soft").GetPrim()
        self.assertTrue(prim.CanApplyAPI("NewtonUIPCDeformableAPI"))
        self.assertTrue(prim.ApplyAPI("NewtonUIPCDeformableAPI"))
        attr = prim.GetAttribute("uipc:constitution")
        self.assertFalse(attr.IsCustom())
        self.assertEqual(attr.Get(), "stable_neo_hookean")
        self.assertEqual(list(attr.GetMetadata("allowedTokens")), ["stable_neo_hookean", "arap", "empty"])
        attr.Set("empty")
        self.assertEqual(attr.Get(), "empty")
        self.assertFalse(UsdGeom.Mesh.Define(stage, "/Cloth").GetPrim().CanApplyAPI("NewtonUIPCDeformableAPI"))

    def test_uipc_rod_schema_defines_constitution(self):
        """Declare the centerline model without marking curves as simulation geometry."""
        stage = Usd.Stage.CreateInMemory()
        prim = UsdGeom.BasisCurves.Define(stage, "/Rod").GetPrim()
        self.assertTrue(prim.CanApplyAPI("NewtonUIPCRodAPI"))
        self.assertTrue(prim.ApplyAPI("NewtonUIPCRodAPI"))
        attribute = prim.GetAttribute("uipc:constitution")
        self.assertFalse(attribute.IsCustom())
        self.assertEqual(attribute.Get(), "hookean_spring")
        self.assertEqual(list(attribute.GetMetadata("allowedTokens")), ["hookean_spring"])
        self.assertFalse(prim.HasAPI("PhysicsCurvesDeformableSimAPI"))
        self.assertFalse(UsdGeom.Mesh.Define(stage, "/Cloth").GetPrim().CanApplyAPI("NewtonUIPCRodAPI"))

    def test_uipc_cloth_schema_defines_constitution(self):
        """Declare membrane choices on a cloth simulation mesh."""
        stage = Usd.Stage.CreateInMemory()
        prim = UsdGeom.Mesh.Define(stage, "/Cloth").GetPrim()
        self.assertTrue(prim.CanApplyAPI("NewtonUIPCClothAPI"))
        self.assertTrue(prim.ApplyAPI("NewtonUIPCClothAPI"))
        attr = prim.GetAttribute("uipc:constitution")
        self.assertFalse(attr.IsCustom())
        self.assertEqual(attr.Get(), "strain_limiting_baraff_witkin")
        self.assertEqual(list(attr.GetMetadata("allowedTokens")), ["strain_limiting_baraff_witkin", "neo_hookean"])
        attr.Set("neo_hookean")
        self.assertEqual(attr.Get(), "neo_hookean")
        self.assertFalse(UsdGeom.TetMesh.Define(stage, "/Soft").GetPrim().CanApplyAPI("NewtonUIPCClothAPI"))

    def test_uipc_rigid_body_schema_defines_constitution(self):
        """Declare ABD and Empty choices for rigid-body and static-collider prims."""
        stage = Usd.Stage.CreateInMemory()
        for path, schema in (("/Body", UsdGeom.Xform), ("/Collider", UsdGeom.Cube)):
            prim = schema.Define(stage, path).GetPrim()
            self.assertTrue(prim.CanApplyAPI("NewtonUIPCRigidBodyAPI"))
            self.assertTrue(prim.ApplyAPI("NewtonUIPCRigidBodyAPI"))
            attr = prim.GetAttribute("uipc:constitution")
            self.assertFalse(attr.IsCustom())
            self.assertIsNone(attr.Get())
            self.assertEqual(list(attr.GetMetadata("allowedTokens")), ["abd", "empty"])
            attr.Set("empty")
            self.assertEqual(attr.Get(), "empty")


if __name__ == "__main__":
    unittest.main()
