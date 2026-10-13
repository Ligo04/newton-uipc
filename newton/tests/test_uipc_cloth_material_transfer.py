# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check cloth material transfer against native UIPC geometry without a solver."""

import importlib.util
import unittest

import numpy as np
import warp as wp

import newton

_HAS_UIPC = importlib.util.find_spec("uipc") is not None
if _HAS_UIPC:
    from uipc import view
    from uipc.constitution import ElasticModuli2D, NeoHookeanShell, StrainLimitingBaraffWitkinShell
    from uipc.core import Scene
    from uipc.geometry import trimesh

    from newton._src.solvers.uipc.cloth import ClothBuilder
    from newton._src.solvers.uipc.converter import UIpcMappingInfo


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCClothMaterialTransfer(unittest.TestCase):
    @staticmethod
    def _build(*, radii=(0.0002,) * 4, cloth_model="strain_limiting_baraff_witkin", registered=True):
        builder = newton.ModelBuilder()
        if registered:
            newton.solvers.SolverUIPC.register_custom_attributes(builder)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=[wp.vec3(0, 0, 0), wp.vec3(2, 0, 0), wp.vec3(0, 1, 0), wp.vec3(2, 3, 0)],
            indices=[0, 1, 2, 1, 3, 2],
            density=0.2,
            particle_radius=0.0002,
            tri_ke=100.0,
            tri_ka=10000.0,
            tri_kd=0.0,
            edge_ke=1.0e-5,
            edge_kd=0.0,
        )
        builder.particle_radius[:] = radii
        model = builder.finalize(device="cpu")
        if registered:
            model.uipc.cloth_model = [cloth_model]
        scene = Scene(Scene.default_config())
        mapping = UIpcMappingInfo()
        converter = ClothBuilder(model, scene, mapping)
        converter.build(scene.contact_tabular().default_element())
        return model, mapping.cloth_geo_slots[0].geometry(), scene

    def test_mass_matches_newton_for_authored_and_inferred_density(self):
        """Preserve total mass with full thickness and uneven vertex radii."""
        for registered in (False, True):
            for radii in ((0.0002,) * 4, (0.0001, 0.0002, 0.0005, 0.002)):
                with self.subTest(registered=registered, radii=radii):
                    model, geometry, _scene = self._build(radii=radii, registered=registered)
                    density = float(view(geometry.meta().find("mass_density"))[0])
                    volume = np.asarray(view(geometry.vertices().find("volume"))).sum()
                    self.assertAlmostEqual(density * volume, float(model.particle_mass.numpy().sum()), places=6)

    def test_membrane_matches_native_constitution_for_lame_inputs(self):
        """Match native constitutions for each material row and triangle radius."""
        radii = (0.0001, 0.0002, 0.0005, 0.002)
        for cloth_model in ("strain_limiting_baraff_witkin", "neo_hookean"):
            with self.subTest(cloth_model=cloth_model):
                model, geometry, scene = self._build(radii=radii, cloth_model=cloth_model)
                # Rebuild with distinct material rows to catch accidental averaging.
                materials = model.tri_materials.numpy()
                materials[1, :2] = (200.0, 4000.0)
                model.tri_materials.assign(materials)
                mapping = UIpcMappingInfo()
                ClothBuilder(model, scene, mapping).build(scene.contact_tabular().default_element())
                geometry = mapping.cloth_geo_slots[0].geometry()
                positions = model.particle_q.numpy()
                for index, face in enumerate(model.tri_indices.numpy()):
                    native = trimesh(positions[face].astype(np.float64), np.array([[0, 1, 2]], dtype=np.int32))
                    mu, lam = materials[index, :2]
                    moduli = ElasticModuli2D.lame(float(lam), float(mu))
                    constitution = (
                        NeoHookeanShell() if cloth_model == "neo_hookean" else StrainLimitingBaraffWitkinShell()
                    )
                    constitution.apply_to(
                        native, moduli, mass_density=1000.0, thickness=float(np.mean(np.array(radii)[face]))
                    )
                    for name in ("mu", "lambda"):
                        self.assertAlmostEqual(
                            float(view(geometry.triangles().find(name))[index]),
                            float(view(native.triangles().find(name))[0]),
                            places=5,
                        )

    def test_cloth_without_particle_radius_uses_full_fallback_thickness(self):
        """Preserve mass and use the fallback radius when particle radii are absent."""
        model, geometry, scene = self._build()
        model.particle_radius = None
        mapping = UIpcMappingInfo()
        ClothBuilder(model, scene, mapping, default_thickness=0.003).build(scene.contact_tabular().default_element())
        geometry = mapping.cloth_geo_slots[0].geometry()
        np.testing.assert_allclose(view(geometry.vertices().find("thickness")), 0.003)
        density = float(view(geometry.meta().find("mass_density"))[0])
        volume = np.asarray(view(geometry.vertices().find("volume"))).sum()
        self.assertAlmostEqual(density * volume, float(model.particle_mass.numpy().sum()), places=6)


if __name__ == "__main__":
    unittest.main()
