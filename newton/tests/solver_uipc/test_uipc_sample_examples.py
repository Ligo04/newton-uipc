# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check native sample materials and contact-coupled scene startup."""

import importlib.util
import unittest

import numpy as np
import warp as wp

import newton.examples
from newton.tests.unittest_utils import get_selected_cuda_test_devices

_HAS_UIPC = importlib.util.find_spec("uipc") is not None
_CUDA_DEVICES = get_selected_cuda_test_devices(mode="basic")


@unittest.skipUnless(_HAS_UIPC and _CUDA_DEVICES, "uipc and a selected CUDA device are required")
class TestUIPCSampleExamples(unittest.TestCase):
    def setUp(self):
        self.enterContext(wp.ScopedDevice(_CUDA_DEVICES[0]))

    def test_sample_88_matches_native_materials(self):
        import uipc  # noqa: PLC0415
        from uipc.constitution import (  # noqa: PLC0415
            DiscreteShellBending,
            ElasticModuli,
            ElasticModuli2D,
            StableNeoHookean,
            StrainLimitingBaraffWitkinShell,
        )

        from newton.examples.uipc.multiphysics.example_uipc_stiff_gipc_benchmark import Example  # noqa: PLC0415

        example = Example(newton.viewer.ViewerNull(), newton.examples.default_args(Example.create_parser()))
        self.assertEqual(example.model.particle_count, 42611)
        self.assertEqual(example.model.tet_count, 159870)
        self.assertEqual(len(example.solver.mapping.deformable_geo_slots), 2)

        for slot in example.solver.mapping.deformable_geo_slots:
            actual = slot.geometry()
            reference = actual.copy()
            StableNeoHookean().apply_to(reference, ElasticModuli.youngs_poisson(1.0e4, 0.49), 1000.0)
            for name in ("lambda", "mu"):
                np.testing.assert_allclose(
                    uipc.view(actual.tetrahedra().find(name)),
                    uipc.view(reference.tetrahedra().find(name)),
                    rtol=2.0e-6,
                )
            np.testing.assert_allclose(uipc.view(actual.meta().find("mass_density")), 1000.0, rtol=1.0e-6)

        cloth = example.solver.mapping.cloth_geo_slots[0].geometry()
        self.assertEqual(len(uipc.view(cloth.triangles().topo())), 8192)
        reference = cloth.copy()
        StrainLimitingBaraffWitkinShell().apply_to(
            reference,
            stretch_moduli=ElasticModuli2D.youngs_poisson(5.0e4, 0.49),
            shear_moduli=ElasticModuli2D.youngs_poisson(10.0, 0.49),
            mass_density=200.0,
            thickness=0.001,
            strain_rate=100.0,
        )
        DiscreteShellBending().apply_to(reference, 3.0e4, 0.49)
        for domain, names in (
            ("triangles", ("lambda", "mu", "strain_rate")),
            ("edges", ("bending_stiffness",)),
            ("vertices", ("thickness",)),
            ("meta", ("mass_density",)),
        ):
            for name in names:
                np.testing.assert_allclose(
                    uipc.view(getattr(cloth, domain)().find(name)),
                    uipc.view(getattr(reference, domain)().find(name)),
                    rtol=2.0e-6,
                )

        for _ in range(3):
            example.step()
            example.test_post_step()
        example.test_final()
        self.assertEqual(example.solver.get_frame_stats()["frame"], 3)

    def test_wrecking_balls_preserves_contact_chain(self):
        import uipc  # noqa: PLC0415

        from newton.examples.uipc.contacts.example_uipc_rigid_wrecking_balls import Example  # noqa: PLC0415

        example = Example(newton.viewer.ViewerNull(), newton.examples.default_args(Example.create_parser()))
        self.assertEqual(example.model.body_count, 574)
        # Newton adds a root Free joint for each independent rigid body.
        self.assertTrue(np.all(example.model.joint_parent.numpy() == -1))
        self.assertTrue(np.all(example.model.joint_type.numpy() == newton.JointType.FREE))
        self.assertEqual(len(example.fixed_bodies), 1)
        self.assertEqual(sum(label.startswith("cube_") for label in example.model.body_label), 560)
        self.assertEqual(sum(label.startswith("link_") for label in example.model.body_label), 13)
        for body, slot in example.solver.mapping.body_geo_slots.items():
            expected = 1000.0 if example.model.body_label[body].startswith("cube_") else 7680.0
            np.testing.assert_allclose(uipc.view(slot.geometry().meta().find("mass_density")), expected, rtol=1.0e-5)

        for _ in range(6):
            example.step()
            example.test_post_step()
        example.test_final()
        self.assertEqual(example.solver.get_frame_stats()["frame"], 6)


if __name__ == "__main__":
    unittest.main()
