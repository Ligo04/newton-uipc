# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for UIPC scene configuration defaults."""

import importlib.util
import tempfile
import unittest

import warp as wp

import newton

_HAS_UIPC = importlib.util.find_spec("uipc") is not None
if _HAS_UIPC:
    import uipc


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCSceneConfig(unittest.TestCase):
    def test_default_scene_config_uses_full_gpu_fused_pcg(self):
        """Use full-GPU FusedPCG CUDA Graph defaults."""
        model = newton.ModelBuilder().finalize()

        solver = newton.solvers.SolverUIPC(model, backend="none", logger_level=uipc.Logger.Error)

        self.assertEqual(solver._scene_config["linear_system"]["solver"], "fused_pcg")
        self.assertEqual(solver._scene_config["linear_system"]["use_cuda_graph"], 2)
        self.assertEqual(solver._scene_config["linear_system"]["fem_preconditioner"], "mas")
        self.assertEqual(solver._scene_config["contact"]["constitution"], "ipc")

    def test_explicit_scene_config_preserves_linear_solver_options(self):
        """Preserve explicit scene configuration overrides."""
        model = newton.ModelBuilder().finalize()
        scene_config = uipc.Scene.default_config()
        scene_config["linear_system"]["solver"] = "linear_pcg"
        scene_config["linear_system"]["use_cuda_graph"] = 0
        scene_config["linear_system"]["fem_preconditioner"] = "diag"
        scene_config["contact"]["constitution"] = "al-ipc"

        solver = newton.solvers.SolverUIPC(
            model,
            backend="none",
            scene_config=scene_config,
            logger_level=uipc.Logger.Error,
        )

        self.assertEqual(solver._scene_config["linear_system"]["solver"], "linear_pcg")
        self.assertEqual(solver._scene_config["linear_system"]["use_cuda_graph"], 0)
        self.assertEqual(solver._scene_config["linear_system"]["fem_preconditioner"], "diag")
        self.assertEqual(solver._scene_config["contact"]["constitution"], "al-ipc")

    def test_explicit_scene_config_defaults_missing_fem_preconditioner_to_mas(self):
        """Use MAS in the UIPC scene when an explicit config omits its selector."""
        model = newton.ModelBuilder().finalize(device="cpu")

        for missing_key in ("fem_preconditioner", "linear_system"):
            with self.subTest(missing_key=missing_key), tempfile.TemporaryDirectory() as workspace:
                scene_config = uipc.Scene.default_config()
                if missing_key == "linear_system":
                    scene_config.pop("linear_system")
                else:
                    scene_config["linear_system"].pop("fem_preconditioner")

                solver = newton.solvers.SolverUIPC(
                    model,
                    backend="none",
                    workspace=workspace,
                    scene_config=scene_config,
                    logger_level=uipc.Logger.Error,
                )
                solver.initialize()

                selector = solver.scene.config().find("linear_system/fem_preconditioner")
                self.assertEqual(uipc.view(selector)[0], "mas")

    def test_default_contact_pairs_use_adaptive_kappa(self):
        """Use UIPC's adaptive-kappa sentinel for built-in contact pairs."""
        model = newton.ModelBuilder().finalize()
        observed_kappas = []

        def inspect_defaults(tabular, _world_index, ground_elem, env_elem, robo_elem, actor_elem):
            pairs = (
                (env_elem, env_elem),
                (env_elem, robo_elem),
                (env_elem, actor_elem),
                (ground_elem, env_elem),
                (ground_elem, robo_elem),
                (ground_elem, actor_elem),
                (robo_elem, robo_elem),
                (robo_elem, actor_elem),
                (actor_elem, actor_elem),
            )
            observed_kappas.extend(tabular.at(lhs.id(), rhs.id()).resistance() for lhs, rhs in pairs)

        solver = newton.solvers.SolverUIPC(model, backend="none", logger_level=uipc.Logger.Error)
        solver.configure_contact_tabular(inspect_defaults)
        solver.initialize()

        self.assertEqual(observed_kappas, [-1.0] * 9)

    def test_adaptive_kappa_bounds_are_public(self):
        """Expose the measured UIPC adaptive contact corridor on SolverUIPC."""
        self.assertEqual(newton.solvers.SolverUIPC.ADAPTIVE_KAPPA_MIN, 9538500988.573645)
        self.assertEqual(newton.solvers.SolverUIPC.ADAPTIVE_KAPPA_MAX, 953850098857.3645)

    def test_fem_contact_elements_are_independent_per_world(self):
        """Bind cloth and volume geometries to independently configurable world elements."""
        world = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverUIPC.register_custom_attributes(world)
        world.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.5),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=[wp.vec3(0.0), wp.vec3(0.1, 0.0, 0.0), wp.vec3(0.0, 0.1, 0.0)],
            indices=[0, 1, 2],
            density=0.2,
            particle_radius=0.0002,
        )
        world.add_soft_mesh(
            pos=wp.vec3(0.3, 0.0, 0.5),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=[wp.vec3(0.0), wp.vec3(0.1, 0.0, 0.0), wp.vec3(0.0, 0.1, 0.0), wp.vec3(0.0, 0.0, 0.1)],
            indices=[0, 1, 2, 3],
            density=1000.0,
            k_mu=50000.0,
            k_lambda=50000.0,
            k_damp=0.0,
        )
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        builder.replicate(world, world_count=2, spacing=(1.0, 0.0, 0.0))
        model = builder.finalize(device="cpu")
        elements = {}

        def configure(tabular, world_index, ground, env, robot, actor, cloth, deformable):
            elements[world_index] = (actor, cloth, deformable)
            tabular.insert(cloth, actor, 0.7, -1.0, False)

        with tempfile.TemporaryDirectory() as workspace:
            solver = newton.solvers.SolverUIPC(
                model, backend="none", workspace=workspace, logger_level=uipc.Logger.Error
            )
            solver.configure_contact_tabular(configure, include_fem=True)
            solver.initialize(model.state())
            self.assertEqual(len(solver.mapping.cloth_geo_slots), 2)
            self.assertEqual(len(solver.mapping.deformable_geo_slots), 2)
            ids = []
            for world_index in range(2):
                actor, cloth, deformable = elements[world_index]
                ids.extend(element.id() for element in (actor, cloth, deformable))
                for slot, element in (
                    (solver.mapping.cloth_geo_slots[world_index], cloth),
                    (solver.mapping.deformable_geo_slots[world_index], deformable),
                ):
                    actual = int(uipc.view(slot.geometry().meta().find("contact_element_id"))[0])
                    self.assertEqual(actual, element.id())
                tabular = solver.scene.contact_tabular()
                self.assertFalse(tabular.at(cloth.id(), actor.id()).is_enabled())
                self.assertTrue(tabular.at(deformable.id(), actor.id()).is_enabled())
                self.assertTrue(tabular.at(cloth.id(), deformable.id()).is_enabled())
                self.assertEqual(tabular.at(cloth.id(), deformable.id()).resistance(), -1.0)
            self.assertEqual(len(set(ids)), 6)
