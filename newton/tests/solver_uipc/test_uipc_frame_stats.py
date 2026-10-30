# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read live UIPC solver counters without enabling timer profiling."""

import importlib.util
import tempfile
import unittest

import warp as wp

import newton
from newton.tests.unittest_utils import get_selected_cuda_test_devices

_HAS_UIPC = importlib.util.find_spec("uipc") is not None
_CUDA_DEVICES = get_selected_cuda_test_devices(mode="basic")


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCFrameStats(unittest.TestCase):
    def test_uninitialized_counters(self):
        """Read zero counters before stepping without initializing the backend."""
        model = newton.ModelBuilder().finalize(device="cpu")
        with tempfile.TemporaryDirectory() as workspace:
            solver = newton.solvers.SolverUIPC(model, backend="none", workspace=workspace)
            stats = solver.get_frame_stats()
            self.assertEqual(stats["step_time_ms"], 0.0)
            self.assertEqual(stats["newton_iterations"], 0)
            self.assertEqual(stats["linear_solver_max_iterations"], 0)
            self.assertFalse(solver._initialized)
            solver.initialize()
            self.assertEqual(solver.get_frame_stats(), stats)

    @unittest.skipUnless(_CUDA_DEVICES, "a selected CUDA device is required")
    def test_live_counters_match_native_frames(self):
        """Refresh the last substep's counters for both PCG solvers and IPC pipelines."""
        import uipc  # noqa: PLC0415

        for device in _CUDA_DEVICES:
            with wp.ScopedDevice(device):
                builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
                builder.add_ground_plane()
                body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1001), wp.quat_identity()))
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
                model = builder.finalize()

                for pipeline, linear_solver in (("ipc", "fused_pcg"), ("ipc", "linear_pcg"), ("al-ipc", "linear_pcg")):
                    with self.subTest(device=str(device), pipeline=pipeline, linear_solver=linear_solver):
                        with tempfile.TemporaryDirectory() as workspace:
                            config = uipc.Scene.default_config()
                            config["contact"]["constitution"] = pipeline
                            config["linear_system"]["solver"] = linear_solver
                            config["linear_system"]["use_cuda_graph"] = 2 if linear_solver == "fused_pcg" else 0
                            solver = newton.solvers.SolverUIPC(model, workspace=workspace, scene_config=config)
                            solver.set_contact(True)
                            state_in, state_out = model.state(), model.state()
                            snapshots = []
                            for frame in (1, 2):
                                solver.step(state_in, state_out, None, dt=1.0 / 60.0)
                                state_in, state_out = state_out, state_in
                                stats = solver.get_frame_stats()
                                native = solver.engine.frame_stats()
                                self.assertEqual({k: v for k, v in stats.items() if k != "step_time_ms"}, native)
                                self.assertEqual(stats["frame"], frame)
                                self.assertEqual(stats["pipeline"], pipeline)
                                self.assertTrue(stats["completed"])
                                self.assertGreater(stats["newton_iterations"], 0)
                                self.assertGreater(stats["step_time_ms"], 0.0)
                                if "linear_solver_max_iterations" in stats:
                                    maximum = stats["linear_solver_max_iterations"]
                                    total = stats["linear_solver_iterations"]
                                    self.assertGreaterEqual(maximum, 0)
                                    self.assertLessEqual(maximum, total)
                                    self.assertLessEqual(total, maximum * stats["newton_iterations"])
                                snapshots.append(stats)
                            self.assertEqual(snapshots[0]["frame"], 1)
                            self.assertIsNone(solver._stats)


if __name__ == "__main__":
    unittest.main()
