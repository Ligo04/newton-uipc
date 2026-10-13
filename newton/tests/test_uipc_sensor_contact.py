# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SensorContact position reporting on SolverUIPC contacts."""

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorContact
from newton.tests.unittest_utils import get_selected_cuda_test_devices
from newton.viewer import ViewerNull

_HAS_UIPC = importlib.util.find_spec("uipc") is not None
_CUDA_TEST_DEVICES = get_selected_cuda_test_devices(mode="basic")

BOX_HALF = 0.5
STACK_GAP = 0.02
BOX_X, BOX_Y = 0.25, -0.15


class TestUIPCSensorContactPositions(unittest.TestCase):
    @unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
    def test_position_matrix_reports_contact_interfaces(self):
        """Report observable contact forces and positions on a native CUDA box stack."""
        import uipc  # noqa: PLC0415

        if not _CUDA_TEST_DEVICES:
            self.skipTest("Requires CUDA")
        for device in _CUDA_TEST_DEVICES:
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
                newton.solvers.SolverUIPC.register_custom_attributes(builder)
                ground_shape = builder.add_ground_plane()

                z0 = BOX_HALF + STACK_GAP
                body_bottom = builder.add_body(
                    xform=wp.transform(wp.vec3(BOX_X, BOX_Y, z0), wp.quat_identity()),
                    label="box_bottom",
                )
                builder.add_shape_box(body_bottom, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, label="shape_bottom")

                z1 = z0 + 2.0 * BOX_HALF + STACK_GAP
                body_top = builder.add_body(
                    xform=wp.transform(wp.vec3(BOX_X, BOX_Y, z1), wp.quat_identity()),
                    label="box_top",
                )
                top_shape = builder.add_shape_box(body_top, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, label="shape_top")

                model = builder.finalize(device=device)
                state_0 = model.state()
                state_1 = model.state()
                control = model.control()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

                dt = 1.0 / 60.0
                solver = newton.solvers.SolverUIPC(model, dt=dt, logger_level=uipc.Logger.Error)
                solver.set_contact(True, 0.001)
                solver.initialize(state_0)

                sensor = SensorContact(
                    model,
                    sensing_shapes=["shape_bottom"],
                    counterpart_shapes=["ground_plane", "shape_top"],
                    request_contact_attributes=False,
                )
                pipeline = newton.CollisionPipeline(
                    model, rigid_contact_max=solver.get_max_contact_count(), soft_contact_max=2
                )
                contacts = pipeline.contacts()
                observables = solver.observables(sensor.solver_observable_kinds, contacts=contacts)
                self.assertIsNone(contacts.force)
                self.assertIsNone(observables.body_qdd)
                self.assertIsNone(observables.body_parent_f)
                sensor.update(state_0, contacts, observables=observables)
                np.testing.assert_array_equal(sensor.total_force.numpy(), 0.0)
                observables.contact_f.fill_(wp.spatial_vector(9.0))
                contacts.soft_contact_count.fill_(1)

                for _ in range(30):
                    state_0.clear_forces()
                    solver.step(state_0, state_1, control, contacts, dt, observables=observables)
                    state_0, state_1 = state_1, state_0

                sensor.update(state_0, contacts, observables=observables)

                col_ground = sensor.counterpart_indices[0].index(ground_shape)
                col_top = sensor.counterpart_indices[0].index(top_shape)

                force_matrix = sensor.force_matrix.numpy()[0]
                self.assertGreater(force_matrix[col_ground][2], 0.1, "expected upward ground reaction on bottom box")
                self.assertLess(force_matrix[col_top][2], -0.1, "expected downward force from top box")

                positions = sensor.position_matrix.numpy()[0]
                pos_ground = positions[col_ground]
                pos_top = positions[col_top]

                # Ground interface sits at z≈0, top-box interface at z≈2*BOX_HALF. The
                # box-box xy is a force-weighted average of per-stencil vertex midpoints,
                # so it only localizes to within the contact face extent (BOX_HALF).
                self.assertAlmostEqual(pos_ground[0], BOX_X, delta=0.1, msg="ground contact x")
                self.assertAlmostEqual(pos_ground[1], BOX_Y, delta=0.1, msg="ground contact y")
                self.assertAlmostEqual(pos_ground[2], 0.0, delta=0.05, msg="ground contact z")
                self.assertAlmostEqual(pos_top[0], BOX_X, delta=BOX_HALF, msg="top contact x")
                self.assertAlmostEqual(pos_top[1], BOX_Y, delta=BOX_HALF, msg="top contact y")
                self.assertAlmostEqual(pos_top[2], z0 + BOX_HALF, delta=0.15, msg="top contact z")

                self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
                self.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 0)
                np.testing.assert_array_equal(observables.contact_f.numpy()[contacts.rigid_contact_max :], 0.0)
                np.testing.assert_array_equal(state_0.body_f.numpy(), 0.0)
                viewer = ViewerNull()
                viewer.set_model(model)
                viewer.show_contacts = True
                viewer.show_contact_forces = True
                with patch.object(viewer, "log_arrows", wraps=viewer.log_arrows) as log_arrows:
                    viewer.log_contacts(contacts, state_0, observables=observables)
                force_arrows = next(call for call in log_arrows.call_args_list if call.args[0] == "/contacts/forces")
                starts, ends = force_arrows.args[1:3]
                self.assertGreater(np.linalg.norm(ends.numpy() - starts.numpy()), 0.0)
                force_snapshot = observables.contact_f.numpy()
                with patch(
                    "newton._src.solvers.uipc.solver_uipc.prepare_contact_gpu_data",
                    side_effect=AssertionError("Unrequested contact readback"),
                ):
                    for request in (None, observables.select(set())):
                        solver.step(state_0, state_1, control, contacts, dt, observables=request)
                        state_0, state_1 = state_1, state_0
                np.testing.assert_array_equal(observables.contact_f.numpy(), force_snapshot)

                # A requested empty contact result must erase the previous forces.
                with patch("newton._src.solvers.uipc.solver_uipc.prepare_contact_gpu_data", return_value=None):
                    solver.step(state_0, state_1, control, contacts, dt, observables=observables)
                self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
                np.testing.assert_array_equal(observables.contact_f.numpy(), 0.0)
                sensor.update(state_1, contacts, observables=observables)
                np.testing.assert_array_equal(sensor.total_force.numpy(), 0.0)


if __name__ == "__main__":
    unittest.main()
