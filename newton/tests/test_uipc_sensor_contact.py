# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SensorContact position reporting on SolverUIPC contacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorContact
from newton.tests.unittest_utils import get_selected_cuda_test_devices, requires
from newton.viewer import ViewerNull

_CUDA_TEST_DEVICES = get_selected_cuda_test_devices(mode="basic")

BOX_HALF = 0.5
STACK_GAP = 0.02
BOX_X, BOX_Y = 0.25, -0.15


class TestUIPCSensorContactPositions(unittest.TestCase):
    @requires("uipc")
    def test_rod_sensor_forces_match_native_contact_gradients(self):
        """Preserve native rod forces through segment filters, readback, and reset on CUDA.

        Half-planes exercise intersecting and parallel plane identities. Empty
        geometry exercises static FEM ownership beside the rod and an affine load.
        """
        import uipc  # noqa: PLC0415
        from uipc.geometry import Geometry  # noqa: PLC0415

        from newton._src.solvers.uipc.rod_contact_forces import _halfplane_topology  # noqa: PLC0415

        if not _CUDA_TEST_DEVICES:
            self.skipTest("Requires CUDA")

        dt = 1.0 / 120.0

        def native_rod_forces(solver, start, count):
            forces = {"N": np.zeros(3), "F": np.zeros(3)}
            plane_shapes = {
                int(np.asarray(uipc.view(slot.geometry().meta().find("global_vertex_offset")))[0]): shape
                for shape, slot in solver.mapping.ground_geo_slots.items()
            }
            plane_wrenches = {shape: np.zeros(6) for shape in plane_shapes.values()}
            rod_positions = solver.get_rod_state().positions.numpy()
            shared_halfplane_point = False
            for key in solver._csf.contact_primitive_types():
                channel = key.rsplit("+", 1)[-1]
                if channel not in forces:
                    continue
                gradients = Geometry()
                solver._csf.contact_gradient(key, gradients)
                indices_attr = gradients.instances().find("i")
                gradient_attr = gradients.instances().find("grad")
                if indices_attr is None or gradient_attr is None:
                    continue
                indices = np.asarray(uipc.view(indices_attr)).reshape(-1)
                values = np.asarray(uipc.view(gradient_attr)).reshape(-1, 3)
                selected = (indices >= start) & (indices < start + count)
                forces[channel] -= values[selected].sum(axis=0, dtype=np.float64) / dt**2
                if key.startswith("PH+"):
                    energy = Geometry()
                    solver._csf.contact_energy(key, energy)
                    topology = np.asarray(uipc.view(energy.instances().find("topo"))).reshape(-1, 2)
                    topology = _halfplane_topology(indices, topology)
                    if channel == "N":
                        _, occurrences = np.unique(indices[selected], return_counts=True)
                        shared_halfplane_point = bool(np.any(occurrences > 1))
                    for plane_vertex, shape in plane_shapes.items():
                        plane_rows = selected & (topology[:, 1] == plane_vertex)
                        plane_forces = -values[plane_rows].astype(np.float64) / dt**2
                        positions = rod_positions[indices[plane_rows] - start]
                        plane_wrenches[shape][:3] += plane_forces.sum(axis=0)
                        plane_wrenches[shape][3:] += np.cross(positions, plane_forces).sum(axis=0)
            return {
                **forces,
                "plane_wrenches": plane_wrenches,
                "shared_halfplane_point": shared_halfplane_point,
            }

        for device in _CUDA_TEST_DEVICES:
            for support in ("half_planes", "parallel_half_planes", "empty_box"):
                with (
                    self.subTest(device=str(device), support=support),
                    wp.ScopedDevice(device),
                    tempfile.TemporaryDirectory(prefix="newton-rod-contact-") as workspace,
                ):
                    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
                    newton.solvers.SolverUIPC.register_custom_attributes(builder)
                    if support == "half_planes":
                        support_shapes = [
                            builder.add_ground_plane(label="floor"),
                            builder.add_shape_plane(plane=(0.0, 1.0, 0.0, 0.05), label="wall"),
                        ]
                        rod_y = -0.03425
                    elif support == "parallel_half_planes":
                        support_shapes = [
                            builder.add_ground_plane(label="upper_floor"),
                            builder.add_ground_plane(height=-0.0005, label="lower_floor"),
                        ]
                        rod_y = 0.0
                    else:
                        support_shapes = [
                            builder.add_shape_box(
                                body=-1,
                                xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()),
                                hx=0.5,
                                hy=0.5,
                                hz=0.05,
                                label="empty_support",
                                custom_attributes={"uipc:shape_constitution": "empty"},
                            )
                        ]
                        rod_y = 0.0

                    rod_z = 0.0165
                    load_body = builder.add_body(
                        xform=wp.transform(wp.vec3(0.0, rod_y, rod_z + 0.04575), wp.quat_identity()),
                        label="load",
                    )
                    load_shape = builder.add_shape_box(body=load_body, hx=0.075, hy=0.012, hz=0.03, label="load_shape")
                    rod_bodies, _ = builder.add_rod(
                        rod=newton.Rod(
                            [(-0.18, rod_y, rod_z), (0.0, rod_y, rod_z), (0.18, rod_y, rod_z)],
                            radius=0.015,
                            youngs_modulus=2.0e5,
                            poissons_ratio=0.3,
                        ),
                        label="cable",
                    )
                    model = builder.finalize(device=device)
                    shape_bodies = model.shape_body.numpy()
                    rod_shapes = np.flatnonzero(np.isin(shape_bodies, rod_bodies)).tolist()
                    self.assertEqual(len(rod_shapes), 2)
                    counterpart_shapes = [*support_shapes, load_shape]

                    def initialize_solver(model, rod_bodies, support, workspace, name):
                        state, result = model.state(), model.state()
                        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                        velocities = state.body_qd.numpy()
                        velocities[rod_bodies, 0] = 0.15
                        velocities[rod_bodies, 1] = -0.08 if support == "half_planes" else 0.0
                        state.body_qd.assign(velocities)
                        solver = newton.solvers.SolverUIPC(
                            model,
                            dt=dt,
                            workspace=f"{workspace}/{name}",
                            logger_level=uipc.Logger.Error,
                            auto_sync_inertia=False,
                        )
                        solver.set_contact(True, 0.003)
                        with self.assertWarnsRegex(newton.NewtonWarning, "ROD.*centerline"):
                            solver.initialize(state)
                        return solver, state, result

                    solver, state, result = initialize_solver(model, rod_bodies, support, workspace, "reported")
                    reference_solver, reference_state, reference_result = initialize_solver(
                        model, rod_bodies, support, workspace, "unreported"
                    )
                    geometry = solver.mapping.rod_geo_slots[0].geometry()
                    global_start = int(np.asarray(uipc.view(geometry.meta().find("global_vertex_offset")))[0])
                    fem_start = int(np.asarray(uipc.view(geometry.meta().find("backend_fem_vertex_offset")))[0])
                    self.assertNotEqual(global_start, fem_start)
                    node_count = geometry.vertices().size()
                    sensor = SensorContact(
                        model,
                        sensing_shapes=[model.shape_label[shape] for shape in rod_shapes],
                        counterpart_shapes=[model.shape_label[shape] for shape in counterpart_shapes],
                        request_contact_attributes=False,
                    )
                    contacts = newton.Contacts(solver.get_max_contact_count(), 0, device=device)
                    observables = solver.observables(sensor.solver_observable_kinds, contacts=contacts)
                    peak_normal = 0.0
                    peak_friction = 0.0
                    max_force_error = 0.0
                    native_force_samples = 0
                    peak_counterpart = np.zeros(len(counterpart_shapes))
                    shared_halfplane_point_seen = False

                    for step in range(72):
                        state.clear_forces()
                        solver.step(state, result, None, contacts, dt, observables=observables)
                        state, result = result, state
                        sensor.update(state, contacts, observables=observables)

                        # The same initial trajectory must not depend on requesting CONTACT_F.
                        if step < 3:
                            reference_state.clear_forces()
                            reference_solver.step(reference_state, reference_result, None, dt=dt)
                            reference_state, reference_result = reference_result, reference_state
                            np.testing.assert_allclose(
                                state.body_q.numpy(), reference_state.body_q.numpy(), rtol=1.0e-5, atol=2.0e-5
                            )
                            np.testing.assert_allclose(
                                state.body_qd.numpy(), reference_state.body_qd.numpy(), rtol=1.0e-4, atol=2.0e-4
                            )
                            nodes = solver.get_rod_state().positions
                            reference_nodes = reference_solver.get_rod_state().positions
                            np.testing.assert_allclose(nodes.numpy(), reference_nodes.numpy(), rtol=1.0e-5, atol=2.0e-5)

                        if step < 8 or (step + 1) % 8 == 0:
                            native_force_samples += 1
                            native = native_rod_forces(solver, global_start, node_count)
                            plane_wrenches = native["plane_wrenches"]
                            shared_halfplane_point_seen |= native["shared_halfplane_point"]
                            measured_force = sensor.total_force.numpy().sum(axis=0)
                            native_force = native["N"] + native["F"]
                            max_force_error = max(max_force_error, float(np.max(np.abs(measured_force - native_force))))
                            np.testing.assert_allclose(
                                measured_force,
                                native_force,
                                rtol=2.0e-3,
                                atol=2.0e-3,
                            )
                            peak_normal = max(peak_normal, float(np.linalg.norm(native["N"])))
                            peak_friction = max(peak_friction, float(np.linalg.norm(native["F"])))
                            matrix = sensor.force_matrix.numpy()
                            np.testing.assert_allclose(
                                matrix.sum(axis=1), sensor.total_force.numpy(), rtol=1.0e-5, atol=1.0e-5
                            )
                            for index, shape in enumerate(counterpart_shapes):
                                column = sensor.counterpart_indices[0].index(shape)
                                peak_counterpart[index] = max(
                                    peak_counterpart[index], np.linalg.norm(matrix[:, column], axis=1).sum()
                                )
                            active = np.linalg.norm(matrix, axis=-1) > 1.0e-5
                            positions = sensor.position_matrix.numpy()[active]
                            self.assertTrue(np.all(np.isfinite(positions)))
                            if positions.size:
                                self.assertLess(float(np.max(np.abs(positions[:, 0]))), 0.5)
                                self.assertLess(float(np.max(np.abs(positions[:, 1]))), 0.2)
                                self.assertGreater(float(np.min(positions[:, 2])), -0.05)
                                self.assertLess(float(np.max(positions[:, 2])), 0.2)
                            count = int(contacts.rigid_contact_count.numpy()[0])
                            self.assertLessEqual(count, contacts.rigid_contact_max)
                            wrenches = observables.contact_f.numpy()[:count]
                            self.assertTrue(np.all(np.isfinite(wrenches)))
                            shape0 = contacts.rigid_contact_shape0.numpy()[:count]
                            shape1 = contacts.rigid_contact_shape1.numpy()[:count]
                            body_positions = state.body_q.numpy()[:, :3]
                            for shape, expected in plane_wrenches.items():
                                column = sensor.counterpart_indices[0].index(shape)
                                np.testing.assert_allclose(
                                    matrix[:, column].sum(axis=0), expected[:3], rtol=2.0e-3, atol=2.0e-3
                                )
                                rows = np.isin(shape0, rod_shapes) & (shape1 == shape)
                                centers = body_positions[shape_bodies[shape0[rows]]]
                                world_moments = wrenches[rows, 3:] + np.cross(centers, wrenches[rows, :3])
                                np.testing.assert_allclose(
                                    world_moments.sum(axis=0), expected[3:], rtol=2.0e-3, atol=2.0e-3
                                )

                    print(
                        "UIPC_ROD_CONTACT_EVIDENCE "
                        + json.dumps(
                            {
                                "device": str(device),
                                "support": support,
                                "dt": dt,
                                "steps": 72,
                                "shared_halfplane_point_seen": shared_halfplane_point_seen,
                                "native_force_samples": native_force_samples,
                                "global_vertex_offset": global_start,
                                "backend_fem_vertex_offset": fem_start,
                                "peak_native_normal_force_n": peak_normal,
                                "peak_native_friction_force_n": peak_friction,
                                "max_force_abs_error_n": max_force_error,
                                "counterpart_peak_force_n": {
                                    model.shape_label[shape]: float(peak_counterpart[index])
                                    for index, shape in enumerate(counterpart_shapes)
                                },
                            },
                            sort_keys=True,
                        )
                    )
                    self.assertGreater(peak_friction, 1.0e-4, "expected nonzero native sliding friction")
                    self.assertTrue(
                        np.all(peak_counterpart > 1.0e-4),
                        f"missing rod contact for counterpart shapes {counterpart_shapes}: {peak_counterpart}",
                    )
                    if support == "parallel_half_planes":
                        self.assertTrue(shared_halfplane_point_seen, "expected one rod node to contact both planes")

                    # A body-state reset must refresh FEM nodes and erase the old cable contacts.
                    reset_state = model.state()
                    reset_pose = state.body_q.numpy()
                    reset_pose[rod_bodies, :3] += np.array([3.0, 2.0, 2.0])
                    reset_state.body_q.assign(reset_pose)
                    reset_state.body_qd.zero_()
                    solver.reset(reset_state, flags=newton.StateFlags.BODY_Q | newton.StateFlags.BODY_QD)
                    state = reset_state
                    observables.contact_f.fill_(wp.spatial_vector(9.0))
                    for _ in range(4):
                        state.clear_forces()
                        solver.step(state, result, None, contacts, dt, observables=observables)
                        state, result = result, state
                    sensor.update(state, contacts, observables=observables)
                    np.testing.assert_allclose(sensor.total_force.numpy(), 0.0, atol=1.0e-6)
                    np.testing.assert_allclose(sensor.force_matrix.numpy(), 0.0, atol=1.0e-6)
                    nodes = solver.get_rod_state().positions
                    self.assertTrue(np.all(nodes.numpy()[:, 2] > 1.5))
                    native = native_rod_forces(solver, global_start, node_count)
                    np.testing.assert_allclose(native["N"] + native["F"], 0.0, atol=1.0e-6)

    @requires("uipc")
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

                # Reuse the arrays with a smaller reporting buffer and a new row layout.
                allocation_contacts = contacts
                contact_f = observables.contact_f
                contacts = newton.Contacts(allocation_contacts.rigid_contact_max // 2, 1, device=device)
                observables.contact_f.fill_(wp.spatial_vector(9.0))
                contacts.soft_contact_count.fill_(1)
                state_0.clear_forces()
                solver.step(state_0, state_1, control, contacts, dt, observables=observables)
                state_0, state_1 = state_1, state_0
                self.assertIs(observables.contacts, contacts)
                self.assertIs(observables.contact_f, contact_f)
                with self.assertRaisesRegex(ValueError, "produced with a different Contacts"):
                    sensor.update(state_0, allocation_contacts, observables=observables)
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
                with patch.object(
                    solver._contact_exporter,
                    "prepare",
                    side_effect=AssertionError("Unrequested contact readback"),
                ):
                    for request in (None, observables.select(set())):
                        solver.step(state_0, state_1, control, contacts, dt, observables=request)
                        state_0, state_1 = state_1, state_0
                np.testing.assert_array_equal(observables.contact_f.numpy(), force_snapshot)

                # A requested empty contact result must erase the previous forces.
                with patch.object(solver._contact_exporter, "prepare", return_value=()):
                    solver.step(state_0, state_1, control, contacts, dt, observables=observables)
                self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
                np.testing.assert_array_equal(observables.contact_f.numpy(), 0.0)
                sensor.update(state_1, contacts, observables=observables)
                np.testing.assert_array_equal(sensor.total_force.numpy(), 0.0)


if __name__ == "__main__":
    unittest.main()
