# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check numerical ownership and wrench conversion for FEM rod reports."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import has_module, requires

_HAS_UIPC = has_module("uipc")
if _HAS_UIPC:
    import uipc

    from newton._src.solvers.uipc.rod_contact_forces import RodContactReporter, _halfplane_topology, _merge_stencils


@requires("uipc")
class TestUIPCRodContactForces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        builder = newton.ModelBuilder()
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        cls.floor = builder.add_ground_plane(label="floor")
        cls.other_floor = builder.add_ground_plane(height=-2.0, label="other_floor")
        body = builder.add_body(xform=wp.transform(wp.vec3(4.0, 1.0, 1.0)))
        cls.box = builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        cls.shapeless = builder.add_link(label="shapeless")
        fixed = builder.add_joint_fixed(-1, cls.shapeless)
        builder.add_articulation([fixed])
        bodies, _ = builder.add_rod(rod=newton.Rod([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0)], radius=0.01))
        builder.body_mass[bodies[1]] *= 3.0
        cls.model = builder.finalize("cpu")
        cls.solver = newton.solvers.SolverUIPC(cls.model, backend="none", logger_level=uipc.Logger.Error)
        cls.solver.initialize(cls.model.state())
        mapping = cls.solver.mapping

        def offset(slot, value):
            geometry = slot.geometry()
            attribute = geometry.meta().find("global_vertex_offset")
            if attribute is None:
                attribute = geometry.meta().create("global_vertex_offset", 0)
            uipc.view(attribute)[:] = value

        for body, slot in mapping.body_geo_slots.items():
            offset(slot, 100 if body == cls.shapeless else 5)
        offset(mapping.rod_geo_slots[0], 200)
        for shape, slot in mapping.ground_geo_slots.items():
            offset(slot, 400 + shape)
        cls.reporter = RodContactReporter(cls.model, mapping, cls.solver._rod_builder)
        cls.reporter._refresh_positions()
        cls.rod_shapes = [mapping.body_shapes[body][0] for body in bodies]

    def test_shared_point_conserves_force_and_world_moment(self):
        """Partition a shared point by nodal weights and publish correct COM wrenches."""

        force = np.array([1.0, 0.0, 2.0])
        topology = np.array([[201, 400 + self.floor]])
        gradients = np.array([[[-0.01, 0.0, -0.02], [0.01, 0.0, 0.02]]])
        data = self.reporter.pack(_merge_stencils([("PH", "N", topology, gradients)], 0.1))
        self.assertEqual(set(data.shape0), set(self.rod_shapes))
        np.testing.assert_array_equal(data.shape1, self.floor)
        np.testing.assert_allclose(data.force, np.array([0.25, 0.75])[:, None] * force, atol=1e-7)
        np.testing.assert_allclose(data.point0, [[1.0, 0.0, 1.0]] * 2)
        np.testing.assert_allclose(data.point1, [[1.0, 0.0, 0.0]] * 2)
        np.testing.assert_allclose(data.force.sum(axis=0), force)
        np.testing.assert_allclose(data.moment.sum(axis=0), np.cross([1.0, 0.0, 1.0], force))

        contacts = newton.Contacts(8, 0, device="cpu")
        observables = self.solver.observables({"contact_f"}, contacts=contacts)
        state = self.model.state()
        self.reporter._publish(data, contacts, state, observables.contact_f)
        np.testing.assert_array_equal(contacts.rigid_contact_count.numpy(), [2])
        wrenches = observables.contact_f.numpy()[:2]
        centers = state.body_q.numpy()[:, :3]
        bodies = self.model.shape_body.numpy()[data.shape0]
        np.testing.assert_allclose(wrenches[:, 3:] + np.cross(centers[bodies], wrenches[:, :3]), data.moment, atol=1e-6)
        np.testing.assert_allclose(contacts.rigid_contact_normal.numpy()[:2], data.normal)

    def test_device_packing_matches_shape_reference_and_reuses_buffers(self):
        """Match canonical host records on CPU/CUDA across force channels and reuse."""
        from unittest.mock import patch  # noqa: PLC0415

        from newton._src.solvers.uipc.contact_export import ContactExporter  # noqa: PLC0415
        from newton.tests.test_uipc_rod import TestUIPCRod  # noqa: PLC0415
        from newton.tests.unittest_utils import get_selected_cuda_test_devices  # noqa: PLC0415

        solvers = [self.solver]
        fixture = TestUIPCRod()
        for device in get_selected_cuda_test_devices(mode="basic"):
            model, _, _ = fixture._model(device=device, colliders=True, ground_plane=True)
            solvers.append(fixture._initialize(model, backend="cuda", contact=True))
        names = ("shape0", "shape1", "point0", "point1", "normal", "force", "moment", "particles", "barycentric")

        def ordered(arrays):
            keys = np.column_stack((arrays[0], arrays[1], arrays[2], arrays[3]))
            order = np.lexsort(keys.T[::-1])
            return [array[order] for array in arrays]

        for solver in solvers:
            model = solver.model
            with self.subTest(device=str(model.device)):
                exporter = ContactExporter(
                    model, solver.mapping, solver._rod_builder, solver._body_affine, solver._body_affine_valid
                )
                reference = RodContactReporter(model, solver.mapping, solver._rod_builder)
                reference._refresh_positions()
                nodes = sorted(reference._nodes)
                planes = sorted(reference._planes)
                rigid = np.flatnonzero(reference._shapes >= 0)
                rigid = rigid[model.shape_body.numpy()[reference._shapes[rigid]] >= 0]
                shape = reference._shapes[rigid[0]]
                rigid = rigid[reference._shapes[rigid] == shape]
                first, second = next(iter(reference._edges))
                topologies = {
                    "PH": [[nodes[0], planes[0]], [nodes[1], planes[0]], [nodes[1], planes[-1]], [nodes[0], planes[0]]],
                    "PP": [[nodes[1], rigid[0]], [rigid[0], nodes[0]]],
                    "PE": [[rigid[0], first, second]],
                    "PT": [[nodes[1], *rigid[:3]]],
                    "EE": [[rigid[0], rigid[1], first, second]],
                }
                permutations = {"PH": [0, 1], "PP": [1, 0], "PE": [0, 2, 1], "PT": [0, 3, 1, 2], "EE": [3, 2, 1, 0]}
                rng = np.random.default_rng(1729)
                channels = []
                for primitive, values in topologies.items():
                    topology = np.asarray(values, dtype=np.int32)
                    normal = rng.normal(size=(*topology.shape, 3))
                    friction_topology = topology[::-1, permutations[primitive]]
                    friction = rng.normal(size=(*friction_topology.shape, 3))
                    if primitive == "PH":
                        friction_topology = np.vstack((friction_topology, [nodes[2], planes[-1]]))
                        friction = np.concatenate((friction, rng.normal(size=(1, 2, 3))))
                        normal[:, 1] = -normal[:, 0]
                        friction[:, 1] = -friction[:, 0]
                    channels.extend(((primitive, "N", topology, normal), (primitive, "F", friction_topology, friction)))
                dt = 0.1
                expected = reference.pack(_merge_stencils(channels, dt))
                expected_arrays = ordered([getattr(expected, name) for name in names])
                state = model.state()
                pointers = None
                for _ in range(2):
                    records = []
                    with patch("newton._src.solvers.uipc.contact_export._read_contact_channels", return_value=channels):
                        for arrays in exporter.prepare(None, dt, state):
                            host = [array.numpy() for array in arrays]
                            selected = host[0] >= 0
                            records.append([array[selected].copy() for array in host])
                    actual = ordered([np.concatenate([record[i] for record in records]) for i in range(len(names))])
                    for name, actual_array, expected_array in zip(names, actual, expected_arrays, strict=True):
                        if np.issubdtype(expected_array.dtype, np.integer):
                            np.testing.assert_array_equal(actual_array, expected_array, err_msg=name)
                        else:
                            np.testing.assert_allclose(
                                actual_array, expected_array, atol=2.0e-6, rtol=2.0e-6, err_msg=name
                            )
                    current = {key: array.ptr for key, array in exporter.buffers.items()}
                    if pointers is not None:
                        self.assertEqual(current, pointers)
                    pointers = current
                self.assertIs(exporter.rods.positions, solver.get_rod_state().positions)
                with patch("newton._src.solvers.uipc.contact_export._read_contact_channels", return_value=[]):
                    self.assertEqual(list(exporter.prepare(None, dt, state)), [])

                contacts = newton.Contacts(2, 0, device=model.device)
                observables = solver.observables({"contact_f"}, contacts=contacts)
                with patch("newton._src.solvers.uipc.contact_export._read_contact_channels", return_value=channels):
                    exporter.export(None, contacts, state, observables.contact_f, dt)
                self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 2)
                self.assertTrue(np.all(np.isfinite(observables.contact_f.numpy())))
                unmapped = (reference._shapes < 0) & (reference._particles < 0)
                unmapped[list(reference._nodes)] = False
                invalid_vertices = [len(reference._shapes), *np.flatnonzero(unmapped)[:1]]
                for vertex in invalid_vertices:
                    invalid = [("PH", "N", np.array([[vertex, planes[0]]]), np.ones((1, 2, 3)))]
                    with patch("newton._src.solvers.uipc.contact_export._read_contact_channels", return_value=invalid):
                        with self.assertRaisesRegex(RuntimeError, "provenance"):
                            exporter.export(None, contacts, state, observables.contact_f, dt)
                    np.testing.assert_array_equal(contacts.rigid_contact_count.numpy(), 0)
                    np.testing.assert_array_equal(contacts.soft_contact_count.numpy(), 0)
                    np.testing.assert_array_equal(observables.contact_f.numpy(), 0.0)

    def test_edge_belongs_to_single_segment(self):
        """Assign an entire rod edge to its segment instead of splitting its endpoints."""

        topology = np.array([[5, 200, 201]])
        normal = np.array([[[-0.03, 0.0, 0.0], [0.012, 0.0, 0.0], [0.018, 0.0, 0.0]]])
        data = self.reporter.pack(_merge_stencils([("PE", "N", topology, normal)], 0.1))
        np.testing.assert_array_equal(data.shape0, [self.box])
        np.testing.assert_array_equal(data.shape1, [self.rod_shapes[0]])
        np.testing.assert_allclose(data.force, [[3.0, 0.0, 0.0]], atol=1e-6)
        np.testing.assert_allclose(data.point1, [[0.6, 0.0, 1.0]], atol=1e-6)
        point = self.reporter._positions[5]
        np.testing.assert_allclose(data.moment[0], np.cross(point, data.force[0]), atol=1e-6)

    def test_channels_join_by_topology_and_keep_friction_only(self):
        """Retain reordered, duplicate, normal-only, and friction-only native stencils."""

        normal_topology = np.array([[200, 400], [201, 400], [200, 400]])
        friction_topology = np.array([[202, 400], [200, 400]])
        normal = np.zeros((3, 2, 3))
        normal[:, 0, 2] = [-0.01, -0.02, -0.03]
        normal[:, 1] = -normal[:, 0]
        friction = np.zeros((2, 2, 3))
        friction[:, 0, 0] = [-0.05, -0.06]
        friction[:, 1] = -friction[:, 0]
        stencils = _merge_stencils(
            [
                ("PH", "N", normal_topology, normal),
                ("PH", "F", friction_topology, friction),
            ],
            0.1,
        )
        self.assertEqual(len(stencils), 3)
        np.testing.assert_allclose(stencils[("PH", 200, 400)][1][0], [0.0, 0.0, 4.0])
        np.testing.assert_allclose(stencils[("PH", 200, 400)][2][0], [6.0, 0.0, 0.0])
        np.testing.assert_allclose(stencils[("PH", 201, 400)][2], 0.0)
        np.testing.assert_allclose(stencils[("PH", 202, 400)][1], 0.0)
        np.testing.assert_allclose(stencils[("PH", 202, 400)][2][0], [5.0, 0.0, 0.0])

        energy_topology = np.array([[8, 11], [9, 11], [10, 11], [4, 12], [5, 12], [8, 12], [9, 12], [10, 12]])
        gradient_indices = np.array([8, 9, 10, 5, 8, 9, 10])
        np.testing.assert_array_equal(
            _halfplane_topology(gradient_indices, energy_topology), energy_topology[[0, 1, 2, 4, 5, 6, 7]]
        )
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            _halfplane_topology(np.array([8]), np.array([[8, 11], [8, 12]]))

    def test_canonical_topology_reorders_gradients_with_features(self):
        """Match reversed points, edges, and triangle order without changing vertex forces."""

        cases = {
            "PP": ([1, 2], [1, 0]),
            "PE": ([1, 2, 3], [0, 2, 1]),
            "PT": ([1, 2, 3, 4], [0, 3, 1, 2]),
            "EE": ([1, 2, 3, 4], [3, 2, 1, 0]),
        }
        for primitive, (vertex_list, permutation) in cases.items():
            with self.subTest(primitive=primitive):
                vertices = np.asarray(vertex_list)
                gradient = np.arange(len(vertices) * 3).reshape(len(vertices), 3).astype(float)
                stencils = _merge_stencils(
                    [
                        (primitive, "N", vertices[None], gradient[None]),
                        (primitive, "F", vertices[permutation][None], gradient[permutation][None]),
                    ],
                    1.0,
                )
                self.assertEqual(len(stencils), 1)
                _, normal, friction = next(iter(stencils.values()))
                np.testing.assert_array_equal(normal, -gradient)
                np.testing.assert_array_equal(friction, -gradient)

    def test_halfplanes_and_current_node_positions(self):
        """Preserve exact plane shape identity and refresh deformed node coordinates."""

        geometry = self.solver.mapping.rod_geo_slots[0].geometry()
        positions = uipc.view(geometry.positions())
        original = positions.copy()
        try:
            positions.reshape(-1, 3)[0] += [0.1, 0.2, 0.3]
            self.reporter._refresh_positions()
            topology = np.array([[200, 400 + self.other_floor]])
            gradient = np.array([[[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]])
            data = self.reporter.pack(_merge_stencils([("PH", "N", topology, gradient)], 1.0))
            np.testing.assert_array_equal(data.shape1, [self.other_floor])
            np.testing.assert_allclose(data.point0, [[0.1, 0.2, 1.3]], atol=1e-6)
            np.testing.assert_allclose(data.point1, [[0.1, 0.2, -2.0]], atol=1e-6)
        finally:
            positions[:] = original
            self.reporter._refresh_positions()

    def test_parallel_halfplanes_preserve_reordered_forces_and_moments(self):
        """Keep per-plane shape forces and world moments when channel orders differ."""

        dt = 0.1
        floor_vertex = 400 + self.floor
        other_vertex = 400 + self.other_floor
        normal_topology = np.array([[201, floor_vertex], [201, other_vertex], [200, floor_vertex]])
        friction_topology = np.array([[201, other_vertex], [200, floor_vertex], [201, floor_vertex]])
        normal = np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 7.0], [0.0, 0.0, 2.0]])
        friction = np.array([[-0.3, 0.0, 0.0], [1.1, 0.0, 0.0], [0.8, 0.0, 0.0]])
        data = self.reporter.pack(
            _merge_stencils(
                [
                    ("PH", "N", normal_topology, dt**2 * np.stack((-normal, normal), axis=1)),
                    ("PH", "F", friction_topology, dt**2 * np.stack((-friction, friction), axis=1)),
                ],
                dt,
            )
        )
        self.assertEqual(len(data.force), 5)
        shared_point = np.array([1.0, 0.0, 1.0])
        first_point = np.array([0.0, 0.0, 1.0])
        floor_force = normal[0] + friction[2]
        other_force = normal[1] + friction[0]
        first_force = normal[2] + friction[1]
        expected = {
            (self.rod_shapes[0], self.floor): (
                0.25 * floor_force + first_force,
                0.25 * np.cross(shared_point, floor_force) + np.cross(first_point, first_force),
            ),
            (self.rod_shapes[1], self.floor): (0.75 * floor_force, 0.75 * np.cross(shared_point, floor_force)),
            (self.rod_shapes[0], self.other_floor): (0.25 * other_force, 0.25 * np.cross(shared_point, other_force)),
            (self.rod_shapes[1], self.other_floor): (0.75 * other_force, 0.75 * np.cross(shared_point, other_force)),
        }
        for (segment, plane), (force, moment) in expected.items():
            with self.subTest(segment=segment, plane=plane):
                selected = (data.shape0 == segment) & (data.shape1 == plane)
                self.assertTrue(np.any(selected))
                np.testing.assert_allclose(data.force[selected].sum(axis=0), force, atol=1.0e-6)
                np.testing.assert_allclose(data.moment[selected].sum(axis=0), moment, atol=1.0e-6)

        contacts = newton.Contacts(8, 0, device="cpu")
        observables = self.solver.observables({"contact_f"}, contacts=contacts)
        state = self.model.state()
        self.reporter._publish(data, contacts, state, observables.contact_f)
        np.testing.assert_array_equal(contacts.rigid_contact_shape0.numpy()[:5], data.shape0)
        np.testing.assert_array_equal(contacts.rigid_contact_shape1.numpy()[:5], data.shape1)
        wrenches = observables.contact_f.numpy()[:5]
        centers = state.body_q.numpy()[self.model.shape_body.numpy()[data.shape0], :3]
        np.testing.assert_allclose(wrenches[:, :3], data.force, atol=1.0e-6)
        np.testing.assert_allclose(wrenches[:, 3:] + np.cross(centers, wrenches[:, :3]), data.moment, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
