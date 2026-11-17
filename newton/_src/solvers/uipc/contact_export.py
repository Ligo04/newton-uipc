# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pack native contact snapshots on the solver device with reusable buffers."""

from __future__ import annotations

import numpy as np
import warp as wp

from .contact_forces import _PRIM_SPLIT, PRIMITIVE_TYPES
from .rod_contact_forces import RodContactReporter, _read_contact_channels, populate_rod_contacts
from .utils import _view_attr


@wp.kernel(enable_backward=False)
def _refresh_positions(
    local: wp.array[wp.vec3d],
    bodies: wp.array[int],
    particles: wp.array[int],
    nodes: wp.array[int],
    affine: wp.array[wp.mat44d],
    affine_valid: wp.array[wp.bool],
    body_q: wp.array[wp.transform],
    particle_q: wp.array[wp.vec3],
    rod_q: wp.array[wp.vec3d],
    positions: wp.array[wp.vec3d],
):
    vertex = wp.tid()
    body, particle, node = bodies[vertex], particles[vertex], nodes[vertex]
    if node >= 0:
        positions[vertex] = rod_q[node]
    elif particle >= 0:
        positions[vertex] = wp.vec3d(particle_q[particle])
    elif body >= 0:
        point = local[vertex]
        if affine_valid[body]:
            transformed = affine[body] * wp.vec4d(point[0], point[1], point[2], wp.float64(1.0))
            positions[vertex] = wp.vec3d(transformed[0], transformed[1], transformed[2])
        else:
            positions[vertex] = wp.vec3d(wp.transform_point(body_q[body], wp.vec3(point)))


@wp.kernel(enable_backward=False)
def _canonicalize(
    raw_topology: wp.array2d[int],
    raw_gradient: wp.array2d[wp.vec3d],
    primitive: int,
    arity: int,
    topology: wp.array2d[int],
    gradient: wp.array2d[wp.vec3d],
):
    row = wp.tid()
    order = wp.vec4i(0, 1, 2, 3)
    if primitive == 4:
        for start in range(0, 4, 2):
            if raw_topology[row, start] > raw_topology[row, start + 1]:
                order[start] = start + 1
                order[start + 1] = start
        first, second = raw_topology[row, order[0]], raw_topology[row, order[2]]
        if first > second or (first == second and raw_topology[row, order[1]] > raw_topology[row, order[3]]):
            order = wp.vec4i(order[2], order[3], order[0], order[1])
    elif primitive > 0:
        start = int(1)
        if primitive == 1:
            start = 0
        for a in range(start, arity):
            for b in range(a + 1, arity):
                if raw_topology[row, order[a]] > raw_topology[row, order[b]]:
                    previous = order[a]
                    order[a] = order[b]
                    order[b] = previous
    for column in range(arity):
        topology[row, column] = raw_topology[row, order[column]]
        gradient[row, column] = raw_gradient[row, order[column]]


@wp.func
def _hash(topology: wp.array2d[int], row: int, arity: int, mask: int):
    value = wp.uint32(-2128831035)
    for column in range(arity):
        value = (value ^ wp.uint32(topology[row, column])) * wp.uint32(16777619)
    return int(value & wp.uint32(mask))


@wp.func
def _same(topology: wp.array2d[int], first: int, second: int, arity: int):
    same = bool(True)
    for column in range(arity):
        same = same and topology[first, column] == topology[second, column]
    return same


@wp.kernel(enable_backward=False)
def _index_contacts(topology: wp.array2d[int], arity: int, table: wp.array[int], representatives: wp.array[int]):
    row = wp.tid()
    index = _hash(topology, row, arity, table.shape[0] - 1)
    while True:
        previous = wp.atomic_cas(table, index, -1, row)
        if previous == -1:
            representatives[row] = row
            break
        if _same(topology, row, previous, arity):
            representatives[row] = previous
            break
        index = (index + 1) & (table.shape[0] - 1)


@wp.kernel(enable_backward=False)
def _sum_forces(
    gradient: wp.array2d[wp.vec3d],
    representatives: wp.array[int],
    normal_count: int,
    scale: wp.float64,
    normal: wp.array2d[wp.vec3d],
    friction: wp.array2d[wp.vec3d],
):
    row, column = wp.tid()
    destination = representatives[row]
    force = -gradient[row, column] * scale
    if row < normal_count:
        wp.atomic_add(normal, destination, column, force)
    else:
        wp.atomic_add(friction, destination, column, force)


@wp.struct
class _Owner:
    shape: int
    weight: wp.float64


@wp.func
def _owner(
    topology: wp.array2d[int],
    row: int,
    start: int,
    stop: int,
    slot: int,
    shapes: wp.array[wp.vec2i],
    weights: wp.array[wp.vec2d],
    particles: wp.array[int],
    neighbors: wp.array[wp.vec2i],
    edge_shapes: wp.array[wp.vec2i],
):
    owner = _Owner()
    owner.shape = -2
    owner.weight = wp.float64(0.0)
    vertex = topology[row, start]
    if vertex < 0 or vertex >= shapes.shape[0]:
        return owner
    if stop - start == 1:
        owner.shape = shapes[vertex][slot]
        owner.weight = weights[vertex][slot]
        if slot == 0 and owner.shape < 0 and particles[vertex] < 0:
            owner.shape = -2
            owner.weight = wp.float64(0.0)
        return owner
    if slot > 0:
        return owner
    if stop - start == 2:
        other = topology[row, start + 1]
        for side in range(2):
            if neighbors[vertex][side] == other:
                owner.shape = edge_shapes[vertex][side]
                owner.weight = wp.float64(1.0)
                return owner
    shape = shapes[vertex][0]
    same_shape = shape >= 0
    all_particles = bool(True)
    for column in range(start, stop):
        current = topology[row, column]
        if current < 0 or current >= shapes.shape[0]:
            return owner
        same_shape = same_shape and shapes[current][0] == shape
        all_particles = all_particles and particles[current] >= 0
    if same_shape or all_particles:
        owner.shape = wp.where(same_shape, shape, -1)
        owner.weight = wp.float64(1.0)
    return owner


@wp.func
def _position(
    topology: wp.array2d[int],
    row: int,
    column: int,
    halfplane: bool,
    positions: wp.array[wp.vec3d],
    plane_normals: wp.array[wp.vec3d],
):
    vertex = topology[row, column]
    point = positions[vertex]
    if halfplane and column == 1:
        first = positions[topology[row, 0]]
        normal = plane_normals[vertex]
        point = first - wp.dot(first - point, normal) * normal
    return point


@wp.func
def _point(
    topology: wp.array2d[int],
    normal: wp.array2d[wp.vec3d],
    friction: wp.array2d[wp.vec3d],
    row: int,
    start: int,
    stop: int,
    halfplane: bool,
    positions: wp.array[wp.vec3d],
    plane_normals: wp.array[wp.vec3d],
):
    normal_weight = wp.float64(0.0)
    friction_weight = wp.float64(0.0)
    for column in range(start, stop):
        normal_weight += wp.length(normal[row, column])
        friction_weight += wp.length(friction[row, column])
    point = wp.vec3d(0.0)
    for column in range(start, stop):
        weight = wp.float64(1.0) / wp.float64(stop - start)
        if normal_weight > wp.float64(0.0):
            weight = wp.length(normal[row, column]) / normal_weight
        elif friction_weight > wp.float64(0.0):
            weight = wp.length(friction[row, column]) / friction_weight
        point += weight * _position(topology, row, column, halfplane, positions, plane_normals)
    return point


@wp.kernel(enable_backward=False)
def _pack_contacts(
    topology: wp.array2d[int],
    normal: wp.array2d[wp.vec3d],
    friction: wp.array2d[wp.vec3d],
    representatives: wp.array[int],
    arity: int,
    split: int,
    halfplane: bool,
    shapes: wp.array[wp.vec2i],
    weights: wp.array[wp.vec2d],
    particles: wp.array[int],
    neighbors: wp.array[wp.vec2i],
    edge_shapes: wp.array[wp.vec2i],
    shape_body: wp.array[int],
    positions: wp.array[wp.vec3d],
    plane_normals: wp.array[wp.vec3d],
    error: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    moments: wp.array[wp.vec3],
    particle_indices: wp.array[wp.vec3i],
    barycentric: wp.array[wp.vec3],
):
    row = wp.tid()
    for slot in range(4):
        shape0[4 * row + slot] = -1
    if representatives[row] != row:
        return
    active = bool(False)
    for column in range(arity):
        vertex = topology[row, column]
        if vertex < 0 or vertex >= shapes.shape[0]:
            wp.atomic_max(error, 0, 1)
            return
        active = active or wp.length(normal[row, column]) > wp.float64(0.0)
        active = active or wp.length(friction[row, column]) > wp.float64(0.0)
    if not active:
        return
    a_start, a_stop, b_start, b_stop = int(0), split, split, arity
    first = _owner(topology, row, a_start, a_stop, 0, shapes, weights, particles, neighbors, edge_shapes)
    second = _owner(topology, row, b_start, b_stop, 0, shapes, weights, particles, neighbors, edge_shapes)
    if first.shape == -2 or second.shape == -2:
        wp.atomic_max(error, 0, 1)
        return
    if first.shape < 0:
        if second.shape < 0:
            return
        a_start, a_stop, b_start, b_stop = split, arity, int(0), split
    force_n, force, moment = wp.vec3d(0.0), wp.vec3d(0.0), wp.vec3d(0.0)
    for column in range(a_start, a_stop):
        vertex_force = normal[row, column] + friction[row, column]
        force_n += normal[row, column]
        force += vertex_force
        moment += wp.cross(_position(topology, row, column, halfplane, positions, plane_normals), vertex_force)
    pa = _point(topology, normal, friction, row, a_start, a_stop, halfplane, positions, plane_normals)
    pb = _point(topology, normal, friction, row, b_start, b_stop, halfplane, positions, plane_normals)
    direction = pb - pa
    if wp.length(force_n) > wp.float64(1.0e-12):
        direction = -force_n / wp.length(force_n)
    if wp.length(direction) > wp.float64(1.0e-12):
        direction = direction / wp.length(direction)
    else:
        direction = wp.vec3d(0.0, 0.0, 1.0)
    indices = wp.vec3i(-1)
    bary = wp.vec3(0.0)
    soft_owner = _owner(topology, row, b_start, b_stop, 0, shapes, weights, particles, neighbors, edge_shapes)
    if soft_owner.shape < 0:
        total_n, total_f = wp.float64(0.0), wp.float64(0.0)
        for column in range(b_start, b_stop):
            total_n += wp.length(normal[row, column])
            total_f += wp.length(friction[row, column])
        for column in range(b_start, b_stop):
            indices[column - b_start] = particles[topology[row, column]]
            value = wp.float64(1.0) / wp.float64(b_stop - b_start)
            if total_n > wp.float64(0.0):
                value = wp.length(normal[row, column]) / total_n
            elif total_f > wp.float64(0.0):
                value = wp.length(friction[row, column]) / total_f
            bary[column - b_start] = float(value)
    for a in range(2):
        oa = _owner(topology, row, a_start, a_stop, a, shapes, weights, particles, neighbors, edge_shapes)
        for b in range(2):
            ob = _owner(topology, row, b_start, b_stop, b, shapes, weights, particles, neighbors, edge_shapes)
            if oa.shape >= 0 and oa.weight > wp.float64(0.0) and ob.weight > wp.float64(0.0):
                if ob.shape >= 0 and shape_body[oa.shape] >= 0 and shape_body[oa.shape] == shape_body[ob.shape]:
                    continue
                destination = 4 * row + 2 * a + b
                weight = oa.weight * ob.weight
                shape0[destination] = oa.shape
                shape1[destination] = ob.shape
                point0[destination] = wp.vec3(pa)
                point1[destination] = wp.vec3(pb)
                normals[destination] = wp.vec3(direction)
                forces[destination] = wp.vec3(weight * force)
                moments[destination] = wp.vec3(weight * moment)
                particle_indices[destination] = indices
                barycentric[destination] = bary


class ContactExporter(RodContactReporter):
    """Reuse shape provenance and pack both force channels on the solver device.

    Native Geometry queries supply host snapshots. Canonical topology matching,
    duplicate reduction, ownership, points and world moments run in Warp. The
    inherited host packer remains the numerical reference for diagnostic use.
    """

    def __init__(self, model, mapping, rods, body_affine, body_affine_valid):
        super().__init__(model, mapping, rods)
        self._refresh_positions()
        self.device = model.device
        self.rods = rods
        self.body_affine = body_affine
        self.body_affine_valid = body_affine_valid
        self.buffers = {}
        extent = len(self._shapes)
        owners = np.column_stack((self._shapes, np.full(extent, -1, dtype=np.int32)))
        weights = np.tile([1.0, 0.0], (extent, 1))
        nodes = np.full(extent, -1, dtype=np.int32)
        bodies = np.full(extent, -1, dtype=np.int32)
        local = np.zeros((extent, 3))
        neighbors = np.full((extent, 2), -1, dtype=np.int32)
        edge_shapes = neighbors.copy()
        plane_normals = np.zeros((extent, 3))
        for body, slot in mapping.body_geo_slots.items():
            geometry = slot.geometry()
            count = geometry.vertices().size()
            start = int(_view_attr(geometry.meta().find("global_vertex_offset"))[0])
            start += mapping.body_instance_ids.get(body, 0) * count
            bodies[start : start + count] = body
            local[start : start + count] = np.asarray(_view_attr(geometry.positions())).reshape(-1, 3)
        for group, slot, rows in rods._built:
            start = int(_view_attr(slot.geometry().meta().find("global_vertex_offset"))[0])
            nodes[start : start + len(group.points)] = np.arange(rows.start, rows.stop)
        for vertex, sources in self._nodes.items():
            owners[vertex] = -1
            weights[vertex] = 0.0
            for side, (shape, weight) in enumerate(sources):
                owners[vertex, side], weights[vertex, side] = shape, weight
        for (first, second), shape in self._edges.items():
            for vertex, other in ((first, second), (second, first)):
                side = int(np.flatnonzero(neighbors[vertex] < 0)[0])
                neighbors[vertex, side], edge_shapes[vertex, side] = other, shape
        for vertex, (_, _, normal) in self._planes.items():
            plane_normals[vertex] = normal
        self.positions = wp.array(self._positions, dtype=wp.vec3d, device=self.device)
        self.local = wp.array(local, dtype=wp.vec3d, device=self.device)
        self.bodies = wp.array(bodies, dtype=int, device=self.device)
        self.particles = wp.array(self._particles, dtype=int, device=self.device)
        self.nodes = wp.array(nodes, dtype=int, device=self.device)
        self.shapes = wp.array(owners, dtype=wp.vec2i, device=self.device)
        self.weights = wp.array(weights, dtype=wp.vec2d, device=self.device)
        self.neighbors = wp.array(neighbors, dtype=wp.vec2i, device=self.device)
        self.edge_shapes = wp.array(edge_shapes, dtype=wp.vec2i, device=self.device)
        self.plane_normals = wp.array(plane_normals, dtype=wp.vec3d, device=self.device)
        self.error = wp.zeros(1, dtype=int, device=self.device)

    def _buffer(self, key, shape, dtype):
        size = int(np.prod(shape))
        array = self.buffers.get(key)
        if array is None or array.size < size:
            array = wp.empty(max(1, 2 * size), dtype=dtype, device=self.device)
            self.buffers[key] = array
        return array[:size].reshape(shape)

    def _upload(self, key, values, dtype):
        host = wp.array(values, dtype=dtype, device="cpu", copy=False)
        destination = self._buffer(key, host.shape, dtype)
        wp.copy(destination, host)
        return destination

    def prepare(self, feature, dt, state):
        """Yield reusable device shape records for each populated primitive."""
        self.error.zero_()
        if self.positions.size:
            wp.launch(
                _refresh_positions,
                dim=self.positions.size,
                inputs=[
                    self.local,
                    self.bodies,
                    self.particles,
                    self.nodes,
                    self.body_affine,
                    self.body_affine_valid,
                    state.body_q,
                    state.particle_q,
                    self.rods.positions,
                    self.positions,
                ],
                device=self.device,
            )
        channels = {
            (primitive, channel): (topology, gradient)
            for primitive, channel, topology, gradient in _read_contact_channels(feature)
        }
        for primitive_index, primitive in enumerate(PRIMITIVE_TYPES):
            normal = channels.get((primitive, "N"))
            friction = channels.get((primitive, "F"))
            if normal is None and friction is None:
                continue
            source = normal if normal is not None else friction
            arity = source[0].shape[1]
            empty = (np.empty((0, arity), dtype=np.int32), np.empty((0, arity, 3)))
            normal = normal if normal is not None else empty
            friction = friction if friction is not None else empty
            count = len(normal[0]) + len(friction[0])
            topology = self._buffer("topology", (count, arity), int)
            gradient = self._buffer("gradient", (count, arity), wp.vec3d)
            raw_topology = self._upload("raw_topology", np.concatenate((normal[0], friction[0])).astype(np.int32), int)
            raw_gradient = self._upload("raw_gradient", np.concatenate((normal[1], friction[1])), wp.vec3d)
            wp.launch(
                _canonicalize,
                dim=count,
                inputs=[raw_topology, raw_gradient, primitive_index, arity, topology, gradient],
                device=self.device,
            )
            table_size = 1 << max(1, (2 * count - 1).bit_length())
            table = self._buffer("table", (table_size,), int)
            table.fill_(-1)
            representatives = self._buffer("representatives", (count,), int)
            wp.launch(_index_contacts, dim=count, inputs=[topology, arity, table, representatives], device=self.device)
            normal_forces = self._buffer("normal", (count, arity), wp.vec3d)
            friction_forces = self._buffer("friction", (count, arity), wp.vec3d)
            normal_forces.zero_()
            friction_forces.zero_()
            wp.launch(
                _sum_forces,
                dim=(count, arity),
                inputs=[
                    gradient,
                    representatives,
                    len(normal[0]),
                    wp.float64(1.0 / (dt * dt)),
                    normal_forces,
                    friction_forces,
                ],
                device=self.device,
            )
            arrays = [
                self._buffer(name, (4 * count,), dtype)
                for name, dtype in (
                    ("shape0", int),
                    ("shape1", int),
                    ("point0", wp.vec3),
                    ("point1", wp.vec3),
                    ("normal_out", wp.vec3),
                    ("force", wp.vec3),
                    ("moment", wp.vec3),
                    ("particle_indices", wp.vec3i),
                    ("barycentric", wp.vec3),
                )
            ]
            wp.launch(
                _pack_contacts,
                dim=count,
                inputs=[
                    topology,
                    normal_forces,
                    friction_forces,
                    representatives,
                    arity,
                    _PRIM_SPLIT[primitive],
                    primitive == "PH",
                    self.shapes,
                    self.weights,
                    self.particles,
                    self.neighbors,
                    self.edge_shapes,
                    self._model.shape_body,
                    self.positions,
                    self.plane_normals,
                    self.error,
                    *arrays,
                ],
                device=self.device,
            )
            yield arrays
        if self.error.numpy()[0]:
            raise RuntimeError("A UIPC contact primitive has no unambiguous Newton shape provenance")

    def export(self, feature, contacts, state, contact_f, dt):
        """Write packed rigid and particle records into existing observable storage."""
        try:
            for arrays in self.prepare(feature, dt, state):
                wp.launch(
                    populate_rod_contacts,
                    dim=arrays[0].size,
                    inputs=[
                        *arrays,
                        self._model.shape_body,
                        self._rod_radii,
                        state.body_q,
                        state.body_qd,
                        self._model.body_com,
                        -1,
                        contacts.rigid_contact_max,
                        contacts.soft_contact_max,
                        contacts.rigid_contact_count,
                        contacts.soft_contact_count,
                    ],
                    outputs=[
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_offset0,
                        contacts.rigid_contact_offset1,
                        contacts.rigid_contact_normal,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_indices,
                        contacts.soft_contact_rigid_indices,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_barycentric,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                        contact_f,
                    ],
                    device=self.device,
                )
        except Exception:
            contacts.rigid_contact_count.zero_()
            contacts.soft_contact_count.zero_()
            contact_f.zero_()
            raise
