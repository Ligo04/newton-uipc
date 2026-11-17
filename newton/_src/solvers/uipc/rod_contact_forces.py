# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Export FEM rod contacts using their Newton segment shapes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import warp as wp
from uipc.core import ContactSystemFeature
from uipc.geometry import Geometry

from newton import Contacts, Model, State

from .contact_forces import _PRIM_ARITY, _PRIM_SPLIT, PRIMITIVE_TYPES, _quat_rotate, _read_gradient
from .converter import UIpcMappingInfo
from .rod import RodBuilder
from .utils import _view_attr


class _ContactStencil(NamedTuple):
    vertices: np.ndarray
    normal_forces: np.ndarray
    friction_forces: np.ndarray


def _merge_stencils(
    channels: list[tuple[str, str, np.ndarray, np.ndarray]], dt: float
) -> dict[tuple[str | int, ...], _ContactStencil]:
    """Join native normal/friction stencils by topology, retaining either channel."""
    stencils = {}
    for primitive, channel, topology, gradients in channels:
        for vertices, gradient in zip(topology, gradients, strict=True):
            order = np.arange(len(vertices))
            if primitive == "PP":
                order = np.argsort(vertices)
            elif primitive in ("PE", "PT"):
                order[1:] = 1 + np.argsort(vertices[1:])
            elif primitive == "EE":
                order[:2] = np.argsort(vertices[:2])
                order[2:] = 2 + np.argsort(vertices[2:])
                if tuple(vertices[order[:2]]) > tuple(vertices[order[2:]]):
                    order = np.concatenate((order[2:], order[:2]))
            canonical_vertices = vertices[order]
            forces = -gradient[order] / (dt * dt)
            key = (primitive, *map(int, canonical_vertices))
            if key not in stencils:
                stencils[key] = _ContactStencil(canonical_vertices, np.zeros_like(forces), np.zeros_like(forces))
            if channel == "N":
                stencils[key].normal_forces[:] += forces
            else:
                stencils[key].friction_forces[:] += forces
    return stencils


def _halfplane_topology(indices: np.ndarray, topology: np.ndarray) -> np.ndarray:
    """Match gradient rows to energy topology only when plane identity is unique."""
    indices = indices.reshape(-1)
    if np.array_equal(indices, topology[:, 0]):
        return topology
    # Native gradient snapshots can omit rows present in energy topology.
    # Both ordered embeddings must identify the same plane for every row.
    embeddings = []
    for reverse in (False, True):
        selected = np.empty(len(indices), dtype=np.int64)
        cursor = len(topology) - 1 if reverse else 0
        step = -1 if reverse else 1
        rows = range(len(indices) - 1, -1, -1) if reverse else range(len(indices))
        for row in rows:
            while 0 <= cursor < len(topology) and topology[cursor, 0] != indices[row]:
                cursor += step
            if not 0 <= cursor < len(topology):
                raise RuntimeError("UIPC halfplane gradient rows do not match the native energy topology")
            selected[row] = cursor
            cursor += step
        embeddings.append(topology[selected])
    if not np.array_equal(embeddings[0], embeddings[1]):
        raise RuntimeError("UIPC halfplane gradient rows have ambiguous plane identities")
    return embeddings[0]


def _read_contact_channels(feature: ContactSystemFeature) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Read and validate native topology together with each gradient channel."""
    channels = []
    for primitive in PRIMITIVE_TYPES:
        arity = _PRIM_ARITY[primitive]
        for channel in ("N", "F"):
            key = f"{primitive}+{channel}"
            data = _read_gradient(feature, key)
            if data is None:
                continue
            indices, gradients = data
            indices = indices.reshape(-1, arity)
            gradients = gradients.reshape(-1, arity, 3).astype(np.float64)
            topology = indices.astype(np.int64)
            if primitive == "PH":
                # PH gradients omit the plane ID. Query the matching normal or
                # frozen-friction topology without advancing the scene.
                geometry = Geometry()
                feature.contact_energy(key, geometry)
                attribute = geometry.instances().find("topo")
                if attribute is None:
                    raise RuntimeError(f"UIPC {key} contact topology is unavailable")
                topology = np.asarray(_view_attr(attribute), dtype=np.int64).reshape(-1, 2)
                topology = _halfplane_topology(indices, topology)
                gradients = np.concatenate((gradients, -gradients), axis=1)
            channels.append((primitive, channel, topology, gradients))
    return channels


def _read_stencils(feature: ContactSystemFeature, dt: float) -> dict[tuple[str | int, ...], _ContactStencil]:
    return _merge_stencils(_read_contact_channels(feature), dt)


@dataclass
class RodContactData:
    """Shape pairs, world-space points, and wrenches about the world origin."""

    shape0: np.ndarray
    shape1: np.ndarray
    point0: np.ndarray
    point1: np.ndarray
    normal: np.ndarray
    force: np.ndarray
    moment: np.ndarray
    particles: np.ndarray
    barycentric: np.ndarray


class RodContactReporter:
    """Map primitive edges to segments and partition shared point contacts.

    Point-contact weights are the rod builder's nodal velocity weights, including
    its kinematic-node rule. Edge contacts belong to their unique segment. Native
    normal and friction channels are joined by topology. Points are force-weighted
    locations of the current FEM nodes; distributed moments use every nodal force.
    """

    def __init__(self, model: Model, mapping: UIpcMappingInfo, rods: RodBuilder):
        self._model = model
        self._shape_body = model.shape_body.numpy()
        self._ranges = []
        self._planes = {}
        self._nodes = {}
        self._edges = {}
        rod_radii = np.zeros(model.shape_count, dtype=np.float32)
        shape_ranges = []
        particle_ranges = []
        extent = 0

        def add_range(slot, shapes, instance=None):
            nonlocal extent
            geometry = slot.geometry()
            attribute = geometry.meta().find("global_vertex_offset")
            if attribute is None:
                raise RuntimeError("UIPC contact reporting requires global_vertex_offset")
            count = geometry.vertices().size()
            start = int(_view_attr(attribute)[0]) + (instance or 0) * count
            self._ranges.append((geometry, start, count, instance))
            shape_ranges.append((start, np.asarray(shapes, dtype=np.int32)))
            extent = max(extent, start + count)
            return start

        for body, slot in mapping.body_geo_slots.items():
            geometry = slot.geometry()
            attribute = geometry.vertices().find("newton_shape_slot")
            if attribute is None:
                raise RuntimeError("UIPC body geometry is missing contact shape provenance")
            ordinal = np.asarray(_view_attr(attribute)).reshape(-1).astype(np.int32)
            shapes = np.asarray(mapping.body_shapes[body], dtype=np.int32)
            resolved = np.full(len(ordinal), -1, dtype=np.int32)
            valid = (ordinal >= 0) & (ordinal < len(shapes))
            resolved[valid] = shapes[ordinal[valid]]
            add_range(slot, resolved, mapping.body_instance_ids.get(body, 0))

        for slot in mapping.static_geo_slots:
            attribute = slot.geometry().vertices().find("newton_shape_index")
            if attribute is None:
                raise RuntimeError("UIPC static geometry is missing contact shape provenance")
            geometry = slot.geometry()
            affine = geometry.instances().find("transform") is not None
            add_range(slot, np.asarray(_view_attr(attribute)).reshape(-1), 0 if affine else None)

        for slots, indices in (
            (mapping.cloth_geo_slots, mapping.cloth_particle_indices),
            (mapping.deformable_geo_slots, mapping.deformable_particle_indices),
        ):
            for slot, particles in zip(slots, indices, strict=True):
                start = add_range(slot, np.full(len(particles), -1))
                particle_ranges.append((start, np.asarray(particles, dtype=np.int32)))

        for group, slot, _ in rods._built:
            start = add_range(slot, np.full(len(group.points), -1))
            for body in group.bodies:
                rod_radii[mapping.body_shapes[int(body)][0]] = group.radius
            for row, (bodies, weights) in enumerate(zip(group.sources, group.weights, strict=True)):
                self._nodes[start + row] = [
                    (mapping.body_shapes[int(body)][0], float(weight))
                    for body, weight in zip(bodies, weights, strict=True)
                    if weight > 0.0
                ]
            for body, edge in zip(group.bodies, group.edges, strict=True):
                self._edges[tuple(sorted(start + edge))] = mapping.body_shapes[int(body)][0]

        transforms = model.shape_transform.numpy()
        for shape, slot in mapping.ground_geo_slots.items():
            attribute = slot.geometry().meta().find("global_vertex_offset")
            if attribute is None:
                raise RuntimeError("UIPC halfplane is missing its global vertex offset")
            vertex = int(_view_attr(attribute)[0])
            transform = transforms[shape]
            self._planes[vertex] = (shape, transform[:3], _quat_rotate(transform[3:], np.array([0.0, 0.0, 1.0])))
            extent = max(extent, vertex + 1)

        self._positions = np.zeros((extent, 3), dtype=np.float64)
        self._shapes = np.full(extent, -1, dtype=np.int32)
        self._particles = np.full(extent, -1, dtype=np.int32)
        for start, shapes in shape_ranges:
            self._shapes[start : start + len(shapes)] = shapes
        for start, particles in particle_ranges:
            self._particles[start : start + len(particles)] = particles
        for vertex, (shape, center, _) in self._planes.items():
            self._shapes[vertex] = shape
            self._positions[vertex] = center
        self._rod_radii = wp.array(rod_radii, dtype=wp.float32, device=model.device)

    def _refresh_positions(self) -> None:
        for geometry, start, count, instance in self._ranges:
            positions = np.asarray(_view_attr(geometry.positions())).reshape(-1, 3)
            if instance is not None:
                transform = np.asarray(_view_attr(geometry.instances().find("transform")))[instance].reshape(4, 4)
                positions = positions @ transform[:3, :3].T + transform[:3, 3]
            self._positions[start : start + count] = positions

    def _owners(self, vertices: np.ndarray) -> list[tuple[int, float]]:
        if len(vertices) == 1 and int(vertices[0]) in self._nodes:
            return self._nodes[int(vertices[0])]
        edge = self._edges.get(tuple(sorted(vertices))) if len(vertices) == 2 else None
        if edge is not None:
            return [(edge, 1.0)]
        shapes = self._shapes[vertices]
        if np.all(shapes == shapes[0]) and shapes[0] >= 0:
            return [(int(shapes[0]), 1.0)]
        if np.all(self._particles[vertices] >= 0):
            return [(-1, 1.0)]
        raise RuntimeError("A UIPC contact primitive has no unambiguous Newton shape provenance")

    def prepare(self, feature: ContactSystemFeature, dt: float) -> RodContactData | None:
        self._refresh_positions()
        return self.pack(_read_stencils(feature, dt))

    def export(
        self,
        feature: ContactSystemFeature,
        contacts: Contacts,
        state: State,
        contact_f: wp.array[wp.spatial_vector],
        dt: float,
    ) -> None:
        """Publish requested rigid and particle-side records into existing storage."""
        self._publish(self.prepare(feature, dt), contacts, state, contact_f)

    def _publish(
        self, data: RodContactData | None, contacts: Contacts, state: State, contact_f: wp.array[wp.spatial_vector]
    ) -> None:
        if data is None:
            return
        rigid = np.flatnonzero(data.shape1 >= 0)[: contacts.rigid_contact_max]
        soft = np.flatnonzero(data.shape1 < 0)[: contacts.soft_contact_max]
        selected = np.concatenate((rigid, soft))
        contacts.rigid_contact_count.fill_(len(rigid))
        contacts.soft_contact_count.fill_(len(soft))
        if len(selected) == 0:
            return
        arrays = [
            wp.from_numpy(getattr(data, name)[selected], dtype=dtype, device=self._model.device)
            for name, dtype in (
                ("shape0", wp.int32),
                ("shape1", wp.int32),
                ("point0", wp.vec3),
                ("point1", wp.vec3),
                ("normal", wp.vec3),
                ("force", wp.vec3),
                ("moment", wp.vec3),
                ("particles", wp.vec3i),
                ("barycentric", wp.vec3),
            )
        ]
        wp.launch(
            populate_rod_contacts,
            dim=len(selected),
            inputs=[
                *arrays,
                self._model.shape_body,
                self._rod_radii,
                state.body_q,
                state.body_qd,
                self._model.body_com,
                len(rigid),
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
            device=self._model.device,
        )

    def pack(self, stencils: dict[tuple[str | int, ...], _ContactStencil]) -> RodContactData | None:
        """Convert canonical native stencils into reporting-only shape records."""
        rows = []
        for key, (vertices, normal_forces, friction_forces) in stencils.items():
            if not np.any(normal_forces) and not np.any(friction_forces):
                continue
            primitive = key[0]
            split = _PRIM_SPLIT[primitive]
            positions = self._positions[vertices].copy()
            if primitive == "PH":
                _, center, normal = self._planes[int(vertices[1])]
                positions[1] = positions[0] - np.dot(positions[0] - center, normal) * normal
            sides = [slice(0, split), slice(split, len(vertices))]
            owners = [self._owners(vertices[side]) for side in sides]
            if owners[0][0][0] < 0:
                if owners[1][0][0] < 0:
                    continue  # Particle-only contacts have no Newton shape-side observable.
                sides.reverse()
                owners.reverse()
            first, second = sides
            force_n = normal_forces[first].sum(axis=0)
            force = force_n + friction_forces[first].sum(axis=0)
            moment = np.cross(positions[first], normal_forces[first] + friction_forces[first]).sum(axis=0)
            points = []
            weights = []
            for side in sides:
                magnitudes = np.linalg.norm(normal_forces[side], axis=1)
                if magnitudes.sum() == 0.0:
                    magnitudes = np.linalg.norm(friction_forces[side], axis=1)
                if magnitudes.sum() == 0.0:
                    magnitudes = np.ones(len(magnitudes))
                barycentric = magnitudes / magnitudes.sum()
                points.append(barycentric @ positions[side])
                weights.append(barycentric)
            length = np.linalg.norm(force_n)
            normal = -force_n / length if length > 1.0e-12 else points[1] - points[0]
            length = np.linalg.norm(normal)
            normal = normal / length if length > 1.0e-12 else np.array([0.0, 0.0, 1.0])
            particles = np.full(3, -1, dtype=np.int32)
            barycentric = np.zeros(3)
            if owners[1][0][0] < 0:
                particles[: len(vertices[second])] = self._particles[vertices[second]]
                barycentric[: len(vertices[second])] = weights[1]
            for shape0, weight0 in owners[0]:
                for shape1, weight1 in owners[1]:
                    if shape1 >= 0 and self._shape_body[shape0] >= 0:
                        if self._shape_body[shape0] == self._shape_body[shape1]:
                            continue
                    weight = weight0 * weight1
                    rows.append(
                        (
                            shape0,
                            shape1,
                            points[0],
                            points[1],
                            normal,
                            weight * force,
                            weight * moment,
                            particles,
                            barycentric,
                        )
                    )
        if not rows:
            return None
        return RodContactData(
            *[
                np.asarray(values, dtype=np.int32 if index in (0, 1, 7) else np.float32)
                for index, values in enumerate(zip(*rows, strict=True))
            ]
        )


@wp.kernel(enable_backward=False)
def populate_rod_contacts(
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    force: wp.array[wp.vec3],
    moment: wp.array[wp.vec3],
    particles: wp.array[wp.vec3i],
    barycentric: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    rod_radii: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    num_rigid: int,
    rigid_capacity: int,
    soft_capacity: int,
    rigid_count: wp.array[wp.int32],
    soft_count: wp.array[wp.int32],
    rigid_shape0: wp.array[wp.int32],
    rigid_shape1: wp.array[wp.int32],
    rigid_point0: wp.array[wp.vec3],
    rigid_point1: wp.array[wp.vec3],
    rigid_offset0: wp.array[wp.vec3],
    rigid_offset1: wp.array[wp.vec3],
    rigid_normal: wp.array[wp.vec3],
    soft_shape: wp.array[wp.int32],
    soft_indices: wp.array[wp.vec3i],
    soft_rigid_indices: wp.array[wp.vec3i],
    soft_particle: wp.array[wp.int32],
    soft_barycentric: wp.array[wp.vec3],
    soft_point: wp.array[wp.vec3],
    soft_velocity: wp.array[wp.vec3],
    soft_normal: wp.array[wp.vec3],
    contact_f: wp.array[wp.spatial_vector],
):
    row = wp.tid()
    if shape0[row] < 0:
        return
    rigid = row < num_rigid
    destination = row
    soft = row - num_rigid
    if num_rigid < 0:
        rigid = shape1[row] >= 0
        if rigid:
            destination = wp.atomic_add(rigid_count, 0, 1)
            if destination >= rigid_capacity:
                wp.atomic_sub(rigid_count, 0, 1)
                return
        else:
            soft = wp.atomic_add(soft_count, 0, 1)
            if soft >= soft_capacity:
                wp.atomic_sub(soft_count, 0, 1)
                return
    body0 = shape_body[shape0[row]]
    center = wp.vec3(0.0)
    # Rod gradients act on centerline nodes. Report the collision surface
    # while retaining the native world moment for the COM wrench.
    surface0 = point0[row] + rod_radii[shape0[row]] * normal[row]
    local0 = surface0
    velocity = wp.vec3(0.0)
    if body0 >= 0:
        transform = body_q[body0]
        center = wp.transform_point(transform, body_com[body0])
        local0 = wp.transform_point(wp.transform_inverse(transform), surface0)
        velocity = wp.spatial_top(body_qd[body0]) + wp.cross(wp.spatial_bottom(body_qd[body0]), surface0 - center)
    torque = moment[row] - wp.cross(center, force[row])
    if rigid:
        surface1 = point1[row] - rod_radii[shape1[row]] * normal[row]
        local1 = surface1
        body1 = shape_body[shape1[row]]
        if body1 >= 0:
            local1 = wp.transform_point(wp.transform_inverse(body_q[body1]), surface1)
        rigid_shape0[destination] = shape0[row]
        rigid_shape1[destination] = shape1[row]
        rigid_point0[destination] = local0
        rigid_point1[destination] = local1
        rigid_offset0[destination] = wp.vec3(0.0)
        rigid_offset1[destination] = wp.vec3(0.0)
        rigid_normal[destination] = normal[row]
        contact_f[destination] = wp.spatial_vector(force[row], torque)
    else:
        soft_shape[soft] = shape0[row]
        soft_indices[soft] = particles[row]
        soft_rigid_indices[soft] = wp.vec3i(-1)
        particle = int(-1)
        if particles[row][1] < 0:
            particle = particles[row][0]
        soft_particle[soft] = particle
        soft_barycentric[soft] = barycentric[row]
        soft_point[soft] = surface0
        soft_velocity[soft] = velocity
        soft_normal[soft] = normal[row]
        contact_f[rigid_capacity + soft] = wp.spatial_vector(force[row], torque)
