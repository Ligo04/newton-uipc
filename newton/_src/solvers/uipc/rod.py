# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Centerline FEM approximation of Newton capsule-segment rods."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import uipc.adapter.warp
import warp as wp
from uipc.constitution import FiniteElementExternalForce, HookeanSpring, KirchhoffRodBending
from uipc.geometry import label_surface, linemesh

from newton import GeoType, ShapeFlags

from ...exceptions import NewtonWarning
from ...math import quat_between_vectors_robust
from ...sim import BodyFlags, JointType, Model, State
from ...sim.articulation import eval_articulation_ik
from .converter import UIpcMappingInfo
from .utils import _view_attr


@wp.kernel(enable_backward=False)
def _gather_nodes(
    rows: wp.array[wp.int32],
    offsets: wp.array[wp.uint32],
    sources: wp.array[wp.vec2i],
    points: wp.array2d[wp.vec3],
    weights: wp.array[wp.vec2],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    positions: wp.array[wp.vec3d],
    velocities: wp.array[wp.vec3d],
):
    row = rows[wp.tid()]
    p = wp.vec3(0.0)
    v = wp.vec3(0.0)
    for side in range(2):
        body = sources[row][side]
        x = wp.transform_point(body_q[body], points[row, side])
        com = wp.transform_point(body_q[body], body_com[body])
        twist = body_qd[body]
        p += weights[row][side] * x
        v += weights[row][side] * (wp.spatial_top(twist) + wp.cross(wp.spatial_bottom(twist), x - com))
    positions[offsets[row]] = wp.vec3d(p)
    velocities[offsets[row]] = wp.vec3d(v)


@wp.kernel(enable_backward=False)
def _read_segments(
    bodies: wp.array[wp.int32],
    nodes: wp.array[wp.vec2i],
    offsets: wp.array[wp.uint32],
    midpoints: wp.array[wp.vec3],
    axes: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    previous_q: wp.array[wp.transform],
    positions: wp.array[wp.vec3d],
    velocities: wp.array[wp.vec3d],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    row = wp.tid()
    body = bodies[row]
    a = offsets[nodes[row][0]]
    b = offsets[nodes[row][1]]
    p0 = wp.vec3(positions[a])
    p1 = wp.vec3(positions[b])
    v0 = wp.vec3(velocities[a])
    v1 = wp.vec3(velocities[b])
    edge = p1 - p0
    length = wp.length(edge)
    rotation = wp.transform_get_rotation(previous_q[body])
    omega = wp.vec3(0.0)
    if length > 1.0e-10:
        tangent = edge / length
        old_tangent = wp.quat_rotate(rotation, axes[row])
        rotation = wp.normalize(quat_between_vectors_robust(old_tangent, tangent) * rotation)
        omega = wp.cross(tangent, v1 - v0) / length
    midpoint = 0.5 * (p0 + p1)
    origin = midpoint - wp.quat_rotate(rotation, midpoints[row])
    com_offset = wp.quat_rotate(rotation, body_com[body] - midpoints[row])
    com_velocity = 0.5 * (v0 + v1) + wp.cross(omega, com_offset)
    body_q[body] = wp.transform(origin, rotation)
    body_qd[body] = wp.spatial_vector(com_velocity, omega)


@wp.kernel(enable_backward=False)
def _gather_forces(
    bodies: wp.array[wp.int32],
    nodes: wp.array[wp.vec2i],
    offsets: wp.array[wp.uint32],
    positions: wp.array[wp.vec3d],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    forces: wp.array[wp.vec3d],
):
    row = wp.tid()
    body = bodies[row]
    a = nodes[row][0]
    b = nodes[row][1]
    p0 = positions[offsets[a]]
    p1 = positions[offsets[b]]
    edge = p1 - p0
    length2 = wp.dot(edge, edge)
    wrench = body_f[body]
    force = wp.vec3d(wp.spatial_top(wrench))
    com = wp.vec3d(wp.transform_point(body_q[body], body_com[body]))
    torque = wp.vec3d(wp.spatial_bottom(wrench)) + wp.cross(com - wp.float64(0.5) * (p0 + p1), force)
    couple = wp.vec3d(0.0)
    if length2 > wp.float64(1.0e-20):
        couple = wp.cross(torque, edge) / length2
    wp.atomic_add(forces, a, wp.float64(0.5) * force - couple)
    wp.atomic_add(forces, b, wp.float64(0.5) * force + couple)


@dataclass
class _RodGroup:
    label: str
    bodies: np.ndarray
    edges: np.ndarray
    points: np.ndarray
    sources: np.ndarray
    local_points: np.ndarray
    weights: np.ndarray
    fixed: np.ndarray
    kinematic: np.ndarray
    mass: np.ndarray
    midpoints: np.ndarray
    axes: np.ndarray
    radius: float
    stretch: np.ndarray
    bending: np.ndarray
    collision_enabled: bool


class RodBuilder:
    """Own centerline geometry, body reconstruction, external forces, and resets.

    Rod joints join capsule endpoints. Their stretch/bend gains are converted
    back to section rigidities using the mean adjacent rest length. The FEM
    model has no directors, shear/twist energies, or intrinsic rest curvature.
    """

    ROD_MODEL_HOOKEAN_SPRING = "hookean_spring"

    def __init__(self, model: Model, scene: Any, mapping: UIpcMappingInfo):
        self._model = model
        self._scene = scene
        self._mapping = mapping
        self._groups: list[_RodGroup] = []
        self._built: list[tuple[_RodGroup, Any, slice]] = []
        self._accessor = None
        self._node_count = 0
        if not model.curve_count and (
            model.joint_type is None or not np.any(model.joint_type.numpy() == int(JointType.ROD))
        ):
            return
        self._shape_body = model.shape_body.numpy()
        self._shape_type = model.shape_type.numpy()
        self._shape_scale = model.shape_scale.numpy()
        self._shape_transform = model.shape_transform.numpy()
        self._shape_flags = model.shape_flags.numpy()
        self._joint_type = model.joint_type.numpy()
        self._parent = model.joint_parent.numpy()
        self._child = model.joint_child.numpy()
        self._xp = model.joint_X_p.numpy()
        self._xc = model.joint_X_c.numpy()
        self._qd_start = model.joint_qd_start.numpy()
        self._ke = model.joint_target_ke.numpy()
        self._enabled = model.joint_enabled.numpy()
        self._body_q = model.body_q.numpy()
        self._body_mass = model.body_mass.numpy()
        self._body_flags = model.body_flags.numpy()
        self._bodies: set[int] = set()

        selectors = getattr(getattr(model, "uipc", None), "rod_model", None)
        if selectors is not None and len(selectors) != model.curve_count:
            raise ValueError("uipc:rod_model must contain one selector per curve object")
        starts = model.curve_body_start.numpy() if model.curve_count else []
        ends = model.curve_body_end.numpy() if model.curve_count else []
        for index, label in enumerate(model.curve_label):
            selector = self.ROD_MODEL_HOOKEAN_SPRING if selectors is None else str(selectors[index]).strip().lower()
            if selector != self.ROD_MODEL_HOOKEAN_SPRING:
                raise ValueError(f"uipc:rod_model[{index}] must be 'hookean_spring', got {selector!r}")
            start = int(starts[index])
            end = int(ends[index])
            if start < end:
                self._add_group(label, list(range(start, end)))

        # Also accept directly authored ROD joints without curve metadata.
        remaining: dict[int, set[int]] = {}
        for joint in np.flatnonzero(self._joint_type == int(JointType.ROD)):
            parent, child = int(self._parent[joint]), int(self._child[joint])
            if parent < 0:
                raise ValueError("UIPC ROD joints must connect two capsule segments")
            if parent in self._bodies or child in self._bodies:
                if parent not in self._bodies or child not in self._bodies:
                    raise ValueError("UIPC ROD joints cannot cross curve object boundaries")
                continue
            remaining.setdefault(parent, set()).add(child)
            remaining.setdefault(child, set()).add(parent)
        while remaining:
            pending = [next(iter(remaining))]
            component = set()
            while pending:
                body = pending.pop()
                if body in component:
                    continue
                component.add(body)
                pending.extend(remaining.pop(body, ()))
            self._add_group(f"rod_{min(component)}", sorted(component))

        mapping.fem_rod_bodies.update(self._bodies)
        self._joint_indices = np.asarray(
            [j for j in range(model.joint_count) if int(self._child[j]) in self._bodies], dtype=np.int32
        )
        if self._groups:
            warnings.warn(
                "SolverUIPC ROD centerline approximation uses HookeanSpring and KirchhoffRodBending; "
                "shear, twist, material damping, and intrinsic rest curvature are omitted. "
                "Segment poses are reconstructed from the centerline without axial spin.",
                NewtonWarning,
                stacklevel=2,
            )

    @property
    def has_rods(self) -> bool:
        return bool(self._groups)

    def _add_group(self, label: str, bodies: list[int]) -> None:
        if self._bodies.intersection(bodies):
            raise ValueError("UIPC rod body ranges must not overlap")
        body_set = set(bodies)
        local = {body: row for row, body in enumerate(bodies)}
        midpoints, axes, lengths, radii, colliding = [], [], [], [], []
        for body in bodies:
            shapes = np.flatnonzero(self._shape_body == body)
            if len(shapes) != 1 or self._shape_type[shapes[0]] != int(GeoType.CAPSULE):
                raise ValueError(f"UIPC rod {label!r}: body {body} requires exactly one capsule shape")
            shape = int(shapes[0])
            transform = wp.transform(*self._shape_transform[shape])
            midpoints.append(np.asarray(wp.transform_get_translation(transform)))
            axes.append(np.asarray(wp.quat_rotate(wp.transform_get_rotation(transform), wp.vec3(0.0, 0.0, 1.0))))
            radii.append(float(self._shape_scale[shape, 0]))
            lengths.append(float(self._shape_scale[shape, 1]) * 2.0)
            colliding.append(bool(int(self._shape_flags[shape]) & int(ShapeFlags.COLLIDE_SHAPES)))
        lengths = np.asarray(lengths)
        midpoints, axes = np.asarray(midpoints), np.asarray(axes)
        if np.any(lengths <= 0.0) or radii[0] <= 0.0 or not np.allclose(radii, radii[0]):
            raise ValueError(f"UIPC rod {label!r} requires positive lengths and a uniform positive radius")
        if any(colliding) != all(colliding):
            raise ValueError(f"UIPC rod {label!r} requires uniform collision enablement")
        endpoints = (
            midpoints[:, None, :] + np.asarray([-0.5, 0.5])[None, :, None] * lengths[:, None, None] * axes[:, None, :]
        )
        roots = list(range(2 * len(bodies)))

        def root(index: int) -> int:
            while roots[index] != index:
                index = roots[index]
            return index

        def endpoint(body: int, anchor: np.ndarray) -> int:
            row = local[body]
            distances = np.linalg.norm(endpoints[row] - anchor[:3], axis=1)
            side = int(np.argmin(distances))
            if distances[side] > 1.0e-6 + 1.0e-5 * lengths[row]:
                raise ValueError(f"UIPC rod {label!r}: joint anchors must lie at capsule centerline endpoints")
            return 2 * row + side

        joints = []
        pinned = []
        for j, (parent_value, child_value) in enumerate(zip(self._parent, self._child, strict=True)):
            parent, child = int(parent_value), int(child_value)
            if parent not in body_set and child not in body_set:
                continue
            kind = JointType(int(self._joint_type[j]))
            if kind == JointType.ROD and parent in body_set and child in body_set:
                if not bool(self._enabled[j]):
                    raise ValueError(f"UIPC rod {label!r} does not support disabled ROD joints")
                a, b = endpoint(parent, self._xp[j]), endpoint(child, self._xc[j])
                roots[max(root(a), root(b))] = min(root(a), root(b))
                joints.append((j, a, parent, child))
            elif parent == -1 and child in body_set and kind == JointType.FREE:
                continue
            elif parent == -1 and child in body_set and kind in (JointType.FIXED, JointType.BALL):
                pinned.extend(
                    [2 * local[child], 2 * local[child] + 1]
                    if kind == JointType.FIXED
                    else [endpoint(child, self._xc[j])]
                )
            else:
                raise ValueError(
                    f"UIPC rod {label!r}: attachments to other bodies or unsupported joint types require a rod coupling constraint"
                )

        node_ids: dict[int, int] = {}
        sources: list[list[tuple[int, int]]] = []
        for end in range(len(roots)):
            owner = root(end)
            if owner not in node_ids:
                node_ids[owner] = len(sources)
                sources.append([])
            sources[node_ids[owner]].append(divmod(end, 2))
        edges = np.asarray(
            [[node_ids[root(2 * row)], node_ids[root(2 * row + 1)]] for row in range(len(bodies))], dtype=np.int32
        )
        if any(len(node) > 2 for node in sources) or len({tuple(sorted(edge)) for edge in edges}) != len(edges):
            raise ValueError(f"UIPC rod {label!r} requires nonbranching chains without duplicate edges")
        points, source_bodies, local_points, weights, masses, fixed, kinematic = [], [], [], [], [], [], []
        for node in sources:
            coordinates = [
                np.asarray(wp.transform_point(wp.transform(*self._body_q[bodies[row]]), wp.vec3(*endpoints[row, side])))
                for row, side in node
            ]
            if len(node) == 2 and np.linalg.norm(coordinates[0] - coordinates[1]) > 1.0e-5:
                raise ValueError(f"UIPC rod {label!r}: initially separated joint anchors are not supported")
            mass = np.asarray([self._body_mass[bodies[row]] * 0.5 for row, _ in node], dtype=np.float64)
            is_kinematic = np.asarray(
                [
                    bool(int(self._body_flags[bodies[row]]) & int(BodyFlags.KINEMATIC))
                    or self._body_mass[bodies[row]] <= 0.0
                    for row, _ in node
                ]
            )
            weight = is_kinematic.astype(float) if is_kinematic.any() else np.maximum(mass, 1.0e-30)
            weight /= weight.sum()
            points.append(np.sum(np.asarray(coordinates) * weight[:, None], axis=0))
            source_bodies.append([bodies[node[0][0]], bodies[node[-1][0]]])
            local_points.append([endpoints[node[0]], endpoints[node[-1]]])
            weights.append([weight[0], weight[1] if len(node) == 2 else 0.0])
            masses.append(mass.sum())
            fixed.append(int(is_kinematic.any()))
            kinematic.append(bool(is_kinematic.any()))
        for end in pinned:
            fixed[node_ids[root(end)]] = 1
        rigidity: list[list[float]] = [[] for _ in bodies]
        bending = np.zeros(len(points), dtype=np.float64)
        section_i = np.pi * radii[0] ** 4 / 4.0
        for j, end, parent, child in joints:
            dual_length = 0.5 * (lengths[local[parent]] + lengths[local[child]])
            dof = int(self._qd_start[j])
            ea = float(self._ke[dof + 2]) * dual_length
            rigidity[local[parent]].append(ea)
            rigidity[local[child]].append(ea)
            if not np.isclose(self._ke[dof + 3], self._ke[dof + 4]):
                raise ValueError(f"UIPC rod {label!r} requires isotropic bending stiffness")
            bending[node_ids[root(end)]] = float(self._ke[dof + 3]) * dual_length / section_i
        stretch = np.asarray(
            [np.mean(values) if values else 1.0e5 * length for values, length in zip(rigidity, lengths, strict=True)]
        ) / (np.pi * radii[0] ** 2)
        points = np.asarray(points, dtype=np.float64)
        edge_vectors = points[edges[:, 1]] - points[edges[:, 0]]
        if np.any(np.linalg.norm(edge_vectors, axis=1) <= 1.0e-10):
            raise ValueError(f"UIPC rod {label!r} has a collapsed centerline edge")
        if np.any(bending > 0.0):
            neighbors: list[list[int]] = [[] for _ in points]
            for a, b in edges:
                neighbors[a].append(b)
                neighbors[b].append(a)
            for node, adjacent in enumerate(neighbors):
                if len(adjacent) != 2:
                    continue
                first, second = points[adjacent] - points[node]
                cosine = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
                if cosine >= 1.0 - 1.0e-10:
                    raise ValueError(f"UIPC rod {label!r} has a backtracking bending hinge at node {node}")
        self._groups.append(
            _RodGroup(
                label,
                np.asarray(bodies, dtype=np.int32),
                edges,
                points,
                np.asarray(source_bodies, dtype=np.int32),
                np.asarray(local_points),
                np.asarray(weights),
                np.asarray(fixed, dtype=np.int32),
                np.asarray(kinematic),
                np.asarray(masses),
                midpoints,
                axes,
                radii[0],
                stretch,
                bending,
                all(colliding),
            )
        )
        self._bodies.update(bodies)

    def build(self, contact_elem: Any, body_range: tuple[int, int], subscene_elem: Any) -> None:
        for group in self._groups:
            if not np.all((group.bodies >= body_range[0]) & (group.bodies < body_range[1])):
                continue
            sc = linemesh(
                np.ascontiguousarray(group.points, dtype=np.float64), np.ascontiguousarray(group.edges, dtype=np.int32)
            )
            HookeanSpring().apply_to(sc, 1.0, mass_density=1000.0, thickness=group.radius)
            _view_attr(sc.edges().find("kappa")).reshape(-1)[:] = group.stretch
            if np.any(group.bending > 0.0):
                KirchhoffRodBending().apply_to(sc, 0.0)
                _view_attr(sc.vertices().find("bending_stiffness")).reshape(-1)[:] = group.bending
            sc.vertices().create("mass", 0.0)
            _view_attr(sc.vertices().find("mass")).reshape(-1)[:] = group.mass
            _view_attr(sc.vertices().find("is_fixed")).reshape(-1)[:] = group.fixed
            if sc.vertices().find("velocity") is None:
                sc.vertices().create("velocity", np.zeros(3, dtype=np.float64))
            velocity = self._initial_velocities(group)
            _view_attr(sc.vertices().find("velocity")).reshape(-1, 3)[:] = velocity
            FiniteElementExternalForce().apply_to(sc, np.zeros(3), enable=False)
            contact_elem.apply_to(sc)
            subscene_elem.apply_to(sc)
            label_surface(sc)
            if not group.collision_enabled:
                for simplices in (sc.vertices(), sc.edges()):
                    _view_attr(simplices.find("is_surf"))[:] = 0
            obj = self._scene.objects().create(group.label)
            slot, rest_slot = obj.geometries().create(sc)
            self._mapping.rod_geo_slots.append(slot)
            self._mapping.rod_rest_geo_slots.append(rest_slot)
            self._mapping.rod_body_indices.append(group.bodies)
            rows = slice(self._node_count, self._node_count + len(group.points))
            self._built.append((group, slot, rows))
            self._node_count = rows.stop

    def _initial_velocities(self, group: _RodGroup) -> np.ndarray:
        velocities = self._model.body_qd.numpy()
        com = self._model.body_com.numpy()
        result = np.zeros_like(group.points)
        for row in range(len(result)):
            for side in range(2):
                body = group.sources[row, side]
                transform = wp.transform(*self._body_q[body])
                point = np.asarray(wp.transform_point(transform, wp.vec3(*group.local_points[row, side])))
                center = np.asarray(wp.transform_point(transform, wp.vec3(*com[body])))
                result[row] += group.weights[row, side] * (
                    velocities[body, :3] + np.cross(velocities[body, 3:], point - center)
                )
        return result

    def attach_backend(self, accessor: Any) -> None:
        self._accessor = accessor
        if not self._built or accessor is None:
            return
        device = self._model.device
        self._positions = uipc.adapter.warp.buffer(accessor.vertex_count(), dtype=wp.vec3d, device=device)
        self._velocities = uipc.adapter.warp.buffer(accessor.vertex_count(), dtype=wp.vec3d, device=device)
        offsets, bodies, nodes, midpoints, axes, sources, points, weights, kinematic = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for group, slot, rows in self._built:
            offset = int(_view_attr(slot.geometry().meta().find("backend_fem_vertex_offset"))[0])
            offsets.extend(offset + np.arange(len(group.points)))
            bodies.extend(group.bodies)
            nodes.extend(group.edges + rows.start)
            midpoints.extend(group.midpoints)
            axes.extend(group.axes)
            sources.extend(group.sources)
            points.extend(group.local_points)
            weights.extend(group.weights)
            kinematic.extend(group.kinematic)
        self._offsets_host = np.asarray(offsets, dtype=np.int64)
        self._offsets = wp.array(offsets, dtype=wp.uint32, device=device)
        self._bodies_wp = wp.array(bodies, dtype=wp.int32, device=device)
        self._nodes = wp.array(nodes, dtype=wp.vec2i, device=device)
        self._midpoints = wp.array(midpoints, dtype=wp.vec3, device=device)
        self._axes = wp.array(axes, dtype=wp.vec3, device=device)
        self._sources_host = np.asarray(sources, dtype=np.int32)
        self._sources = wp.array(sources, dtype=wp.vec2i, device=device)
        self._points = wp.array(np.asarray(points), dtype=wp.vec3, ndim=2, device=device)
        self._weights = wp.array(weights, dtype=wp.vec2, device=device)
        self._kinematic = np.flatnonzero(kinematic)
        self._forces = wp.zeros(self._node_count, dtype=wp.vec3d, device=device)
        self._force_host = wp.empty(self._node_count, dtype=wp.vec3d, device="cpu", pinned=device.is_cuda)
        self._joint_starts = wp.array(self._joint_indices, dtype=wp.int32, device=device)
        self._joint_ends = wp.array(self._joint_indices + 1, dtype=wp.int32, device=device)
        self._read_buffers()

    def _read_buffers(self) -> None:
        count = self._accessor.vertex_count()
        self._accessor.copy_position_to(self._positions.buffer_view(), 0, count)
        self._accessor.copy_velocity_to(self._velocities.buffer_view(), 0, count)

    def read_state(self, state_in: State, state_out: State) -> None:
        if not self.has_rods or self._accessor is None:
            return
        self._read_buffers()
        model = self._model
        wp.launch(
            _read_segments,
            dim=self._bodies_wp.size,
            inputs=[
                self._bodies_wp,
                self._nodes,
                self._offsets,
                self._midpoints,
                self._axes,
                model.body_com,
                state_in.body_q,
                self._positions.warp(),
                self._velocities.warp(),
                state_out.body_q,
                state_out.body_qd,
            ],
            device=model.device,
        )
        # One-joint ranges also include loop joints outside articulation ranges.
        wp.launch(
            eval_articulation_ik,
            dim=(len(self._joint_indices), 1),
            inputs=[
                self._joint_starts,
                self._joint_ends,
                len(self._joint_indices),
                None,
                None,
                state_out.body_q,
                state_out.body_qd,
                model.body_com,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.joint_dof_dim,
                model.joint_q_start,
                model.joint_qd_start,
                model.body_flags,
                int(BodyFlags.ALL),
            ],
            outputs=[state_out.joint_q, state_out.joint_qd],
            device=model.device,
        )

    def cache_forces(self, state: State) -> None:
        if not self.has_rods or self._accessor is None:
            return
        self._forces.zero_()
        if state.body_f is not None:
            wp.launch(
                _gather_forces,
                dim=self._bodies_wp.size,
                inputs=[
                    self._bodies_wp,
                    self._nodes,
                    self._offsets,
                    self._positions.warp(),
                    state.body_q,
                    self._model.body_com,
                    state.body_f,
                    self._forces,
                ],
                device=self._model.device,
            )
        wp.copy(self._force_host, self._forces)

    def publish_forces(self) -> None:
        if not self.has_rods or self._accessor is None:
            return
        values = self._force_host.numpy()
        for _, slot, rows in self._built:
            attributes = slot.geometry().vertices()
            _view_attr(attributes.find("external_force")).reshape(-1, 3)[:] = values[rows]
            _view_attr(attributes.find("external_force/is_constrained")).reshape(-1)[:] = np.any(
                values[rows] != 0.0, axis=1
            )

    def sync_state(
        self,
        body_q: wp.array,
        body_qd: wp.array,
        *,
        rows: np.ndarray | None = None,
        world_mask: np.ndarray | None = None,
        write_position: bool = True,
        write_velocity: bool = True,
    ) -> None:
        if not self.has_rods or self._accessor is None:
            return
        if rows is None:
            rows = np.arange(self._node_count, dtype=np.int32)
        if world_mask is not None:
            worlds = self._model.body_world.numpy()[self._sources_host[rows, 0]]
            selected = np.full(len(rows), bool(np.any(world_mask)))
            local = worlds >= 0
            selected[local] = world_mask[worlds[local]]
            rows = rows[selected]
        if not len(rows):
            return
        wp.launch(
            _gather_nodes,
            dim=len(rows),
            inputs=[
                wp.array(rows, dtype=wp.int32, device=self._model.device),
                self._offsets,
                self._sources,
                self._points,
                self._weights,
                body_q,
                body_qd,
                self._model.body_com,
                self._positions.warp(),
                self._velocities.warp(),
            ],
            device=self._model.device,
        )
        if not hasattr(self, "_state_geometry"):
            self._state_geometry = self._accessor.create_geometry()
            self._state_geometry.vertices().create("position", np.zeros(3))
            self._state_geometry.vertices().create("velocity", np.zeros(3))
        geometry = self._state_geometry
        self._accessor.copy_to(geometry)
        offsets = self._offsets_host[rows]
        if write_position:
            _view_attr(geometry.vertices().find("position")).reshape(-1, 3)[offsets] = self._positions.warp().numpy()[
                offsets
            ]
        if write_velocity:
            _view_attr(geometry.vertices().find("velocity")).reshape(-1, 3)[offsets] = self._velocities.warp().numpy()[
                offsets
            ]
        self._accessor.copy_from(geometry)
        self._read_buffers()

    def update_kinematic(self, state: State) -> None:
        if self.has_rods and self._accessor is not None and len(self._kinematic):
            self.sync_state(state.body_q, state.body_qd, rows=self._kinematic)
