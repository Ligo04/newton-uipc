# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-body contact force extraction from UIPC.

Two paths:
  - GPU (hot path): ``prepare_contact_gpu_data`` → warp kernels export contact
    records and forces into solver observables.
  - CPU (diagnostic): ``retrieve_contact_forces`` → ``ContactForceReadback``
    for ``get_contact_forces()`` readback API.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import uipc.builtin as uipc_builtin
import warp as wp
from uipc.core import ContactSystemFeature
from uipc.geometry import Geometry

from .converter import UIpcMappingInfo
from .utils import _view_attr

PRIMITIVE_TYPES: tuple[str, ...] = ("PH", "PP", "PE", "PT", "EE")
FORCE_CHANNELS: tuple[str, ...] = ("N", "F")

_PRIM_SPLIT: dict[str, int] = {"PH": 1, "PP": 1, "PE": 1, "PT": 1, "EE": 2}

# Gradient doublets use fixed arity per contact stencil.
_PRIM_ARITY: dict[str, int] = {"PH": 1, "PP": 2, "PE": 3, "PT": 4, "EE": 4}

_VERTEX_UNMAPPED = -1


# Shared gradient reading


def _read_gradient(csf: ContactSystemFeature, key: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Read one primitive+channel gradient from UIPC.

    Returns (i_view, grad_flat) or None if empty.
    i_view: (num_instances, arity) int array of vertex indices.
    grad_flat: (num_instances, arity, 3) float array of gradients.
    """
    geo = Geometry()
    csf.contact_gradient(key, geo)

    i_attr = geo.instances().find("i")
    grad_attr = geo.instances().find("grad")
    if i_attr is None or grad_attr is None:
        return None

    i_view = _view_attr(i_attr)
    grad_view = _view_attr(grad_attr)

    num_instances = i_view.shape[0]
    if num_instances == 0:
        return None

    if i_view.ndim == 1:
        i_view = i_view.reshape(-1, 1)
    arity = i_view.shape[1]
    grad_flat = grad_view.reshape(num_instances, arity, 3)
    return i_view, grad_flat


# CPU diagnostic path (ContactForceReadback)


@dataclass
class PerBodyPrimitiveForce:
    """Per-vertex contact forces on one body from one primitive type + channel."""

    vertex_indices: np.ndarray  # (K,) int64
    forces: np.ndarray  # (K, 3) float64
    torques: np.ndarray  # (K, 3) float64

    @classmethod
    def empty(cls) -> PerBodyPrimitiveForce:
        return cls(
            vertex_indices=np.empty(0, dtype=np.int64),
            forces=np.empty((0, 3), dtype=np.float64),
            torques=np.empty((0, 3), dtype=np.float64),
        )


@dataclass
class ContactForceReadback:
    """Full dump of one retrieve pass.

    Layout: ``data[body_idx][prim_type][channel] -> PerBodyPrimitiveForce``.
    """

    data: dict[int, dict[str, dict[str, PerBodyPrimitiveForce]]] = field(default_factory=dict)

    def body_total(self, body_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Sum all force/torque into a single (f, tau) pair, both shape (3,)."""
        f = np.zeros(3, dtype=np.float64)
        tau = np.zeros(3, dtype=np.float64)
        for prim_dict in self.data.get(body_idx, {}).values():
            for pbf in prim_dict.values():
                if pbf.forces.size:
                    f += pbf.forces.sum(axis=0)
                if pbf.torques.size:
                    tau += pbf.torques.sum(axis=0)
        return f, tau


def _quat_rotate(quat: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion (x, y, z, w)."""
    q_xyz = quat[:3]
    q_w = quat[3]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def retrieve_contact_forces(
    csf: ContactSystemFeature,
    mapping: UIpcMappingInfo,
    body_q_np: np.ndarray | None,
    body_com_np: np.ndarray | None,
    dt: float = 1.0,
) -> ContactForceReadback:
    """CPU diagnostic: read all contact gradients and bucket by body.

    Args:
        csf: UIPC contact system feature.
        mapping: Newton↔UIPC mapping info (needs vertex_to_body CPU data).
        body_q_np: (body_count, 7) body transforms [px,py,pz, qx,qy,qz,qw].
        body_com_np: (body_count, 3) body center-of-mass offsets.
        dt: Simulation time step [s] for gradient→force conversion.

    Returns:
        ContactForceReadback with per-body per-primitive forces and torques.
    """
    inv_dt2 = 1.0 / (dt * dt) if dt != 0.0 else 1.0
    result = ContactForceReadback()

    vtb_np = _build_vertex_to_body_np(mapping)
    if vtb_np is None:
        return result

    for prim in PRIMITIVE_TYPES:
        split_a = _PRIM_SPLIT[prim]
        for chan in FORCE_CHANNELS:
            key = f"{prim}+{chan}"
            data = _read_gradient(csf, key)
            if data is None:
                continue
            i_view, grad_flat = data
            n_inst = i_view.shape[0]
            arity = i_view.shape[1]

            for inst_idx in range(n_inst):
                verts = i_view[inst_idx]
                grads = grad_flat[inst_idx]

                for v_local in range(arity):
                    gv = int(verts[v_local])
                    if gv < 0 or gv >= len(vtb_np):
                        continue
                    body_idx = int(vtb_np[gv])
                    if body_idx == _VERTEX_UNMAPPED:
                        continue

                    force = -grads[v_local].astype(np.float64) * inv_dt2

                    torque = np.zeros(3, dtype=np.float64)
                    if body_q_np is not None and body_com_np is not None and 0 <= body_idx < len(body_q_np):
                        q = body_q_np[body_idx]
                        com_local = body_com_np[body_idx]
                        pos = q[:3]
                        quat = q[3:]
                        com_world = pos + _quat_rotate(quat, com_local)
                        # Approximate: use body COM for torque arm
                        torque = np.cross(-com_world + pos, force)

                    side = "A" if v_local < split_a else "B"
                    prim_key = f"{prim}.{side}"

                    if body_idx not in result.data:
                        result.data[body_idx] = {}
                    if prim_key not in result.data[body_idx]:
                        result.data[body_idx][prim_key] = {}
                    if chan not in result.data[body_idx][prim_key]:
                        result.data[body_idx][prim_key][chan] = PerBodyPrimitiveForce(
                            vertex_indices=np.array([gv], dtype=np.int64),
                            forces=force.reshape(1, 3),
                            torques=torque.reshape(1, 3),
                        )
                    else:
                        pbf = result.data[body_idx][prim_key][chan]
                        pbf.vertex_indices = np.append(pbf.vertex_indices, gv)
                        pbf.forces = np.vstack([pbf.forces, force])
                        pbf.torques = np.vstack([pbf.torques, torque])

    return result


def _build_vertex_to_body_np(mapping: UIpcMappingInfo) -> np.ndarray | None:
    """Build CPU vertex→body lookup from mapping (for diagnostic path)."""
    if not mapping.body_geo_slots:
        return None

    max_gv = 0
    body_vertex_ranges: list[tuple[int, int, int]] = []

    for body_idx, geo_slot in mapping.body_geo_slots.items():
        geo = geo_slot.geometry()
        offset_attr = geo.meta().find(uipc_builtin.global_vertex_offset)
        if offset_attr is None:
            continue
        base_offset = int(_view_attr(offset_attr)[0])
        n_verts = geo.vertices().size()
        instance_id = mapping.body_instance_ids.get(body_idx, 0)
        start = base_offset + instance_id * n_verts
        end = start + n_verts
        body_vertex_ranges.append((body_idx, start, end))
        max_gv = max(max_gv, end)

    # Rod vertices have no unique Newton body owner at shared nodes. Keep
    # their range addressable while leaving their reporting entries unmapped.
    for slot in mapping.rod_geo_slots:
        geo = slot.geometry()
        offset = geo.meta().find(uipc_builtin.global_vertex_offset) or geo.meta().find("backend_fem_vertex_offset")
        if offset is not None:
            max_gv = max(max_gv, int(_view_attr(offset)[0]) + geo.vertices().size())

    if max_gv == 0:
        return None

    vtb = np.full(max_gv, _VERTEX_UNMAPPED, dtype=np.int32)
    for body_idx, start, end in body_vertex_ranges:
        vtb[start:end] = body_idx

    return vtb


# GPU path: vertex map building


def build_gpu_vertex_maps(mapping: UIpcMappingInfo, body_count: int, device: wp.Device) -> None:
    """Build GPU lookup tables for contact force scatter.

    Populates ``mapping.vertex_to_body_wp``, ``mapping.vertex_to_particle_wp``,
    and ``mapping.body_to_first_shape_wp``.

    Args:
        mapping: The mapping info to populate (modified in-place).
        body_count: Number of rigid bodies in the model.
        device: Warp device for output arrays.
    """
    max_gv = 0
    body_vertex_ranges: list[tuple[int, int, int, np.ndarray]] = []

    for body_idx, geo_slot in mapping.body_geo_slots.items():
        geo = geo_slot.geometry()
        offset_attr = geo.meta().find(uipc_builtin.global_vertex_offset)
        if offset_attr is None:
            continue
        base_offset = int(_view_attr(offset_attr)[0])
        n_verts = geo.vertices().size()
        instance_id = mapping.body_instance_ids.get(body_idx, 0)
        start = base_offset + instance_id * n_verts
        end = start + n_verts
        # Transform ABD vertices from body-local to world coordinates.
        pos_np = np.ascontiguousarray(_view_attr(geo.positions()), dtype=np.float32).reshape(-1, 3)
        body_vertex_ranges.append((body_idx, start, end, pos_np))
        max_gv = max(max_gv, end)

    particle_vertex_ranges: list[tuple[np.ndarray, int, int]] = []
    for geo_slot, particle_indices in [
        *zip(mapping.cloth_geo_slots, mapping.cloth_particle_indices, strict=False),
        *zip(mapping.deformable_geo_slots, mapping.deformable_particle_indices, strict=False),
    ]:
        geo = geo_slot.geometry()
        offset_attr = geo.meta().find("backend_fem_vertex_offset") or geo.meta().find(uipc_builtin.global_vertex_offset)
        if offset_attr is None:
            continue
        base_offset = int(_view_attr(offset_attr)[0])
        pi = np.asarray(particle_indices, dtype=np.int32)
        end = base_offset + len(pi)
        particle_vertex_ranges.append((pi, base_offset, end))
        max_gv = max(max_gv, end)

    for slot in mapping.rod_geo_slots:
        geo = slot.geometry()
        offset = geo.meta().find(uipc_builtin.global_vertex_offset) or geo.meta().find("backend_fem_vertex_offset")
        if offset is not None:
            max_gv = max(max_gv, int(_view_attr(offset)[0]) + geo.vertices().size())

    if max_gv == 0:
        mapping.max_global_vertex = 0
        return

    vtb_np = np.full(max_gv, _VERTEX_UNMAPPED, dtype=np.int32)
    vtp_np = np.full(max_gv, _VERTEX_UNMAPPED, dtype=np.int32)
    vlp_np = np.zeros((max_gv, 3), dtype=np.float32)

    for body_idx, start, end, pos_np in body_vertex_ranges:
        vtb_np[start:end] = body_idx
        vlp_np[start:end] = pos_np

    for pi, start, end in particle_vertex_ranges:
        vtp_np[start:end] = pi

    mapping.vertex_to_body_wp = wp.from_numpy(vtb_np, dtype=wp.int32, device=device)
    mapping.vertex_to_body_np = vtb_np
    mapping.vertex_to_particle_wp = wp.from_numpy(vtp_np, dtype=wp.int32, device=device)
    mapping.vertex_local_pos_wp = wp.from_numpy(vlp_np, dtype=wp.vec3f, device=device)
    mapping.max_global_vertex = max_gv

    b2s_np = np.zeros(body_count, dtype=np.int32)
    for body_idx, shapes in mapping.body_shapes.items():
        if 0 <= body_idx < body_count and shapes:
            b2s_np[body_idx] = shapes[0]
    mapping.body_to_first_shape_wp = wp.from_numpy(b2s_np, dtype=wp.int32, device=device)


# GPU path: warp kernels


@wp.kernel
def _populate_contact_pairs_kernel(
    vert_a: wp.array[wp.int32],
    vert_b: wp.array[wp.int32],
    forces_n: wp.array[wp.vec3],
    forces_f: wp.array[wp.vec3],
    vertex_to_body: wp.array[wp.int32],
    vertex_to_particle: wp.array[wp.int32],
    vertex_local_pos: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    particle_q: wp.array[wp.vec3],
    body_to_first_shape: wp.array[wp.int32],
    rod_body_a: wp.array[wp.int32],
    rod_body_b: wp.array[wp.int32],
    rod_point_a: wp.array[wp.vec3],
    rod_point_b: wp.array[wp.vec3],
    body_count: int,
    max_global_vertex: int,
    ground_shape: int,
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_force: wp.array[wp.spatial_vector],
    counter: wp.array[wp.int32],
    max_contacts: int,
):
    tid = wp.tid()
    gv_a = vert_a[tid]
    gv_b = vert_b[tid]

    if gv_a < 0 or gv_a >= max_global_vertex:
        return

    body_a = vertex_to_body[gv_a]
    override_a = int(-1)
    override_b = int(-1)
    if rod_body_a:
        override_a = rod_body_a[tid]
        override_b = rod_body_b[tid]
    if override_a >= 0:
        body_a = override_a
    if body_a < 0 or body_a >= body_count:
        return

    shape_a = body_to_first_shape[body_a]

    shape_b = ground_shape
    body_b = int(-1)
    particle_b = int(-1)
    if override_b >= 0:
        body_b = override_b
        shape_b = body_to_first_shape[body_b]
    elif gv_b >= 0 and gv_b < max_global_vertex:
        bb = vertex_to_body[gv_b]
        if bb >= 0 and bb < body_count:
            body_b = bb
            shape_b = body_to_first_shape[bb]
        else:
            particle_b = vertex_to_particle[gv_b]

    slot = wp.atomic_add(counter, 0, 1)
    if slot >= max_contacts:
        return

    fn = forces_n[tid]
    ff = forces_f[tid]
    total_f = fn + ff
    n_len = wp.length(fn)
    normal = wp.vec3(0.0, 1.0, 0.0)
    if n_len > 1.0e-12:
        normal = fn / n_len

    contact_shape0[slot] = shape_a
    contact_shape1[slot] = shape_b
    contact_normal[slot] = normal
    contact_force[slot] = wp.spatial_vector(total_f[0], total_f[1], total_f[2], 0.0, 0.0, 0.0)

    # Map contact points into SensorContact.position_matrix.
    point_a = vertex_local_pos[gv_a]
    if override_a >= 0:
        point_a = wp.transform_point(wp.transform_inverse(body_q[body_a]), rod_point_a[tid])
    contact_point0[slot] = point_a
    if override_b >= 0:
        contact_point1[slot] = wp.transform_point(wp.transform_inverse(body_q[body_b]), rod_point_b[tid])
    elif body_b >= 0:
        contact_point1[slot] = vertex_local_pos[gv_b]
    elif particle_b >= 0 and particle_q:
        contact_point1[slot] = particle_q[particle_b]
    else:
        contact_point1[slot] = wp.transform_point(body_q[body_a], point_a)


# GPU path: data preparation (vectorized numpy, no per-element Python loop)


@dataclass
class GpuContactData:
    """Packed contact data ready for GPU transfer."""

    vert_a: np.ndarray  # (M,) int32 — per-instance body-A representative vertex
    vert_b: np.ndarray  # (M,) int32 — per-instance body-B representative vertex
    forces_n: np.ndarray  # (M, 3) float32 — normal channel force sum per instance
    forces_f: np.ndarray  # (M, 3) float32 — friction channel force sum per instance
    # Rod sides override the vertex-to-body lookup and supply world-space surface points.
    body_a: np.ndarray | None = None
    body_b: np.ndarray | None = None
    point_a: np.ndarray | None = None
    point_b: np.ndarray | None = None


@dataclass
class _RodContactVertex:
    bodies: np.ndarray
    weights: np.ndarray
    position: np.ndarray
    radius: float


@dataclass
class _ContactOwner:
    body: int
    vertex: int
    weight: float = 1.0
    point: np.ndarray | None = None


def _rod_contact_maps(
    mapping: UIpcMappingInfo,
) -> tuple[dict[int, _RodContactVertex], dict[tuple[int, int], int]]:
    """Resolve native rod vertices and edges to their Newton segment bodies."""
    vertices = {}
    edges = {}
    for slot, bodies, sources, weights in zip(
        mapping.rod_geo_slots,
        mapping.rod_body_indices,
        mapping.rod_vertex_sources,
        mapping.rod_vertex_weights,
        strict=True,
    ):
        geo = slot.geometry()
        offset_attr = geo.meta().find(uipc_builtin.global_vertex_offset) or geo.meta().find("backend_fem_vertex_offset")
        if offset_attr is None:
            continue
        offset = int(_view_attr(offset_attr)[0])
        positions = np.asarray(_view_attr(geo.positions())).reshape(-1, 3)
        radii = np.asarray(_view_attr(geo.vertices().find("thickness"))).reshape(-1)
        for local, position in enumerate(positions):
            vertices[offset + local] = _RodContactVertex(sources[local], weights[local], position, float(radii[local]))
        topology = np.asarray(_view_attr(geo.edges().topo())).reshape(-1, 2)
        for body, edge in zip(bodies, topology, strict=True):
            edges[tuple(sorted((offset + int(edge[0]), offset + int(edge[1]))))] = int(body)
    return vertices, edges


def _rod_contact_owners(
    indices: np.ndarray,
    forces: np.ndarray,
    vertices: dict[int, _RodContactVertex],
    edges: dict[tuple[int, int], int],
    vertex_to_body: np.ndarray,
) -> list[_ContactOwner]:
    """Resolve one stencil side, splitting shared point loads with rod node weights."""
    if len(indices) == 2 and tuple(sorted(indices)) in edges:
        nodes = [vertices[int(index)] for index in indices]
        magnitudes = np.linalg.norm(forces, axis=1)
        weights = magnitudes / magnitudes.sum() if magnitudes.sum() > 0.0 else np.full(2, 0.5)
        point = np.sum(np.asarray([node.position for node in nodes]) * weights[:, None], axis=0)
        owners = [_ContactOwner(edges[tuple(sorted(indices))], int(indices[0]), point=point)]
        radius = nodes[0].radius
    elif len(indices) == 1 and int(indices[0]) in vertices:
        node = vertices[int(indices[0])]
        owners = [
            _ContactOwner(int(body), int(indices[0]), float(weight), node.position.copy())
            for body, weight in zip(node.bodies, node.weights, strict=True)
            if weight > 0.0
        ]
        radius = node.radius
    else:
        for index in indices:
            if 0 <= index < len(vertex_to_body) and vertex_to_body[index] >= 0:
                return [_ContactOwner(int(vertex_to_body[index]), int(index))]
        return [_ContactOwner(-1, int(indices[0]) if len(indices) else -1)]
    force = forces.sum(axis=0)
    magnitude = np.linalg.norm(force)
    if magnitude > 0.0:
        for owner in owners:
            owner.point = owner.point - radius * force / magnitude
    return owners


def _pack_rod_contacts(
    primitive: str,
    topology: np.ndarray,
    normal: np.ndarray,
    friction: np.ndarray,
    vertices: dict[int, _RodContactVertex],
    edges: dict[tuple[int, int], int],
    vertex_to_body: np.ndarray,
    *,
    friction_topology: np.ndarray | None = None,
) -> GpuContactData | None:
    """Pack rod stencils as shape pairs while conserving each contact's total force."""
    if friction_topology is not None and not np.array_equal(topology, friction_topology):
        friction_by_stencil = dict(zip(map(tuple, friction_topology), friction, strict=True))
        aligned = np.zeros_like(normal)
        for row, indices in enumerate(topology):
            values = friction_by_stencil.get(tuple(indices))
            if values is not None:
                aligned[row] = values
        friction = aligned
    records = []
    split = _PRIM_SPLIT[primitive]
    for indices, fn, ff in zip(topology, normal, friction, strict=True):
        owners_a = _rod_contact_owners(indices[:split], fn[:split], vertices, edges, vertex_to_body)
        owners_b = _rod_contact_owners(indices[split:], fn[split:], vertices, edges, vertex_to_body)
        force_n, force_f = fn[:split].sum(axis=0), ff[:split].sum(axis=0)
        if all(owner.body < 0 for owner in owners_a):
            owners_a, owners_b = owners_b, owners_a
            force_n, force_f = fn[split:].sum(axis=0), ff[split:].sum(axis=0)
        for owner_a in owners_a:
            if owner_a.body < 0:
                continue
            for owner_b in owners_b:
                weight = owner_a.weight * owner_b.weight
                records.append(
                    (
                        owner_a.vertex,
                        owner_b.vertex,
                        force_n * weight,
                        force_f * weight,
                        owner_a.body if owner_a.point is not None else -1,
                        owner_b.body if owner_b.point is not None else -1,
                        owner_a.point if owner_a.point is not None else np.zeros(3),
                        owner_b.point if owner_b.point is not None else np.zeros(3),
                    )
                )
    if not records:
        return None
    columns = list(zip(*records, strict=True))
    return GpuContactData(
        vert_a=np.asarray(columns[0], dtype=np.int32),
        vert_b=np.asarray(columns[1], dtype=np.int32),
        forces_n=np.asarray(columns[2], dtype=np.float32),
        forces_f=np.asarray(columns[3], dtype=np.float32),
        body_a=np.asarray(columns[4], dtype=np.int32),
        body_b=np.asarray(columns[5], dtype=np.int32),
        point_a=np.asarray(columns[6], dtype=np.float32),
        point_b=np.asarray(columns[7], dtype=np.float32),
    )


def prepare_contact_gpu_data(
    csf: ContactSystemFeature,
    dt: float = 1.0,
    vertex_to_body: np.ndarray | None = None,
    mapping: UIpcMappingInfo | None = None,
) -> GpuContactData | None:
    """Read all UIPC contact gradients and pack into flat arrays.

    UIPC gradients are derivatives of the incremental potential which
    includes a ``kappa · dt²`` scaling.  Dividing by ``dt²`` converts
    them to physical forces [N].

    Pair export reconstructs contact stencils from the fixed per-primitive gradient
    layout (:data:`_PRIM_ARITY`): each stencil becomes one contact instance whose
    force is the gradient sum over its A-side vertices. When ``vertex_to_body``
    (global vertex → Newton body, CPU) is given, the A side is the body group of the
    first rigid vertex and ``vert_b`` is the first vertex of the other side, so
    body-body pairs resolve exactly; otherwise a first-``_PRIM_SPLIT`` heuristic
    is used.

    ABD-only stencils use vectorized NumPy packing. Rod stencils additionally
    resolve segment ownership and shared-node weights from ``mapping``.
    Returns None if no contact data is available.
    """
    inv_dt2 = 1.0 / (dt * dt) if dt != 0.0 else 1.0

    # Per-instance data for contact pairs (keyed by primitive type)
    inst_vert_a: list[np.ndarray] = []
    inst_vert_b: list[np.ndarray] = []
    inst_force_n: list[np.ndarray] = []
    inst_force_f: list[np.ndarray] = []
    rod_vertices, rod_edges = _rod_contact_maps(mapping) if mapping is not None else ({}, {})
    rod_indices = np.fromiter(rod_vertices, dtype=np.int32)
    inst_body_a, inst_body_b, inst_point_a, inst_point_b = [], [], [], []

    for prim in PRIMITIVE_TYPES:
        arity = _PRIM_ARITY[prim]

        pair_force: dict[str, np.ndarray] = {}
        pair_topo: tuple[np.ndarray, np.ndarray] | None = None
        rod_channels = {}

        for chan_label in FORCE_CHANNELS:
            data = _read_gradient(csf, f"{prim}+{chan_label}")
            if data is None:
                continue
            i_view, grad_flat = data

            verts_flat = i_view.reshape(-1).astype(np.int32)
            forces_flat = (-grad_flat.reshape(-1, 3) * inv_dt2).astype(np.float32)

            if verts_flat.shape[0] % arity != 0:
                continue  # unexpected layout: skip pair export
            sv = verts_flat.reshape(-1, arity)
            sf = forces_flat.reshape(-1, arity, 3)
            if rod_vertices:
                rod_rows = np.isin(sv, rod_indices).any(axis=1)
                rod_channels[chan_label] = (sv[rod_rows], sf[rod_rows])
                sv, sf = sv[~rod_rows], sf[~rod_rows]
            n_stencils = sv.shape[0]
            rows = np.arange(n_stencils)

            if vertex_to_body is not None and vertex_to_body.shape[0] > 0:
                mgv = vertex_to_body.shape[0]
                body_of = np.where((sv >= 0) & (sv < mgv), vertex_to_body[np.clip(sv, 0, mgv - 1)], -2)
                # Anchor side A on the first rigid vertex for ABD/FEM stencils.
                has_abd = body_of >= 0
                anchor = np.where(has_abd.any(axis=1), has_abd.argmax(axis=1), 0)
                a_side = body_of == body_of[rows, anchor][:, None]
            else:
                anchor = np.zeros(n_stencils, dtype=np.int64)
                a_side = np.zeros(sv.shape, dtype=bool)
                a_side[:, : _PRIM_SPLIT[prim]] = True

            force_a = (sf * a_side[:, :, None]).sum(axis=1).astype(np.float32)
            pair_force[chan_label] = force_a

            if chan_label == "N":
                b_side = ~a_side
                has_b = b_side.any(axis=1)
                vb = np.where(has_b, sv[rows, b_side.argmax(axis=1)], -1).astype(np.int32)
                pair_topo = (sv[rows, anchor].astype(np.int32), vb)

        if pair_topo is None or "N" not in pair_force:
            continue
        va, vb = pair_topo
        fn = pair_force["N"]
        ff = pair_force.get("F")
        inst_vert_a.append(va)
        inst_vert_b.append(vb)
        inst_force_n.append(fn)
        # Friction stencils mirror the normal set; align by index, drop on mismatch.
        if ff is not None and ff.shape[0] == fn.shape[0]:
            inst_force_f.append(ff)
        else:
            inst_force_f.append(np.zeros_like(fn))
        if rod_vertices:
            inst_body_a.append(np.full(len(va), -1, dtype=np.int32))
            inst_body_b.append(np.full(len(va), -1, dtype=np.int32))
            inst_point_a.append(np.zeros((len(va), 3), dtype=np.float32))
            inst_point_b.append(np.zeros((len(va), 3), dtype=np.float32))
            topology, normal = rod_channels["N"]
            friction_topology, friction = rod_channels.get("F", (None, None))
            if friction_topology is None:
                friction = np.zeros_like(normal)
            rod_data = _pack_rod_contacts(
                prim,
                topology,
                normal,
                friction,
                rod_vertices,
                rod_edges,
                vertex_to_body,
                friction_topology=friction_topology,
            )
            if rod_data is not None:
                inst_vert_a.append(rod_data.vert_a)
                inst_vert_b.append(rod_data.vert_b)
                inst_force_n.append(rod_data.forces_n)
                inst_force_f.append(rod_data.forces_f)
                inst_body_a.append(rod_data.body_a)
                inst_body_b.append(rod_data.body_b)
                inst_point_a.append(rod_data.point_a)
                inst_point_b.append(rod_data.point_b)

    if not inst_vert_a:
        return None

    vert_a_cat = np.concatenate(inst_vert_a)
    vert_b_cat = np.concatenate(inst_vert_b)
    forces_n_cat = np.concatenate(inst_force_n)
    forces_f_cat = np.concatenate(inst_force_f)

    return GpuContactData(
        vert_a=vert_a_cat,
        vert_b=vert_b_cat,
        forces_n=forces_n_cat,
        forces_f=forces_f_cat,
        body_a=np.concatenate(inst_body_a) if rod_vertices else None,
        body_b=np.concatenate(inst_body_b) if rod_vertices else None,
        point_a=np.concatenate(inst_point_a) if rod_vertices else None,
        point_b=np.concatenate(inst_point_b) if rod_vertices else None,
    )
