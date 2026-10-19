# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Transfer Newton State forces to UIPC's independent external-force channels."""

from __future__ import annotations

import numpy as np
import warp as wp
from uipc.constitution import AffineBodyExternalBodyForce, FiniteElementExternalForce
from uipc.geometry import SimplicialComplexSlot

from ...sim import Model, State
from .converter import UIpcMappingInfo
from .utils import _view_attr


@wp.kernel(enable_backward=False)
def _gather_body_forces(
    body_indices: wp.array[wp.int32],
    body_f: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_affine: wp.array[wp.mat44d],
    forces: wp.array2d[wp.float64],
):
    row = wp.tid()
    body = body_indices[row]
    wrench = body_f[body]
    force = wp.vec3d(wp.float64(wrench[0]), wp.float64(wrench[1]), wp.float64(wrench[2]))
    torque = wp.vec3d(wp.float64(wrench[3]), wp.float64(wrench[4]), wp.float64(wrench[5]))
    com = wp.vec3d(body_com[body])

    # x = A X + t: F acts at the local COM; torque is about the world COM.
    # G : (skew(dtheta) A) = torque . dtheta, so G_tau = skew(torque) A^-T / 2.
    affine_force = wp.outer(force, com)
    if wp.dot(torque, torque) > wp.float64(0.0):
        transform = body_affine[body]
        affine = wp.mat33d(
            transform[0, 0],
            transform[0, 1],
            transform[0, 2],
            transform[1, 0],
            transform[1, 1],
            transform[1, 2],
            transform[2, 0],
            transform[2, 1],
            transform[2, 2],
        )
        affine_force += wp.float64(0.5) * wp.skew(torque) @ wp.transpose(wp.inverse(affine))

    for axis in range(3):
        forces[row, axis] = force[axis]
    for component in range(9):
        forces[row, component + 3] = affine_force[component // 3, component % 3]


@wp.kernel(enable_backward=False)
def _gather_particle_forces(
    particle_indices: wp.array[wp.int32],
    particle_f: wp.array[wp.vec3],
    forces: wp.array2d[wp.float64],
):
    row = wp.tid()
    force = particle_f[particle_indices[row]]
    for axis in range(3):
        forces[row, axis] = wp.float64(force[axis])


class ExternalForces:
    """Register force constitutions once, then stage and publish each step's inputs."""

    def __init__(self, model: Model, mapping: UIpcMappingInfo):
        self._model = model
        self._body_groups: list[tuple[SimplicialComplexSlot, np.ndarray, np.ndarray]] = []
        self._particle_groups: list[tuple[SimplicialComplexSlot, slice]] = []

        bodies = sorted(mapping.body_geo_slots)
        groups: dict[int, tuple[SimplicialComplexSlot, list[int], list[int]]] = {}
        for row, body in enumerate(bodies):
            slot = mapping.body_geo_slots[body]
            if slot.id() not in groups:
                groups[slot.id()] = (slot, [], [])
            _, rows, instances = groups[slot.id()]
            rows.append(row)
            instances.append(mapping.body_instance_ids.get(body, 0))
        body_force = AffineBodyExternalBodyForce()
        for slot, rows, instances in groups.values():
            body_force.apply_to(slot.geometry(), np.zeros(12), enable=False)
            self._body_groups.append((slot, np.asarray(rows), np.asarray(instances)))

        particles = []
        particle_force = FiniteElementExternalForce()
        for slots, indices in (
            (mapping.cloth_geo_slots, mapping.cloth_particle_indices),
            (mapping.deformable_geo_slots, mapping.deformable_particle_indices),
        ):
            for slot, particle_indices in zip(slots, indices, strict=True):
                start = len(particles)
                particles.extend(particle_indices)
                particle_force.apply_to(slot.geometry(), np.zeros(3), enable=False)
                self._particle_groups.append((slot, slice(start, len(particles))))

        self._body_indices = wp.array(bodies, dtype=wp.int32, device=model.device)
        self._particle_indices = wp.array(particles, dtype=wp.int32, device=model.device)
        self._body_forces = wp.zeros((len(bodies), 12), dtype=wp.float64, device=model.device)
        self._particle_forces = wp.zeros((len(particles), 3), dtype=wp.float64, device=model.device)
        self._body_host = wp.empty(self._body_forces.shape, dtype=wp.float64, device="cpu", pinned=model.device.is_cuda)
        self._particle_host = wp.empty(
            self._particle_forces.shape, dtype=wp.float64, device="cpu", pinned=model.device.is_cuda
        )
        self._body_values = self._body_host.numpy()
        self._particle_values = self._particle_host.numpy()

    def cache(self, state: State, body_affine: wp.array) -> None:
        """Enqueue force gathers and D2H copies on the solver's Warp stream."""
        if self._body_indices.size:
            if state.body_f is None:
                self._body_forces.zero_()
            else:
                wp.launch(
                    _gather_body_forces,
                    dim=self._body_indices.size,
                    inputs=[self._body_indices, state.body_f, self._model.body_com, body_affine, self._body_forces],
                    device=self._model.device,
                )
            wp.copy(self._body_host, self._body_forces)
        if self._particle_indices.size:
            if state.particle_f is None:
                self._particle_forces.zero_()
            else:
                wp.launch(
                    _gather_particle_forces,
                    dim=self._particle_indices.size,
                    inputs=[self._particle_indices, state.particle_f, self._particle_forces],
                    device=self._model.device,
                )
            wp.copy(self._particle_host, self._particle_forces)

    def publish(self) -> None:
        """Write staged values after the shared control-transfer stream barrier."""
        for slot, rows, instances in self._body_groups:
            attributes = slot.geometry().instances()
            values = self._body_values[rows]
            _view_attr(attributes.find("external_force")).reshape(-1, 12)[instances] = values
            _view_attr(attributes.find("external_force/is_constrained"))[instances] = np.any(values != 0.0, axis=1)
        for slot, rows in self._particle_groups:
            attributes = slot.geometry().vertices()
            values = self._particle_values[rows]
            _view_attr(attributes.find("external_force")).reshape(-1, 3)[:] = values
            _view_attr(attributes.find("external_force/is_constrained"))[:] = np.any(values != 0.0, axis=1)
