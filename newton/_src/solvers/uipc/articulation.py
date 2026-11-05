# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Articulation runtime state for the UIPC solver backend.

Each :class:`Articulation` holds geometry references and views into the shared
joint buffers owned by :mod:`.articulation_builder`. Transfers and animation
updates are batched across articulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warp as wp
from uipc.geometry import SimplicialComplexSlot

from newton import JointTargetMode, JointType
from newton.math import velocity_at_point


@wp.func
def com_twist_to_point_velocity(qd: wp.spatial_vector, X_wb: wp.transform, body_com: wp.vec3, point: wp.vec3):
    """Evaluate a point velocity from a COM-referenced body twist.

    Local copy of Newton's COM-twist point-velocity helper so the UIPC backend
    depends only on the public :func:`newton.math.velocity_at_point`.
    """
    return velocity_at_point(qd, point - wp.transform_point(X_wb, body_com))


@dataclass
class FreeJointReadbackContext:
    """Device-side body / model arrays for FREE-joint readback.

    The builder rebuilds this context for each output State, since body arrays
    can change when callers swap their input and output states.
    """

    body_q: wp.array
    body_qd: wp.array
    body_com: wp.array
    joint_parent: wp.array
    joint_child: wp.array
    joint_X_p: wp.array
    joint_X_c: wp.array
    joint_q_start: wp.array
    joint_qd_start: wp.array


# Batched joint transfers


@wp.kernel
def _cache_control_kernel(
    active_joints: wp.array[wp.int32],
    local_q_start: wp.array[wp.int32],
    local_qd_start: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_target_mode: wp.array[wp.int32],
    target_pos: wp.array[wp.float32],
    target_vel: wp.array[wp.float32],
    joint_f: wp.array[wp.float32],
    has_target_pos: int,
    has_target_vel: int,
    has_joint_f: int,
    blend_aims: int,
    control_values: wp.array2d[wp.float64],
    control_flags: wp.array2d[wp.int32],
):
    local = wp.tid()
    newton_idx = active_joints[local]
    jtype = joint_type[newton_idx]

    if jtype != JointType.REVOLUTE and jtype != JointType.PRISMATIC:
        return

    q_idx = local_q_start[local]
    qd_idx = local_qd_start[local]
    mode = joint_target_mode[qd_idx]

    # Drive position targets, including velocity aims when blending is enabled.
    if mode == JointTargetMode.POSITION or mode == JointTargetMode.POSITION_VELOCITY:
        control_flags[0, local] = 1
        if has_target_pos != 0:
            control_values[0, local] = wp.float64(target_pos[q_idx])
    elif blend_aims != 0 and mode == JointTargetMode.VELOCITY:
        control_flags[0, local] = 1
    else:
        control_flags[0, local] = 0

    # Forward velocity targets for POSITION_VELOCITY and blended VELOCITY modes.
    forward_vel = mode == JointTargetMode.POSITION_VELOCITY
    if blend_aims != 0 and mode == JointTargetMode.VELOCITY:
        forward_vel = True
    if forward_vel and has_target_vel != 0:
        control_values[1, local] = wp.float64(target_vel[qd_idx])
    else:
        control_values[1, local] = wp.float64(0.0)

    # Apply force or torque only in EFFORT mode.
    if mode == JointTargetMode.EFFORT and has_joint_f != 0:
        control_values[2, local] = wp.float64(joint_f[qd_idx])
        control_flags[1, local] = 1
    else:
        control_flags[1, local] = 0


@wp.kernel
def _write_readback_kernel(
    local_q_start: wp.array[wp.int32],
    local_qd_start: wp.array[wp.int32],
    joint_state: wp.array2d[wp.float64],
    joint_q_out: wp.array[wp.float32],
    joint_qd_out: wp.array[wp.float32],
    has_qd: int,
):
    local = wp.tid()
    joint_q_out[local_q_start[local]] = wp.float32(joint_state[0, local])
    if has_qd != 0:
        joint_qd_out[local_qd_start[local]] = wp.float32(joint_state[1, local])


@wp.func
def _free_joint_state(
    X_wp: wp.transform,
    v_wp: wp.spatial_vector,
    parent_com: wp.vec3,
    X_wc: wp.transform,
    v_wc: wp.spatial_vector,
    child_com: wp.vec3,
    X_pj: wp.transform,
    X_cj: wp.transform,
    has_parent: bool,
):
    """Recover a FREE coordinate and its child-COM-referenced relative twist."""
    X_wpj = X_pj
    w_p = wp.vec3()
    if has_parent:
        X_wpj = X_wp * X_pj
        w_p = wp.spatial_bottom(v_wp)
    X_wcj = X_wc * X_cj
    q_p = wp.transform_get_rotation(X_wpj)
    q_c = wp.transform_get_rotation(X_wcj)
    x_err = wp.transform_get_translation(X_wcj) - wp.transform_get_translation(X_wpj)
    q_pc = wp.quat_inverse(q_p) * q_c
    x_err_c = wp.quat_rotate_inv(q_p, x_err)
    x_child_com_world = wp.transform_point(X_wc, child_com)
    v_com_err = wp.spatial_top(v_wc)
    if has_parent:
        v_com_err = v_com_err - com_twist_to_point_velocity(v_wp, X_wp, parent_com, x_child_com_world)
    v_err_c = wp.quat_rotate_inv(q_p, v_com_err)
    w_err_c = wp.quat_rotate_inv(q_p, wp.spatial_bottom(v_wc) - w_p)
    return wp.transform(x_err_c, q_pc), wp.spatial_vector(v_err_c, w_err_c)


@wp.func
def _store_free_joint_state(
    joint: int,
    position: wp.transform,
    velocity: wp.spatial_vector,
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    q_start = joint_q_start[joint]
    qd_start = joint_qd_start[joint]
    p = wp.transform_get_translation(position)
    q = wp.transform_get_rotation(position)
    for k in range(3):
        joint_q[q_start + k] = p[k]
    for k in range(4):
        joint_q[q_start + 3 + k] = q[k]
    for k in range(6):
        joint_qd[qd_start + k] = velocity[k]


@wp.kernel(enable_backward=False)
def _free_joint_readback_kernel(
    free_joint_indices: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    """Recover all remaining FREE joints after body conversion has completed."""
    joint = free_joint_indices[wp.tid()]
    parent = joint_parent[joint]
    child = joint_child[joint]
    X_wp = wp.transform_identity()
    v_wp = wp.spatial_vector()
    parent_com = wp.vec3()
    if parent >= 0:
        X_wp = body_q[parent]
        v_wp = body_qd[parent]
        parent_com = body_com[parent]
    position, velocity = _free_joint_state(
        X_wp,
        v_wp,
        parent_com,
        body_q[child],
        body_qd[child],
        body_com[child],
        joint_X_p[joint],
        joint_X_c[joint],
        parent >= 0,
    )
    _store_free_joint_state(joint, position, velocity, joint_q_start, joint_qd_start, joint_q, joint_qd)


# Placeholder for empty warp arrays passed to kernels
_EMPTY_F32_CACHE: dict[str, wp.array] = {}


def _empty_f32(device: Any) -> wp.array:
    key = str(device)
    arr = _EMPTY_F32_CACHE.get(key)
    if arr is None:
        arr = wp.zeros(1, dtype=wp.float32, device=device)
        _EMPTY_F32_CACHE[key] = arr
    return arr


class Articulation:
    """Per-articulation metadata and CPU views into the builder's joint buffers."""

    def __init__(self, name: str, dt: float, device: Any) -> None:
        self.name = name
        self._dt = dt
        self._device = device

        # Joint metadata (populated by ArticulationBuilder)
        self.active_joint_indices: list[int] = []
        """Newton joint indices for active (driven) joints."""

        self._joint_to_local: dict[int, int] = {}
        """Newton joint index → local (0-based) index."""

        # Newton model index mapping (populated by register_joint)
        self._joint_q_start: dict[int, int] = {}
        self._joint_qd_start: dict[int, int] = {}

        # Positive entries route revolute effort through the armature potential.
        self._effort_armature: dict[int, float] = {}

        # UIPC geometry references (populated by ArticulationBuilder)
        self.joint_geo_slots: dict[int, SimplicialComplexSlot] = {}
        self.joint_mesh: dict[int, Any] = {}
        # Per-joint edge index and type for post-retrieve readback.
        self._joint_edge_idx: dict[int, int] = {}
        self._joint_is_revolute: dict[int, bool] = {}

        # Animator-facing CPU views (bound after all worlds build)
        self.joint_position: wp.array | None = None  # (J,) float64
        self.joint_velocity: wp.array | None = None  # (J,) float64
        self.target_position: wp.array | None = None  # (J,) float64
        self.target_velocity: wp.array | None = None  # (J,) float64
        self.target_force: wp.array | None = None  # (J,) float64
        self.is_constrained: wp.array | None = None  # (J,) int32
        self.is_force_constrained: wp.array | None = None  # (J,) int32

        # Offset into the shared active-joint buffers, assigned after all worlds build.
        self._buffer_offset = 0

        # Aim blending (populated by ArticulationBuilder)
        self.aim_blend_weights: dict[int, float] = {}

        # FREE joint readback (populated by register_free_joint)
        self._free_joint_indices: list[int] = []

    # Properties

    @property
    def num_active_joints(self) -> int:
        """Number of active (driven) joints in this articulation."""
        return len(self.active_joint_indices)

    @property
    def num_free_joints(self) -> int:
        """Number of FREE joints in this articulation (read back from body state)."""
        return len(self._free_joint_indices)

    # Build-time registration

    def register_joint(
        self,
        newton_idx: int,
        q_start: int,
        qd_start: int,
    ) -> int:
        """Register an active joint and return its local index.

        Args:
            newton_idx: Newton joint index.
            q_start: Start index in ``joint_q`` for this joint.
            qd_start: Start index in ``joint_qd`` for this joint.

        Returns:
            Local (0-based) index within this articulation.
        """
        local = len(self.active_joint_indices)
        self.active_joint_indices.append(newton_idx)
        self._joint_to_local[newton_idx] = local
        self._joint_q_start[newton_idx] = q_start
        self._joint_qd_start[newton_idx] = qd_start
        return local

    def register_free_joint(self, newton_idx: int) -> None:
        """Register a FREE joint for body-state readback.

        Only the global Newton joint index is stored; the kernel reads
        ``q_start`` / ``qd_start`` from the model arrays directly.

        Args:
            newton_idx: Newton joint index of the FREE joint.
        """
        self._free_joint_indices.append(newton_idx)

    def bind_state(self, state: wp.array, control: wp.array, flags: wp.array, offset: int) -> None:
        """Bind contiguous views of the shared pinned CPU buffers."""
        self._buffer_offset = offset
        end = offset + self.num_active_joints
        self.joint_position = state[0, offset:end]
        self.joint_velocity = state[1, offset:end]
        self.target_position = control[0, offset:end]
        self.target_velocity = control[1, offset:end]
        self.target_force = control[2, offset:end]
        self.is_constrained = flags[0, offset:end]
        self.is_force_constrained = flags[1, offset:end]
