from __future__ import annotations

from typing import Any

import numpy as np
import uipc.builtin as uipc_builtin
from uipc import view


def _view_attr(slot: Any) -> np.ndarray:
    return view(slot)  # pyright: ignore[reportArgumentType]


def _dhat_from_usd(_value: Any, context: dict[str, Any]) -> float | None:
    """Read a geometry's SI contact distance, including the older USD spelling."""
    prim = context["prim"]
    for name in ("uipc:dhat", "uipc:d_hat"):
        attribute = prim.GetAttribute(name)
        if attribute and attribute.HasAuthoredValue():
            value = attribute.Get()
            return None if value is None else float(value)
    return None


def _dhat_values(model: Any, name: str, count: int) -> np.ndarray:
    """Read one distance per owning entity; negative values inherit the scene default."""
    attribute = getattr(getattr(model, "uipc", None), name, None)
    if attribute is None:
        return np.full(count, -1.0)
    values = np.asarray(attribute.numpy(), dtype=np.float64).reshape(-1)
    if len(values) != count:
        raise ValueError(f"uipc:{name} requires {count} values, got {len(values)}")
    if not np.isfinite(values).all() or np.any(values == 0.0):
        raise ValueError(f"uipc:{name} must be finite and nonzero; negative values inherit contact/d_hat")
    return np.where(values > 0.0, values, -1.0)


def _uniform_dhat(values: np.ndarray, label: str) -> float:
    """Resolve explicit overrides for entities sharing one native collision geometry."""
    values = np.asarray(values)
    overrides = values[values > 0.0]
    if not len(overrides):
        return -1.0
    if np.any(overrides != overrides[0]):
        raise ValueError(f"UIPC geometry {label!r} requires a uniform uipc:dhat")
    return float(overrides[0])


def _apply_geometry_dhat(geometry: Any, dhat: float) -> None:
    """Store an explicit distance before native geometry initialization."""
    if dhat > 0.0:
        geometry.meta().create(uipc_builtin.d_hat, dhat)


def _rest_positions_from_model(model: Any, particle_indices: np.ndarray) -> np.ndarray | None:
    """Return authored UIPC rest points for a particle subset, when present.

    IsaacLab registers these arrays only for imported USD deformables. Procedural Newton
    cloth and soft-body builders leave the validity marker at zero, so they continue to
    use ``model.particle_q`` as both the initial and reference geometry.
    """
    uipc_attrs = getattr(model, "uipc", None)
    rest_attr = getattr(uipc_attrs, "rest_position", None) if uipc_attrs is not None else None
    valid_attr = getattr(uipc_attrs, "rest_position_valid", None) if uipc_attrs is not None else None
    if rest_attr is None or valid_attr is None:
        return None
    selected_valid = np.asarray(valid_attr.numpy()[particle_indices], dtype=np.int32).reshape(-1)
    if selected_valid.size == 0 or not np.all(selected_valid != 0):
        return None
    return np.asarray(rest_attr.numpy()[particle_indices], dtype=np.float64).reshape(-1, 3)


def _clone_geometry_with_positions(geometry: Any, positions: np.ndarray) -> Any:
    """Clone a UIPC geometry while replacing only its vertex positions."""
    rest_geometry: Any = geometry.clone()
    rest_view = _view_attr(rest_geometry.positions())
    rest_view[...] = np.asarray(positions, dtype=rest_view.dtype).reshape(rest_view.shape)
    return rest_geometry
