from __future__ import annotations

from typing import Any

import numpy as np
from uipc import view


def _view_attr(slot: Any) -> np.ndarray:
    return view(slot)  # pyright: ignore[reportArgumentType]


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
