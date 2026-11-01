# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Assets and solver settings shared by the libuipc-samples scene ports."""

from pathlib import Path

import numpy as np
import uipc
from uipc.geometry import (
    SimplicialComplexIO,
    extract_surface,
    flip_inward_triangles,
    label_surface,
    label_triangle_orient,
)

import newton
import newton.examples

SAMPLES_REVISION = "b70f32294e31e9e3dcc12445b72fa9f4698ff5b1"
SAMPLES_URL = "https://github.com/spiriMirror/libuipc-samples.git"


def asset_directory(override: str | None) -> Path:
    """Locate the original meshes, downloading a pinned revision when needed."""
    if override is not None:
        return Path(override).expanduser().resolve()
    return Path(newton.examples.download_external_git_folder(SAMPLES_URL, "assets/sim_data", ref=SAMPLES_REVISION))


def read_mesh(path: Path, *, surface: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Read sample topology, extracting an outward surface for affine bodies."""
    geometry = SimplicialComplexIO().read(str(path))
    if surface and geometry.dim() == 3:
        label_surface(geometry)
        label_triangle_orient(geometry)
        geometry = extract_surface(flip_inward_triangles(geometry))
    domain = geometry.tetrahedra() if geometry.dim() == 3 else geometry.triangles()
    arity = 4 if geometry.dim() == 3 else 3
    positions = np.array(uipc.view(geometry.positions()), dtype=np.float64, copy=True).reshape(-1, 3)
    topology = np.array(uipc.view(domain.topo()), dtype=np.int32, copy=True).reshape(-1, arity)
    return positions, topology


def _contact_table(tabular, _world_index, *elements):
    """Use the source scenes' fixed contact resistance for every geometry pair."""
    tabular.default_model(0.2, 1.0e8)
    for i, left in enumerate(elements):
        for right in elements[i:]:
            tabular.insert(left, right, 0.2, 1.0e8, True)


def create_solver(model, *, dt: float, workspace: str, fem_preconditioner: str | None = None):
    """Preserve the native samples' scene-relative tolerances and PCG settings."""
    solver = newton.solvers.SolverUIPC(
        model,
        dt=dt,
        workspace=workspace,
        scene_config=uipc.Scene.default_config(),
        kappa=1.0e8,
        logger_level=uipc.Logger.Warn,
        enable_soft_position_constraint=False,
    )
    solver.set_contact(True)
    solver.configure_scene(
        {
            "contact": {"d_hat_relative": 1.0e-3, "eps_velocity_relative": 1.0e-2},
            "newton": {"velocity_tol_relative": 1.0e-2, "transrate_tol": 10.0},
            "linear_system": {"tol_rate": 1.0e-4},
        }
    )
    if fem_preconditioner is not None:
        solver.configure_scene(
            {
                "linear_system": {"fem_preconditioner": fem_preconditioner},
                "newton": {"semi_implicit": {"enable": 1, "beta_tol": 1.0e-2, "K_min": 6}},
            }
        )
    solver.configure_contact_tabular(_contact_table)
    return solver
