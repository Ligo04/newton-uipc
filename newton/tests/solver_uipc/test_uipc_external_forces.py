# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check State force inputs against native UIPC motion."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import get_selected_cuda_test_devices

_HAS_UIPC = importlib.util.find_spec("uipc") is not None
_CUDA_TEST_DEVICES = get_selected_cuda_test_devices(mode="basic")


@unittest.skipUnless(_HAS_UIPC and _CUDA_TEST_DEVICES, "uipc and a selected CUDA device are required")
class TestUIPCExternalForces(unittest.TestCase):
    def test_body_wrenches_and_clearing(self):
        """Apply world wrenches at COM across instances and independent worlds."""
        for device in _CUDA_TEST_DEVICES:
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                scene = newton.ModelBuilder(gravity=wp.vec3(0.0))
                for world_id in range(2):
                    world = newton.ModelBuilder(gravity=wp.vec3(0.0))
                    for index in range(2):
                        offset_com = world_id == 1 and index == 0
                        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), 0.4)
                        body = world.add_body(
                            xform=wp.transform(wp.vec3(0.0, index * 2.0, 0.0), rotation),
                            is_kinematic=world_id == 1 and index == 1,
                        )
                        world.add_shape_box(
                            body,
                            xform=wp.transform(wp.vec3(0.13, 0.08, -0.04)) if offset_com else None,
                            hx=0.1,
                            hy=0.15,
                            hz=0.2,
                        )
                    if world_id == 1:
                        proxy = world.add_link(
                            xform=wp.transform(wp.vec3(0.0, 4.0, 0.0)),
                            mass=2.0,
                            inertia=wp.mat33(0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4),
                            com=wp.vec3(0.02, -0.03, 0.01),
                        )
                        joint = world.add_joint_prismatic(
                            parent=-1,
                            child=proxy,
                            parent_xform=wp.transform(wp.vec3(0.0, 4.0, 0.0)),
                            axis=newton.Axis.X,
                            actuator_mode=newton.JointTargetMode.EFFORT,
                        )
                        world.add_articulation([joint])
                    scene.add_world(world, xform=wp.transform(wp.vec3(world_id * 3.0, 0.0, 0.0)))
                model = scene.finalize(device=device)
                state, next_state = model.state(), model.state()
                dt = 1.0 / 120.0
                workspace = self.enterContext(tempfile.TemporaryDirectory())
                solver = newton.solvers.SolverUIPC(model, dt=dt, workspace=workspace)
                initial_q = state.body_q.numpy().copy()
                acceleration = np.array(
                    [[0.0, 0.0, 0.0], [2.0, -1.0, 0.5], [-0.5, 1.5, 0.75], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
                )
                forces = np.zeros((model.body_count, 6), dtype=np.float32)
                forces[:, :3] = model.body_mass.numpy()[:, None] * acceleration
                forces[3] = 100.0
                state.body_f.assign(forces)

                # The first lazy-initialized step must consume the supplied forces.
                solver.step(state, next_state, None, dt=dt)
                np.testing.assert_allclose(next_state.body_qd.numpy()[:, :3], acceleration * dt, atol=2.0e-5)
                np.testing.assert_allclose(next_state.body_qd.numpy()[:, 3:], 0.0, atol=2.0e-5)
                np.testing.assert_allclose(next_state.body_q.numpy()[[0, 3]], initial_q[[0, 3]], atol=1.0e-6)
                np.testing.assert_array_equal(state.body_f.numpy(), forces)
                np.testing.assert_array_equal(next_state.body_f.numpy(), 0.0)
                self.assertEqual(solver.mapping.body_geo_slots[0].id(), solver.mapping.body_geo_slots[1].id())
                self.assertGreater(np.linalg.norm(model.body_com.numpy()[2]), 0.1)
                state, next_state = next_state, state

                # A pure world torque about an offset COM must not accelerate that COM.
                rotation = np.asarray(wp.quat_to_matrix(wp.quat(*state.body_q.numpy()[2, 3:]))).reshape(3, 3)
                inertia = rotation @ model.body_inertia.numpy()[2] @ rotation.T
                angular_acceleration = np.array([0.7, -0.4, 0.3])
                forces.fill(0.0)
                forces[2, 3:] = inertia @ angular_acceleration
                state.body_f.assign(forces)
                local_com = np.append(model.body_com.numpy()[2], 1.0)
                com_before = solver.get_body_affine_transforms()[0].numpy()[2] @ local_com
                solver.step(state, next_state, None, dt=dt)
                com_after = solver.get_body_affine_transforms()[0].numpy()[2] @ local_com
                np.testing.assert_allclose(com_after[:3], com_before[:3] + acceleration[2] * dt * dt, atol=2.0e-6)
                np.testing.assert_allclose(next_state.body_qd.numpy()[:, :3], acceleration * dt, atol=2.0e-5)
                np.testing.assert_allclose(next_state.body_qd.numpy()[2, 3:], angular_acceleration * dt, atol=3.0e-5)
                state, next_state = next_state, state

                # Both an absent input and explicit zeros must retire native force entries.
                for absent in (True, False):
                    previous_velocity = state.body_qd.numpy().copy()
                    force_buffer = state.body_f
                    if absent:
                        state.body_f = None
                        solver.reset(state, flags=newton.StateFlags.BODY_QD)
                    else:
                        state.clear_forces()
                    solver.step(state, next_state, None, dt=dt)
                    state.body_f = force_buffer
                    np.testing.assert_allclose(next_state.body_qd.numpy(), previous_velocity, atol=3.0e-5)
                    state, next_state = next_state, state

                forces.fill(0.0)
                forces[1, :3] = model.body_mass.numpy()[1] * acceleration[1]
                forces[4, :3] = model.body_mass.numpy()[4] * acceleration[4]
                state.body_f.assign(forces)
                control = model.control()
                efforts = control.joint_f.numpy().copy()
                slider = np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.PRISMATIC))[0]
                efforts[model.joint_qd_start.numpy()[slider]] = forces[4, 0]
                control.joint_f.assign(efforts)
                previous_velocity = state.body_qd.numpy()[:, :3].copy()
                solver.step(state, state, control, dt=dt)
                np.testing.assert_allclose(
                    state.body_qd.numpy()[1, :3], previous_velocity[1] + acceleration[1] * dt, atol=2.0e-5
                )
                np.testing.assert_allclose(
                    state.body_qd.numpy()[4, :3], previous_velocity[4] + 2.0 * acceleration[4] * dt, atol=2.0e-5
                )
                np.testing.assert_array_equal(state.body_f.numpy(), 0.0)

    def test_particle_forces_and_clearing(self):
        """Gather cloth and tet forces through irregular world particle offsets."""
        for device in _CUDA_TEST_DEVICES:
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                scene = newton.ModelBuilder(gravity=wp.vec3(0.0))
                newton.solvers.SolverUIPC.register_custom_attributes(scene)
                accelerations = []
                for world_id in range(2):
                    world = newton.ModelBuilder(gravity=wp.vec3(0.0))
                    newton.solvers.SolverUIPC.register_custom_attributes(world)
                    for _ in range(world_id + 1):
                        world.add_particle(wp.vec3(-1.0, 0.0, 0.0), wp.vec3(0.0), 1.0)
                        accelerations.append((0.0, 0.0, 0.0))
                    world.add_cloth_mesh(
                        pos=wp.vec3(0.0),
                        rot=wp.quat_identity(),
                        scale=1.0,
                        vel=wp.vec3(0.0),
                        vertices=[wp.vec3(0.0), wp.vec3(0.15, 0.0, 0.0), wp.vec3(0.0, 0.15, 0.0)],
                        indices=[0, 1, 2],
                        density=1.0,
                        tri_ke=1.0e3,
                        tri_ka=1.0e3,
                        tri_kd=0.0,
                    )
                    accelerations.extend([(2.0, 0.0, 0.0) if world_id == 0 else (0.0, -1.5, 0.0)] * 3)
                    world.add_particle(wp.vec3(-2.0, 0.0, 0.0), wp.vec3(0.0), 1.0)
                    accelerations.append((0.0, 0.0, 0.0))
                    world.add_soft_mesh(
                        pos=wp.vec3(0.0, 1.0, 0.0),
                        rot=wp.quat_identity(),
                        scale=1.0,
                        vel=wp.vec3(0.0),
                        vertices=[
                            wp.vec3(0.0),
                            wp.vec3(0.1, 0.0, 0.0),
                            wp.vec3(0.0, 0.1, 0.0),
                            wp.vec3(0.0, 0.0, 0.1),
                        ],
                        indices=[0, 1, 2, 3],
                        density=1.0e3,
                        k_mu=1.0e4,
                        k_lambda=1.0e4,
                        k_damp=0.0,
                    )
                    accelerations.extend([(0.0, 0.0, 0.0) if world_id == 0 else (0.0, 0.0, 0.75)] * 4)
                    scene.add_world(world, xform=wp.transform(wp.vec3(world_id * 3.0, 0.0, 0.0)))
                model = scene.finalize(device=device)
                state, next_state = model.state(), model.state()
                initial_positions = state.particle_q.numpy().copy()
                acceleration = np.array(accelerations, dtype=np.float32)
                forces = model.particle_mass.numpy()[:, None] * acceleration
                # Use authored particle ranges to distinguish loose particles from mesh vertices.
                mapped = np.zeros(model.particle_count, dtype=bool)
                for prefix in ("cloth", "deformable_body"):
                    first = getattr(model.uipc, f"{prefix}_particle_first").numpy()
                    last = getattr(model.uipc, f"{prefix}_particle_last").numpy()
                    for start, end in zip(first, last, strict=True):
                        mapped[start : end + 1] = True
                unmapped = np.flatnonzero(~mapped)
                forces[unmapped] = 100.0
                state.particle_f.assign(forces)
                dt = 1.0 / 120.0
                workspace = self.enterContext(tempfile.TemporaryDirectory())
                solver = newton.solvers.SolverUIPC(model, dt=dt, workspace=workspace)
                solver.step(state, next_state, None, dt=dt)
                np.testing.assert_allclose(next_state.particle_qd.numpy(), acceleration * dt, atol=3.0e-5)
                np.testing.assert_allclose(
                    next_state.particle_q.numpy(), initial_positions + acceleration * dt * dt, atol=2.0e-6
                )
                np.testing.assert_array_equal(state.particle_f.numpy(), forces)
                np.testing.assert_array_equal(next_state.particle_f.numpy(), 0.0)
                state, next_state = next_state, state
                for absent in (False, True):
                    previous_velocity = state.particle_qd.numpy().copy()
                    force_buffer = state.particle_f
                    if absent:
                        state.particle_f = None
                    else:
                        state.clear_forces()
                    solver.step(state, next_state, None, dt=dt)
                    state.particle_f = force_buffer
                    np.testing.assert_allclose(next_state.particle_qd.numpy(), previous_velocity, atol=3.0e-5)
                    state, next_state = next_state, state


if __name__ == "__main__":
    unittest.main()
