# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise sparse ArticulationView access through the native UIPC solver."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
from newton.selection import ArticulationView
from newton.tests.unittest_utils import get_selected_cuda_test_devices

_HAS_UIPC = importlib.util.find_spec("uipc") is not None
_CUDA_TEST_DEVICES = get_selected_cuda_test_devices(mode="basic")


def _make_sparse_world_model(device):
    """Interleave three floating robots with worlds and bodies of different sizes."""

    def add_ballast(world, index):
        body = world.add_body(label=f"ballast_{index}", xform=wp.transform(wp.vec3(0.0, index + 2.0, 0.0)))
        world.add_shape_box(body, hx=0.03 + 0.01 * index, hy=0.05, hz=0.05)

    scene = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for world_id, (robot, before, after) in enumerate(
        ((True, 0, 0), (False, 1, 0), (True, 1, 1), (False, 2, 0), (True, 2, 0))
    ):
        world = newton.ModelBuilder(gravity=wp.vec3(0.0))

        for index in range(before):
            add_ballast(world, index)
        if robot:
            root = world.add_link(label="robot/root")
            hinge = world.add_link(label="robot/hinge")
            slider = world.add_link(label="robot/slider")
            # Shape order deliberately differs from link order.
            world.add_shape_box(slider, hx=0.05, hy=0.05, hz=0.05)
            world.add_shape_box(root, hx=0.1, hy=0.1, hz=0.1)
            world.add_shape_box(hinge, hx=0.05, hy=0.05, hz=0.05)
            joints = [world.add_joint_free(child=root, label="robot/root_joint")]
            joints.append(
                world.add_joint_revolute(
                    root,
                    hinge,
                    axis=newton.Axis.Z,
                    actuator_mode=newton.JointTargetMode.POSITION,
                    label="robot/hinge_joint",
                )
            )
            joints.append(
                world.add_joint_prismatic(
                    hinge,
                    slider,
                    axis=newton.Axis.X,
                    actuator_mode=newton.JointTargetMode.POSITION,
                    label="robot/slider_joint",
                )
            )
            world.add_articulation(joints, label="robot")
        for index in range(before, before + after):
            add_ballast(world, index)
        scene.add_world(world, xform=wp.transform(wp.vec3(3.0 * world_id, 0.0, 0.0)))
    return scene.finalize(device=device)


@unittest.skipUnless(_HAS_UIPC and _CUDA_TEST_DEVICES, "uipc and a selected CUDA device are required")
class TestUIPCSelection(unittest.TestCase):
    def test_sparse_world_control_readback_and_reset(self):
        """Drive sparse robots and reset selected roots without changing other worlds."""
        for device in _CUDA_TEST_DEVICES:
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                model = _make_sparse_world_model(device)
                view = ArticulationView(model, "robot", homogeneous=False)
                drives = ArticulationView(
                    model, "robot", homogeneous=False, exclude_joint_types=[newton.JointType.FREE]
                )
                np.testing.assert_array_equal(view.world_ids.numpy(), [0, 2, 4])
                self.assertTrue(view.is_sparse)
                self.assertTrue(view.uses_explicit_model_indices)
                self.assertEqual((view.world_count, view.count_per_world), (3, 1))

                # Use model topology as an independent reference for every readback.
                driven_joints = [
                    joint
                    for joint, label in enumerate(model.joint_label)
                    if label.endswith(("/hinge_joint", "/slider_joint"))
                ]
                q_rows = model.joint_q_start.numpy()[driven_joints].reshape(3, 2)
                qd_rows = model.joint_qd_start.numpy()[driven_joints].reshape(3, 2)
                self.assertTrue(np.any(q_rows != qd_rows))
                root_joints = [joint for joint, label in enumerate(model.joint_label) if label.endswith("/root_joint")]
                root_bodies = model.joint_child.numpy()[root_joints]
                ballast_bodies = [body for body, label in enumerate(model.body_label) if "ballast_" in label]

                state, next_state, control = model.state(), model.state(), model.control()
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                initial_body_q = state.body_q.numpy().copy()
                targets = np.array([[[0.1, 0.02]], [[-0.2, -0.03]], [[0.3, 0.04]]], dtype=np.float32)
                drives.set_attribute("joint_target_q", control, targets)

                workspace = self.enterContext(tempfile.TemporaryDirectory())
                dt = 1.0 / 60.0
                solver = newton.solvers.SolverUIPC(model, dt=dt, workspace=workspace)
                solver.initialize(state)

                # Reuse the scene for the first commands and a masked target change.
                for masked in (False, True):
                    with self.subTest(masked=masked):
                        if masked:
                            replacements = np.array([[[0.8, 0.1]], [[-0.35, 0.01]], [[-0.8, -0.1]]], dtype=np.float32)
                            drives.set_attribute("joint_target_q", control, replacements, mask=[False, True, False])
                            targets[1] = replacements[1]
                        np.testing.assert_array_equal(control.joint_target_q.numpy()[q_rows], targets[:, 0])
                        before_q = state.joint_q.numpy()[q_rows].copy()
                        solver.step(state, next_state, control, dt=dt)
                        state, next_state = next_state, state
                        actual_q = state.joint_q.numpy()[q_rows]
                        np.testing.assert_allclose(actual_q, targets[:, 0], atol=5.0e-4)
                        np.testing.assert_allclose(
                            drives.get_dof_positions(state).numpy(), actual_q[:, None], atol=1.0e-6
                        )
                        np.testing.assert_allclose(
                            drives.get_dof_velocities(state).numpy()[:, 0],
                            (actual_q - before_q) / dt,
                            atol=1.0e-4,
                        )
                        np.testing.assert_allclose(
                            state.body_q.numpy()[ballast_bodies], initial_body_q[ballast_bodies], atol=1.0e-5
                        )
                        for _ in range(2):
                            solver.step(state, next_state, control, dt=dt)
                            state, next_state = next_state, state

                before_reset = solver.get_body_affine_transforms()[0].numpy().copy()
                roots = view.get_root_transforms(state).numpy().copy()
                roots[:, 0, 2] += [0.3, 0.7, 0.5]
                view_mask = np.array([True, False, True])
                view.set_root_transforms(state, wp.array(roots, dtype=wp.transform, device=device), mask=view_mask)
                view.set_root_velocities(
                    state, wp.zeros((3, 1), dtype=wp.spatial_vector, device=device), mask=view_mask
                )

                # Solver masks use model worlds; view masks use the compact world axis.
                world_mask = np.zeros(model.world_count, dtype=bool)
                world_mask[view.world_ids.numpy()[view_mask]] = True
                unselected_bodies = ~world_mask[model.body_world.numpy()]
                poisoned_body_q = state.body_q.numpy().copy()
                poisoned_body_q[unselected_bodies, 2] += 10.0
                state.body_q.assign(poisoned_body_q)
                solver.reset(
                    state,
                    world_mask=wp.array(world_mask, dtype=bool, device=device),
                    flags=newton.StateFlags.JOINT_Q | newton.StateFlags.JOINT_QD,
                )
                np.testing.assert_array_equal(
                    solver.get_body_affine_transforms()[0].numpy()[unselected_bodies], before_reset[unselected_bodies]
                )
                solver.step(state, next_state, control, dt=dt)
                state, next_state = next_state, state

                np.testing.assert_allclose(
                    state.body_q.numpy()[root_bodies[view_mask], :3], roots[view_mask, 0, :3], atol=1.0e-4
                )
                np.testing.assert_allclose(
                    view.get_root_transforms(state).numpy()[:, 0], state.body_q.numpy()[root_bodies], atol=1.0e-5
                )
                self.assertTrue(np.all(np.isfinite(state.joint_q.numpy())))
                self.assertTrue(np.all(np.isfinite(state.joint_qd.numpy())))


if __name__ == "__main__":
    unittest.main()
