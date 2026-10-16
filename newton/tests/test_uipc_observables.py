# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate UIPC observable requests before native simulation begins."""

import importlib.util
import unittest

import numpy as np

import newton


@unittest.skipUnless(importlib.util.find_spec("uipc"), "uipc is not installed")
class TestUIPCObservables(unittest.TestCase):
    """Check capability, allocation, and ownership through the public API."""

    def setUp(self):
        """Construct a CPU model with the non-simulating native backend."""
        import uipc  # noqa: PLC0415

        self.model = newton.ModelBuilder().finalize(device="cpu")
        self.solver = newton.solvers.SolverUIPC(self.model, backend="none", logger_level=uipc.Logger.Error)
        self.kinds = {self.solver.ObservableKind.CONTACT_F}

    def test_contact_allocation_and_unsupported_kinds(self):
        """Size outputs from reporting storage and reject unavailable diagnostics."""
        self.assertEqual(self.solver.supported_observables, self.kinds)
        with self.assertRaisesRegex(ValueError, "Pass Contacts"):
            self.solver.observables(self.kinds)
        contacts = newton.Contacts(4, 2, device=self.model.device)
        observables = self.solver.observables(self.kinds, contacts=contacts)
        self.assertEqual(observables.contact_f.shape, (6,))
        self.assertIs(observables.contacts, contacts)
        self.assertIsNone(contacts.force)
        np.testing.assert_array_equal(observables.contact_f.numpy(), 0.0)
        for kind in (self.solver.ObservableKind.BODY_QDD, self.solver.ObservableKind.BODY_PARENT_F):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "does not support"):
                self.solver.observables({kind})

    def test_step_rejects_wrong_owner_or_contacts_before_initializing(self):
        """Reject invalid requests before advancing or initializing native state."""
        import uipc  # noqa: PLC0415

        contacts = newton.Contacts(4, 0, device=self.model.device)
        observables = self.solver.observables(self.kinds, contacts=contacts)
        other_solver = newton.solvers.SolverUIPC(self.model, backend="none", logger_level=uipc.Logger.Error)
        state_in, state_out = self.model.state(), self.model.state()
        invalid_contacts = (
            (None, "Pass the step's Contacts"),
            (newton.Contacts(5, 0, device=self.model.device), "need 5 rows"),
        )
        for wrong_contacts, message in invalid_contacts:
            with self.subTest(contacts=wrong_contacts), self.assertRaisesRegex(ValueError, message):
                self.solver.step(state_in, state_out, None, wrong_contacts, 1.0 / 60.0, observables=observables)
        with self.assertRaisesRegex(ValueError, "solver instance"):
            self.solver.step(
                state_in, state_out, None, contacts, 1.0 / 60.0, observables=other_solver.observables(set())
            )

    def test_zero_capacity_contact_request(self):
        """Keep zero-capacity contact requests bound and explicitly requested."""
        contacts = newton.CollisionPipeline(self.model, rigid_contact_max=0, soft_contact_max=0).contacts()
        observables = self.solver.observables(self.kinds, contacts=contacts)
        self.assertEqual(observables.contact_f.shape, (0,))
        self.assertTrue(observables.is_requested(self.solver.ObservableKind.CONTACT_F))
        self.solver.validate_observables(observables, contacts)


if __name__ == "__main__":
    unittest.main()
