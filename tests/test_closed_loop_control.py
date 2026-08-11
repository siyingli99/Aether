import unittest
from unittest.mock import patch

import numpy as np

from scripts.closed_loop import _traj_to_action, _trajectory_states


class TrajectoryControlTest(unittest.TestCase):
    def test_trajectory_states_uses_segment_deltas(self) -> None:
        plan = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 2.0]])

        states = _trajectory_states(plan)

        np.testing.assert_allclose(
            states[1:, :2], [[1.0, -0.0], [2.0, -1.0], [2.0, -2.0]]
        )
        np.testing.assert_allclose(
            states[1:, 2], [np.pi / 2.0, -np.pi / 4.0, 0.0],
            atol=1e-6,
        )

    def test_trajectory_states_updates_previous_waypoint(self) -> None:
        plan = np.asarray([[1.0, 1.0], [2.0, 1.0]])

        states = _trajectory_states(plan)

        np.testing.assert_allclose(states[1:, 2], [-np.pi / 4.0, 0.0])

    def test_trajectory_states_rejects_non_finite_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _trajectory_states(np.asarray([[0.0, np.nan]]))

    def test_traj_to_action_passes_segment_headings_to_ilqr(self) -> None:
        captured = {}

        def fake_plan2control(states, current):
            captured["states"] = states.copy()
            captured["current"] = current.copy()
            return 1.25, -0.5

        with patch(
            "aether.control.ilqr.lqr.plan2control",
            side_effect=fake_plan2control,
        ):
            action = _traj_to_action(
                np.asarray([[1.0, 1.0], [2.0, 1.0]]),
                {"ego_velo": 3.0, "ego_steer": 0.2},
            )

        self.assertEqual(action, (1.25, -0.5))
        np.testing.assert_allclose(
            captured["states"][1:, 2], [-np.pi / 4.0, 0.0]
        )
        np.testing.assert_allclose(
            captured["current"], [0.0, 0.0, 0.0, 3.0, 0.2]
        )


if __name__ == "__main__":
    unittest.main()
