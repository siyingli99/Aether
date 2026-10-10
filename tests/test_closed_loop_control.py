import unittest
from unittest.mock import patch

import numpy as np

from scripts.closed_loop import _traj_to_action, _trajectory_states


class TrajectoryControlTest(unittest.TestCase):
    def test_trajectory_states_uses_segment_deltas(self) -> None:
        # UniAD columns are (right, forward): negative lateral is a left turn.
        plan = np.asarray([[0.0, 1.0], [-1.0, 2.0], [-2.0, 2.0]])

        states = _trajectory_states(plan)

        np.testing.assert_allclose(
            states[1:, :2], [[1.0, -0.0], [2.0, 1.0], [2.0, 2.0]]
        )
        np.testing.assert_allclose(
            states[1:, 2], [0.0, np.pi / 4.0, np.pi / 2.0],
            atol=1e-6,
        )

    def test_trajectory_states_updates_previous_waypoint(self) -> None:
        plan = np.asarray([[-1.0, 1.0], [-2.0, 1.0]])

        states = _trajectory_states(plan)

        np.testing.assert_allclose(states[1:, 2], [np.pi / 4.0, np.pi / 2.0])

    def test_trajectory_states_rejects_non_finite_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _trajectory_states(np.asarray([[0.0, np.nan]]))

    def test_traj_to_action_passes_segment_headings_to_ilqr(self) -> None:
        captured = {}

        def fake_plan2control(states, current):
            captured["states"] = states.copy()
            captured["current"] = current.copy()
            return 1.25, 0.5

        with patch(
            "aether.control.ilqr.lqr.plan2control",
            side_effect=fake_plan2control,
        ):
            action = _traj_to_action(
                np.asarray([[-1.0, 1.0], [-2.0, 1.0]]),
                {"ego_velo": 3.0, "ego_steer": 0.2},
            )

        # iLQR left-positive steering is converted to BicycleEgo right-positive.
        self.assertEqual(action, (1.25, -0.5))
        np.testing.assert_allclose(
            captured["states"][1:, 2], [np.pi / 4.0, np.pi / 2.0]
        )
        np.testing.assert_allclose(
            captured["current"], [0.0, 0.0, 0.0, 3.0, -0.2]
        )

    def test_left_and_right_waypoints_keep_execution_sign(self) -> None:
        captured = []

        def semantic_controller(states, _current):
            captured.append(states.copy())
            return 0.0, float(np.sign(states[-1, 1]))

        with patch(
            "aether.control.ilqr.lqr.plan2control",
            side_effect=semantic_controller,
        ):
            left_action = _traj_to_action(
                np.asarray([[-0.5, 1.0], [-1.0, 2.0]]),
                {"ego_velo": 2.0, "ego_steer": 0.0},
            )
            right_action = _traj_to_action(
                np.asarray([[0.5, 1.0], [1.0, 2.0]]),
                {"ego_velo": 2.0, "ego_steer": 0.0},
            )

        self.assertLess(left_action[1], 0.0)
        self.assertGreater(right_action[1], 0.0)
        self.assertGreater(captured[0][-1, 1], 0.0)
        self.assertLess(captured[1][-1, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
