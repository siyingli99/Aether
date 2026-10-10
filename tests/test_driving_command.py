import unittest

import numpy as np

from aether.envs.driving_command import _commands_from_bev_path


class DrivingCommandTest(unittest.TestCase):
    def test_future_offset_uses_right_positive_axis(self):
        # Path coordinates are (forward, lateral-right). At the first point,
        # the heading is forward; negative lateral displacement means left.
        left = np.asarray([[0, 0], [1, 0], [2, -1], [3, -2]], dtype=float)
        right = left.copy()
        right[:, 1] *= -1
        self.assertEqual(int(_commands_from_bev_path(left, lat_thresh=1)[0]), 1)
        self.assertEqual(int(_commands_from_bev_path(right, lat_thresh=1)[0]), 0)


if __name__ == "__main__":
    unittest.main()
