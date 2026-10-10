import numpy as np
import os
import pickle
import pytest
import threading

from aether.agents.sdk.bridge import BaseAgent, _trajectory, serve


def test_trajectory_accepts_array_or_plan_dict():
    expected = np.asarray([[0.0, 1.0], [-0.5, 2.0]], dtype=np.float32)
    np.testing.assert_array_equal(_trajectory(expected), expected)
    np.testing.assert_array_equal(_trajectory({"plan_traj": expected}), expected)


@pytest.mark.parametrize(
    "value",
    [[], [1.0, 2.0], [[1.0]], [[0.0, float("nan")]]],
)
def test_trajectory_rejects_invalid_plans(value):
    with pytest.raises(ValueError):
        _trajectory(value)


def test_serve_exchanges_one_frame_and_stops(tmp_path):
    class Agent(BaseAgent):
        def __init__(self):
            super().__init__()
            self.closed = False

        def infer(self, obs, info):
            assert obs["rgb"]["CAM_FRONT"].shape == (2, 3, 3)
            assert info["timestamp"] == 0.0
            return [[0.0, 1.0], [0.25, 2.0]]

        def teardown(self):
            self.closed = True

    obs_pipe = tmp_path / "obs_pipe"
    plan_pipe = tmp_path / "plan_pipe"
    os.mkfifo(obs_pipe)
    os.mkfifo(plan_pipe)
    agent = Agent()
    thread = threading.Thread(
        target=serve, args=(agent, "0", str(tmp_path)), daemon=True
    )
    thread.start()

    payload = (
        {"rgb": {"CAM_FRONT": np.zeros((2, 3, 3), dtype=np.uint8)}},
        {"timestamp": 0.0},
    )
    with obs_pipe.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    with plan_pipe.open("rb") as handle:
        plan = pickle.load(handle)
    np.testing.assert_array_equal(
        plan, np.asarray([[0.0, 1.0], [0.25, 2.0]], dtype=np.float32)
    )

    with obs_pipe.open("wb") as handle:
        pickle.dump("Done", handle, protocol=4)
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert agent.closed
