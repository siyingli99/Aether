"""Studio actor manager: plugin controllers are stepped once per sim time."""
from types import SimpleNamespace

from aether.studio.runtime import StudioActorManager
from aether.studio.schemas import ActorEdit, ControllerSpec, RoadPose


class CountingPlugin:
    def __init__(self):
        self.calls = []

    def step(self, time_s, actor, ego):
        self.calls.append(time_s)
        return RoadPose(x=10.0 * time_s, y=1.0)

    def reset(self):
        self.calls.clear()


def _manager(plugin):
    env = SimpleNamespace(timestamp=0.0, data_frame=0, _ground_ref=2.0,
                          _traffic_ego_state=lambda: None)
    manager = object.__new__(StudioActorManager)
    manager.env = env
    manager.dt = 0.25
    manager.spec = SimpleNamespace(actors=[ActorEdit(
        id="npc", source="external", operation="add", asset_id="car",
        start=RoadPose(), controller=ControllerSpec(kind="plugin", plugin="counting"),
    )])
    manager.plugins = {"npc": plugin}
    manager.external_slots = {"npc": 7}
    manager._asset_index = {"npc": 0}
    manager.pool = None
    manager.limsim = None
    manager.log = {}
    manager._entries = []
    manager._entries_key = None
    return manager


def test_plugin_is_stepped_once_per_time_and_uses_sim_clock():
    plugin = CountingPlugin()
    manager = _manager(plugin)
    manager.reset()
    for _ in range(4):                      # render, collision, info, ...
        manager.pose_at(manager.env.data_frame)
    assert plugin.calls == [0.0]

    manager.env.timestamp, manager.env.data_frame = 0.75, 8
    entries = manager.advance(0.75, ego=None)
    for _ in range(3):
        manager.pose_at(8)
    assert plugin.calls == [0.0, 0.75]
    assert abs(entries[0]["x"] - 7.5) < 1e-5
    assert entries[0]["instance_id"] == 7
