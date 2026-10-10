from pathlib import Path
import tempfile
import unittest

import numpy as np

from aether.traffic.protocol import EgoState, TrafficFrame, VehicleState
from aether.traffic.visualization import LimSimTopDownRenderer


class TrafficVisualizationTest(unittest.TestCase):
    def test_renders_complete_network_and_live_actors(self):
        network = """<net>
          <edge id="road-a"><lane id="a_0" width="3.2" shape="0,0 25,0"/></edge>
          <edge id="road-b"><lane id="b_0" width="3.2" shape="25,0 25,20"/></edge>
          <edge id=":internal" function="internal"><lane id=":i" shape="0,0 1,1"/></edge>
        </net>"""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "map.net.xml"
            path.write_text(network, encoding="utf-8")
            renderer = LimSimTopDownRenderer(path, width=320, height=240)
            image = renderer.render(
                TrafficFrame(1, 0.25, (
                    VehicleState("bg-1", 10, 0, 0, 5),
                )),
                EgoState(0, 0, 0, 2),
            )
        self.assertEqual(len(renderer.lanes), 2)
        self.assertEqual(image.shape, (240, 320, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(int(image.max()), 200)


if __name__ == "__main__":
    unittest.main()
