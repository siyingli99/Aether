import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from aether.render.external_assets import ExternalVehiclePool, GaussianVehicleAsset


class ExternalAssetTest(unittest.TestCase):
    def _write_asset(self, root: Path):
        n = 8
        state = [None] * 9
        state[0] = 3
        state[1] = torch.tensor([[float(i), -1.0, 0.5] for i in range(n)])
        state[2] = torch.zeros(n, 1, 3)
        state[3] = torch.zeros(n, 15, 3)
        state[4] = torch.zeros(n, 20)
        state[5] = torch.full((n, 3), -3.0)
        state[6] = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)
        state[7] = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        state[8] = None
        torch.save((state, 30000), root / "gs.pth")
        (root / "wlh.json").write_text(json.dumps([2.0, 4.0, 1.5]))

    def test_load_prunes_and_converts_axes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_asset(root)
            asset = GaussianVehicleAsset.load(root, max_gaussians=4)
        self.assertEqual(asset.num_gaussians, 4)
        self.assertEqual(asset.size_lwh, (4.0, 2.0, 1.5))
        self.assertTrue(torch.allclose(asset.means[:, 1], torch.full((4,), 0.5)))
        self.assertTrue(torch.allclose(asset.means[:, 2], torch.full((4,), 0.25)))

    def test_round_trip_aether_asset(self):
        count = 7
        asset = GaussianVehicleAsset(
            means=torch.randn(count, 3),
            features_dc=torch.randn(count, 3),
            features_rest=torch.randn(count, 15, 3),
            opacities=torch.randn(count, 1),
            scales=torch.full((count, 3), -3.0),
            quats=torch.nn.functional.normalize(torch.randn(count, 4), dim=-1),
            size_lwh=(4.5, 1.9, 1.6),
            source="3DRealCar:test-sequence",
        )
        with tempfile.TemporaryDirectory() as temp:
            asset.save(temp)
            loaded = GaussianVehicleAsset.load(temp, max_gaussians=5)
        self.assertEqual(loaded.num_gaussians, 5)
        self.assertEqual(loaded.size_lwh, asset.size_lwh)
        self.assertEqual(loaded.source, asset.source)

    def test_pool_appends_slots_and_hides_originals(self):
        n, frames, instances = 3, 2, 2
        model = SimpleNamespace(
            _means=torch.nn.Parameter(torch.zeros(n, 3)),
            _features_dc=torch.nn.Parameter(torch.zeros(n, 3)),
            _features_rest=torch.nn.Parameter(torch.zeros(n, 15, 3)),
            _opacities=torch.nn.Parameter(torch.zeros(n, 1)),
            _scales=torch.nn.Parameter(torch.zeros(n, 3)),
            _quats=torch.nn.Parameter(torch.zeros(n, 4)),
            point_ids=torch.tensor([[0], [0], [1]]),
            instances_trans=torch.nn.Parameter(torch.zeros(frames, instances, 3)),
            instances_quats=torch.nn.Parameter(torch.zeros(frames, instances, 4)),
            instances_fv=torch.ones(frames, instances, dtype=torch.bool),
            instances_size=torch.ones(instances, 3),
        )
        deformable = SimpleNamespace(
            instances_fv=torch.ones(frames, 3, dtype=torch.bool),
        )
        renderer = SimpleNamespace(trainer=SimpleNamespace(models={
            "RigidNodes": model,
            "DeformableNodes": deformable,
        }))
        asset = GaussianVehicleAsset(
            means=torch.zeros(2, 3), features_dc=torch.zeros(2, 3),
            features_rest=torch.zeros(2, 15, 3), opacities=torch.zeros(2, 1),
            scales=torch.zeros(2, 3), quats=torch.tensor([[1., 0, 0, 0]]).repeat(2, 1),
            size_lwh=(4.0, 2.0, 1.5), source="unit-test",
        )
        pool = ExternalVehiclePool(renderer, asset, capacity=2)
        self.assertEqual(pool.instance_ids, (2, 3))
        self.assertEqual(tuple(model._means.shape), (7, 3))
        self.assertEqual(tuple(model.instances_trans.shape), (2, 4, 3))
        self.assertFalse(bool(model.instances_fv.any()))
        self.assertFalse(bool(deformable.instances_fv.any()))
        self.assertEqual(
            pool.assert_original_dynamic_hidden(),
            {"RigidNodes": 2, "DeformableNodes": 3},
        )
        self.assertEqual(model.point_ids[-4:].squeeze().tolist(), [2, 2, 3, 3])
        self.assertEqual(pool.original_gaussian_indices(0).tolist(), [0, 1])
        self.assertEqual(pool.original_gaussian_counts(), {0: 2, 1: 1})
        extracted = pool.extract_original(0)
        self.assertEqual(extracted.num_gaussians, 2)
        self.assertEqual(extracted.source, "checkpoint:RigidNodes:0")
        pool.set_original_visible(0, True)
        self.assertTrue(bool(model.instances_fv[:, 0].all()))
        self.assertFalse(bool(model.instances_fv[:, 1].any()))
        pool.set_original_visible(0, False)
        self.assertFalse(bool(model.instances_fv.any()))


if __name__ == "__main__":
    unittest.main()
