"""Unit tests for AETHER-Diag v2."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aether.diagnose import (
    DiagnosisEngine, DiagnosisConfig, MockRollout, PGDConfig,
    SemanticPerturbation, TOPO_DIMS, PHOTO_DIMS, DIM, perturb_pose,
)
from aether.diagnose.risk import extract_critical_window
from aether.diagnose.losses import plan_loss, PlanLossWeights
from aether.diagnose.pgd import dual_track_pgd
from aether.diagnose.causal import causal_target_localisation
from aether.diagnose.probes import compute_step_probes, CHANNEL_NAMES


class TestSemantics13D(unittest.TestCase):
    def test_dim_is_13(self):
        self.assertEqual(DIM, 13)
        self.assertEqual(len(TOPO_DIMS) + len(PHOTO_DIMS), 13)

    def test_zero_perturbation_neutral(self):
        z = SemanticPerturbation.zero()
        self.assertEqual(z.z.size, 13)
        self.assertTrue(z.is_identity())
        c2w = np.eye(4)
        out = perturb_pose(c2w, z, t=1.0)
        np.testing.assert_allclose(out, np.eye(4), atol=1e-5)

    def test_delta_d_lat_translates(self):
        z = SemanticPerturbation.zero(); z.z[1] = 2.0  # delta_d_lat
        c2w = np.eye(4)
        out = perturb_pose(c2w, z, t=0.0)
        self.assertAlmostEqual(float(out[1, 3]), 2.0, places=4)

    def test_freeze_topo_resets(self):
        z = SemanticPerturbation.zero()
        z.z[0] = 1.0; z.z[6] = 0.3   # topo + photo
        z2 = z.freeze_topo()
        self.assertEqual(float(z2.z[0]), 0.0)
        self.assertAlmostEqual(float(z2.z[6]), 0.3, places=6)


class TestRiskAndWindow(unittest.TestCase):
    def test_window_identifies_peak(self):
        roll = MockRollout(n_steps=20, danger_idx=0)
        trace = roll()
        win = extract_critical_window(trace, k_back=3, f_fwd=3)
        self.assertGreaterEqual(win.t_critical, 0)
        self.assertLess(win.t_critical, len(trace))
        self.assertLessEqual(win.start, win.t_critical)
        self.assertLessEqual(win.t_critical, win.end)
        self.assertGreater(win.peak_risk, 0.0)


class TestProbes(unittest.TestCase):
    def test_channels_present(self):
        roll = MockRollout(n_steps=12, danger_idx=0)
        trace = roll()
        p = compute_step_probes(trace)
        for k in CHANNEL_NAMES:
            self.assertIn(k, p)
            self.assertEqual(p[k].shape[0], len(trace))


class TestCausalAblation(unittest.TestCase):
    def test_planted_actor_ranks_first(self):
        roll = MockRollout(n_steps=14, danger_idx=1)

        def _rollout_keep(keep):
            return roll(actor_keep=keep, perturbation=SemanticPerturbation.zero())

        res = causal_target_localisation(
            actor_pool=roll.actor_pool, rollout_keep=_rollout_keep,
            rollout_displace=None, rollout_swap=None,
        )
        self.assertEqual(res.target_index_in_pool, 1)
        self.assertEqual(res.target_class, "RigidNodes")


class TestPGDIncreasesLossV2(unittest.TestCase):
    def test_topo_and_photo_loss_increase(self):
        roll = MockRollout(n_steps=14, danger_idx=0)
        target_idx = 0

        def _rollout_with_z(z):
            return roll(actor_keep=[target_idx], perturbation=z)

        cfg = PGDConfig(iters=2, step_size=0.25, fd_eps=0.05)
        dual = dual_track_pgd(_rollout_with_z, cfg)

        # final loss should be >= base for both tracks
        self.assertGreaterEqual(dual.topo.loss_history[-1],
                                dual.topo.base_loss - 1e-6)
        self.assertGreaterEqual(dual.photo.loss_history[-1],
                                dual.photo.base_loss - 1e-6)
        # topo track must not move photo dims
        for i in PHOTO_DIMS:
            self.assertEqual(dual.topo.grad_final[i], 0.0)
        for i in TOPO_DIMS:
            self.assertEqual(dual.photo.grad_final[i], 0.0)


class TestEngineEndToEnd(unittest.TestCase):
    def test_full_pipeline_with_viz(self):
        roll = MockRollout(n_steps=18, danger_idx=2)
        cfg = DiagnosisConfig(
            pgd=PGDConfig(iters=2, step_size=0.25, fd_eps=0.05),
            enable_l2_displacement=False,   # MockRollout has no displace
            emit_viz=True,
        )
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "diag.json"
            out = DiagnosisEngine(roll, tested_model="mock", cfg=cfg).run(
                output_path=out_path)
            self.assertTrue(out.report_path.exists())
            r = out.report
            self.assertEqual(r["schema_version"], "aether-diag-v2")
            self.assertEqual(r["metadata"]["target_object_id"], "SMPLNodes#3")
            self.assertIn("causal_localisation", r)
            self.assertIn("pgd", r)
            self.assertIn("observability_probes", r)
            self.assertIn(r["macro_pathology_metrics"]["diagnosis_conclusion"], {
                "no_observable_pathology",
                "kinematic_topology_dominated",
                "photometric_dominated",
                "balanced_dual_track",
            })
            # viz files should exist
            fig_dir = Path(td) / "figs"
            self.assertTrue(fig_dir.exists())
            for fname in ("risk_curve.png", "causal_bars.png",
                          "sensitivity_heatmap.png", "pgd_loss_curves.png",
                          "plan_overlay.png", "obs_channels.png"):
                self.assertTrue((fig_dir / fname).exists(),
                                f"missing viz: {fname}")


if __name__ == "__main__":
    unittest.main()
