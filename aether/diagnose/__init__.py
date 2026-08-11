"""AETHER-Diag v2 -- semantic-causal diagnosis for end-to-end driving stacks."""
from .engine import DiagnosisEngine, DiagnosisConfig, DiagnosisOutput
from .risk import compute_risk, extract_critical_window, RiskWeights, CriticalWindow
from .causal import causal_target_localisation, CausalLocalisation, CausalScores
from .semantics import (
    SemanticPerturbation, SEMANTIC_VAR_NAMES, SEMANTIC_BOUNDS,
    TOPO_DIMS, PHOTO_DIMS, DIM, perturb_pose, TargetGaussianHandle,
    neutral_vec, per_axis_range,
)
from .pgd import dual_track_pgd, PGDConfig, DualTrackResult, TrackResult
from .losses import plan_loss, PlanLossWeights
from .probes import compute_step_probes, aggregate_channels, CHANNEL_NAMES
from .sensitivity import compute_normalized_sensitivities, Sensitivities
from .report import build_report, write_report
from .rollout import Rollout, MockRollout
from . import viz, fingerprint

__all__ = [
    "DiagnosisEngine", "DiagnosisConfig", "DiagnosisOutput",
    "compute_risk", "extract_critical_window", "RiskWeights", "CriticalWindow",
    "causal_target_localisation", "CausalLocalisation", "CausalScores",
    "SemanticPerturbation", "SEMANTIC_VAR_NAMES", "SEMANTIC_BOUNDS",
    "TOPO_DIMS", "PHOTO_DIMS", "DIM", "perturb_pose", "TargetGaussianHandle",
    "neutral_vec", "per_axis_range",
    "dual_track_pgd", "PGDConfig", "DualTrackResult", "TrackResult",
    "plan_loss", "PlanLossWeights",
    "compute_step_probes", "aggregate_channels", "CHANNEL_NAMES",
    "compute_normalized_sensitivities", "Sensitivities",
    "build_report", "write_report",
    "Rollout", "MockRollout",
    "viz", "fingerprint",
]
