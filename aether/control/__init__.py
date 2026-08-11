"""Fine-grained per-instance foreground control.

This package exposes a single high-level entry point :class:`ForegroundController`
that lets *any* AETHER task (diagnosis, scenario authoring, replay editing,
counterfactual studies, ...) tweak individual trained 3DGS instances:

    * pose: rigid SE(3) overrides per frame (delegates to
      ``DriveStudioRenderer.set_actor_pose``).
    * kinematics: window-level semantic perturbations using the
      ``v_lat / a_lon / omega_yaw / kappa_len`` schema.
    * appearance: SH0 / opacity edits with a transient context manager
      (``glare / hue / fade`` schema).
    * visibility: per-frame instance toggling via ``instances_fv``.

It is intentionally read-mostly: the real per-frame Gaussians are *not*
permanently mutated -- every override is a snapshot/restore so the trainer
state is left clean between calls.
"""
from .foreground import (
    ForegroundController,
    InstanceInfo,
    TrainedActorAdapter,
    AppearancePerturbation,
    KinematicPerturbation,
)

__all__ = [
    "ForegroundController",
    "InstanceInfo",
    "TrainedActorAdapter",
    "AppearancePerturbation",
    "KinematicPerturbation",
]
