"""AETHER — Autonomous Evaluation Testbed with High-fidelity Embodied Renderer.

A closed-loop driving simulator that wraps DriveStudio's native rendering stack
behind a HUGSIM-compatible AD-agent IPC interface.
"""
from . import paths  # noqa: F401  (side-effect: extends sys.path)

__version__ = "0.1.0"
