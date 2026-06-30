"""Benchmark adapters for running Sanjaya against external suites."""

from .mmou_adapter import SanjayaMMOUAdapter
from .mmou_cli import main as run_mmou_cli

__all__ = ["SanjayaMMOUAdapter", "run_mmou_cli"]
