"""Frozen non-relational perturbation-generation entry point."""
from study import make_nonrelational_control

GENERATOR_ID = "fixed-document-marker-enumeration-v1-with-one-fixed-fallback"
MAX_ATTEMPTS = 2

__all__ = ("GENERATOR_ID", "MAX_ATTEMPTS", "make_nonrelational_control")
