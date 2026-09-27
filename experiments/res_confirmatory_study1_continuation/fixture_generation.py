"""Frozen fixture-generation entry points.

The implementation lives in :mod:`study`; this separately hashed module makes
the fixture generator's role explicit in the cryptographic freeze.
"""
from study import build_pool, messages_for, structural_metrics, validate_fixture_invariants

__all__ = ("build_pool", "messages_for", "structural_metrics", "validate_fixture_invariants")
