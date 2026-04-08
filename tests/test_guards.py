# tests/test_guards.py
"""Tests for LLM Guard scanner wiring."""

import pytest


def test_guards_module_imports():
    """Verify the module can be imported and scanners are configured."""
    pytest.importorskip("llm_guard")

    from src.guards import scan_input, scan_response

    assert callable(scan_input)
    assert callable(scan_response)
