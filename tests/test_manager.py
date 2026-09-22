"""Tests for the lazy ``ROOT`` manager."""

import pytest

from spey_roofit.manager import RootManager, manager


def test_root_is_available():
    """``PyROOT`` is importable and reports a version."""
    assert manager.is_available
    assert manager.version


def test_batch_mode_is_enabled():
    """The manager puts ``ROOT`` into batch mode so no canvas is created."""
    assert manager.ROOT.gROOT.IsBatch()


def test_silence_rejects_unknown_level():
    """An unknown message level is reported instead of silently ignored."""
    local = RootManager()
    with pytest.raises(ValueError, match="Unknown RooFit message level"):
        local.silence("LOUD")


def test_silence_accepts_known_levels():
    """Every documented level is accepted, in any capitalisation."""
    local = RootManager()
    for level in ("debug", "INFO", "Progress", "warning", "ERROR", "fatal"):
        local.silence(level)


def test_keep_alive_and_release():
    """Objects handed to the manager are held until explicitly released."""
    local = RootManager()
    sentinel = object()
    local.keep_alive(sentinel)
    assert sentinel in local._keep_alive  # pylint: disable=protected-access
    local.release()
    assert not local._keep_alive  # pylint: disable=protected-access
