"""Shared fixtures for the ``spey-roofit`` test suite."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytest.importorskip("ROOT", reason="spey-roofit requires PyROOT")

REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATLAS_DIRECTORY = os.path.join(REPOSITORY, "atlas-exot-2018-06")

BACKGROUND = [50.0, 60.0, 30.0]
SIGNAL = [12.0, 11.0, 5.0]
UNCERTAINTY = [3.0, 5.0, 2.0]
OBSERVED = [51.0, 48.0, 32.0]
COVARIANCE = [[9.0, 2.0, 1.0], [2.0, 25.0, 3.0], [1.0, 3.0, 4.0]]


def pytest_configure(config):
    """Register the custom markers used by the suite."""
    config.addinivalue_line("markers", "atlas: tests that need the ATLAS ROOT files")
    config.addinivalue_line("markers", "slow: tests that run a full hypothesis test")


@pytest.fixture(scope="session")
def toy_yields():
    """
    Yields shared by every toy model.

    Returns:
        ``Dict[str, List[float]]``:
        Background, signal, uncertainty, observation and covariance inputs.
    """
    return {
        "background": list(BACKGROUND),
        "signal": list(SIGNAL),
        "uncertainty": list(UNCERTAINTY),
        "data": list(OBSERVED),
        "covariance": [list(row) for row in COVARIANCE],
    }


@pytest.fixture(scope="session")
def uncorrelated_workspace(tmp_path_factory, toy_yields):
    """
    Path to a ``HistFactory`` twin of ``default.uncorrelated_background``.

    Returns:
        ``str``:
        Path of the ``.root`` file.
    """
    from toy_models import uncorrelated_background_workspace

    directory = str(tmp_path_factory.mktemp("uncorrelated"))
    return uncorrelated_background_workspace(
        directory,
        toy_yields["background"],
        toy_yields["signal"],
        toy_yields["data"],
        toy_yields["uncertainty"],
    )


@pytest.fixture(scope="session")
def background_only_workspace(tmp_path_factory, toy_yields):
    """
    Path to a background-only ``HistFactory`` workspace without any POI.

    Returns:
        ``str``:
        Path of the ``.root`` file.
    """
    from toy_models import uncorrelated_background_workspace

    directory = str(tmp_path_factory.mktemp("background_only"))
    return uncorrelated_background_workspace(
        directory,
        toy_yields["background"],
        None,
        toy_yields["data"],
        toy_yields["uncertainty"],
        name="bkgonly",
    )


@pytest.fixture(scope="session")
def correlated_workspace(tmp_path_factory, toy_yields):
    """
    Path to a ``HistFactory`` twin of ``default.correlated_background``.

    Returns:
        ``str``:
        Path of the ``.root`` file.
    """
    from toy_models import correlated_background_workspace

    directory = str(tmp_path_factory.mktemp("correlated"))
    return correlated_background_workspace(
        directory,
        toy_yields["background"],
        toy_yields["signal"],
        toy_yields["data"],
        toy_yields["covariance"],
    )


@pytest.fixture(scope="session")
def atlas_background_only():
    """
    Path to the published ATLAS background-only workspace.

    Returns:
        ``str``:
        Path of ``bOnlyFit_postFit.root``.
    """
    path = os.path.join(ATLAS_DIRECTORY, "bOnlyFit_postFit.root")
    if not os.path.exists(path):
        pytest.skip("ATLAS validation material is not available")
    return path


@pytest.fixture(scope="session")
def atlas_signal_plus_background():
    """
    Path to the published ATLAS signal-plus-background workspace.

    Returns:
        ``str``:
        Path of ``sPlusBFit_postFit.root``.
    """
    path = os.path.join(ATLAS_DIRECTORY, "sPlusBFit_postFit.root")
    if not os.path.exists(path):
        pytest.skip("ATLAS validation material is not available")
    return path
