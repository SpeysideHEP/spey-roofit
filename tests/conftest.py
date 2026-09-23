"""Shared fixtures for the ``spey-roofit`` test suite."""

import os
import re
import sys

import numpy as np
import pytest
import spey

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
    config.addinivalue_line("markers", "topq: tests that need the ATLAS HS3 files")
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


TOPQ_DIRECTORY = os.path.join(REPOSITORY, "atlas-topq-2018-24")

#: The workspaces of the ATLAS ttbar+gamma record, HS3 file and pyhf twin.
TOPQ_WORKSPACES = {
    "ljets_1NF": ("workspace_ljets_1NF_HS3.json", "workspace_ljets_1NF.json"),
    "ljets_2NF": ("workspace_ljets_2NF_HS3.json", "workspace_ljets_2NF.json"),
    "dilep_1NF": ("workspace_dilep_1NF_HS3.json", "workspace_dilep_1NF.json"),
    "dilep_2NF": ("workspace_dilep_2NF_HS3.json", "workspace_dilep_2NF.json"),
    "combined_1NF": ("workspace_combined_1NF_HS3.json", "workspace_combined_1NF.json"),
    "combined_2NF": ("workspace_combined_2NF_HS3.json", "workspace_combined_2NF.json"),
}


@pytest.fixture(scope="session")
def uncorrelated_hs3(toy_yields):
    """
    HS3 twin of ``default.uncorrelated_background``.

    Returns:
        ``Dict[str, Any]``:
        The HS3 document.
    """
    from hs3_models import uncorrelated_background_document

    return uncorrelated_background_document(
        toy_yields["background"],
        toy_yields["signal"],
        toy_yields["data"],
        toy_yields["uncertainty"],
    )


@pytest.fixture(scope="session")
def background_only_hs3(toy_yields):
    """
    Background-only HS3 document, without a parameter of interest.

    Returns:
        ``Dict[str, Any]``:
        The HS3 document.
    """
    from hs3_models import uncorrelated_background_document

    return uncorrelated_background_document(
        toy_yields["background"],
        None,
        toy_yields["data"],
        toy_yields["uncertainty"],
    )


@pytest.fixture(scope="session")
def correlated_hs3(toy_yields):
    """
    HS3 twin of ``default.correlated_background``.

    Returns:
        ``Dict[str, Any]``:
        The HS3 document.
    """
    from hs3_models import correlated_background_document

    return correlated_background_document(
        toy_yields["background"],
        toy_yields["signal"],
        toy_yields["data"],
        toy_yields["covariance"],
    )


@pytest.fixture(scope="session")
def poisson_hs3(toy_yields):
    """
    HS3 twin of ``default.poisson``, a model without nuisance parameters.

    Returns:
        ``Dict[str, Any]``:
        The HS3 document.
    """
    from hs3_models import poisson_document

    return poisson_document(
        toy_yields["background"], toy_yields["signal"], toy_yields["data"]
    )


def topq_path(name: str, pyhf_twin: bool = False) -> str:
    """
    Path of one of the ATLAS ttbar+gamma workspaces.

    Args:
        name (``str``): key of :data:`TOPQ_WORKSPACES`, e.g. ``"dilep_1NF"``.
        pyhf_twin (``bool``, default ``False``): return the ``pyhf``
          serialisation instead of the HS3 one.

    Returns:
        ``str``:
        Path of the requested file. The calling test is skipped when the
        validation material has not been downloaded.
    """
    path = os.path.join(TOPQ_DIRECTORY, TOPQ_WORKSPACES[name][1 if pyhf_twin else 0])
    if not os.path.exists(path):
        pytest.skip(
            "ATLAS ttbar+gamma validation material is not available; "
            "run ./atlas-topq-2018-24/download.sh"
        )
    return path


@pytest.fixture(scope="session")
def pyhf_backend():
    """
    The ``spey-pyhf`` plug-in, configured for a reproducible comparison.

    ``spey-pyhf`` selects the ``jax`` tensor backend whenever ``jax`` is
    importable. That is the right default for it, but not for a test that
    compares numbers: ``jax`` brings its own precision and platform behaviour —
    on Apple silicon it picks the experimental Metal platform, which has no
    ``float64`` at all. The comparison therefore runs against ``numpy``, which
    is ``pyhf``'s own reference implementation.

    Returns:
        ``module``:
        The imported ``spey_pyhf`` module. The calling test is skipped when the
        plug-in is not installed, or is too old for the running ``spey``.
    """
    spey_pyhf = pytest.importorskip("spey_pyhf", reason="spey-pyhf is not installed")
    try:
        spey.get_backend("pyhf")
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"the installed spey-pyhf cannot be used: {error}")
    spey_pyhf.manager.backend = "numpy"
    return spey_pyhf


def translate_parameters(values, source_names, target_names, initial=None):
    """
    Rearrange a parameter vector from one backend's ordering into another's.

    Every backend is free to order its parameters as it likes — ``spey-roofit``
    sorts the nuisance parameters by name and puts the parameter of interest
    first, ``spey-pyhf`` keeps ``pyhf``'s own order — so a comparison at fixed
    parameters has to go through the names.

    The two models need not have the *same* parameters either. ``pyhf`` carries
    a ``lumi`` parameter that the ``RooFit`` model holds constant and therefore
    never exposes, and a parameter left out of the translation has to keep the
    value the target backend suggests for it — not zero, which for ``lumi``
    would multiply every expected yield by nothing.

    Args:
        values (``Sequence[float]``): parameter values in the source ordering.
        source_names (``Sequence[str]``): names of ``values``, in order.
        target_names (``Sequence[str]``): names in the target ordering.
        initial (``Sequence[float]``, default ``None``): values for the target
          parameters that ``source_names`` does not mention, typically
          ``config().suggested_init``. Zeros are used when omitted.

    Raises:
        ``KeyError``: if a source name has no counterpart in the target.
        ``ValueError``: if ``initial`` does not match ``target_names`` in length.

    Returns:
        ``np.ndarray``:
        The parameter vector in the target ordering.
    """
    target = list(target_names)
    if initial is None:
        out = np.zeros(len(target), dtype=float)
    else:
        out = np.asarray(initial, dtype=float).copy()
        if out.size != len(target):
            raise ValueError(
                f"`initial` has {out.size} entries but the target ordering has "
                f"{len(target)}."
            )
    for name, value in zip(source_names, values):
        try:
            out[target.index(name)] = float(value)
        except ValueError as error:
            raise KeyError(f"'{name}' is not a parameter of the other backend") from error
    return out


def roofit_to_pyhf_name(name: str) -> str:
    """
    Translate a ``RooFit`` parameter name into its ``pyhf`` spelling.

    ``HistFactory`` prefixes a constrained nuisance parameter with ``alpha_``
    and names the Monte-Carlo statistics parameters one per bin, whereas
    ``pyhf`` drops the prefix and keeps one vector-valued parameter per region.

    Args:
        name (``str``): the ``RooFit`` name.

    Returns:
        ``str``:
        The corresponding entry of ``spey-pyhf``'s ``parameter_names``.
    """
    if name.startswith("alpha_"):
        return name[len("alpha_") :]
    match = re.fullmatch(r"gamma_stat_(.+)_bin_(\d+)", name)
    if match:
        return f"staterror_{match.group(1)}[{match.group(2)}]"
    return name
