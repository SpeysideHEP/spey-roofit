r"""
Validation against the published ATLAS monojet workspaces.

The material is taken from the HEPData record of
*Search for new phenomena in events with an energetic jet and missing
transverse momentum in pp collisions at* :math:`\sqrt{s}=13` *TeV with the
ATLAS detector* (:xref:`2102.10874`, HEPData record ``ins1847779``):

* ``bOnlyFit_postFit.root`` — background-only ``RooFit`` workspace with the
  nuisance parameters at their post-fit values;
* ``sPlusBFit_postFit.root`` — signal-plus-background workspace for the
  ``DMA_1_2000_gq0p25`` benchmark, also at its post-fit values;
* ``hepdata/Histogram_*.json`` — the published data and post-fit background
  predictions of the signal and control regions.

Three independent checks are performed: the interpreter must reproduce the
published yields, the profile likelihood must reproduce the published
:math:`\hat\mu`, and a signal injected into the background-only workspace must
behave the way the published signal-plus-background workspace does.
"""

import json
import os

import numpy as np
import pytest

from spey_roofit import WorkspaceInterpreter

pytestmark = pytest.mark.atlas

CUTS = [200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
REGIONS = {
    "SR": "Histogram_SR",
    "CR1mu0b": "Histogram_CR1mu0b",
    "CR1e0b": "Histogram_CR1e0b",
    "CR1L1b": "Histogram_CR1L1b",
    "CR2mu": "Histogram_CR2mu",
    "CR2e": "Histogram_CR2e",
}
PUBLISHED_MU_HAT = 0.12684446636721747
"""Signal strength stored in the published signal-plus-background workspace."""


def _rounding_tolerance(text):
    """
    Half a unit in the last significant digit of a published number.

    The HEPData tables quote the yields with a handful of significant digits,
    so a comparison can only be as tight as that rounding allows.

    Args:
        text (``str``): the number exactly as published.

    Returns:
        ``float``:
        Absolute tolerance implied by the quoted precision.
    """
    digits = text.strip().lstrip("+-").replace(".", "").lstrip("0").rstrip("0")
    significant = max(len(digits), 1)
    exponent = np.floor(np.log10(abs(float(text))))
    return 0.5 * 10.0 ** (exponent - significant + 1)


def _hepdata(name):
    """
    Read one of the published HEPData histogram tables.

    Args:
        name (``str``): base name of the table file.

    Returns:
        ``List[Dict]``:
        The rows of the table, or ``None`` if the file is not available.
    """
    from conftest import ATLAS_DIRECTORY

    path = os.path.join(ATLAS_DIRECTORY, "hepdata", f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["values"]


@pytest.fixture(name="background_only", scope="module")
def background_only_fixture(atlas_background_only):
    """
    Interpreter for the published background-only workspace.

    Returns:
        ~spey_roofit.helper_functions.WorkspaceInterpreter:
        Interpreter of ``bOnlyFit_postFit.root``.
    """
    return WorkspaceInterpreter(atlas_background_only)


@pytest.fixture(name="signal_plus_background", scope="module")
def signal_plus_background_fixture(atlas_signal_plus_background):
    """
    Interpreter for the published signal-plus-background workspace.

    Returns:
        ~spey_roofit.helper_functions.WorkspaceInterpreter:
        Interpreter of ``sPlusBFit_postFit.root``.
    """
    return WorkspaceInterpreter(atlas_signal_plus_background)


def test_region_structure(background_only):
    """The interpreter finds the 78 single-bin regions of the analysis."""
    assert len(background_only.channels) == 78
    assert set(background_only.bin_map.values()) == {1}
    assert len(background_only.guess_signal_regions()) == 13
    assert len(background_only.guess_control_regions()) == 65
    assert background_only.poi_names == []


def test_signal_plus_background_structure(signal_plus_background):
    """The published signal workspace declares its own parameter of interest."""
    assert signal_plus_background.poi_names == ["mu_SIG"]
    assert signal_plus_background.signal_samples["SR_200_cuts"] == ["DMA_1_2000_gq0p25"]
    assert signal_plus_background.signal_samples["CR1mu0b_200_cuts"] == []


@pytest.mark.parametrize("region", sorted(REGIONS))
def test_yields_match_hepdata(background_only, region):
    """Observed counts and post-fit background reproduce the published tables."""
    rows = _hepdata(REGIONS[region])
    if rows is None:
        pytest.skip("HEPData tables are not available")

    observed = background_only.observed_data
    background = background_only.expected_background_yields
    for row, cut in zip(rows, CUTS):
        channel = f"{region}_{cut}_cuts"
        published_data = row["y"][0]["value"]
        published_background = row["y"][1]["value"]
        assert observed[channel][0] == pytest.approx(float(published_data))
        assert background[channel][0] == pytest.approx(
            float(published_background),
            abs=_rounding_tolerance(published_background),
        )


def test_signal_template_matches_the_published_workspace(signal_plus_background):
    r"""
    The signal of the published workspace scales linearly with :math:`\mu`.

    The yields stored in the workspace correspond to the fitted
    :math:`\hat\mu`, so dividing them by it gives the template at
    :math:`\mu=1`.
    """
    from spey_roofit.data import WorkspaceData

    model = WorkspaceData(signal_plus_background.workspace)
    nominal = model.nominal_parameters
    assert nominal[0] == pytest.approx(PUBLISHED_MU_HAT)

    at_zero = np.array(model.expected_data(np.concatenate([[0.0], nominal[1:]])))
    at_one = np.array(model.expected_data(np.concatenate([[1.0], nominal[1:]])))
    at_two = np.array(model.expected_data(np.concatenate([[2.0], nominal[1:]])))
    template = at_one - at_zero
    assert (at_two - at_zero)[: model.nbins] == pytest.approx(
        2.0 * template[: model.nbins], rel=1e-9
    )
    # the signal only lives in the signal regions
    for index, channel in enumerate(model.channel_names):
        if not channel.startswith("SR"):
            assert template[index] == pytest.approx(0.0, abs=1e-6)
    assert np.sum(template[: model.nbins]) > 0.0


@pytest.mark.slow
def test_profile_likelihood_recovers_the_published_mu_hat(
    atlas_signal_plus_background,
):
    r"""
    Maximising the likelihood returns the :math:`\hat\mu` of the workspace.

    The published workspace stores the result of the ATLAS fit, so recovering
    it validates the whole chain: yield evaluation, constraint terms, data
    vector layout and the ``spey`` optimisation.
    """
    import spey

    model = spey.get_backend("root.roofit")(
        workspace=atlas_signal_plus_background, analysis="atlas-exot-2018-06"
    )
    mu_hat, _ = model.maximize_likelihood()
    assert mu_hat == pytest.approx(PUBLISHED_MU_HAT, rel=1e-4)


@pytest.mark.slow
def test_injected_signal_reproduces_the_published_workspace(
    atlas_background_only, signal_plus_background
):
    r"""
    Injecting the published signal template into the background-only
    workspace gives back the signal-plus-background expectation.

    The two workspaces are not identical — the published one also carries the
    signal theory systematics — so the comparison is done on the expected
    yields at fixed nuisance parameters rather than on the likelihood.
    """
    from spey_roofit.data import WorkspaceData

    published = WorkspaceData(signal_plus_background.workspace)
    nominal = published.nominal_parameters
    template = np.array(
        published.expected_data(np.concatenate([[1.0], nominal[1:]]))
    ) - np.array(published.expected_data(np.concatenate([[0.0], nominal[1:]])))

    signal = {
        channel: [float(template[index])]
        for index, channel in enumerate(published.channel_names)
        if template[index] > 0.0
    }
    assert set(signal) == set(
        name for name in published.channel_names if name.startswith("SR")
    )

    interpreter = WorkspaceInterpreter(atlas_background_only)
    interpreter.inject_signals(signal)
    injected = WorkspaceData(interpreter.patch)

    zero = np.array(injected.expected_data([0.0] + [0.0] * (injected.npar - 1)))
    one = np.array(injected.expected_data([1.0] + [0.0] * (injected.npar - 1)))
    injected_template = (one - zero)[: injected.nbins]
    expected = np.array(
        [signal.get(channel, [0.0])[0] for channel in injected.channel_names]
    )
    assert injected_template == pytest.approx(expected, rel=1e-9, abs=1e-9)
