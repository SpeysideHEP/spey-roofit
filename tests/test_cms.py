r"""
Tests against the published CMS dijet angular statistical model.

The material is the one CMS released for CMS-EXO-24-011, a search for new
physics in dijet angular distributions. Every test is marked ``cms`` and skips
itself when the datacards have not been downloaded, so a fresh clone still runs
everything that does not need them:

.. code-block:: bash

    ./cms-exo-24-011/download.sh
    pytest -m cms
    pytest -m "not cms"       # skip them explicitly
"""

import numpy as np
import pytest
import spey

from conftest import cms_path
from spey_roofit import DatacardInterpreter
from spey_roofit.combine import SWAPPED_TWO_HYPOTHESIS_MODEL, TWO_HYPOTHESIS_MODEL

# pylint: disable=redefined-outer-name, protected-access

pytestmark = pytest.mark.cms

pytest.importorskip(
    "HiggsAnalysis.CombinedLimit.DatacardParser",
    reason="the CMS Combine backend needs cms-combine",
)

#: The card is a two hypothesis fit: eight bins of the dijet mass, each holding
#: the ``chi`` distribution in twelve bins, and two signal processes — QCD+BSM
#: and QCD alone — that ``x`` interpolates between.
STRUCTURE = {
    "channels": 8,
    "bins": 96,
    "systematics": 149,
    "processes": ["QCDADD12000", "QCDADD12000_ALT", "QCD"],
}


@pytest.fixture(scope="module")
def interpreter():
    """
    The reference datacard, read but not compiled.

    Returns:
        ~spey_roofit.combine.DatacardInterpreter:
        The interpreter of ``chi_datacard13TeV3_ADD12000_run2.txt``.
    """
    return DatacardInterpreter(cms_path())


@pytest.fixture(scope="module")
def model(interpreter):
    r"""
    The published model, with the physics model the analysis uses.

    The two signal processes are QCD+BSM and QCD; :math:`x=1` selects the
    first, which is the convention of the record and therefore the one that
    makes ``spey``'s background the Standard Model.

    Returns:
        ~spey.StatisticalModel:
        The compiled model.
    """
    return spey.get_backend("root.cms-combine")(
        datacard=cms_path(),
        physics_model=SWAPPED_TWO_HYPOTHESIS_MODEL,
        analysis="cms-exo-24-011",
    )


def test_the_card_describes_the_published_analysis(interpreter):
    """The structure of the card is the one the paper describes."""
    assert len(interpreter.channels) == STRUCTURE["channels"]
    assert sum(interpreter.bin_map.values()) == STRUCTURE["bins"]
    assert interpreter.processes == STRUCTURE["processes"]
    assert len(interpreter.nuisance_parameters) == STRUCTURE["systematics"]
    assert interpreter.signal_processes == ["QCDADD12000", "QCDADD12000_ALT"]


def test_the_nuisance_parameters_were_renamed(interpreter):
    r"""
    The card renames its nuisance parameters after the templates were written.

    ``nuisance edit rename jes1 CMS_scale_j_AbsoluteScale`` and its fifty-odd
    companions are appended to the card by the collaboration's own
    ``copyCards.sh``; reading them is what makes the published names visible.
    """
    names = interpreter.nuisance_parameters
    assert "CMS_scale_j_AbsoluteScale" in names
    assert "jes1" not in names
    # the templates keep the old name, which is why the card still compiles
    assert interpreter.datacard.systematicsShapeMap


def test_a_ragged_channel_is_padded(interpreter):
    r"""
    The highest mass channel is rebinned for the signals but not for the dummy.

    ``chi7000_13000`` holds five bins of data and of both signal templates, and
    twelve of the negligible ``QCD`` one. ``combine`` pads the short templates,
    and the interpreter reports the padded length.
    """
    assert interpreter.bin_map["bin7"] == 12
    observed = interpreter.observed_data["bin7"]
    assert observed[:5] == pytest.approx([14.0, 11.0, 5.0, 10.0, 10.0])
    assert observed[5:] == pytest.approx([0.0] * 7)


def test_the_two_hypotheses_share_their_normalisation(interpreter):
    r"""
    The signal of this search is a shape, not a rate.

    Both hypotheses are normalised to the same number of events in every mass
    bin, so what distinguishes them is only how those events are distributed in
    :math:`\chi`. ``spey``'s signal template therefore integrates to nothing
    while being far from zero bin by bin.
    """
    yields = interpreter.sample_yields("bin7")["bin7"]
    primary = np.asarray(yields["QCDADD12000"])
    alternative = np.asarray(yields["QCDADD12000_ALT"])
    assert float(np.sum(primary)) == pytest.approx(float(np.sum(alternative)), rel=1e-6)
    assert np.max(np.abs(primary - alternative)) > 1.0


def test_the_compiled_model_matches_the_card(model, interpreter):
    """The workspace ``combine`` builds holds what the card describes."""
    data = model.backend.model
    assert data.nbins == STRUCTURE["bins"]
    assert data.naux == STRUCTURE["systematics"]
    assert data.npar == STRUCTURE["systematics"] + 1
    assert data.poi.GetName() == "x"
    assert model.backend.workspace.pdf("model_s").InheritsFrom("RooSimultaneous")

    flat = np.concatenate(
        [np.asarray(interpreter.observed_data[channel]) for channel in data.channel_names]
    )
    assert data.observed_counts == pytest.approx(flat, rel=1e-12)


def test_the_expected_yields_match_the_card(model, interpreter):
    r"""
    At :math:`x=1` the model is the QCD+BSM hypothesis of the datacard.

    This is the check that the compilation kept the rates and the shapes of the
    templates: the workspace is asked for its expectation and compared against
    the yields read straight from the card.

    The padded bins of the highest mass channel are the one exception.
    ``combine`` floors every morphed template at :math:`10^{-9}` of its
    integral to keep the pdf positive, so a bin the card leaves empty holds a
    few times :math:`10^{-8}` events instead of nothing. The data are empty
    there as well, so the floor costs the log-likelihood the same
    :math:`10^{-8}`.
    """
    data = model.backend.model
    nominal = data.nominal_parameters.copy()
    nominal[0] = 1.0
    data.set_parameters(nominal)

    expected = []
    for channel in data.channel_names:
        samples = interpreter.sample_yields(channel)[channel]
        total = np.zeros(interpreter.bin_map[channel])
        for name, values in samples.items():
            if name != "QCDADD12000_ALT":
                total = total + np.asarray(values)
        expected.append(total)
    assert data.expected_yields() == pytest.approx(
        np.concatenate(expected), rel=1e-9, abs=1e-6
    )


def test_the_two_physics_models_mirror_each_other(interpreter):
    r"""
    ``combine``'s own two hypothesis model puts :math:`x` on the other process.

    ``HiggsJPC`` scales the process whose name carries the ``ALT`` marker,
    which for this card is QCD alone; the record states that :math:`x=1` is
    QCD+BSM, so the model this plug-in contributes is the one that reproduces
    it. The two differ by :math:`x\to1-x`, which is why the mistake is worth a
    test rather than a comment.
    """
    stock = spey.get_backend("root.cms-combine")(
        datacard=cms_path(), physics_model=TWO_HYPOTHESIS_MODEL, analysis="stock"
    )
    swapped = spey.get_backend("root.cms-combine")(
        datacard=cms_path(),
        physics_model=SWAPPED_TWO_HYPOTHESIS_MODEL,
        analysis="swapped",
    )
    _, stock_signal = stock.backend._background_and_signal()
    _, swapped_signal = swapped.backend._background_and_signal()
    assert stock_signal == pytest.approx(-np.asarray(swapped_signal), abs=1e-6)

    # the BSM hypothesis is the one that fills the low-chi bins
    swapped_background, _ = swapped.backend._background_and_signal()
    assert swapped_signal[84] > 0.0
    assert swapped_background[84] > 0.0


@pytest.mark.slow
def test_the_data_prefer_the_standard_model(model):
    r"""
    The fitted :math:`\hat{x}` is close to zero, i.e. to QCD alone.

    CMS-EXO-24-011 reports no excess, so the mixing parameter has to come out
    compatible with the Standard Model hypothesis.
    """
    fitted, _ = model.maximize_likelihood()
    assert -0.2 < fitted < 0.5


@pytest.mark.slow
def test_twelve_tev_is_excluded(model):
    r"""
    :math:`\Lambda_{\rm ADD}=12` TeV is excluded, as the paper reports.

    The published limit on the ADD scale is above this mass point, so the
    :math:`x=1` hypothesis — the full QCD+BSM prediction — has to be rejected.
    """
    assert model.exclusion_confidence_level(poi_test=1.0)[0] > 0.95


@pytest.mark.slow
def test_a_signal_can_be_injected_into_the_published_card():
    r"""
    The published card can be reinterpreted with a signal of one's own.

    Its two signal processes are stripped, which leaves the negligible ``QCD``
    dummy as the only background, and a flat excess is attached to the highest
    mass channel. What is checked is that the model is alive and that the
    injected yields survive the round trip through ``combine``.
    """
    interpreter = DatacardInterpreter(cms_path())
    yields = [4.0, 3.0, 2.0, 1.0, 1.0] + [0.0] * 7
    model = spey.get_backend("root.cms-combine")(
        datacard=interpreter,
        signal_yields={"bin7": yields},
        strip_existing_signal=True,
        analysis="reinterpreted",
    )
    assert model.backend.is_alive
    _, signal = model.backend._background_and_signal()
    assert signal[84:96] == pytest.approx(yields, rel=1e-9, abs=1e-6)
    assert np.all(np.abs(signal[:84]) < 1e-6)
