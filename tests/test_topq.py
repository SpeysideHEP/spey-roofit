r"""
Validation of ``root.hs3`` against a published analysis.

The ATLAS :math:`t\bar{t}\gamma` measurement (:xref:`2403.09452`, HEPData
``ins2768921``) publishes six likelihoods, and publishes each of them twice:
once as an HS3 document and once as a ``pyhf`` workspace. The tests here read
the HS3 side with ``root.hs3`` and the ``pyhf`` twin through the ``spey-pyhf``
plug-in, so that both are driven by the same ``spey`` interface and only the
likelihood implementation differs.

That comparison needs one caveat spelled out, because it is easy to mistake for
a bug in one of the two tools:

* the **rate model is identical** — same samples, same data, same yields to
  machine precision at the nominal nuisance parameters and for any value of the
  signal strength. Once the nuisance parameters are displaced the two drift
  apart by :math:`\mathcal{O}(10^{-5})` relative, because ``ROOT`` and ``pyhf``
  implement the ``HistFactory`` interpolation slightly differently;
* the **constraint model differs by construction**: the HS3 document asks for a
  Poisson constraint on the Monte-Carlo statistics parameters :math:`\gamma`,
  which is what ``HistFactory`` does, whereas ``pyhf`` always uses a Gaussian
  one. The two log-likelihoods therefore differ by an amount that depends on
  :math:`\gamma`, and no tolerance can paper over that.

The exact, number-by-number validation of the constraint model lives in
:mod:`test_hs3`, where both sides are known to be the same model.

Everything here is marked ``topq`` and skipped when the material has not been
downloaded with ``./atlas-topq-2018-24/download.sh``.
"""

import json

import numpy as np
import pytest
import spey

from conftest import roofit_to_pyhf_name, topq_path, translate_parameters
from spey_roofit.hs3_interface import _interpreter_class

pytestmark = pytest.mark.topq


def _require_interpreter():
    """
    Skip the module when ``spey-hs3`` is absent or too old.

    The backend needs a ``WorkspaceInterpreter`` newer than the last released
    ``spey-hs3``, so the requirement is checked against the class rather than
    against a version string.

    Returns:
        ``None``
    """
    try:
        _interpreter_class()
    except ImportError as error:
        pytest.skip(str(error), allow_module_level=True)


_require_interpreter()

#: Structure of every workspace of the record: channels, bins, parameters and
#: auxiliary measurements. The ``2NF`` variants hold one more parameter
#: constant, hence one auxiliary measurement fewer.
STRUCTURE = {
    "dilep_1NF": (1, 10, 226, 226),
    "dilep_2NF": (1, 10, 226, 225),
    "ljets_1NF": (4, 21, 249, 249),
    "ljets_2NF": (4, 21, 249, 248),
    "combined_1NF": (5, 31, 271, 271),
    "combined_2NF": (5, 31, 271, 270),
}


@pytest.fixture(name="dilep", scope="module")
def dilep_fixture():
    """
    The dilepton inclusive fit, the smallest workspace of the record.

    Returns:
        ~spey.StatisticalModel:
        The ``root.hs3`` model.
    """
    return spey.get_backend("root.hs3")(
        workspace=topq_path("dilep_1NF"), analysis="ttgamma-dilep-1NF"
    )


@pytest.fixture(name="pyhf_twin", scope="module")
def pyhf_twin_fixture(pyhf_backend):
    """
    The ``pyhf`` serialisation of the same dilepton fit, through ``spey-pyhf``.

    Going through the plug-in rather than through ``pyhf`` directly means both
    sides of the comparison are asked the same questions by the same ``spey``
    interface, and only the likelihood implementation differs.

    The signal patch is empty because this workspace already carries its signal.
    ``spey-pyhf`` judges :attr:`~spey.BackendBase.is_alive` from the patch
    rather than from the model and therefore reports ``False``; that only
    affects :func:`~spey.StatisticalModel.poi_upper_limit`, which these tests do
    not call.

    Returns:
        ~spey.StatisticalModel:
        The ``spey-pyhf`` model of ``workspace_dilep_1NF.json``.
    """
    assert pyhf_backend.manager.backend == "numpy"
    with open(topq_path("dilep_1NF", pyhf_twin=True), encoding="utf-8") as handle:
        workspace = json.load(handle)
    return spey.get_backend("pyhf")(
        background_only_model=workspace,
        signal_patch=[],
        analysis="ttgamma-dilep-1NF-pyhf",
    )


def parameter_order(ours, theirs):
    """
    Match the two backends' parameter vectors up by name.

    Args:
        ours (~spey.StatisticalModel): the ``root.hs3`` model.
        theirs (~spey.StatisticalModel): the ``spey-pyhf`` model.

    Returns:
        ``Tuple[List[str], List[str], List[float]]``:
        Our parameter names translated into ``pyhf``'s spelling, the names of
        the ``spey-pyhf`` parameter vector in its own order, and the values
        ``spey-pyhf`` suggests — which is what the parameters we do not carry,
        such as ``lumi``, have to keep.
    """
    config = theirs.backend.config()
    renamed = [roofit_to_pyhf_name(name) for name in ours.backend.model.parameter_names]
    assert set(renamed) <= set(config.parameter_names)
    return renamed, config.parameter_names, list(config.suggested_init)


# --------------------------------------------------------------------------- #
# Structure                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(STRUCTURE))
def test_every_published_workspace_compiles(name):
    """All six published likelihoods read back with the expected structure."""
    channels, nbins, npar, naux = STRUCTURE[name]
    model = spey.get_backend("root.hs3")(workspace=topq_path(name), analysis=name)
    view = model.backend.model

    assert len(view.channels) == channels
    assert view.nbins == nbins
    assert view.npar == npar
    assert view.naux == naux
    assert view.poi.GetName() == "SigXsecOverSM"
    assert sum(view.bin_map.values()) == nbins
    assert model.backend.is_alive


def test_the_combined_fit_is_a_simultaneous_model():
    r"""
    The combined likelihood keeps its five regions apart.

    A multi-channel model becomes a ``RooSimultaneous``, whose channels are
    named after the index category — ``SR_alt1`` — rather than after the HS3
    distribution that produced them, ``model_SR_alt1``. Signal injection goes
    through the HS3 interpreter and therefore uses the *distribution* names;
    the two namings are checked against each other here.
    """
    from spey_hs3.helper_functions import WorkspaceInterpreter

    from spey_roofit import load_document

    path = topq_path("combined_1NF")
    model = spey.get_backend("root.hs3")(workspace=path, analysis="combined")
    view = model.backend.model

    assert view.index_category is not None
    assert set(view.bin_map) == {
        "CR_fakes",
        "CR_photon",
        "CR_tty_dec",
        "SR",
        "SR_alt1",
    }
    assert view.bin_map["SR_alt1"] == 10

    interpreter = WorkspaceInterpreter(load_document(path))
    assert interpreter.bin_map == {
        f"model_{channel}": nbins for channel, nbins in view.bin_map.items()
    }


def test_the_interpreter_describes_the_document():
    """The region structure is readable without touching ``RooFit``."""
    from spey_hs3.helper_functions import WorkspaceInterpreter

    from spey_roofit import load_document

    interpreter = WorkspaceInterpreter(load_document(topq_path("dilep_1NF")))
    assert interpreter.bin_map == {"model_SR_alt1": 10}
    assert interpreter.poi_parameters == ["SigXsecOverSM"]
    assert len(interpreter.signal_samples["model_SR_alt1"]) == 2
    assert len(interpreter.observed_data["model_SR_alt1"]) == 10


# --------------------------------------------------------------------------- #
# Against the pyhf twin                                                       #
# --------------------------------------------------------------------------- #


def test_expected_yields_match_the_pyhf_twin(dilep, pyhf_twin):
    r"""
    The rate model is the same in both serialisations.

    At the nominal nuisance parameters, and for any value of :math:`\mu`, the
    expected yields have to agree to machine precision: the two files describe
    the same samples, the same data and the same binning. The yields are read
    out of each backend's Asimov dataset, whose first entries are exactly
    :math:`\lambda_i(\mu,\theta)`.
    """
    renamed, their_names, their_init = parameter_order(dilep, pyhf_twin)
    view = dilep.backend.model
    nominal = np.asarray(view.nominal_parameters, dtype=float)
    nbins = view.nbins

    for poi in (0.0, 1.0, 2.0):
        pars = nominal.copy()
        pars[0] = poi
        translated = translate_parameters(pars, renamed, their_names, their_init)

        ours = np.asarray(dilep.backend.expected_data(list(pars)))[:nbins]
        theirs = np.asarray(pyhf_twin.backend.expected_data(list(translated)))[:nbins]
        assert ours == pytest.approx(theirs, rel=1e-12)

    view.set_parameters(nominal)


def test_the_two_serialisations_use_different_statistics_constraints(dilep, pyhf_twin):
    r"""
    The :math:`\gamma` constraint differs, and it is the *only* thing that does.

    With the Monte-Carlo statistics parameters held at one, the offset between
    the two log-likelihoods is constant to better than a tenth of a log unit —
    it is a normalisation. Letting the :math:`\gamma` float makes the offset
    move by two orders of magnitude more, which is the Poisson-versus-Gaussian
    constraint showing up. Asserting this keeps the difference documented
    rather than mistaken for a defect.
    """
    renamed, their_names, their_init = parameter_order(dilep, pyhf_twin)
    view = dilep.backend.model
    names = view.parameter_names
    nominal = np.asarray(view.nominal_parameters, dtype=float)
    ours = dilep.backend.get_logpdf_func()
    theirs = pyhf_twin.backend.get_logpdf_func()
    is_gamma = np.array([name.startswith("gamma_") for name in names])

    def offset(pars):
        """Difference between the two log-likelihoods at one parameter point."""
        translated = translate_parameters(pars, renamed, their_names, their_init)
        return ours(pars) - theirs(translated)

    def spread(gamma_scale):
        """Range of that difference over a handful of parameter points."""
        rng = np.random.default_rng(3)
        values = []
        for _ in range(6):
            pars = nominal.copy()
            pars[0] += rng.uniform(-0.3, 0.3)
            step = np.where(
                is_gamma,
                rng.normal(0.0, gamma_scale, pars.size),
                rng.normal(0.0, 0.6, pars.size),
            )
            step[0] = 0.0
            values.append(offset(pars + step))
        return max(values) - min(values)

    fixed_gamma = spread(0.0)
    floating_gamma = spread(0.03)

    assert fixed_gamma < 0.1
    assert floating_gamma > 10.0 * fixed_gamma
    view.set_parameters(nominal)


# --------------------------------------------------------------------------- #
# Inference                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_recovers_the_published_signal_strength(dilep):
    r"""
    The fit reproduces the measurement stored in the workspace.

    The workspace was published after ATLAS fitted it, so :math:`\hat\mu` has
    to come back close to the measured value; the tolerance here is loose
    enough to survive a different minimiser but tight enough to catch a model
    that has been read wrongly.
    """
    muhat, nll = dilep.maximize_likelihood()
    assert muhat == pytest.approx(1.1099, abs=5e-3)
    assert np.isfinite(nll)
    assert 0.0 < dilep.sigma_mu(muhat) < 1.0


@pytest.mark.slow
def test_reinterpretation_replaces_the_published_signal():
    """The published signal can be stripped and a different one injected."""
    my_signal = [4.0, 12.0, 15.0, 12.0, 9.0, 6.0, 4.0, 3.0, 2.0, 1.0]
    model = spey.get_backend("root.hs3")(
        workspace=topq_path("dilep_1NF"),
        signal_yields={"model_SR_alt1": my_signal},
        strip_existing_signal=True,
        poi_name="mu",
        analysis="reinterpreted",
    )
    assert model.backend.model.poi.GetName() == "mu"

    background, signal = model.backend._background_and_signal()
    assert signal == pytest.approx(my_signal, rel=1e-8)
    assert np.all(background > 0.0)

    confidence_level = model.exclusion_confidence_level()[0]
    assert 0.0 <= confidence_level <= 1.0
