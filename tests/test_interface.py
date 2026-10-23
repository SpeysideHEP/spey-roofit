r"""
Tests of the ``spey`` backend, validated against the built-in ``spey``
likelihoods and against the ``spey-pyhf`` plug-in.

The toy workspaces are built so that they are mathematically identical to
``default.uncorrelated_background`` and ``default.correlated_background``,
which makes an exact number-by-number comparison possible. Two caveats apply
and are encoded in the assertions below:

* ``default.correlated_background`` writes the background constraint as a
  correlated Gaussian on :math:`\theta`, whereas the ``HistFactory`` twin uses
  the uncorrelated basis :math:`\theta=D^{-1}Lz`. The two log-likelihoods
  therefore differ by the constant :math:`\tfrac12\log\det\rho`.
* ``spey`` before 0.2.8 kept the auxiliary data of an Asimov dataset at the
  centre of the constraint, while ``RooFit`` — like ``pyhf`` and like
  :xref:`1007.1727` — sets it to the profiled :math:`\hat\theta`. From 0.2.8
  the two agree, and the assertions below require it; against an older
  ``spey`` the Asimov comparisons have to go through ``pyhf`` instead.
"""

import numpy as np
import pytest
import spey

from conftest import assert_equal_up_to_constant, likelihood_profile
from spey_roofit import RooFitInterface, WorkspaceInterpreter


@pytest.fixture(name="wrapper", scope="module")
def wrapper_fixture():
    """
    The registered ``root.roofit`` backend factory.

    Returns:
        ``Callable``:
        Factory producing :class:`~spey.StatisticalModel` instances.
    """
    assert "root.roofit" in spey.AvailableBackends()
    return spey.get_backend("root.roofit")


@pytest.fixture(name="uncorrelated", scope="module")
def uncorrelated_fixture(wrapper, uncorrelated_workspace):
    """
    RooFit and built-in models for the uncorrelated background toy.

    Returns:
        ``Tuple[~spey.StatisticalModel, ~spey.StatisticalModel]``:
        The ``RooFit`` backed model and its ``spey`` twin.
    """
    from conftest import BACKGROUND, OBSERVED, SIGNAL, UNCERTAINTY

    roofit = wrapper(workspace=uncorrelated_workspace, analysis="roofit")
    reference = spey.get_backend("default.uncorrelated_background")(
        signal_yields=SIGNAL,
        background_yields=BACKGROUND,
        data=OBSERVED,
        absolute_uncertainties=UNCERTAINTY,
        analysis="reference",
    )
    return roofit, reference


def test_backend_metadata():
    """The backend carries the metadata ``spey``'s plug-in registry needs."""
    assert RooFitInterface.name == "root.roofit"
    assert RooFitInterface.spey_requires
    assert RooFitInterface.author
    assert RooFitInterface.version


def test_config(uncorrelated, toy_yields):
    """Parameter layout, bounds and the minimum POI follow ``spey``'s rules."""
    roofit, reference = uncorrelated
    config = roofit.backend.config()
    assert config.poi_index == 0
    assert config.npar == len(toy_yields["background"]) + 1
    assert config.parameter_names[0] == "mu"
    assert config.minimum_poi == pytest.approx(reference.backend.config().minimum_poi)
    assert config.suggested_bounds[0] == (config.minimum_poi, 10.0)

    positive = roofit.backend.config(allow_negative_signal=False)
    assert positive.suggested_bounds[0] == (0.0, 10.0)


def test_is_alive(uncorrelated, wrapper, background_only_workspace):
    """A model is alive when it has a signal, and only then."""
    roofit, _ = uncorrelated
    assert roofit.is_alive

    interpreter = WorkspaceInterpreter(background_only_workspace)
    interpreter.inject_signal("ch", [0.0, 0.0, 0.0])
    assert not wrapper(workspace=interpreter, analysis="empty").is_alive


def test_logpdf_matches_spey(uncorrelated):
    """The log-likelihood is the built-in ``spey`` backend's, up to a constant."""
    roofit, reference = uncorrelated
    ours = roofit.backend.get_logpdf_func()
    theirs = reference.backend.get_logpdf_func()
    rng = np.random.default_rng(2024)
    points = [
        np.concatenate([[rng.uniform(-2.0, 3.0)], rng.normal(0.0, 0.8, 3)])
        for _ in range(10)
    ]
    assert_equal_up_to_constant(
        [ours(pars) for pars in points], [theirs(pars) for pars in points], abs=1e-9
    )


def test_likelihood_and_maximum_likelihood_match_spey(uncorrelated):
    """Observed likelihood profile and its maximum agree with the built-in backend."""
    roofit, reference = uncorrelated
    pois = (0.0, 0.5, 1.0, 2.0)
    assert likelihood_profile(roofit, pois) == pytest.approx(
        likelihood_profile(reference, pois), abs=1e-6
    )
    assert roofit.maximize_likelihood()[0] == pytest.approx(
        reference.maximize_likelihood()[0], abs=1e-3
    )


def test_expected_data_matches_spey(uncorrelated):
    r"""
    The whole Asimov dataset agrees with the built-in backend.

    Both halves have to match: the expected bin counts, and the auxiliary
    measurements, which follow the nuisance parameters the dataset was
    generated at rather than staying at the centre of the constraint. That is
    the defining property of an Asimov dataset — refitting it returns the
    parameters it was built from — and eq. (25) of :xref:`1007.1727` requires
    it. ``spey`` only satisfies it from 0.2.8 onwards.
    """
    roofit, reference = uncorrelated
    pars = [1.0, 0.2, -0.3, 0.4]
    ours = roofit.backend.expected_data(pars)
    theirs = reference.backend.expected_data(pars)
    nbins = roofit.backend.model.nbins

    assert ours[:nbins] == pytest.approx(theirs[:nbins], rel=1e-10)
    assert ours[nbins:] == pytest.approx(pars[1:])
    assert ours == pytest.approx(theirs, rel=1e-10, abs=1e-12)


def test_correlated_background_matches_spey(wrapper, correlated_workspace, toy_yields):
    r"""
    The correlated-background twin agrees up to a constant.

    The two models use different bases for the nuisance parameters, which
    changes the log-likelihood by the constant Jacobian term
    :math:`\tfrac12\log\det\rho` but leaves every likelihood ratio — and
    therefore every inference result — untouched.
    """
    covariance = np.asarray(toy_yields["covariance"], dtype=np.float64)
    factor = np.linalg.cholesky(covariance)
    scale = np.diag(np.sqrt(np.diag(covariance)))

    roofit = wrapper(workspace=correlated_workspace, analysis="roofit")
    reference = spey.get_backend("default.correlated_background")(
        signal_yields=toy_yields["signal"],
        background_yields=toy_yields["background"],
        data=toy_yields["data"],
        covariance_matrix=covariance,
        analysis="reference",
    )
    ours = roofit.backend.get_logpdf_func()
    theirs = reference.backend.get_logpdf_func()

    rng = np.random.default_rng(7)
    differences = []
    for _ in range(5):
        poi = rng.uniform(-1.0, 2.0)
        latent = rng.normal(0.0, 1.0, 3)
        theta = np.linalg.inv(scale) @ (factor @ latent)
        differences.append(
            ours(np.concatenate([[poi], latent])) - theirs(np.concatenate([[poi], theta]))
        )
    # RooFit's bin-by-bin offset adds a data-only constant on top of the
    # Jacobian term, so only the spread of the difference is checked.
    assert np.ptp(differences) == pytest.approx(0.0, abs=1e-8)

    assert roofit.maximize_likelihood()[0] == pytest.approx(
        reference.maximize_likelihood()[0], abs=1e-3
    )


def test_sampler(uncorrelated):
    """Pseudo-data have the right shape and the right mean."""
    roofit, _ = uncorrelated
    sampler = roofit.backend.get_sampler(np.array([1.0, 0.0, 0.0, 0.0]))
    model = roofit.backend.model

    main_only = sampler(64, include_auxiliary=False)
    assert main_only.shape == (64, model.nbins)

    samples = sampler(4000)
    assert samples.shape == (4000, model.nbins + model.naux)
    rates = np.array(roofit.backend.expected_data([1.0, 0.0, 0.0, 0.0]))
    assert samples[:, : model.nbins].mean(axis=0) == pytest.approx(
        rates[: model.nbins], rel=0.1
    )
    assert samples[:, model.nbins :].mean(axis=0) == pytest.approx(
        np.zeros(model.naux), abs=0.15
    )


def test_toy_calculator_is_available(uncorrelated):
    """Implementing the sampler unlocks ``spey``'s toy calculator."""
    roofit, _ = uncorrelated
    assert "toy" in roofit.available_calculators
    assert "asymptotic" in roofit.available_calculators


def test_apriori_uses_the_background_only_expectation(uncorrelated):
    """``apriori`` fits the background-only Asimov dataset."""
    roofit, _ = uncorrelated
    model = roofit.backend.model
    # pylint: disable=protected-access
    data = roofit.backend._resolve_data(spey.ExpectationType.apriori, None)
    background = model.expected_data([0.0] + [0.0] * 3)
    assert data == pytest.approx(background)


def test_signal_injection_through_the_backend(
    wrapper, background_only_workspace, toy_yields, uncorrelated
):
    """Injecting a signal reproduces the published signal-plus-background model."""
    roofit, _ = uncorrelated
    injected = wrapper(
        workspace=background_only_workspace,
        signal_yields={"ch": toy_yields["signal"]},
        analysis="injected",
    )
    ours = injected.backend.get_logpdf_func()
    theirs = roofit.backend.get_logpdf_func()
    rng = np.random.default_rng(99)
    points = [
        np.concatenate([[rng.uniform(-1.0, 2.0)], rng.normal(0.0, 0.5, 3)])
        for _ in range(5)
    ]
    assert_equal_up_to_constant([ours(p) for p in points], [theirs(p) for p in points])


def test_a_stripped_workspace_is_saved_and_reused(
    wrapper, uncorrelated_workspace, toy_yields, uncorrelated, tmp_path
):
    """Signal-plus-background becomes a background-only file that takes a new signal."""
    roofit, _ = uncorrelated
    path = str(tmp_path / "background_only.root")

    interpreter = WorkspaceInterpreter(uncorrelated_workspace)
    assert interpreter.is_alive
    assert interpreter.strip_signal().save_patch(path) == path

    saved = WorkspaceInterpreter(path)
    assert not saved.is_alive
    assert saved.poi_names == []
    assert saved.expected_background_yields["ch"] == pytest.approx(
        toy_yields["background"], rel=1e-12
    )

    reinjected = wrapper(
        workspace=saved, signal_yields={"ch": toy_yields["signal"]}, analysis="saved"
    )
    ours = reinjected.backend.get_logpdf_func()
    theirs = roofit.backend.get_logpdf_func()
    rng = np.random.default_rng(7)
    points = [
        np.concatenate([[rng.uniform(-1.0, 2.0)], rng.normal(0.0, 0.5, 3)])
        for _ in range(5)
    ]
    assert_equal_up_to_constant([ours(p) for p in points], [theirs(p) for p in points])


def test_a_patched_workspace_is_saved_with_its_signal(
    background_only_workspace, toy_yields, tmp_path
):
    """A reinterpretation is stored as it is and read back by the interpreter."""
    path = str(tmp_path / "reinterpretation.root")
    interpreter = WorkspaceInterpreter(background_only_workspace)
    interpreter.inject_signal("ch", toy_yields["signal"]).save_patch(path)

    saved = WorkspaceInterpreter(path)
    assert saved.is_alive
    assert saved.expected_signal_yields["ch"] == pytest.approx(toy_yields["signal"])


def test_agrees_with_spey_pyhf(uncorrelated, toy_yields, pyhf_backend):
    r"""
    Log-likelihood, Asimov data and exclusion agree with ``spey-pyhf``.

    The comparison goes through the ``spey-pyhf`` plug-in rather than through
    ``pyhf`` directly, so that both sides are asked the same questions by the
    same ``spey`` interface and only the likelihood implementation differs.

    ``pyhf`` implements the same ``HistFactory`` model, including the
    convention that the auxiliary data of an Asimov dataset follow the nuisance
    parameters, so the agreement is expected to be exact.
    """
    from conftest import roofit_to_pyhf_name, translate_parameters
    from toy_models import uncorrelated_background_pyhf_workspace

    assert pyhf_backend.manager.backend == "numpy"

    workspace, patch = uncorrelated_background_pyhf_workspace(
        toy_yields["background"],
        toy_yields["signal"],
        toy_yields["data"],
        toy_yields["uncertainty"],
    )
    theirs = spey.get_backend("pyhf")(
        background_only_model=workspace, signal_patch=patch, analysis="spey-pyhf"
    )
    assert theirs.backend.is_alive
    roofit, _ = uncorrelated

    our_names = roofit.backend.model.parameter_names
    their_names = theirs.backend.config().parameter_names
    # HistFactory calls a constrained nuisance parameter alpha_theta0, pyhf
    # calls it theta0.
    renamed = [roofit_to_pyhf_name(name) for name in our_names]
    assert sorted(renamed) == sorted(their_names)

    ours = roofit.backend.get_logpdf_func()
    reference = theirs.backend.get_logpdf_func()

    rng = np.random.default_rng(5)
    values = []
    for _ in range(8):
        pars = np.concatenate([[rng.uniform(-1.0, 2.0)], rng.normal(0.0, 0.6, 3)])
        translated = translate_parameters(pars, renamed, their_names)
        values.append((ours(pars), reference(translated)))

        # Both halves of the Asimov dataset, bin counts and auxiliary
        # measurements; in this model both backends order them the same way.
        assert roofit.backend.expected_data(pars) == pytest.approx(
            theirs.backend.expected_data(list(translated)), rel=1e-10, abs=1e-12
        )

    assert_equal_up_to_constant(*zip(*values), abs=1e-9)
    assert roofit.exclusion_confidence_level() == pytest.approx(
        theirs.exclusion_confidence_level(), rel=1e-4
    )
    assert roofit.poi_upper_limit() == pytest.approx(theirs.poi_upper_limit(), rel=1e-4)
