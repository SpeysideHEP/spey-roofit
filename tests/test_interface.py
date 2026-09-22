r"""
Tests of the ``spey`` backend, validated against the built-in ``spey``
likelihoods and against ``pyhf``.

The toy workspaces are built so that they are mathematically identical to
``default.uncorrelated_background`` and ``default.correlated_background``,
which makes an exact number-by-number comparison possible. Two caveats apply
and are encoded in the assertions below:

* ``default.correlated_background`` writes the background constraint as a
  correlated Gaussian on :math:`\theta`, whereas the ``HistFactory`` twin uses
  the uncorrelated basis :math:`\theta=D^{-1}Lz`. The two log-likelihoods
  therefore differ by the constant :math:`\tfrac12\log\det\rho`.
* the built-in ``spey`` backends keep the auxiliary data of an Asimov dataset
  at the centre of the constraint, while ``RooFit`` — like ``pyhf`` and like
  :xref:`1007.1727` — sets it to the profiled :math:`\hat\theta`. Asimov
  quantities are consequently validated against ``pyhf`` rather than against
  the built-in backends.
"""

import numpy as np
import pytest
import spey

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
    """The log-likelihood is identical to the built-in ``spey`` backend."""
    roofit, reference = uncorrelated
    ours = roofit.backend.get_logpdf_func()
    theirs = reference.backend.get_logpdf_func()
    rng = np.random.default_rng(2024)
    for _ in range(10):
        pars = np.concatenate([[rng.uniform(-2.0, 3.0)], rng.normal(0.0, 0.8, 3)])
        assert ours(pars) == pytest.approx(theirs(pars), rel=1e-10, abs=1e-10)


def test_likelihood_and_maximum_likelihood_match_spey(uncorrelated):
    """Observed likelihood and its maximum agree with the built-in backend."""
    roofit, reference = uncorrelated
    for poi_test in (0.0, 0.5, 1.0, 2.0):
        assert roofit.likelihood(poi_test) == pytest.approx(
            reference.likelihood(poi_test), rel=1e-6
        )
    ours = roofit.maximize_likelihood()
    theirs = reference.maximize_likelihood()
    assert ours[0] == pytest.approx(theirs[0], abs=1e-3)
    assert ours[1] == pytest.approx(theirs[1], rel=1e-6)


def test_expected_data_matches_spey(uncorrelated):
    """The main part of the Asimov data agrees with the built-in backend."""
    roofit, reference = uncorrelated
    pars = [1.0, 0.2, -0.3, 0.4]
    ours = roofit.backend.expected_data(pars)
    theirs = reference.backend.expected_data(pars)
    nbins = roofit.backend.model.nbins
    assert ours[:nbins] == pytest.approx(theirs[:nbins], rel=1e-10)
    # spey keeps the auxiliary data at the centre of the constraint, RooFit
    # moves it to the nuisance parameter values, as pyhf and RooStats do.
    assert ours[nbins:] == pytest.approx(pars[1:])


def test_correlated_background_matches_spey(wrapper, correlated_workspace, toy_yields):
    r"""
    The correlated-background twin agrees up to :math:`\tfrac12\log\det\rho`.

    The two models use different bases for the nuisance parameters, which
    changes the log-likelihood by a constant Jacobian term but leaves every
    likelihood ratio — and therefore every inference result — untouched.
    """
    covariance = np.asarray(toy_yields["covariance"], dtype=np.float64)
    factor = np.linalg.cholesky(covariance)
    scale = np.diag(np.sqrt(np.diag(covariance)))
    correlation = np.linalg.inv(scale) @ covariance @ np.linalg.inv(scale)
    constant = 0.5 * np.log(np.linalg.det(correlation))

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
    for _ in range(5):
        poi = rng.uniform(-1.0, 2.0)
        latent = rng.normal(0.0, 1.0, 3)
        theta = np.linalg.inv(scale) @ (factor @ latent)
        assert ours(np.concatenate([[poi], latent])) - theirs(
            np.concatenate([[poi], theta])
        ) == pytest.approx(constant, abs=1e-8)

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
    for _ in range(5):
        pars = np.concatenate([[rng.uniform(-1.0, 2.0)], rng.normal(0.0, 0.5, 3)])
        assert ours(pars) == pytest.approx(theirs(pars), rel=1e-10, abs=1e-10)


def test_agrees_with_pyhf(uncorrelated, toy_yields):
    """
    Log-likelihood, Asimov data and exclusion agree with ``pyhf``.

    ``pyhf`` implements the same ``HistFactory`` model, including the
    convention that the auxiliary data of an Asimov dataset follow the nuisance
    parameters, so the agreement here is expected to be exact.
    """
    pyhf = pytest.importorskip("pyhf")
    pyhf.set_backend("numpy", "scipy")

    channels, observations = [], []
    zipped = zip(
        toy_yields["background"],
        toy_yields["signal"],
        toy_yields["uncertainty"],
        toy_yields["data"],
    )
    for index, (background, signal, uncertainty, observed) in enumerate(zipped):
        channels.append(
            {
                "name": f"b{index}",
                "samples": [
                    {
                        "name": "bkg",
                        "data": [background],
                        "modifiers": [
                            {
                                "name": f"theta{index}",
                                "type": "histosys",
                                "data": {
                                    "hi_data": [background + uncertainty],
                                    "lo_data": [background - uncertainty],
                                },
                            }
                        ],
                    },
                    {
                        "name": "sig",
                        "data": [signal],
                        "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
                    },
                ],
            }
        )
        observations.append({"name": f"b{index}", "data": [observed]})
    workspace = pyhf.Workspace(
        {
            "channels": channels,
            "observations": observations,
            "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": []}}],
            "version": "1.0.0",
        }
    )
    model = workspace.model()
    data = workspace.data(model)

    roofit, _ = uncorrelated
    ours = roofit.backend.get_logpdf_func()
    order = model.config.par_order
    index = [order.index(name) for name in ("mu", "theta0", "theta1", "theta2")]

    rng = np.random.default_rng(5)
    for _ in range(8):
        pars = np.concatenate([[rng.uniform(-1.0, 2.0)], rng.normal(0.0, 0.6, 3)])
        translated = np.zeros(model.config.npars)
        for position, target in enumerate(index):
            translated[target] = pars[position]
        assert ours(pars) == pytest.approx(
            float(model.logpdf(translated, data)[0]), rel=1e-10, abs=1e-10
        )
        # pyhf orders the auxiliary data by `auxdata_order`, spey-roofit sorts
        # the global observables by name; both give theta0, theta1, theta2 here.
        assert model.config.auxdata_order == ["theta0", "theta1", "theta2"]
        assert roofit.backend.expected_data(pars) == pytest.approx(
            np.asarray(model.expected_data(translated)).tolist(), rel=1e-10
        )

    observed_cls = pyhf.infer.hypotest(1.0, data, model, test_stat="qtilde")
    assert roofit.exclusion_confidence_level()[0] == pytest.approx(
        1.0 - float(observed_cls), rel=1e-4
    )
