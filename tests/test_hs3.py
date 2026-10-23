r"""
Tests of the ``root.hs3`` backend and of the HS3 compilation layer.

The toy documents of :mod:`hs3_models` are written so that they are
mathematically identical to ``default.uncorrelated_background``,
``default.correlated_background`` and ``default.poisson``, which makes an exact
number-by-number comparison possible. The same two caveats as in
:mod:`test_interface` apply:

* ``default.correlated_background`` writes the background constraint as a
  correlated Gaussian on :math:`\theta`, whereas the ``HistFactory`` twin uses
  the uncorrelated basis :math:`\theta=D^{-1}Lz`. The two log-likelihoods
  therefore differ by the constant :math:`\tfrac12\log\det\rho`.
* ``spey`` before 0.2.8 kept the auxiliary data of an Asimov dataset at the
  centre of the constraint, while ``RooFit`` — like ``pyhf`` and like
  :xref:`1007.1727` — sets it to the profiled :math:`\hat\theta`. From 0.2.8
  the two agree and the Asimov assertions below require it.

The strongest statement the suite makes is
:func:`test_matches_the_roofit_backend`: the very same model, read once from a
``.root`` file by ``root.roofit`` and once from its HS3 serialisation by
``root.hs3``, has to give bit-for-bit the same log-likelihood.
"""

import copy
import json

import numpy as np
import pytest
import spey

from conftest import assert_equal_up_to_constant, likelihood_profile
from spey_roofit import HS3Interface, load_document, repair_document
from spey_roofit.hs3 import (
    compile_workspace,
    ensure_default_domain,
    ensure_parameter,
    resolve_model_config,
)
from spey_roofit.hs3_interface import (
    _interpreter_class,
    normalise_signal_yields,
)


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


@pytest.fixture(name="wrapper", scope="module")
def wrapper_fixture():
    """
    The ``root.hs3`` backend as ``spey`` exposes it.

    Returns:
        ``Callable``:
        The statistical-model factory.
    """
    return spey.get_backend("root.hs3")


@pytest.fixture(name="uncorrelated", scope="module")
def uncorrelated_fixture(wrapper, uncorrelated_hs3, toy_yields):
    """
    The HS3 model and its built-in ``spey`` twin.

    Returns:
        ``Tuple[~spey.StatisticalModel, ~spey.StatisticalModel]``:
        The ``root.hs3`` model and ``default.uncorrelated_background``.
    """
    ours = wrapper(workspace=uncorrelated_hs3, analysis="hs3")
    theirs = spey.get_backend("default.uncorrelated_background")(
        signal_yields=toy_yields["signal"],
        background_yields=toy_yields["background"],
        data=toy_yields["data"],
        absolute_uncertainties=toy_yields["uncertainty"],
        analysis="reference",
    )
    return ours, theirs


# --------------------------------------------------------------------------- #
# Registration                                                                #
# --------------------------------------------------------------------------- #


def test_backend_is_registered():
    """``root.hs3`` is discoverable through ``spey``."""
    assert "root.hs3" in spey.AvailableBackends()
    assert HS3Interface.name == "root.hs3"
    assert HS3Interface.spey_requires


# --------------------------------------------------------------------------- #
# Document handling                                                           #
# --------------------------------------------------------------------------- #


def test_load_document_accepts_every_spelling(tmp_path, uncorrelated_hs3):
    """A mapping, a path and a JSON string all give the same document."""
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps(uncorrelated_hs3), encoding="utf-8")

    assert load_document(uncorrelated_hs3) is uncorrelated_hs3
    assert load_document(str(path)) == uncorrelated_hs3
    assert load_document(path) == uncorrelated_hs3
    assert load_document(json.dumps(uncorrelated_hs3)) == uncorrelated_hs3


def test_load_document_rejects_nonsense():
    """Unreadable input is reported rather than passed on to ``ROOT``."""
    with pytest.raises(TypeError):
        load_document(42)
    with pytest.raises(ValueError, match="not an existing file"):
        load_document("no-such-workspace.json")
    with pytest.raises(ValueError, match="Could not parse"):
        load_document('{"distributions": [,]}')


def test_repair_restores_a_missing_constraint_parameter(uncorrelated_hs3):
    r"""
    A constraint without its ``x`` key is repaired from the naming convention.

    ``ROOT`` 6.36 writes ``HistFactory`` constraints this way, and its own
    reader then rejects them.
    """
    document = copy.deepcopy(uncorrelated_hs3)
    document["distributions"].append(
        {
            "type": "gaussian_dist",
            "name": "theta0Constraint",
            "mean": "nom_theta0",
            "sigma": 1.0,
        }
    )
    repaired, names = repair_document(document)
    assert names == ["theta0Constraint"]
    assert repaired["distributions"][-1]["x"] == "theta0"
    # The input is left untouched.
    assert "x" not in document["distributions"][-1]


def test_repair_reports_an_unidentifiable_constraint(uncorrelated_hs3):
    """A constraint that cannot be resolved raises instead of guessing."""
    document = copy.deepcopy(uncorrelated_hs3)
    document["distributions"].append(
        {"type": "gaussian_dist", "name": "mystery", "mean": "somewhere", "sigma": 1.0}
    )
    with pytest.raises(ValueError, match="does not declare the parameter"):
        repair_document(document)


def test_repair_is_idempotent(uncorrelated_hs3):
    """A document that needs nothing is returned unchanged."""
    repaired, names = repair_document(uncorrelated_hs3)
    assert names == []
    assert repaired == uncorrelated_hs3


def test_default_domain_is_added_when_missing(uncorrelated_hs3):
    r"""
    The fallback domain the ``ROOT`` reader insists on is added if absent.

    Without it the reader aborts the process with an uncatchable C++
    exception, so this is a repair rather than a convenience.
    """
    document = copy.deepcopy(uncorrelated_hs3)
    document["domains"][0]["name"] = "something_else"
    for analysis in document["analyses"]:
        analysis["domains"] = ["something_else"]

    assert ensure_default_domain(document) is True
    names = [domain["name"] for domain in document["domains"]]
    assert "default_domain" in names
    # The axes of the original domain are carried over unchanged.
    fallback = next(d for d in document["domains"] if d["name"] == "default_domain")
    assert fallback["axes"] == document["domains"][0]["axes"]
    # Running it again changes nothing.
    assert ensure_default_domain(document) is False


def test_ensure_parameter_declares_value_and_range():
    """An undeclared parameter is given a value and a domain."""
    document = {}
    assert ensure_parameter(document, "mu", value=2.0, bounds=(-3.0, 4.0)) is True
    axis = document["domains"][0]["axes"][0]
    assert (axis["name"], axis["min"], axis["max"]) == ("mu", -3.0, 4.0)
    assert document["parameter_points"][0]["parameters"][0] == {
        "name": "mu",
        "value": 2.0,
    }
    assert ensure_parameter(document, "mu") is False


def test_normalise_signal_yields():
    """Both spellings of ``signal_yields`` end up in the nested form."""
    assert normalise_signal_yields({"model_a": [1.0, 2.0]}, "sig") == {
        "model_a": {"sig": [1.0, 2.0]}
    }
    assert normalise_signal_yields({"model_a": {"contents": [1.0]}}, "sig") == {
        "model_a": {"sig": {"contents": [1.0]}}
    }
    assert normalise_signal_yields({"model_a": {"mine": [1.0]}}) == {
        "model_a": {"mine": [1.0]}
    }
    with pytest.raises(ValueError, match="have to be a sequence"):
        normalise_signal_yields({"model_a": 3.0})


# --------------------------------------------------------------------------- #
# Compilation                                                                 #
# --------------------------------------------------------------------------- #


def test_compile_workspace_builds_a_usable_model(uncorrelated_hs3):
    """The compiled workspace carries a pdf, data and a parameter of interest."""
    workspace, model_config = compile_workspace(uncorrelated_hs3)
    assert model_config.GetPdf()
    assert [par.GetName() for par in model_config.GetParametersOfInterest()] == ["mu"]
    assert workspace.data("obsData_ch")


def test_compile_workspace_completes_the_global_observables(uncorrelated_hs3):
    r"""
    The auxiliary measurements the reader leaves out are recovered.

    ``RooJSONFactoryWSTool`` creates the constraint terms of a
    ``histfactory_dist`` on the fly but does not declare their global
    observables, so without this step the data vector would be missing its
    whole auxiliary half.
    """
    workspace, model_config = compile_workspace(uncorrelated_hs3, complete=False)
    assert len(model_config.GetGlobalObservables() or []) == 0

    workspace, model_config = compile_workspace(uncorrelated_hs3)
    assert sorted(par.GetName() for par in model_config.GetGlobalObservables()) == [
        "nom_theta0",
        "nom_theta1",
        "nom_theta2",
    ]
    assert workspace is not None


def test_resolve_model_config_reports_what_is_available(uncorrelated_hs3):
    """Asking for a model configuration that does not exist is an error."""
    workspace, _ = compile_workspace(uncorrelated_hs3)
    with pytest.raises(ValueError, match="does not contain a ModelConfig"):
        resolve_model_config(workspace, "NotThere")


# --------------------------------------------------------------------------- #
# The likelihood                                                              #
# --------------------------------------------------------------------------- #


def test_structure(uncorrelated):
    """Channels, bins and parameters are read off the document correctly."""
    ours, _ = uncorrelated
    model = ours.backend.model
    assert model.bin_map == {"model_ch": 3}
    assert model.nbins == 3
    assert model.naux == 3
    assert model.parameter_names == ["mu", "theta0", "theta1", "theta2"]
    assert ours.backend.is_alive


def test_logpdf_matches_spey(uncorrelated):
    r"""
    :math:`\log\mathcal{L}(\mu,\theta)` agrees with the built-in backend.

    The comparison is made point by point at fixed parameters, which isolates
    the structure of the likelihood from any convention about which dataset it
    is evaluated on.
    """
    ours, theirs = uncorrelated
    mine, reference = ours.backend.get_logpdf_func(), theirs.backend.get_logpdf_func()

    rng = np.random.default_rng(7)
    points = [
        np.concatenate([[rng.uniform(-1.0, 3.0)], rng.normal(0.0, 1.2, 3)])
        for _ in range(10)
    ]
    assert_equal_up_to_constant([mine(p) for p in points], [reference(p) for p in points])


def test_maximum_likelihood_matches_spey(uncorrelated):
    """The fitted signal strength and its likelihood agree with ``spey``."""
    ours, theirs = uncorrelated
    mine, reference = ours.maximize_likelihood(), theirs.maximize_likelihood()
    assert mine[0] == pytest.approx(reference[0], abs=1e-3)
    pois = (0.0, 0.5, 1.0, 2.0)
    assert likelihood_profile(ours, pois) == pytest.approx(
        likelihood_profile(theirs, pois), abs=1e-6
    )


def test_chi2_matches_spey(uncorrelated):
    """The test statistic agrees with the built-in backend."""
    ours, theirs = uncorrelated
    assert ours.chi2(poi_test=1.0) == pytest.approx(theirs.chi2(poi_test=1.0), rel=1e-6)


def test_exclusion_matches_spey(uncorrelated):
    """Observed and expected exclusion agree with the built-in backend."""
    ours, theirs = uncorrelated
    assert ours.exclusion_confidence_level() == pytest.approx(
        theirs.exclusion_confidence_level(), abs=1e-6
    )
    apriori = spey.ExpectationType.apriori
    assert ours.exclusion_confidence_level(expected=apriori) == pytest.approx(
        theirs.exclusion_confidence_level(expected=apriori), abs=1e-4
    )


def test_asimov_data_matches_spey(uncorrelated):
    r"""
    The whole Asimov dataset agrees with the built-in backend.

    Both halves have to match: the expected bin counts, and the auxiliary
    measurements, which follow the nuisance parameters the dataset was
    generated at. That is the defining property of an Asimov dataset — eq. (25)
    of :xref:`1007.1727` — and ``spey`` satisfies it from 0.2.8 onwards.
    """
    ours, theirs = uncorrelated
    pars = [1.0, 0.3, -0.7, 0.2]
    mine = ours.backend.expected_data(pars)
    reference = theirs.backend.expected_data(pars)
    nbins = ours.backend.model.nbins

    assert mine[nbins:] == pytest.approx(pars[1:])
    assert mine == pytest.approx(reference, rel=1e-10, abs=1e-12)


def test_upper_limit_matches_spey(uncorrelated):
    """The 95% CL upper limit on the signal strength agrees with ``spey``."""
    ours, theirs = uncorrelated
    assert ours.poi_upper_limit() == pytest.approx(theirs.poi_upper_limit(), rel=1e-4)


def test_poisson_document_matches_spey(wrapper, poisson_hs3, toy_yields):
    """A document without constraints reproduces ``default.poisson``."""
    ours = wrapper(workspace=poisson_hs3, analysis="hs3")
    theirs = spey.get_backend("default.poisson")(
        signal_yields=toy_yields["signal"],
        background_yields=toy_yields["background"],
        data=[int(value) for value in toy_yields["data"]],
        analysis="reference",
    )
    assert ours.backend.model.naux == 0
    assert ours.backend.model.npar == 1
    pois = (0.0, 0.5, 1.0, 2.0)
    assert likelihood_profile(ours, pois) == pytest.approx(
        likelihood_profile(theirs, pois), abs=1e-9
    )
    assert ours.exclusion_confidence_level() == pytest.approx(
        theirs.exclusion_confidence_level(), rel=1e-6
    )


def test_correlated_document_matches_spey(wrapper, correlated_hs3, toy_yields):
    r"""
    The correlated twin agrees up to a constant.

    The two models use different bases for the nuisance parameters, which
    changes the log-likelihood by a constant Jacobian term, :math:`\tfrac12\log
    \det\rho`, but leaves every likelihood ratio — and therefore every inference
    result — untouched. ``RooFit``'s own additive terms are constant as well.
    """
    covariance = np.asarray(toy_yields["covariance"], dtype=np.float64)
    factor = np.linalg.cholesky(covariance)
    scale = np.diag(np.sqrt(np.diag(covariance)))

    ours = wrapper(workspace=correlated_hs3, analysis="hs3")
    theirs = spey.get_backend("default.correlated_background")(
        signal_yields=toy_yields["signal"],
        background_yields=toy_yields["background"],
        data=toy_yields["data"],
        covariance_matrix=covariance,
        analysis="reference",
    )
    mine, reference = ours.backend.get_logpdf_func(), theirs.backend.get_logpdf_func()

    rng = np.random.default_rng(7)
    values, expected = [], []
    for _ in range(5):
        poi = rng.uniform(-1.0, 2.0)
        latent = rng.normal(0.0, 1.0, 3)
        theta = np.linalg.inv(scale) @ (factor @ latent)
        values.append(mine(np.concatenate([[poi], latent])))
        expected.append(reference(np.concatenate([[poi], theta])))
    assert_equal_up_to_constant(values, expected, abs=1e-8)

    assert ours.maximize_likelihood()[0] == pytest.approx(
        theirs.maximize_likelihood()[0], abs=1e-3
    )


# --------------------------------------------------------------------------- #
# Cross-check against the RooFit backend                                      #
# --------------------------------------------------------------------------- #


def test_matches_the_roofit_backend(wrapper, uncorrelated_workspace, tmp_path):
    r"""
    The same model read from ``.root`` and from HS3 gives the same likelihood.

    The workspace is exported to HS3 by ``ROOT`` itself and read back by
    ``root.hs3``; since both backends end up evaluating a ``RooFit`` model
    built from the same description, the agreement has to be exact rather than
    merely close.
    """
    from spey_roofit.helper_functions import load_workspace
    from spey_roofit.manager import manager

    workspace = load_workspace(uncorrelated_workspace)
    document = json.loads(
        manager.ROOT.RooJSONFactoryWSTool(workspace).exportJSONtoString()
    )
    path = tmp_path / "exported.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    from_root = spey.get_backend("root.roofit")(
        workspace=uncorrelated_workspace, analysis="roofit"
    )
    from_hs3 = wrapper(workspace=str(path), analysis="hs3")

    assert (
        from_hs3.backend.model.parameter_names == from_root.backend.model.parameter_names
    )

    mine = from_hs3.backend.get_logpdf_func()
    reference = from_root.backend.get_logpdf_func()
    rng = np.random.default_rng(1)
    for _ in range(6):
        pars = np.concatenate([[rng.uniform(-0.5, 2.0)], rng.normal(0.0, 0.8, 3)])
        assert mine(pars) == reference(pars)

    assert from_hs3.exclusion_confidence_level() == pytest.approx(
        from_root.exclusion_confidence_level(), rel=1e-10
    )


# --------------------------------------------------------------------------- #
# Signal injection                                                            #
# --------------------------------------------------------------------------- #


def test_injects_a_signal_into_a_background_only_document(
    wrapper, background_only_hs3, uncorrelated, toy_yields
):
    """A background-only document plus yields reproduces the full model."""
    injected = wrapper(
        workspace=background_only_hs3,
        signal_yields={"model_ch": toy_yields["signal"]},
        poi_name="mu",
        analysis="injected",
    )
    assert injected.backend.model.poi.GetName() == "mu"
    assert injected.backend.is_alive

    reference, _ = uncorrelated
    mine = injected.backend.get_logpdf_func()
    theirs = reference.backend.get_logpdf_func()
    rng = np.random.default_rng(3)
    for _ in range(5):
        pars = np.concatenate([[rng.uniform(-0.5, 2.0)], rng.normal(0.0, 0.8, 3)])
        assert mine(pars) == pytest.approx(theirs(pars), rel=1e-10, abs=1e-10)


def test_background_only_document_without_signal_is_rejected(
    wrapper, background_only_hs3
):
    """Without a signal there is nothing to set a limit on, and it is said so."""
    with pytest.raises(ValueError, match="parameter of interest"):
        wrapper(workspace=background_only_hs3, analysis="nothing")


def test_strip_existing_signal_allows_reinterpretation(
    wrapper, uncorrelated_hs3, toy_yields
):
    """The published signal can be replaced by a different one."""
    replacement = [2.0 * value for value in toy_yields["signal"]]
    model = wrapper(
        workspace=uncorrelated_hs3,
        signal_yields={"model_ch": replacement},
        strip_existing_signal=True,
        analysis="reinterpreted",
    )
    background, signal = model.backend._background_and_signal()
    assert signal == pytest.approx(replacement, rel=1e-8)
    assert background == pytest.approx(toy_yields["background"], rel=1e-8)


def test_stripping_the_signal_of_a_document_without_constraint_keys(
    wrapper, uncorrelated_hs3
):
    """
    Parameters that only the signal uses are not pruned from under their constraint.

    A constraint without its ``x`` key, as ``ROOT`` 6.36 writes them, does not
    refer to its parameter as far as the interpreter can tell. Stripping the
    signal would then prune the parameter and leave the constraint pointing at
    nothing, so the document has to be repaired before the interpreter sees it.
    """
    document = copy.deepcopy(uncorrelated_hs3)
    document["distributions"][0]["samples"][1]["modifiers"].append(
        {
            "type": "normsys",
            "name": "alpha_sig",
            "parameter": "alpha_sig",
            "constraint": "Gauss",
            "data": {"hi": 1.1, "lo": 0.9},
        }
    )
    document["distributions"].append(
        {
            "type": "gaussian_dist",
            "name": "alpha_sigConstraint",
            "mean": "nom_alpha_sig",
            "sigma": 1.0,
        }
    )
    document["parameter_points"][0]["parameters"] += [
        {"name": "alpha_sig", "value": 0.0},
        {"name": "nom_alpha_sig", "value": 0.0},
    ]
    document["domains"][0]["axes"] += [
        {"name": "alpha_sig", "min": -5.0, "max": 5.0},
        {"name": "nom_alpha_sig", "min": -5.0, "max": 5.0},
    ]

    model = wrapper(
        workspace=document,
        signal_yields={"model_ch": [1.0, 2.0, 3.0]},
        strip_existing_signal=True,
        analysis="no-constraint-keys",
    )
    assert model.backend.document["distributions"][-1]["x"] == "alpha_sig"
    assert model.backend.is_alive


def test_signal_yields_with_a_wrong_bin_count_is_rejected(wrapper, background_only_hs3):
    """The interpreter checks the shape of the yields against the binning."""
    with pytest.raises(ValueError, match="[Bb]in count"):
        wrapper(
            workspace=background_only_hs3,
            signal_yields={"model_ch": [1.0, 2.0]},
            poi_name="mu",
            analysis="wrong-shape",
        )


def test_interpreter_and_document_are_exposed(wrapper, uncorrelated_hs3):
    """The backend keeps the interpreter, the document and the workspace."""
    model = wrapper(workspace=uncorrelated_hs3, analysis="hs3")
    assert model.backend.interpreter.bin_map == {"model_ch": 3}
    assert "distributions" in model.backend.document
    assert model.backend.workspace.pdf("model_ch")


# --------------------------------------------------------------------------- #
# Sampling and configuration                                                  #
# --------------------------------------------------------------------------- #


def test_sampler(uncorrelated):
    """Pseudo-data have the right shape and the right mean."""
    ours, _ = uncorrelated
    model = ours.backend.model
    sampler = ours.backend.get_sampler(np.array([1.0, 0.0, 0.0, 0.0]))

    assert sampler(32, include_auxiliary=False).shape == (32, model.nbins)

    samples = sampler(4000)
    assert samples.shape == (4000, model.nbins + model.naux)
    rates = np.array(ours.backend.expected_data([1.0, 0.0, 0.0, 0.0]))
    assert samples[:, : model.nbins].mean(axis=0) == pytest.approx(
        rates[: model.nbins], rel=0.1
    )
    assert samples[:, model.nbins :].mean(axis=0) == pytest.approx(
        np.zeros(model.naux), abs=0.15
    )


def test_config_respects_the_requested_bounds(uncorrelated):
    """The parameter of interest is given the range ``spey`` asks for."""
    ours, _ = uncorrelated
    config = ours.backend.config(allow_negative_signal=False, poi_upper_bound=7.0)
    assert config.poi_index == 0
    assert config.suggested_bounds[0] == (0.0, 7.0)
    assert config.parameter_names == ["mu", "theta0", "theta1", "theta2"]
    assert ours.backend.model.poi.getMax() == pytest.approx(7.0)


def test_both_calculators_are_available(uncorrelated):
    """Implementing the sampler unlocks ``spey``'s toy calculator."""
    ours, _ = uncorrelated
    assert {"toy", "asymptotic"} <= set(ours.available_calculators)


def test_a_renamed_domain_still_compiles(uncorrelated_hs3):
    r"""
    A document whose domain is not called ``default_domain`` is still readable.

    The fallback domain is added whatever ``repair`` says, because the reader
    does not raise when it is missing — it aborts the process. A regression
    here would therefore take the whole test session down, which is the loudest
    possible failure and exactly what is wanted.
    """
    document = copy.deepcopy(uncorrelated_hs3)
    document["domains"][0]["name"] = "my_domain"
    for analysis in document["analyses"]:
        analysis["domains"] = ["my_domain"]

    _, model_config = compile_workspace(document, repair=False)
    assert [par.GetName() for par in model_config.GetParametersOfInterest()] == ["mu"]
    # The caller's document is left alone.
    assert [domain["name"] for domain in document["domains"]] == ["my_domain"]


def test_the_backend_does_not_modify_the_document_it_is_given(uncorrelated_hs3):
    """Building a model twice from the same mapping gives the same model."""
    original = copy.deepcopy(uncorrelated_hs3)
    model = spey.get_backend("root.hs3")(workspace=uncorrelated_hs3, analysis="first")
    assert uncorrelated_hs3 == original
    # The backend keeps its own copy, with the fallback domain added.
    assert model.backend.document is not uncorrelated_hs3
