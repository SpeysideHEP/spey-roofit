"""Tests for the flat ``RooFit`` model view."""

import numpy as np
import pytest

from spey_roofit import load_workspace
from spey_roofit.data import WorkspaceData


@pytest.fixture(name="model", scope="module")
def model_fixture(uncorrelated_workspace):
    """
    A :class:`~spey_roofit.data.WorkspaceData` built from the toy workspace.

    Returns:
        ~spey_roofit.data.WorkspaceData:
        Model view of the uncorrelated-background toy.
    """
    return WorkspaceData(load_workspace(uncorrelated_workspace))


def test_layout(model, toy_yields):
    """Channels, bins, parameters and auxiliary data have the expected shapes."""
    assert model.channel_names == ["ch"]
    assert model.nbins == len(toy_yields["background"])
    assert model.naux == len(toy_yields["background"])
    assert model.npar == len(toy_yields["background"]) + 1
    assert model.parameter_names[0] == "mu"
    assert model.parameter_names[1:] == sorted(model.parameter_names[1:])
    assert model.observed_counts == pytest.approx(toy_yields["data"])
    assert model.observed_data[: model.nbins] == pytest.approx(toy_yields["data"])
    assert model.observed_data[model.nbins :] == pytest.approx([0.0, 0.0, 0.0])


def test_split_accepts_both_layouts(model, toy_yields):
    """A data vector may or may not carry the auxiliary part."""
    counts, auxiliary = model.split(toy_yields["data"])
    assert counts == pytest.approx(toy_yields["data"])
    assert auxiliary == pytest.approx([0.0, 0.0, 0.0])

    counts, auxiliary = model.split(list(toy_yields["data"]) + [0.1, 0.2, 0.3])
    assert counts == pytest.approx(toy_yields["data"])
    assert auxiliary == pytest.approx([0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="Expected a data vector of length"):
        model.split([1.0, 2.0])


def test_set_parameters_validates_length(model):
    """A parameter vector of the wrong length is rejected."""
    with pytest.raises(ValueError, match="Expected 4 parameters"):
        model.set_parameters([1.0, 0.0])


def test_set_parameters_widens_range_instead_of_clamping(model):
    """``RooFit`` would silently clamp; the model widens the range instead."""
    model.set_parameters([1e3] + [0.0] * 3)
    assert model.poi.getVal() == pytest.approx(1e3)
    model.reset_parameters()


def test_expected_data_matches_inputs(model, toy_yields):
    """Expected yields reproduce the inputs of the toy at nominal parameters."""
    background = model.expected_data([0.0] + [0.0] * 3)
    assert background[: model.nbins] == pytest.approx(toy_yields["background"])
    assert background[model.nbins :] == pytest.approx([0.0, 0.0, 0.0])

    with_signal = model.expected_data([1.0] + [0.0] * 3)
    assert np.subtract(
        with_signal[: model.nbins], background[: model.nbins]
    ) == pytest.approx(toy_yields["signal"])


def test_expected_auxiliary_data_follows_the_nuisances(model):
    r"""Asimov auxiliary data are set to :math:`\hat\theta`, as ``RooFit`` does."""
    thetas = [0.3, -0.7, 1.1]
    expected = model.expected_data([1.0] + thetas)
    assert expected[model.nbins :] == pytest.approx(thetas)
    model.reset_parameters()


def test_nuisance_parameters_shift_the_yields(model, toy_yields):
    r"""A unit shift of :math:`\theta_i` moves bin ``i`` by :math:`\sigma_i`."""
    background = np.array(model.expected_data([0.0] + [0.0] * 3)[: model.nbins])
    shifted = np.array(model.expected_data([0.0, 1.0, 0.0, 0.0])[: model.nbins])
    assert shifted - background == pytest.approx([toy_yields["uncertainty"][0], 0.0, 0.0])
    model.reset_parameters()


def test_dataset_round_trip(model):
    """Bin counts survive a round trip through a ``RooDataSet``."""
    counts = [11.0, 22.0, 33.0]
    dataset = model.make_dataset(counts, name="round_trip")
    assert dataset.sumEntries() == pytest.approx(sum(counts))
    # pylint: disable=protected-access
    assert model._read_counts(dataset) == pytest.approx(counts)

    with pytest.raises(ValueError, match="Expected 3 bin counts"):
        model.make_dataset([1.0])


def test_roofit_and_reference_likelihood_agree(model):
    """
    The ``RooFit`` NLL and the closed-form expression differ by a constant.

    This is the assumption the calibration of the likelihood normalisation
    rests on, so it is checked explicitly over a scan of parameter points.
    """
    # pylint: disable=protected-access
    record = model._record(model.observed_data)
    counts = model.observed_counts
    offsets = []
    rng = np.random.default_rng(1234)
    for _ in range(10):
        pars = np.concatenate([[rng.uniform(-1.0, 2.0)], rng.normal(0.0, 0.5, 3)])
        model.set_parameters(pars)
        offsets.append(record.nll.getVal() + model.reference_logpdf(counts))
    model.reset_parameters()
    assert np.std(offsets) == pytest.approx(0.0, abs=1e-9)


def test_logpdf_is_insensitive_to_evaluation_order(model):
    """
    Re-evaluating an old parameter point gives the old answer back.

    ``RooFit`` caches the result of the previous evaluation, which is exactly
    the failure mode the model guards against by flushing the cache.
    """
    logpdf = model.logpdf_func()
    first = logpdf(np.array([1.0, 0.0, 0.0, 0.0]))
    logpdf(np.array([3.0, 1.0, -1.0, 0.5]))
    logpdf(np.array([-2.0, -1.0, 1.0, 0.0]))
    assert logpdf(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(first)


def test_alternating_datasets_do_not_leak_auxiliary_data(model):
    """Two likelihoods built for different auxiliary data stay independent."""
    observed = model.logpdf_func()
    shifted = model.logpdf_func(np.concatenate([model.observed_counts, [0.5, 0.5, 0.5]]))
    pars = np.array([1.0, 0.0, 0.0, 0.0])
    first_observed = observed(pars)
    first_shifted = shifted(pars)
    assert observed(pars) == pytest.approx(first_observed)
    assert shifted(pars) == pytest.approx(first_shifted)
    assert first_observed != pytest.approx(first_shifted)


def test_flush_and_clear_cache(model):
    """Flushing and clearing the cache leave the likelihood unchanged."""
    logpdf = model.logpdf_func()
    pars = np.array([1.0, 0.1, -0.1, 0.2])
    reference = logpdf(pars)
    model.flush()
    assert logpdf(pars) == pytest.approx(reference)
    model.clear_cache()
    assert model.logpdf_func()(pars) == pytest.approx(reference)
