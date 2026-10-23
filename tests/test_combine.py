"""Tests for the ``root.cms-combine`` backend."""

import os

import numpy as np
import pytest
import spey

from combine_models import uncorrelated_background_datacard, write_templates
from conftest import assert_equal_up_to_constant, likelihood_profile
from spey_roofit import DatacardInterpreter, patch_datacard
from spey_roofit.combine import (
    SWAPPED_TWO_HYPOTHESIS_MODEL,
    TWO_HYPOTHESIS_MODEL,
    combine_options,
    parse_datacard,
    resolve_shapes,
)
from spey_roofit.combine_interface import CombineInterface

# pylint: disable=redefined-outer-name, protected-access

pytest.importorskip(
    "HiggsAnalysis.CombinedLimit.DatacardParser",
    reason="the CMS Combine backend needs cms-combine",
)


# --------------------------------------------------------------------------- #
# Registration                                                                 #
# --------------------------------------------------------------------------- #


def test_the_backend_is_registered():
    """``root.cms-combine`` is available through the entry point."""
    assert "root.cms-combine" in spey.AvailableBackends()


def test_metadata():
    """The backend declares everything the plug-in registry needs."""
    assert CombineInterface.name == "root.cms-combine"
    assert CombineInterface.author
    assert CombineInterface.spey_requires
    assert "2404.06614" in CombineInterface.arXiv


# --------------------------------------------------------------------------- #
# The interpreter                                                              #
# --------------------------------------------------------------------------- #


def test_a_counting_card_is_read(poisson_datacard, toy_yields):
    """Channels, processes, rates and observations come out of the text."""
    interpreter = DatacardInterpreter(poisson_datacard)

    assert interpreter.channels == ["ch0", "ch1", "ch2"]
    assert interpreter.processes == ["sig", "bkg"]
    assert interpreter.signal_processes == ["sig"]
    assert interpreter.bin_map == {"ch0": 1, "ch1": 1, "ch2": 1}
    assert interpreter.observed_data["ch1"] == pytest.approx([toy_yields["data"][1]])
    assert interpreter.expected_background_yields["ch1"] == pytest.approx(
        [toy_yields["background"][1]]
    )
    assert interpreter.expected_signal_yields["ch1"] == pytest.approx(
        [toy_yields["signal"][1]]
    )
    assert interpreter.is_alive


def test_a_shape_card_is_read(uncorrelated_datacard, toy_yields):
    """The templates are read through ``combine``'s own shape builder."""
    interpreter = DatacardInterpreter(uncorrelated_datacard)

    assert interpreter.bin_map == {"ch1": 3}
    assert interpreter.samples == {"ch1": ["sig", "bkg"]}
    assert interpreter.signal_samples == {"ch1": ["sig"]}
    assert interpreter.observed_data["ch1"] == pytest.approx(toy_yields["data"])
    assert interpreter.expected_background_yields["ch1"] == pytest.approx(
        toy_yields["background"]
    )
    assert interpreter.expected_signal_yields["ch1"] == pytest.approx(
        toy_yields["signal"]
    )
    assert "ch1" in interpreter.datacard.binParFlags


def test_the_summary_mentions_every_channel(poisson_datacard):
    """:meth:`summary` is the human readable entry point of the interpreter."""
    text = DatacardInterpreter(poisson_datacard).summary(print_out=False)
    for channel in ("ch0", "ch1", "ch2"):
        assert channel in text
    assert "counting" in text


def test_templates_of_different_length_are_padded(tmp_path):
    """
    A channel is as long as its longest template.

    Published cards do mix binnings within a channel: the tail of a
    distribution is rebinned for the processes that matter but not for a dummy
    one, and ``combine`` pads the short ones.
    """
    rootfile = write_templates(
        str(tmp_path),
        "ragged",
        {
            "bkg": [10.0, 20.0, 5.0, 1.0],
            "sig": [1.0, 2.0],
            "data_obs": [11.0, 21.0],
        },
    )
    card = (
        "imax 1\njmax 1\nkmax *\n"
        f"shapes * ch1 {rootfile} $PROCESS\n"
        "bin ch1\nobservation 32\n"
        "bin ch1 ch1\nprocess sig bkg\nprocess 0 1\nrate -1 -1\n"
    )
    interpreter = DatacardInterpreter(card)
    assert interpreter.bin_map == {"ch1": 4}
    assert interpreter.observed_data["ch1"] == pytest.approx([11.0, 21.0, 0.0, 0.0])
    assert interpreter.expected_signal_yields["ch1"] == pytest.approx(
        [1.0, 2.0, 0.0, 0.0]
    )


def test_injecting_the_wrong_number_of_yields_raises(background_only_datacard):
    """The shape of ``signal_yields`` has to follow ``bin_map``."""
    interpreter = DatacardInterpreter(background_only_datacard)
    with pytest.raises(ValueError, match="bin"):
        interpreter.inject_signal("ch1", [1.0, 2.0])


def test_injecting_twice_raises(background_only_datacard, toy_yields):
    """A second injection into the same channel is a mistake, not an addition."""
    interpreter = DatacardInterpreter(background_only_datacard)
    interpreter.inject_signal("ch1", toy_yields["signal"])
    with pytest.raises(ValueError, match="already has an injected signal"):
        interpreter.inject_signal("ch1", toy_yields["signal"])
    assert interpreter.reset_signal().signal_per_channel == {}


def test_an_unknown_channel_raises(poisson_datacard):
    """The error lists the channels that do exist."""
    with pytest.raises(ValueError, match="ch0"):
        DatacardInterpreter(poisson_datacard).inject_signal("nope", [1.0])


# --------------------------------------------------------------------------- #
# Rewriting the card                                                           #
# --------------------------------------------------------------------------- #


def test_patching_adds_one_column_per_channel(poisson_datacard, toy_yields):
    """The injected process becomes an ordinary column of the datacard."""
    text, _ = patch_datacard(poisson_datacard, injections={"ch1": [7.0]})
    card, _ = parse_datacard(text)

    assert "spey_signal" in card.processes
    assert card.isSignal["spey_signal"]
    assert card.exp["ch1"]["spey_signal"] == pytest.approx(7.0)
    assert "spey_signal" not in card.exp["ch0"]
    # everything else is untouched
    assert card.exp["ch0"]["bkg"] == pytest.approx(toy_yields["background"][0])


def test_patching_keeps_the_systematics_table_square(log_normal_datacard):
    """Every row of the table grows with the new column."""
    text, _ = patch_datacard(log_normal_datacard, injections={"ch0": [1.0]})
    card, _ = parse_datacard(text)

    for _, _, _, _, errline in card.systs:
        assert "spey_signal" in errline["ch0"]
        assert errline["ch0"]["spey_signal"] == 0.0


def test_stripping_removes_the_signal_columns(uncorrelated_datacard):
    """A published card can be reduced to its background."""
    text, _ = patch_datacard(uncorrelated_datacard, strip_signal=True)
    card, _ = parse_datacard(text, options=combine_options())

    assert card.processes == ["bkg"]
    assert "sig" not in card.exp["ch1"]


def test_stripping_everything_raises(tmp_path):
    """A card whose only process is signal cannot be stripped."""
    rootfile = write_templates(
        str(tmp_path), "onlysig", {"sig": [3.0], "data_obs": [3.0]}
    )
    card = (
        "imax 1\njmax 0\nkmax *\n"
        f"shapes * ch1 {rootfile} $PROCESS\n"
        "bin ch1\nobservation 3\n"
        "bin ch1\nprocess sig\nprocess 0\nrate -1\n"
    )
    with pytest.raises(ValueError, match="empty|without any process"):
        patch_datacard(card, strip_signal=True)


def test_shape_paths_are_resolved(tmp_path, toy_yields):
    """
    A template is found next to the card even under the producer's own path.

    Published datacards routinely carry the absolute path of the machine they
    were produced on, which no longer exists anywhere else.
    """
    text = uncorrelated_background_datacard(
        str(tmp_path),
        toy_yields["background"],
        toy_yields["signal"],
        toy_yields["data"],
        toy_yields["uncertainty"],
        name="elsewhere",
    )
    moved = text.replace(str(tmp_path), "/nowhere/at/all")
    assert "/nowhere/at/all" in moved

    resolved = resolve_shapes(moved, str(tmp_path))
    assert "/nowhere/at/all" not in resolved
    assert DatacardInterpreter(resolved, directory=str(tmp_path)).bin_map == {"ch1": 3}


def test_a_card_read_from_a_file_finds_its_templates(tmp_path, toy_yields):
    """Relative paths resolve against the directory holding the card."""
    text = uncorrelated_background_datacard(
        str(tmp_path),
        toy_yields["background"],
        toy_yields["signal"],
        toy_yields["data"],
        toy_yields["uncertainty"],
        name="relative",
    )
    path = os.path.join(str(tmp_path), "card.txt")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(text.replace(str(tmp_path) + os.sep, ""))
    assert DatacardInterpreter(path).bin_map == {"ch1": 3}


# --------------------------------------------------------------------------- #
# Agreement with other implementations                                         #
# --------------------------------------------------------------------------- #


def test_matches_spey_poisson(poisson_datacard, toy_yields):
    """A counting card without systematics is exactly ``default.poisson``."""
    ours = spey.get_backend("root.cms-combine")(
        datacard=poisson_datacard, analysis="combine"
    )
    theirs = spey.get_backend("default.poisson")(
        signal_yields=toy_yields["signal"],
        background_yields=toy_yields["background"],
        data=toy_yields["data"],
        analysis="spey",
    )
    pois = (0.0, 0.5, 1.0, 2.0)
    assert likelihood_profile(ours, pois) == pytest.approx(
        likelihood_profile(theirs, pois), abs=1e-9
    )
    assert ours.maximize_likelihood()[0] == pytest.approx(
        theirs.maximize_likelihood()[0], abs=1e-5
    )
    assert ours.poi_upper_limit() == pytest.approx(theirs.poi_upper_limit(), rel=1e-6)


def test_matches_spey_uncorrelated_background(uncorrelated_datacard, toy_yields):
    r"""
    ``autoMCStats`` in its Gaussian mode is ``default.uncorrelated_background``.

    Barlow-Beeston-lite gives every bin one parameter with a unit Gaussian
    constraint and adds :math:`\beta_i\sigma_i` to the expectation, which is
    the additive Gaussian model of that backend. ``lnN`` would *not* be: it is
    multiplicative.
    """
    ours = spey.get_backend("root.cms-combine")(
        datacard=uncorrelated_datacard, analysis="combine"
    )
    theirs = spey.get_backend("default.uncorrelated_background")(
        signal_yields=toy_yields["signal"],
        background_yields=toy_yields["background"],
        data=toy_yields["data"],
        absolute_uncertainties=toy_yields["uncertainty"],
        analysis="spey",
    )
    pois = (0.0, 0.5, 1.0, 2.0)
    assert likelihood_profile(ours, pois) == pytest.approx(
        likelihood_profile(theirs, pois), abs=1e-5
    )
    assert ours.maximize_likelihood()[0] == pytest.approx(
        theirs.maximize_likelihood()[0], abs=1e-4
    )
    assert ours.exclusion_confidence_level()[0] == pytest.approx(
        theirs.exclusion_confidence_level()[0], abs=1e-5
    )
    assert ours.poi_upper_limit() == pytest.approx(theirs.poi_upper_limit(), rel=1e-5)


def test_matches_the_roofit_backend(shape_datacard, tmp_path):
    """
    A yield-preserving ``shape`` is a ``HistFactory`` ``HistoSys``.

    Both morph the templates vertically with the same interpolation, but one
    reaches it from a datacard through ``combine`` and the other from a
    ``HistFactory`` measurement through ``ROOT``.
    """
    from conftest import BACKGROUND, OBSERVED, SHIFTS, SIGNAL
    from toy_models import _build

    path = _build(
        str(tmp_path),
        "histosys",
        BACKGROUND,
        SIGNAL,
        OBSERVED,
        [np.array(shift) for shift in SHIFTS],
    )
    ours = spey.get_backend("root.cms-combine")(
        datacard=shape_datacard, analysis="combine"
    )
    theirs = spey.get_backend("root.roofit")(workspace=path, analysis="roofit")

    assert ours.backend.model.npar == theirs.backend.model.npar
    ours_logpdf = ours.backend.get_logpdf_func()
    theirs_logpdf = theirs.backend.get_logpdf_func()
    mine, reference = [], []
    for poi in (0.0, 1.0, 2.0):
        for theta in (-1.5, -0.4, 0.0, 0.8):
            point = np.array([poi, theta, -0.5 * theta])
            ours.backend.model.set_parameters(point)
            theirs.backend.model.set_parameters(point)
            assert ours.backend.model.expected_yields() == pytest.approx(
                theirs.backend.model.expected_yields(), rel=1e-12
            )
            mine.append(ours_logpdf(point))
            reference.append(theirs_logpdf(point))
    assert_equal_up_to_constant(mine, reference)


def test_matches_spey_pyhf(
    log_normal_datacard, log_normal_kappas, toy_yields, pyhf_backend
):
    r"""
    A symmetric ``lnN`` is a ``pyhf`` ``normsys``.

    ``pyhf`` interpolates a normalisation systematic with the piecewise power
    law, which for :math:`\kappa^-=1/\kappa^+` is :math:`\kappa^\theta`
    everywhere, and so is ``combine``'s ``AsymPow``.
    """
    channels, observations, patch = [], [], []
    zipped = zip(
        toy_yields["background"],
        toy_yields["signal"],
        toy_yields["data"],
        log_normal_kappas,
    )
    for index, (background, signal, observed, kappa) in enumerate(zipped):
        channels.append(
            {
                "name": f"ch{index}",
                "samples": [
                    {
                        "name": "bkg",
                        "data": [float(background)],
                        "modifiers": [
                            {
                                "name": f"norm{index}",
                                "type": "normsys",
                                "data": {"hi": kappa, "lo": 1.0 / kappa},
                            }
                        ],
                    }
                ],
            }
        )
        observations.append({"name": f"ch{index}", "data": [float(observed)]})
        patch.append(
            {
                "op": "add",
                "path": f"/channels/{index}/samples/1",
                "value": {
                    "name": "sig",
                    "data": [float(signal)],
                    "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
                },
            }
        )
    workspace = {
        "channels": channels,
        "observations": observations,
        "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": []}}],
        "version": "1.0.0",
    }

    ours = spey.get_backend("root.cms-combine")(
        datacard=log_normal_datacard, analysis="combine"
    )
    theirs = spey.get_backend("pyhf")(
        background_only_model=workspace, signal_patch=patch, analysis="pyhf"
    )
    assert theirs.backend.is_alive
    pois = (0.0, 0.5, 1.0, 2.0)
    assert likelihood_profile(ours, pois) == pytest.approx(
        likelihood_profile(theirs, pois), abs=1e-5
    )
    assert ours.poi_upper_limit() == pytest.approx(theirs.poi_upper_limit(), rel=1e-5)


# --------------------------------------------------------------------------- #
# The backend                                                                  #
# --------------------------------------------------------------------------- #


def test_the_workspace_is_the_one_combine_builds(uncorrelated_datacard):
    """The compiled workspace carries ``combine``'s own objects."""
    model = spey.get_backend("root.cms-combine")(
        datacard=uncorrelated_datacard, analysis="combine"
    )
    workspace = model.backend.workspace
    assert workspace.pdf("model_s")
    assert workspace.data("data_obs")
    assert workspace.obj("ModelConfig")
    assert workspace.pdf("model_s").InheritsFrom("RooSimultaneous")
    assert model.backend.datacard.bins == ["ch1"]
    assert model.backend.interpreter.bin_map == {"ch1": 3}


def test_asimov_data_matches_spey(uncorrelated_datacard, toy_yields):
    """Asimov datasets carry the auxiliary measurements, as ``spey`` expects."""
    ours = spey.get_backend("root.cms-combine")(
        datacard=uncorrelated_datacard, analysis="combine"
    )
    theirs = spey.get_backend("default.uncorrelated_background")(
        signal_yields=toy_yields["signal"],
        background_yields=toy_yields["background"],
        data=toy_yields["data"],
        absolute_uncertainties=toy_yields["uncertainty"],
        analysis="spey",
    )
    point = [1.0, 0.4, -0.2, 0.7]
    assert ours.backend.expected_data(point) == pytest.approx(
        theirs.backend.expected_data(point), rel=1e-12
    )


def test_the_sampler_generates_bins_and_auxiliary_data(uncorrelated_datacard):
    """The toy calculator needs bin counts and auxiliary measurements."""
    model = spey.get_backend("root.cms-combine")(
        datacard=uncorrelated_datacard, analysis="combine"
    )
    sampler = model.backend.get_sampler(np.array([1.0, 0.0, 0.0, 0.0]))
    samples = sampler(64)
    assert samples.shape == (64, 6)
    assert np.all(samples[:, :3] >= 0.0)
    assert sampler(8, include_auxiliary=False).shape == (8, 3)


def test_the_configuration_names_every_parameter(uncorrelated_datacard):
    """``config`` reports the POI first and the nuisance parameters by name."""
    model = spey.get_backend("root.cms-combine")(
        datacard=uncorrelated_datacard, analysis="combine"
    )
    config = model.backend.config()
    assert config.poi_index == 0
    assert config.parameter_names[0] == "r"
    assert len(config.suggested_init) == 4
    assert config.minimum_poi < 0.0


def test_a_signal_can_be_injected_into_a_background_only_card(
    background_only_datacard, uncorrelated_datacard, toy_yields
):
    """Injection reproduces the card that declares the signal itself."""
    injected = spey.get_backend("root.cms-combine")(
        datacard=background_only_datacard,
        signal_yields={"ch1": toy_yields["signal"]},
        analysis="injected",
    )
    declared = spey.get_backend("root.cms-combine")(
        datacard=uncorrelated_datacard, analysis="declared"
    )
    assert injected.backend.is_alive
    for poi in (0.0, 1.0, 2.0):
        assert injected.likelihood(poi_test=poi) == pytest.approx(
            declared.likelihood(poi_test=poi), abs=1e-6
        )


def test_stripping_the_signal_leaves_the_background(uncorrelated_datacard, toy_yields):
    """A published card can be reinterpreted with a different signal."""
    other = [3.0, 4.0, 1.0]
    reinterpreted = spey.get_backend("root.cms-combine")(
        datacard=uncorrelated_datacard,
        signal_yields={"ch1": other},
        strip_existing_signal=True,
        analysis="reinterpreted",
    )
    background, signal = reinterpreted.backend._background_and_signal()
    assert background == pytest.approx(toy_yields["background"], rel=1e-12)
    assert signal == pytest.approx(other, rel=1e-12)


def test_a_stripped_card_is_saved_and_reused(uncorrelated_datacard, toy_yields, tmp_path):
    """A published card is reduced to its background, written and read back."""
    path = str(tmp_path / "background_only.txt")
    interpreter = DatacardInterpreter(uncorrelated_datacard)
    assert interpreter.is_alive
    assert interpreter.strip_signal().save_patch(path) == path

    saved = DatacardInterpreter(path)
    assert saved.signal_processes == []
    assert not saved.is_alive
    assert saved.expected_background_yields["ch1"] == pytest.approx(
        toy_yields["background"], rel=1e-12
    )

    model = spey.get_backend("root.cms-combine")(
        datacard=path, signal_yields={"ch1": toy_yields["signal"]}, analysis="saved"
    )
    declared = spey.get_backend("root.cms-combine")(
        datacard=uncorrelated_datacard, analysis="declared"
    )
    for poi in (0.0, 1.0):
        assert model.likelihood(poi_test=poi) == pytest.approx(
            declared.likelihood(poi_test=poi), abs=1e-6
        )


def test_a_reinterpretation_is_saved_with_its_templates(uncorrelated_datacard, tmp_path):
    """The templates of an injected signal travel with the saved card."""
    other = [3.0, 4.0, 1.0]
    target = tmp_path / "elsewhere"
    path = str(target / "reinterpretation.txt")

    interpreter = DatacardInterpreter(uncorrelated_datacard)
    interpreter.strip_signal().inject_signal("ch1", other).save_patch(path)

    assert (target / "spey_signal.root").is_file()
    saved = DatacardInterpreter(path)
    assert saved.signal_processes == ["spey_signal"]
    assert saved.expected_signal_yields["ch1"] == pytest.approx(other)


def test_the_two_hypothesis_models_are_mirror_images(tmp_path):
    r"""
    ``spey``'s background is the :math:`x=0` hypothesis, so which of the two
    signals :math:`x` scales decides what is being tested. ``combine``'s own
    model scales the one carrying the ``ALT`` marker; the one contributed by
    :mod:`spey_roofit.physics_models` scales the other.
    """
    rootfile = write_templates(
        str(tmp_path),
        "hypotheses",
        {
            "new": [30.0, 10.0],
            "new_ALT": [20.0, 20.0],
            "dummy": [1e-9, 1e-9],
            "data_obs": [25.0, 15.0],
        },
    )
    card = (
        "imax 1\njmax 2\nkmax *\n"
        f"shapes * ch1 {rootfile} $PROCESS\n"
        "bin ch1\nobservation 40\n"
        "bin ch1 ch1 ch1\nprocess new new_ALT dummy\nprocess -1 0 1\n"
        "rate -1 -1 -1\n"
    )
    stock = spey.get_backend("root.cms-combine")(
        datacard=card, physics_model=TWO_HYPOTHESIS_MODEL, analysis="stock"
    )
    swapped = spey.get_backend("root.cms-combine")(
        datacard=card, physics_model=SWAPPED_TWO_HYPOTHESIS_MODEL, analysis="swapped"
    )
    assert stock.backend.model.poi.GetName() == "x"

    stock_background, stock_signal = stock.backend._background_and_signal()
    swapped_background, swapped_signal = swapped.backend._background_and_signal()
    assert stock_background == pytest.approx([30.0, 10.0], abs=1e-6)
    assert swapped_background == pytest.approx([20.0, 20.0], abs=1e-6)
    assert stock_signal == pytest.approx(-np.asarray(swapped_signal), abs=1e-6)


def test_combine_options_reach_combine(log_normal_datacard):
    """The switches of ``text2workspace.py`` are available through the backend."""
    model = spey.get_backend("root.cms-combine")(
        datacard=log_normal_datacard,
        options=combine_options(stat=True),
        analysis="stat-only",
    )
    # `--stat` drops every systematic uncertainty, which leaves only the POI
    assert model.backend.model.parameter_names == ["r"]


def test_an_unreadable_card_is_reported(poisson_datacard):
    """``combine``'s own error reaches the caller."""
    with pytest.raises(RuntimeError, match="Unsupported pdf"):
        DatacardInterpreter(poisson_datacard + "weird gmZ 1 1 1 1 1 1\n")


@pytest.mark.slow
def test_a_card_with_every_directive_runs(mixed_datacard):
    """
    A card exercising the directives only ``combine`` has runs end to end.

    Asymmetric ``lnN``, a ``gmN`` sideband, a ``param`` with an asymmetric
    width and a ``rateParam`` have no counterpart in another tool, so what is
    checked here is that ``combine`` builds them and that ``spey`` can do its
    work on the result.
    """
    model = spey.get_backend("root.cms-combine")(
        datacard=mixed_datacard, analysis="mixed"
    )
    names = model.backend.model.parameter_names
    for expected in ("symmetric", "asymmetric", "sideband", "shifted", "scale"):
        assert expected in names
    assert 0.0 < model.exclusion_confidence_level()[0] < 1.0
    assert model.poi_upper_limit() > 0.0


@pytest.mark.slow
def test_a_shape_card_runs_a_full_hypothesis_test(shape_datacard):
    """The whole ``spey`` machinery works on a card with shape uncertainties."""
    model = spey.get_backend("root.cms-combine")(
        datacard=shape_datacard, analysis="combine"
    )
    assert 0.0 < model.exclusion_confidence_level()[0] < 1.0
    assert model.poi_upper_limit() > 0.0
    assert np.isfinite(model.significance()[0])
