"""Tests for the workspace interpreter."""

import numpy as np
import pytest

from spey_roofit import WorkspaceInterpreter, load_workspace
from spey_roofit.data import WorkspaceData


def test_load_workspace_rejects_missing_file(tmp_path):
    """A missing file is reported as an ``IOError``."""
    with pytest.raises(IOError):
        load_workspace(str(tmp_path / "does-not-exist.root"))


def test_load_workspace_rejects_unknown_name(uncorrelated_workspace):
    """Asking for a workspace that is not in the file fails loudly."""
    with pytest.raises(ValueError, match="does not contain a RooWorkspace named"):
        load_workspace(uncorrelated_workspace, workspace_name="not-there")


def test_interpreter_rejects_missing_model_config(uncorrelated_workspace):
    """A wrong ``ModelConfig`` name is reported with the available objects."""
    with pytest.raises(ValueError, match="does not contain a ModelConfig"):
        WorkspaceInterpreter(uncorrelated_workspace, model_config_name="Nope")


def test_structure(uncorrelated_workspace, toy_yields):
    """Channels, binning and observed data are read back correctly."""
    interpreter = WorkspaceInterpreter(uncorrelated_workspace)
    assert interpreter.channels == ["ch"]
    assert interpreter.bin_map == {"ch": len(toy_yields["background"])}
    assert interpreter.poi_names == ["mu"]
    assert len(interpreter.nuisance_parameters) == len(toy_yields["background"])
    assert interpreter.observed_data["ch"] == pytest.approx(toy_yields["data"])


def test_sample_decomposition(uncorrelated_workspace, toy_yields):
    """Samples are split into background and signal, with the right yields."""
    interpreter = WorkspaceInterpreter(uncorrelated_workspace)
    assert set(interpreter.samples["ch"]) == {"bkg", "sig"}
    assert interpreter.signal_samples["ch"] == ["sig"]
    assert interpreter.expected_background_yields["ch"] == pytest.approx(
        toy_yields["background"]
    )
    assert interpreter.expected_signal_yields["ch"] == pytest.approx(toy_yields["signal"])
    per_sample = interpreter.sample_yields("ch")["ch"]
    assert per_sample["bkg"] == pytest.approx(toy_yields["background"])
    assert per_sample["sig"] == pytest.approx(toy_yields["signal"])


def test_background_only_workspace_has_no_poi(background_only_workspace):
    """A background-only workspace declares no parameter of interest."""
    interpreter = WorkspaceInterpreter(background_only_workspace)
    assert interpreter.poi_names == []
    assert interpreter.signal_samples["ch"] == []
    assert interpreter.expected_signal_yields["ch"] == pytest.approx([0.0, 0.0, 0.0])
    assert not interpreter.is_alive


def test_inject_signal_creates_poi(background_only_workspace, toy_yields):
    """Injecting a signal into a background-only workspace creates ``mu``."""
    interpreter = WorkspaceInterpreter(background_only_workspace)
    interpreter.inject_signal("ch", toy_yields["signal"])
    assert interpreter.is_alive
    assert interpreter.signal_per_channel["ch"] == pytest.approx(toy_yields["signal"])

    patched = interpreter.patch
    assert patched is not interpreter.workspace
    model = WorkspaceData(patched)
    assert model.poi.GetName() == "mu"

    background = model.expected_data([0.0] + [0.0] * 3)[: model.nbins]
    with_signal = model.expected_data([1.0] + [0.0] * 3)[: model.nbins]
    assert background == pytest.approx(toy_yields["background"])
    assert np.subtract(with_signal, background) == pytest.approx(toy_yields["signal"])


def test_inject_signal_input_validation(background_only_workspace):
    """Bad channel names and bad shapes are rejected."""
    interpreter = WorkspaceInterpreter(background_only_workspace)
    with pytest.raises(ValueError, match="is not a channel"):
        interpreter.inject_signal("nope", [1.0])
    with pytest.raises(ValueError, match="signal yield"):
        interpreter.inject_signal("ch", [1.0, 2.0])
    interpreter.inject_signal("ch", [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="already has an injected signal"):
        interpreter.inject_signal("ch", [1.0, 2.0, 3.0])
    interpreter.reset_signal()
    assert interpreter.signal_per_channel == {}


def test_patch_is_identity_without_edits(uncorrelated_workspace):
    """Without edits the interpreter hands back the original workspace."""
    interpreter = WorkspaceInterpreter(uncorrelated_workspace)
    assert interpreter.patch is interpreter.workspace


def test_strip_signal_removes_existing_signal(uncorrelated_workspace, toy_yields):
    """``strip_signal`` leaves a background-only model behind."""
    interpreter = WorkspaceInterpreter(uncorrelated_workspace)
    interpreter.strip_signal()
    stripped = WorkspaceInterpreter(interpreter.patch)
    assert stripped.poi_names == []
    assert stripped.samples["ch"] == ["bkg"]
    assert stripped.expected_background_yields["ch"] == pytest.approx(
        toy_yields["background"]
    )


def test_strip_and_reinject(uncorrelated_workspace, toy_yields):
    """A published signal can be replaced by a different one."""
    interpreter = WorkspaceInterpreter(uncorrelated_workspace)
    replacement = [1.0, 2.0, 3.0]
    interpreter.strip_signal().inject_signal("ch", replacement)
    model = WorkspaceData(interpreter.patch, poi_name="mu")
    background = model.expected_data([0.0] + [0.0] * 3)[: model.nbins]
    with_signal = model.expected_data([1.0] + [0.0] * 3)[: model.nbins]
    assert background == pytest.approx(toy_yields["background"])
    assert np.subtract(with_signal, background) == pytest.approx(replacement)


def test_channel_type_guessing(uncorrelated_workspace):
    """Region types are guessed from the channel name."""
    interpreter = WorkspaceInterpreter(uncorrelated_workspace)
    assert interpreter.guess_channel_type("SR_200_cuts") == "SR"
    assert interpreter.guess_channel_type("CR1mu0b_200_cuts") == "CR"
    assert interpreter.guess_channel_type("VR_something") == "VR"
    assert interpreter.guess_channel_type("ch") == "__unknown__"


def test_summary_is_returned_and_mentions_channels(uncorrelated_workspace):
    """The summary can be captured instead of printed."""
    interpreter = WorkspaceInterpreter(uncorrelated_workspace)
    text = interpreter.summary(print_out=False)
    assert "ch" in text
    assert "bkg" in text and "sig" in text
