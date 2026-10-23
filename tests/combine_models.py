r"""
Builders for the small ``CMS Combine`` datacards used to validate the plug-in.

The cards produced here are constructed so that they are *exactly* equivalent
to a likelihood some other tool already implements, which makes it possible to
compare the two number by number:

* :func:`poisson_datacard` reproduces ``default.poisson``: a counting
  experiment with no nuisance parameter at all.
* :func:`uncorrelated_background_datacard` reproduces
  ``default.uncorrelated_background``. ``combine``'s ``lnN`` is multiplicative
  and therefore *not* the additive Gaussian of that backend, but its
  ``autoMCStats`` directive is: the Barlow-Beeston-lite treatment gives every
  bin one parameter :math:`\beta_i` with a unit Gaussian constraint and adds
  :math:`\beta_i\sigma_i` to the expectation, which is exactly
  :math:`\lambda_i=b_i+\mu s_i+\theta_i\sigma_i`.
* :func:`log_normal_datacard` reproduces a ``pyhf`` ``normsys``. ``pyhf``'s
  default interpolation for a normalisation systematic is the piecewise power
  law, which for a symmetric :math:`\kappa` is :math:`\kappa^\theta`
  everywhere — the same as ``combine``'s ``AsymPow``.
* :func:`shape_datacard` reproduces a ``HistFactory`` ``HistoSys``. A
  ``shape`` uncertainty whose varied templates have the *same integral* as the
  nominal one contributes no normalisation factor, and its morphing of the
  fractions is then term for term the vertical interpolation ``HistFactory``
  applies to the absolute bin contents.

The remaining builder, :func:`mixed_datacard`, is not the twin of anything: it
exercises the directives that only ``combine`` has — asymmetric ``lnN``,
``gmN``, ``param`` and ``rateParam`` — so what is checked with it is that
``combine`` builds them and that ``spey`` can do its work on the result.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

__all__ = [
    "poisson_datacard",
    "log_normal_datacard",
    "uncorrelated_background_datacard",
    "shape_datacard",
    "mixed_datacard",
    "write_templates",
]


def __dir__():
    return __all__


def _root():
    """
    Return the ``ROOT`` module through the plug-in's manager.

    Returns:
        ``module``:
        The ``ROOT`` module, in batch mode and with a quiet ``RooMsgService``.
    """
    from spey_roofit.manager import manager  # pylint: disable=import-outside-toplevel

    manager.silence("FATAL")
    return manager.ROOT


def _counting_header(channels: Sequence[str], observation: Sequence[float]) -> list[str]:
    """
    First half of a datacard, up to the ``observation`` line.

    Args:
        channels (``Sequence[str]``): names of the channels.
        observation (``Sequence[float]``): observed counts per channel.

    Returns:
        ``List[str]``:
        The lines of the block.
    """
    return [
        f"imax {len(channels)}",
        "jmax *",
        "kmax *",
        "-" * 20,
        "bin " + " ".join(channels),
        "observation " + " ".join(f"{value:.10g}" for value in observation),
        "-" * 20,
    ]


def _process_block(columns: Sequence[tuple], rates: Sequence[float]) -> list[str]:
    """
    The ``bin``/``process``/``rate`` block of a datacard.

    Args:
        columns (``Sequence[tuple]``): one ``(channel, process, index)`` per
          column, with the index following ``combine``'s convention that a
          value :math:`\\leq 0` marks a signal.
        rates (``Sequence[float]``): expected yield of every column.

    Returns:
        ``List[str]``:
        The lines of the block.
    """
    return [
        "bin " + " ".join(channel for channel, _, _ in columns),
        "process " + " ".join(process for _, process, _ in columns),
        "process " + " ".join(str(index) for _, _, index in columns),
        "rate " + " ".join(f"{rate:.10g}" for rate in rates),
        "-" * 20,
    ]


def write_templates(
    path: str,
    name: str,
    histograms: dict[str, Sequence[float]],
    errors: dict[str, Sequence[float]] | None = None,
) -> str:
    """
    Write a flat ``ROOT`` file of ``TH1D`` templates.

    Args:
        path (``str``): directory the file is written to.
        name (``str``): base name of the file.
        histograms (``Dict[str, Sequence[float]]``): bin contents, keyed by the
          name the histogram gets in the file.
        errors (``Dict[str, Sequence[float]]``, default ``None``): per-bin
          uncertainties of the same histograms. Zero when omitted.

    Returns:
        ``str``:
        Path of the ``.root`` file.
    """
    ROOT = _root()

    errors = errors or {}
    output = os.path.join(path, f"{name}.root")
    handle = ROOT.TFile.Open(output, "RECREATE")
    for key, values in histograms.items():
        histogram = ROOT.TH1D(key, key, len(values), 0.0, float(len(values)))
        histogram.SetDirectory(0)
        for index, value in enumerate(values):
            histogram.SetBinContent(index + 1, float(value))
            histogram.SetBinError(
                index + 1, float(errors.get(key, [0.0] * len(values))[index])
            )
        histogram.Write()
    handle.Close()
    return output


def poisson_datacard(
    background: Sequence[float], signal: Sequence[float], data: Sequence[float]
) -> str:
    r"""
    Build the counting twin of ``default.poisson``.

    One single-bin channel per entry and no systematic uncertainty, so that

    .. math::

        \mathcal{L}(\mu) =
            \prod_i{\rm Pois}\left(n_i\vert\mu s_i + b_i\right)\ .

    Args:
        background (``Sequence[float]``): background yields :math:`b_i`.
        signal (``Sequence[float]``): signal yields :math:`s_i`.
        data (``Sequence[float]``): observed counts :math:`n_i`.

    Returns:
        ``str``:
        The datacard.
    """
    channels = [f"ch{index}" for index in range(len(background))]
    columns, rates = [], []
    for channel, yields, rate in zip(channels, signal, background):
        columns += [(channel, "sig", 0), (channel, "bkg", 1)]
        rates += [float(yields), float(rate)]
    lines = _counting_header(channels, data) + _process_block(columns, rates)
    return "\n".join(lines) + "\n"


def log_normal_datacard(
    background: Sequence[float],
    signal: Sequence[float],
    data: Sequence[float],
    kappas: Sequence[float],
) -> str:
    r"""
    Build a counting card with one ``lnN`` uncertainty per channel.

    The likelihood is

    .. math::

        \mathcal{L}(\mu,\theta) =
            \prod_i{\rm Pois}\left(n_i\vert\mu s_i + b_i\kappa_i^{\theta_i}\right)
            \cdot\prod_i{\rm Gauss}\left(0\vert\theta_i,1\right)\ ,

    which is what ``pyhf`` builds from a symmetric ``normsys``.

    Args:
        background (``Sequence[float]``): background yields :math:`b_i`.
        signal (``Sequence[float]``): signal yields :math:`s_i`.
        data (``Sequence[float]``): observed counts :math:`n_i`.
        kappas (``Sequence[float]``): the factors :math:`\kappa_i`.

    Returns:
        ``str``:
        The datacard.
    """
    channels = [f"ch{index}" for index in range(len(background))]
    columns, rates = [], []
    for channel, yields, rate in zip(channels, signal, background):
        columns += [(channel, "sig", 0), (channel, "bkg", 1)]
        rates += [float(yields), float(rate)]
    lines = _counting_header(channels, data) + _process_block(columns, rates)
    for index, kappa in enumerate(kappas):
        entries = []
        for position in range(len(channels)):
            entries += ["-", f"{kappa:.10g}" if position == index else "-"]
        lines.append(f"norm{index} lnN " + " ".join(entries))
    return "\n".join(lines) + "\n"


def uncorrelated_background_datacard(
    path: str,
    background: Sequence[float],
    signal: Sequence[float] | None,
    data: Sequence[float],
    uncertainties: Sequence[float],
    name: str = "uncorrelated",
) -> str:
    r"""
    Build the ``combine`` twin of ``default.uncorrelated_background``.

    One channel with as many bins as there are entries, and an ``autoMCStats``
    directive whose threshold is zero so that every bin is treated in the
    Gaussian Barlow-Beeston-lite mode. The Monte Carlo uncertainty of the
    background template carries :math:`\sigma_i`, giving

    .. math::

        \mathcal{L}(\mu,\theta) =
            \prod_i{\rm Pois}\left(n_i\vert\mu s_i + b_i + \theta_i\sigma_i\right)
            \cdot\prod_i{\rm Gauss}\left(0\vert\theta_i,1\right)\ .

    Args:
        path (``str``): directory the templates are written to.
        background (``Sequence[float]``): background yields :math:`b_i`.
        signal (``Sequence[float]``): signal yields :math:`s_i`, or ``None``
          for a background-only card.
        data (``Sequence[float]``): observed counts :math:`n_i`.
        uncertainties (``Sequence[float]``): absolute background uncertainties
          :math:`\sigma_i`.
        name (``str``, default ``"uncorrelated"``): base name of the templates.

    Returns:
        ``str``:
        The datacard.
    """
    histograms = {"bkg": list(background), "data_obs": list(data)}
    errors = {"bkg": list(uncertainties)}
    if signal is not None:
        histograms["sig"] = list(signal)
    rootfile = write_templates(path, name, histograms, errors)

    columns = [("ch1", "bkg", 1)]
    rates = [-1.0]
    if signal is not None:
        columns.insert(0, ("ch1", "sig", 0))
        rates.insert(0, -1.0)
    lines = _counting_header(["ch1"], [float(np.sum(data))])
    lines.insert(4, f"shapes * ch1 {rootfile} $PROCESS")
    lines += _process_block(columns, rates)
    lines.append("ch1 autoMCStats 0 0 1")
    return "\n".join(lines) + "\n"


def shape_datacard(
    path: str,
    background: Sequence[float],
    signal: Sequence[float],
    data: Sequence[float],
    shifts: Sequence[Sequence[float]],
    name: str = "shapes",
) -> str:
    r"""
    Build a card whose ``shape`` uncertainties preserve the total yield.

    Every shift vector has to sum to zero. The varied templates then integrate
    to the same total as the nominal one, ``combine`` adds no normalisation
    factor, and the model is exactly the ``HistFactory`` one

    .. math::

        \lambda_i = \mu s_i + b_i + \sum_k \alpha_k(\theta_k)\,t_{ki}\ ,

    with :math:`\alpha` the vertical interpolation both tools use.

    Args:
        path (``str``): directory the templates are written to.
        background (``Sequence[float]``): background yields :math:`b_i`.
        signal (``Sequence[float]``): signal yields :math:`s_i`.
        data (``Sequence[float]``): observed counts :math:`n_i`.
        shifts (``Sequence[Sequence[float]]``): one shift vector
          :math:`t_{ki}` per uncertainty, each summing to zero.
        name (``str``, default ``"shapes"``): base name of the templates.

    Raises:
        ``ValueError``: if a shift vector does not sum to zero.

    Returns:
        ``str``:
        The datacard.
    """
    nominal = np.asarray(background, dtype=float)
    histograms = {
        "bkg": list(nominal),
        "sig": list(signal),
        "data_obs": list(data),
    }
    for index, shift in enumerate(shifts):
        values = np.asarray(shift, dtype=float)
        if abs(float(np.sum(values))) > 1e-12:
            raise ValueError(
                f"The shift vector {index} sums to {float(np.sum(values)):g}; it has "
                f"to preserve the total yield for the card to be a HistFactory twin."
            )
        histograms[f"bkg_theta{index}Up"] = list(nominal + values)
        histograms[f"bkg_theta{index}Down"] = list(nominal - values)
    rootfile = write_templates(path, name, histograms)

    lines = _counting_header(["ch1"], [float(np.sum(data))])
    lines.insert(4, f"shapes * ch1 {rootfile} $PROCESS $PROCESS_$SYSTEMATIC")
    lines += _process_block([("ch1", "sig", 0), ("ch1", "bkg", 1)], [-1.0, -1.0])
    for index in range(len(shifts)):
        lines.append(f"theta{index} shape - 1")
    return "\n".join(lines) + "\n"


def mixed_datacard(
    background: Sequence[float], signal: Sequence[float], data: Sequence[float]
) -> str:
    """
    Build a counting card exercising the directives only ``combine`` has.

    The card carries a symmetric and an asymmetric ``lnN``, a ``gmN`` sideband,
    a ``param`` with an asymmetric width and a ``rateParam``. None of them has
    a counterpart in another tool, so what the card checks is that ``combine``
    builds every one of them and that ``spey`` can run a full hypothesis test
    on the result.

    Args:
        background (``Sequence[float]``): background yields, at least three.
        signal (``Sequence[float]``): signal yields.
        data (``Sequence[float]``): observed counts.

    Returns:
        ``str``:
        The datacard.
    """
    channels = [f"ch{index}" for index in range(len(background))]
    columns, rates = [], []
    for channel, yields, rate in zip(channels, signal, background):
        columns += [(channel, "sig", 0), (channel, "bkg", 1)]
        rates += [float(yields), float(rate)]
    # the gmN sideband replaces the rate of the last background by alpha * N
    sideband, alpha = 12, float(background[-1]) / 12.0
    rates[-1] = alpha * sideband

    lines = _counting_header(channels, data) + _process_block(columns, rates)
    columns_count = len(columns)

    def row(entries: dict[int, str]) -> str:
        """Render one line of the systematics table."""
        return " ".join(entries.get(index, "-") for index in range(columns_count))

    lines.append("symmetric lnN " + row({1: "1.10", 3: "1.05"}))
    lines.append("asymmetric lnN " + row({1: "0.92/1.17", 5: "0.88/1.30"}))
    lines.append(f"sideband gmN {sideband} " + row({columns_count - 1: f"{alpha:.10g}"}))
    lines.append("shifted lnN " + row({0: "1.08"}))
    lines.append("shifted param 0 -0.7/+1.3")
    lines.append("scale rateParam ch0 bkg 1 [0.2,3.0]")
    return "\n".join(lines) + "\n"
