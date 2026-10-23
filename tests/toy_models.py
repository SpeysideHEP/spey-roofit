r"""
Builders for small ``HistFactory`` workspaces used to validate the plug-in.

The workspaces produced here are constructed so that they are *exactly*
equivalent to the built-in ``spey`` likelihoods, which makes it possible to
compare the two implementations number by number:

* :func:`uncorrelated_background_workspace` reproduces
  ``default.uncorrelated_background``. Every bin gets its own ``HistoSys``
  whose up and down templates are :math:`b_i\pm\sigma_i`, so that the
  piecewise-linear interpolation of ``HistFactory`` gives exactly
  :math:`\lambda_i=b_i+\theta_i\sigma_i+\mu s_i` with a unit Gaussian
  constraint on each :math:`\theta_i`.
* :func:`correlated_background_workspace` reproduces
  ``default.correlated_background``. The covariance matrix is factorised as
  :math:`\Sigma=LL^T` and the columns of :math:`L` are used as ``HistoSys``
  templates, which turns the correlated Gaussian constraint into a set of
  independent unit Gaussians without changing the model.

Both builders switch the ``HistFactory`` luminosity parameter off, so that no
extra constraint term enters the likelihood.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "uncorrelated_background_workspace",
    "correlated_background_workspace",
    "uncorrelated_background_pyhf_workspace",
]


def __dir__():
    return __all__


def _root():
    """
    Return the ``ROOT`` module through the plug-in's manager.

    Going through the manager guarantees batch mode and a quiet
    ``RooMsgService``, which ``HistFactory`` is otherwise very generous with.

    Returns:
        ``module``:
        The ``ROOT`` module.
    """
    from spey_roofit.manager import manager  # pylint: disable=import-outside-toplevel

    manager.silence("FATAL")
    return manager.ROOT


def _histogram(name: str, values: Sequence[float]):
    """
    Build a ``TH1D`` with one bin per entry of ``values``.

    Args:
        name (``str``): name of the histogram.
        values (``Sequence[float]``): bin contents.

    Returns:
        ``ROOT.TH1D``:
        Histogram spanning ``[0, len(values)]`` with unit-width bins.
    """
    ROOT = _root()

    nbins = len(values)
    histogram = ROOT.TH1D(name, name, nbins, 0.0, float(nbins))
    histogram.SetDirectory(0)
    # HistFactory takes the histogram over (``HistRef`` deletes it with its
    # sample). Without handing the ownership to C++, ``PyROOT`` deletes it a
    # second time on garbage collection, which segfaults with ROOT >= 6.40.
    ROOT.SetOwnership(histogram, False)
    for index, value in enumerate(values):
        histogram.SetBinContent(index + 1, float(value))
    return histogram


def _build(
    path: str,
    name: str,
    background: Sequence[float],
    signal: Optional[Sequence[float]],
    data: Sequence[float],
    templates: Sequence[Sequence[float]],
) -> str:
    r"""
    Assemble and write a single-channel ``HistFactory`` workspace.

    Args:
        path (``str``): directory the workspace is written to.
        name (``str``): base name of the measurement and of the output file.
        background (``Sequence[float]``): nominal background yields.
        signal (``Sequence[float]``): signal yields scaled by ``mu``, or
          ``None`` for a background-only workspace.
        data (``Sequence[float]``): observed counts.
        templates (``Sequence[Sequence[float]]``): one shift vector per
          nuisance parameter; the ``HistoSys`` up and down templates are
          :math:`b\pm t_k`.

    Returns:
        ``str``:
        Path of the ``.root`` file holding the workspace, whose ``RooWorkspace``
        is stored under the key ``"combined"``.
    """
    ROOT = _root()

    HF = ROOT.RooStats.HistFactory
    keep: List = []

    measurement = HF.Measurement(name, name)
    measurement.SetOutputFilePrefix(os.path.join(path, name))
    measurement.SetPOI("mu")
    measurement.SetLumi(1.0)
    measurement.SetLumiRelErr(0.0)
    measurement.AddConstantParam("Lumi")

    channel = HF.Channel("ch")
    observed = _histogram(f"{name}_data", data)
    channel.SetData(observed)
    keep.append(observed)

    nominal = _histogram(f"{name}_bkg", background)
    sample = HF.Sample("bkg")
    sample.SetHisto(nominal)
    sample.SetNormalizeByTheory(False)
    keep.append(nominal)

    for index, template in enumerate(templates):
        high = _histogram(
            f"{name}_bkg_up{index}", np.asarray(background) + np.asarray(template)
        )
        low = _histogram(
            f"{name}_bkg_dn{index}", np.asarray(background) - np.asarray(template)
        )
        systematic = HF.HistoSys()
        systematic.SetName(f"theta{index}")
        systematic.SetHistoHigh(high)
        systematic.SetHistoLow(low)
        sample.AddHistoSys(systematic)
        keep += [high, low, systematic]
    channel.AddSample(sample)

    if signal is not None:
        signal_histogram = _histogram(f"{name}_sig", signal)
        signal_sample = HF.Sample("sig")
        signal_sample.SetHisto(signal_histogram)
        signal_sample.SetNormalizeByTheory(False)
        signal_sample.AddNormFactor("mu", 1.0, -10.0, 10.0)
        channel.AddSample(signal_sample)
        keep += [signal_histogram, signal_sample]

    measurement.AddChannel(channel)
    workspace = HF.MakeModelAndMeasurementFast(measurement)
    output = os.path.join(path, f"{name}.root")
    workspace.writeToFile(output)
    return output


def uncorrelated_background_workspace(
    path: str,
    background: Sequence[float],
    signal: Optional[Sequence[float]],
    data: Sequence[float],
    uncertainties: Sequence[float],
    name: str = "uncorrelated",
) -> str:
    r"""
    Build the ``HistFactory`` twin of ``default.uncorrelated_background``.

    The resulting likelihood is

    .. math::

        \mathcal{L}(\mu,\theta) =
            \prod_i{\rm Pois}\left(n_i\vert\mu s_i + b_i + \theta_i\sigma_i\right)
            \cdot\prod_i{\rm Gauss}\left(0\vert\theta_i,1\right)\ .

    Args:
        path (``str``): directory the workspace is written to.
        background (``Sequence[float]``): background yields :math:`b_i`.
        signal (``Sequence[float]``): signal yields :math:`s_i`, or ``None``
          for a background-only workspace.
        data (``Sequence[float]``): observed counts :math:`n_i`.
        uncertainties (``Sequence[float]``): absolute background uncertainties
          :math:`\sigma_i`.
        name (``str``, default ``"uncorrelated"``): base name of the workspace.

    Returns:
        ``str``:
        Path of the ``.root`` file holding the workspace.
    """
    nbins = len(background)
    templates = []
    for index, sigma in enumerate(uncertainties):
        template = np.zeros(nbins)
        template[index] = sigma
        templates.append(template)
    return _build(path, name, background, signal, data, templates)


def correlated_background_workspace(
    path: str,
    background: Sequence[float],
    signal: Optional[Sequence[float]],
    data: Sequence[float],
    covariance: Sequence[Sequence[float]],
    name: str = "correlated",
) -> str:
    r"""
    Build the ``HistFactory`` twin of ``default.correlated_background``.

    The covariance matrix is factorised as :math:`\Sigma=LL^T`; using the
    columns of :math:`L` as ``HistoSys`` templates gives

    .. math::

        \lambda_i = \mu s_i + b_i + \sum_k L_{ik}z_k\ ,\qquad
        z_k\sim{\rm Gauss}(0,1)\ ,

    which has exactly the correlated Gaussian background model of
    ``default.correlated_background``, written in an uncorrelated basis.

    Args:
        path (``str``): directory the workspace is written to.
        background (``Sequence[float]``): background yields :math:`b_i`.
        signal (``Sequence[float]``): signal yields :math:`s_i`, or ``None``
          for a background-only workspace.
        data (``Sequence[float]``): observed counts :math:`n_i`.
        covariance (``Sequence[Sequence[float]]``): background covariance
          matrix :math:`\Sigma`.
        name (``str``, default ``"correlated"``): base name of the workspace.

    Returns:
        ``str``:
        Path of the ``.root`` file holding the workspace.
    """
    factor = np.linalg.cholesky(np.asarray(covariance, dtype=np.float64))
    templates = [factor[:, k] for k in range(factor.shape[1])]
    return _build(path, name, background, signal, data, templates)


def uncorrelated_background_pyhf_workspace(
    background: Sequence[float],
    signal: Sequence[float],
    data: Sequence[float],
    uncertainties: Sequence[float],
) -> Tuple[dict, List[dict]]:
    r"""
    Build the ``pyhf`` twin of ``default.uncorrelated_background``.

    One single-bin channel per bin, each carrying a ``histosys`` whose up and
    down templates are :math:`b_i\pm\sigma_i`. ``HistFactory``'s piecewise
    interpolation is linear between them, so the model is exactly

    .. math::

        \mathcal{L}(\mu,\theta) =
            \prod_i{\rm Pois}\left(n_i\vert\mu s_i + b_i + \theta_i\sigma_i\right)
            \cdot\prod_i{\rm Gauss}\left(0\vert\theta_i,1\right)\ .

    The result is split the way ``spey-pyhf`` expects it: a background-only
    workspace, and a ``JSON`` patch adding the signal sample. Handing the
    plug-in a workspace that already contains its signal together with an empty
    patch also evaluates correctly, but leaves
    :attr:`~spey.BackendBase.is_alive` ``False`` — the plug-in judges that from
    the patch — and ``spey`` then reports an infinite upper limit.

    .. attention::

        ``histosys`` is the modifier to use here, **not** ``shapesys``. The
        latter scales the background by a multiplicative :math:`\gamma` with an
        asymmetric Poisson constraint, which is a genuinely different
        likelihood — the two disagree by several log units at the same
        parameter point.

    Args:
        background (``Sequence[float]``): background yields :math:`b_i`.
        signal (``Sequence[float]``): signal yields :math:`s_i`.
        data (``Sequence[float]``): observed counts :math:`n_i`.
        uncertainties (``Sequence[float]``): absolute background uncertainties
          :math:`\sigma_i`.

    Returns:
        ``Tuple[dict, List[dict]]``:
        The background-only workspace and the signal patch, ready to be passed
        to ``spey.get_backend("pyhf")`` as ``background_only_model`` and
        ``signal_patch``.
    """
    channels, observations, patch = [], [], []
    zipped = zip(background, signal, uncertainties, data)
    for index, (nominal, yields, sigma, observed) in enumerate(zipped):
        channels.append(
            {
                "name": f"b{index}",
                "samples": [
                    {
                        "name": "bkg",
                        "data": [float(nominal)],
                        "modifiers": [
                            {
                                "name": f"theta{index}",
                                "type": "histosys",
                                "data": {
                                    "hi_data": [float(nominal) + float(sigma)],
                                    "lo_data": [float(nominal) - float(sigma)],
                                },
                            }
                        ],
                    }
                ],
            }
        )
        observations.append({"name": f"b{index}", "data": [float(observed)]})
        patch.append(
            {
                "op": "add",
                "path": f"/channels/{index}/samples/1",
                "value": {
                    "name": "sig",
                    "data": [float(yields)],
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
    return workspace, patch
