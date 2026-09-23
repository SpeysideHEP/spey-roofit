r"""
Builders for small HS3 documents used to validate the ``root.hs3`` backend.

The documents produced here are constructed so that they are *exactly*
equivalent to the built-in ``spey`` likelihoods, which makes it possible to
compare the two implementations number by number:

* :func:`uncorrelated_background_document` reproduces
  ``default.uncorrelated_background``. Every bin gets its own ``histosys``
  whose up and down templates are :math:`b_i\pm\sigma_i`, so that the
  piecewise-linear interpolation of ``HistFactory`` gives exactly
  :math:`\lambda_i=b_i+\theta_i\sigma_i+\mu s_i` with a unit Gaussian
  constraint on each :math:`\theta_i`.
* :func:`correlated_background_document` reproduces
  ``default.correlated_background``. The covariance matrix is factorised as
  :math:`\Sigma=LL^T` and the columns of :math:`L` are used as ``histosys``
  templates, which turns the correlated Gaussian constraint into a set of
  independent unit Gaussians without changing the model.
* :func:`poisson_document` reproduces ``default.poisson``: no modifier other
  than the signal strength, hence no constraint term and no auxiliary data.

The three builders share :func:`histfactory_document`, which assembles the
sections an HS3 document needs — ``distributions``, ``data``, ``domains``,
``likelihoods``, ``analyses`` and ``parameter_points``.

References
~~~~~~~~~~
* HS3 specification, https://hep-statistics-serialization-standard.github.io/
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "histfactory_document",
    "uncorrelated_background_document",
    "correlated_background_document",
    "poisson_document",
]


def __dir__():
    return __all__


def _axes(channel: str, nbins: int) -> List[Dict[str, Any]]:
    """
    Build the axis description of a single-observable channel.

    Args:
        channel (``str``): name of the channel.
        nbins (``int``): number of bins.

    Returns:
        ``List[Dict[str, Any]]``:
        The ``axes`` section, one unit-width bin per entry.
    """
    return [{"name": f"obs_x_{channel}", "min": 0.0, "max": float(nbins), "nbins": nbins}]


def histfactory_document(
    background: Sequence[float],
    signal: Optional[Sequence[float]],
    data: Sequence[float],
    templates: Sequence[Sequence[float]] = (),
    channel: str = "ch",
    poi: str = "mu",
    poi_bounds: Sequence[float] = (-10.0, 10.0),
) -> Dict[str, Any]:
    r"""
    Assemble a single-channel ``histfactory_dist`` HS3 document.

    Args:
        background (``Sequence[float]``): nominal background yields :math:`b_i`.
        signal (``Sequence[float]``): signal yields :math:`s_i` scaled by the
          parameter of interest, or ``None`` for a background-only document.
        data (``Sequence[float]``): observed counts :math:`n_i`.
        templates (``Sequence[Sequence[float]]``, default ``()``): one shift
          vector per nuisance parameter; the ``histosys`` up and down templates
          are :math:`b\pm t_k` and the constraint is a unit Gaussian.
        channel (``str``, default ``"ch"``): name of the channel. The
          distribution is called ``model_<channel>``.
        poi (``str``, default ``"mu"``): name of the parameter of interest.
        poi_bounds (``Sequence[float]``, default ``(-10.0, 10.0)``): range of
          the parameter of interest.

    Returns:
        ``Dict[str, Any]``:
        A complete HS3 document, ready for ``spey.get_backend("root.hs3")``.
    """
    background = np.asarray(background, dtype=float)
    axes = _axes(channel, background.size)

    modifiers: List[Dict[str, Any]] = []
    parameters: List[Dict[str, Any]] = []
    domain_axes: List[Dict[str, Any]] = []

    for index, template in enumerate(templates):
        shift = np.asarray(template, dtype=float)
        name = f"theta{index}"
        modifiers.append(
            {
                "type": "histosys",
                "name": name,
                "parameter": name,
                "constraint": "Gauss",
                "data": {
                    "hi": {"contents": (background + shift).tolist()},
                    "lo": {"contents": (background - shift).tolist()},
                },
            }
        )
        parameters.append({"name": name, "value": 0.0})
        domain_axes.append({"name": name, "min": -5.0, "max": 5.0})

    samples: List[Dict[str, Any]] = [
        {
            "name": "bkg",
            "data": {"contents": background.tolist()},
            "modifiers": modifiers,
        }
    ]
    if signal is not None:
        samples.append(
            {
                "name": "sig",
                "data": {"contents": [float(value) for value in signal]},
                "modifiers": [{"type": "normfactor", "name": poi, "parameter": poi}],
            }
        )
        parameters.insert(0, {"name": poi, "value": 1.0})
        domain_axes.insert(
            0, {"name": poi, "min": float(poi_bounds[0]), "max": float(poi_bounds[1])}
        )

    return {
        "metadata": {"hs3_version": "0.2"},
        "distributions": [
            {
                "type": "histfactory_dist",
                "name": f"model_{channel}",
                "axes": axes,
                "samples": samples,
            }
        ],
        "data": [
            {
                "name": f"obsData_{channel}",
                "type": "binned",
                "axes": axes,
                "contents": [float(value) for value in data],
            }
        ],
        "domains": [
            {"name": "default_domain", "type": "product_domain", "axes": domain_axes}
        ],
        "likelihoods": [
            {
                "name": "simPdf_obsData",
                "distributions": [f"model_{channel}"],
                "data": [f"obsData_{channel}"],
            }
        ],
        "analyses": [
            {
                "name": "simPdf_obsData",
                "likelihood": "simPdf_obsData",
                "domains": ["default_domain"],
                "parameters_of_interest": [poi] if signal is not None else [],
            }
        ],
        "parameter_points": [{"name": "default_values", "parameters": parameters}],
    }


def uncorrelated_background_document(
    background: Sequence[float],
    signal: Optional[Sequence[float]],
    data: Sequence[float],
    uncertainties: Sequence[float],
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""
    Build the HS3 twin of ``default.uncorrelated_background``.

    The resulting likelihood is

    .. math::

        \mathcal{L}(\mu,\theta) =
            \prod_i{\rm Pois}\left(n_i\vert\mu s_i + b_i + \theta_i\sigma_i\right)
            \cdot\prod_i{\rm Gauss}\left(0\vert\theta_i,1\right)\ .

    Args:
        background (``Sequence[float]``): background yields :math:`b_i`.
        signal (``Sequence[float]``): signal yields :math:`s_i`, or ``None``
          for a background-only document.
        data (``Sequence[float]``): observed counts :math:`n_i`.
        uncertainties (``Sequence[float]``): absolute background uncertainties
          :math:`\sigma_i`.
        kwargs (``Any``): forwarded to :func:`histfactory_document`.

    Returns:
        ``Dict[str, Any]``:
        The HS3 document.
    """
    nbins = len(background)
    templates = [
        np.eye(nbins)[index] * float(sigma) for index, sigma in enumerate(uncertainties)
    ]
    return histfactory_document(background, signal, data, templates, **kwargs)


def correlated_background_document(
    background: Sequence[float],
    signal: Optional[Sequence[float]],
    data: Sequence[float],
    covariance: Sequence[Sequence[float]],
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""
    Build the HS3 twin of ``default.correlated_background``.

    The covariance matrix is factorised as :math:`\Sigma=LL^T`; using the
    columns of :math:`L` as ``histosys`` templates gives

    .. math::

        \lambda_i = \mu s_i + b_i + \sum_k L_{ik}z_k\ ,\qquad
        z_k\sim{\rm Gauss}(0,1)\ ,

    which is exactly the correlated Gaussian background model of
    ``default.correlated_background``, written in an uncorrelated basis.

    Args:
        background (``Sequence[float]``): background yields :math:`b_i`.
        signal (``Sequence[float]``): signal yields :math:`s_i`, or ``None``
          for a background-only document.
        data (``Sequence[float]``): observed counts :math:`n_i`.
        covariance (``Sequence[Sequence[float]]``): background covariance
          matrix :math:`\Sigma`.
        kwargs (``Any``): forwarded to :func:`histfactory_document`.

    Returns:
        ``Dict[str, Any]``:
        The HS3 document.
    """
    factor = np.linalg.cholesky(np.asarray(covariance, dtype=np.float64))
    templates = [factor[:, column] for column in range(factor.shape[1])]
    return histfactory_document(background, signal, data, templates, **kwargs)


def poisson_document(
    background: Sequence[float],
    signal: Sequence[float],
    data: Sequence[float],
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""
    Build the HS3 twin of ``default.poisson``.

    Without a single constrained modifier the model has no nuisance parameter
    and no auxiliary measurement,

    .. math::

        \mathcal{L}(\mu) =
            \prod_i{\rm Pois}\left(n_i\vert\mu s_i + b_i\right)\ .

    Args:
        background (``Sequence[float]``): background yields :math:`b_i`.
        signal (``Sequence[float]``): signal yields :math:`s_i`.
        data (``Sequence[float]``): observed counts :math:`n_i`.
        kwargs (``Any``): forwarded to :func:`histfactory_document`.

    Returns:
        ``Dict[str, Any]``:
        The HS3 document.
    """
    return histfactory_document(background, signal, data, (), **kwargs)
