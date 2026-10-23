r"""
``spey`` plug-in for ``RooFit`` workspaces.

The backend exposes any binned ``RooFit`` model — a ``HistFactory`` workspace
read from a ``.root`` file, a workspace built by ``RooJSONFactoryWSTool`` from
an HS3 ``JSON`` document, or a workspace assembled by hand — through the
``spey`` :class:`~spey.BackendBase` contract, so that exclusion limits,
significances and upper limits are computed by ``spey`` exactly as for any
other likelihood prescription.

The statistical model is the usual binned one,

.. math::

    \mathcal{L}(\mu,\theta) =
        \prod_{i\in\rm bins}{\rm Pois}\left(n_i\vert\lambda_i(\mu,\theta)\right)
        \cdot\prod_{c\in\rm constraints} p_c(a_c\vert\theta)\ ,

where :math:`\lambda_i` is evaluated by ``RooFit`` from the extended pdf of the
channel the bin belongs to, and :math:`a_c` are the global observables of the
``RooStats::ModelConfig``. Both the bin counts :math:`n_i` and the auxiliary
measurements :math:`a_c` are part of the data vector ``spey`` passes around, so
that Asimov datasets carry the profiled auxiliary data as they must.

References
~~~~~~~~~~
* RooFit manual, https://root.cern/manual/roofit/
* K. Cranmer et al., *HistFactory: A tool for creating statistical models for
  use with RooFit and RooStats*, CERN-OPEN-2012-016
* G. Cowan, K. Cranmer, E. Gross, O. Vitells, :xref:`1007.1727`
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from collections.abc import Sequence

import numpy as np

from ._version import __version__
from .base import RooFitBackendBase
from .data import WorkspaceData
from .helper_functions import WorkspaceInterpreter

__all__ = ["RooFitInterface"]


def __dir__():
    return __all__


log = logging.getLogger("Spey")

# pylint: disable=invalid-name, too-many-arguments, too-many-instance-attributes


class RooFitInterface(RooFitBackendBase):
    r"""
    ``spey`` backend for a ``RooFit`` workspace.

    The backend accepts either a signal-plus-background workspace, in which
    case its parameter of interest is used directly, or a background-only
    workspace together with ``signal_yields``, in which case a signal sample
    scaled by a freshly created parameter of interest is injected into the
    requested channels through
    :class:`~spey_roofit.helper_functions.WorkspaceInterpreter`.

    Args:
        workspace (``Union[RooWorkspace, str, WorkspaceInterpreter]``): the
          model. A ``str`` is interpreted as the path of a ``.root`` file.
        signal_yields (``Dict[str, Sequence[float]]``, default ``None``):
          per-channel signal yields to inject, ``{channel: [y_1, ..., y_n]}``
          with one entry per bin of the channel. Use
          :attr:`~spey_roofit.helper_functions.WorkspaceInterpreter.bin_map` to
          discover the expected shape. Nothing is injected when omitted.
        model_config_name (``str``, default ``"ModelConfig"``): name of the
          ``RooStats::ModelConfig`` object stored in the workspace.
        data_name (``str``, default ``None``): name of the observed dataset.
          Resolved automatically when omitted, preferring ``"obsData"``.
        workspace_name (``str``, default ``None``): key of the workspace inside
          the ``ROOT`` file, only used when ``workspace`` is a path.
        poi_name (``str``, default ``None``): name of the parameter of
          interest. The first entry of the model configuration is used when
          omitted.
        strip_existing_signal (``bool``, default ``False``): remove the samples
          that are already scaled by a parameter of interest before injecting
          ``signal_yields``. Use this to reinterpret a published
          signal-plus-background workspace with a different signal.
        strict_cache_flush (``bool``, default ``True``): invalidate the
          ``RooFit`` value cache before every likelihood evaluation. ``RooFit``
          caches the result of the previous evaluation and, depending on the
          ROOT version and on constant-term optimisation, can return a stale
          value after the parameters have been changed from Python.

    Raises:
        ``ValueError``: if the workspace holds no model configuration, no
          dataset, or no parameter of interest and no signal was injected.
        ``ImportError``: if ``PyROOT`` is not installed.

    Example:

    .. code-block:: python3
        :linenos:

        >>> import spey
        >>> from spey_roofit import WorkspaceInterpreter

        >>> interpreter = WorkspaceInterpreter("bOnlyFit_postFit.root")
        >>> interpreter.summary(max_channels=3)

        >>> wrapper = spey.get_backend("root.roofit")
        >>> statistical_model = wrapper(
        ...     workspace="bOnlyFit_postFit.root",
        ...     signal_yields={"SR_200_cuts": [120.0], "SR_250_cuts": [80.0]},
        ...     analysis="atlas-exot-2018-06",
        ... )
        >>> statistical_model.exclusion_confidence_level()

    .. note::

        The log-likelihood is ``RooFit``'s own, returned as :math:`-\mathrm{NLL}`
        without any rescaling. ``RooFit`` adds data dependent, parameter
        independent terms such as the category normalisation of a
        ``RooSimultaneous``, so absolute values are not comparable with those of
        another ``spey`` backend for the same model, whereas likelihood ratios,
        fits and limits are unaffected.
    """

    name: str = "root.roofit"
    """Name of the backend"""
    version: str = __version__
    """Version of the backend"""
    author: str = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: str = ">=0.2.0"
    """Spey version required for the backend"""
    arXiv: list[str] = ["physics/0306116", "1009.1003"]
    """arXiv references: the RooFit toolkit and the RooStats project"""

    __slots__ = ["_interpreter", "_signal_yields"]

    def __init__(
        self,
        workspace: Any | str | WorkspaceInterpreter,
        signal_yields: dict[str, float | Sequence[float]] | None = None,
        model_config_name: str = "ModelConfig",
        data_name: str | None = None,
        workspace_name: str | None = None,
        poi_name: str | None = None,
        strip_existing_signal: bool = False,
        strict_cache_flush: bool = True,
    ):
        if isinstance(workspace, WorkspaceInterpreter):
            interpreter = workspace
        else:
            interpreter = WorkspaceInterpreter(
                workspace,
                model_config_name=model_config_name,
                data_name=data_name,
                workspace_name=workspace_name,
            )

        if strip_existing_signal:
            interpreter.strip_signal()
        if signal_yields:
            interpreter.inject_signals(signal_yields)

        self._interpreter: WorkspaceInterpreter = interpreter
        self._signal_yields: dict[str, list[float]] = interpreter.signal_per_channel

        self._model = WorkspaceData(
            interpreter.patch,
            model_config_name=model_config_name,
            data_name=data_name,
            poi_name=poi_name,
            strict_cache_flush=strict_cache_flush,
        )
        self._minimum_poi: float | None = None
        self._template: tuple[np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ #
    # Accessors                                                           #
    # ------------------------------------------------------------------ #

    @property
    def interpreter(self) -> WorkspaceInterpreter:
        """
        The interpreter that produced the patched workspace.

        Returns:
            ~spey_roofit.helper_functions.WorkspaceInterpreter:
            Interpreter holding the original workspace and the injected signal.
        """
        return self._interpreter
