r"""
``spey`` plug-in for ``CMS Combine`` datacards.

A datacard is the input format of `Combine
<https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/>`_, the statistical
tool of the CMS collaboration, and is what CMS publishes when it publishes a
likelihood. This backend hands such a card to ``combine`` itself — its
``DatacardParser`` and its ``ShapeBuilder``, driven in process, so the
workspace is the one ``text2workspace.py`` writes — and then evaluates that
workspace with the same machinery that backs
:class:`~spey_roofit.interface.RooFitInterface` and
:class:`~spey_roofit.hs3_interface.HS3Interface`. Exclusion limits,
significances and upper limits are computed by ``spey`` exactly as for any
other likelihood prescription.

The model is the binned one ``combine`` builds,

.. math::

    \mathcal{L}(\mu,\theta) =
        \prod_{i\in\rm bins}{\rm Pois}\left(n_i\vert\lambda_i(\mu,\theta)\right)
        \cdot\prod_{e} p_e(a_e\vert\theta)\ .

Both the bin counts :math:`n_i` and the auxiliary measurements :math:`a_e` are
part of the data vector ``spey`` passes around, so that Asimov datasets carry
the profiled auxiliary data as required by :xref:`1007.1727`.

What the model looks like before it is compiled — which channels exist, how
many bins each has, what every process is expected to yield — is available from
:class:`~spey_roofit.combine.DatacardInterpreter`, which also attaches a signal
to a published background model without any ``combine`` or ``RooFit``
knowledge.

References
~~~~~~~~~~
* CMS Collaboration, *The CMS statistical analysis and combination tool:
  Combine*, :xref:`2404.06614`
* RooFit manual, https://root.cern/manual/roofit/
* G. Cowan, K. Cranmer, E. Gross, O. Vitells, :xref:`1007.1727`
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from ._version import __version__
from .base import RooFitBackendBase
from .combine import DEFAULT_SAMPLE_NAME, DatacardInterpreter
from .data import WorkspaceData

__all__ = ["CombineInterface"]


def __dir__():
    return __all__


log = logging.getLogger("Spey")

# pylint: disable=invalid-name, too-many-arguments, too-many-instance-attributes


class CombineInterface(RooFitBackendBase):
    r"""
    ``spey`` backend for a ``CMS Combine`` datacard.

    The backend accepts a card that already contains its signal, in which case
    the parameter of interest of ``combine``'s physics model is used directly,
    or a background-only card together with ``signal_yields``, in which case an
    extra process scaled by the parameter of interest is added to the requested
    channels. A published signal-plus-background card can be reinterpreted by
    combining ``strip_existing_signal`` with ``signal_yields``; in every case
    the model is compiled by ``combine``.

    Args:
        datacard (``Union[str, os.PathLike,
          ~spey_roofit.combine.DatacardInterpreter]``): the model. A ``str`` is
          read as the path of a ``.txt`` datacard, or as the card itself when
          it is not a path. An interpreter is used as it is, together with
          whatever mutations have already been queued on it.
        signal_yields (``Dict[str, Union[float, Sequence[float]]]``, default
          ``None``): per-channel signal yields to inject, ``{channel: [y_1,
          ..., y_n]}`` with one entry per bin of the channel. Use
          :attr:`~spey_roofit.combine.DatacardInterpreter.bin_map` to discover
          the expected shape. Nothing is injected when omitted.
        mass (``Union[int, float]``, default ``0.0``): value substituted for
          ``$MASS`` in the ``shapes`` directives, i.e. the ``-m`` option of
          ``text2workspace.py``.
        physics_model (``str``, default ``None``): the ``-P`` option of
          ``text2workspace.py``, in the form ``"module:object"``. ``combine``'s
          default model — every signal process scaled by :math:`r`, every
          background by one — is used when omitted. A two hypothesis card such
          as CMS-EXO-24-011 asks for
          ``"HiggsAnalysis.CombinedLimit.HiggsJPC:twoHypothesisHiggs"``, whose
          parameter of interest interpolates between the two signal processes;
          ``spey``'s background is then the :math:`x=0` hypothesis and its
          signal the difference between the two.
        physics_options (``Sequence[str]``, default ``()``): the ``--PO``
          options of the physics model.
        sample_name (``str``, default ``"spey_signal"``): name of the process
          created for ``signal_yields``.
        strip_existing_signal (``bool``, default ``False``): remove the
          processes the card declares as signal before injecting
          ``signal_yields``.
        directory (``str``, default ``None``): directory the relative paths of
          the ``shapes`` directives are resolved against. The directory holding
          the card is used when it is given as a path.
        options (``optparse.Values``, default ``None``): further options for
          ``combine``, see :func:`~spey_roofit.combine.combine_options`.
        data_name (``str``, default ``None``): name of the observed dataset.
          ``combine``'s ``"data_obs"`` is used when omitted.
        strict_cache_flush (``bool``, default ``True``): invalidate the
          ``RooFit`` value cache before every likelihood evaluation.

    Raises:
        ``ValueError``: if the card is malformed or if the signal cannot be
          attached to it.
        ``ImportError``: if ``combine`` or ``PyROOT`` is not installed.
        ``RuntimeError``: propagated from ``combine`` when the card cannot be
          compiled.

    Example:

    .. code-block:: python3
        :linenos:

        >>> import spey
        >>> from spey_roofit import DatacardInterpreter

        >>> interpreter = DatacardInterpreter("chi_datacard.txt")
        >>> interpreter.bin_map            # which channels and how many bins
        >>> interpreter.summary(max_channels=3)

        >>> wrapper = spey.get_backend("root.cms-combine")
        >>> statistical_model = wrapper(
        ...     datacard="chi_datacard.txt",
        ...     physics_model="HiggsAnalysis.CombinedLimit.HiggsJPC:twoHypothesisHiggs",
        ...     analysis="cms-exo-24-011",
        ... )
        >>> statistical_model.exclusion_confidence_level()

    .. note::

        ``combine`` writes its Gaussian constraints without the
        :math:`1/\sqrt{2\pi}` of each term, so absolute log-likelihood values
        follow ``combine``'s normalisation rather than ``spey``'s. The offset
        is constant and is calibrated away once per dataset, and every
        likelihood ratio is unaffected in any case.
    """

    name: str = "root.cms-combine"
    """Name of the backend"""
    version: str = __version__
    """Version of the backend"""
    author: str = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: str = ">=0.2.0"
    """Spey version required for the backend"""
    arXiv: list[str] = ["2404.06614", "physics/0306116"]
    """arXiv references: the Combine tool and the RooFit toolkit"""

    __slots__ = ["_interpreter", "_workspace", "_signal_yields"]

    def __init__(
        self,
        datacard: str | os.PathLike | DatacardInterpreter,
        signal_yields: dict[str, float | Sequence[float]] | None = None,
        mass: int | float = 0.0,
        physics_model: str | None = None,
        physics_options: Sequence[str] = (),
        sample_name: str = DEFAULT_SAMPLE_NAME,
        strip_existing_signal: bool = False,
        directory: str | None = None,
        options: Any | None = None,
        data_name: str | None = None,
        strict_cache_flush: bool = True,
    ):
        if isinstance(datacard, DatacardInterpreter):
            interpreter = datacard
        else:
            interpreter = DatacardInterpreter(
                datacard, mass=mass, directory=directory, options=options
            )

        if strip_existing_signal:
            interpreter.strip_signal()
        if signal_yields:
            interpreter.inject_signals(signal_yields)

        self._interpreter: DatacardInterpreter = interpreter
        self._signal_yields: dict[str, list[float]] = interpreter.signal_per_channel

        workspace, model_config = interpreter.workspace(
            physics_model=physics_model,
            physics_options=physics_options,
            options=options,
        )
        self._workspace = workspace
        self._model = WorkspaceData(
            workspace,
            model_config_name=model_config.GetName(),
            data_name=data_name,
            strict_cache_flush=strict_cache_flush,
        )
        self._minimum_poi: float | None = None
        self._template: tuple[np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ #
    # Accessors                                                           #
    # ------------------------------------------------------------------ #

    @property
    def interpreter(self) -> DatacardInterpreter:
        """
        The interpreter that produced the compiled workspace.

        Returns:
            ~spey_roofit.combine.DatacardInterpreter:
            Interpreter holding the datacard, its templates and the injected
            signal.
        """
        return self._interpreter

    @property
    def datacard(self) -> Any:
        """
        The card as ``combine`` parsed it.

        Returns:
            ``HiggsAnalysis.CombinedLimit.Datacard.Datacard``:
            ``combine``'s own container.
        """
        return self._interpreter.datacard

    @property
    def workspace(self) -> Any:
        """
        The ``RooFit`` workspace ``combine`` built.

        Returns:
            ``ROOT.RooWorkspace``:
            Workspace holding the model, its data and its model configuration.
        """
        return self._workspace
