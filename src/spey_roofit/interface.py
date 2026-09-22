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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from spey.base import ModelConfig
from spey.base.backend_base import BackendBase
from spey.utils import ExpectationType

from ._version import __version__
from .data import WorkspaceData
from .helper_functions import WorkspaceInterpreter

__all__ = ["RooFitInterface"]


def __dir__():
    return __all__


log = logging.getLogger("Spey")

# pylint: disable=invalid-name, too-many-arguments, too-many-instance-attributes


class RooFitInterface(BackendBase):
    r"""
    ``spey`` backend for a ``RooFit`` workspace.

    The backend accepts either a signal-plus-background workspace, in which
    case its parameter of interest is used directly, or a background-only
    workspace together with ``signal_yields``, in which case a signal sample
    scaled by a freshly created parameter of interest is injected into the
    requested channels through
    :class:`~spey_roofit.helper_functions.WorkspaceInterpreter`.

    Args:
        workspace (``ROOT.RooWorkspace``, ``str`` or
          ~spey_roofit.helper_functions.WorkspaceInterpreter): the model. A
          ``str`` is interpreted as the path of a ``.root`` file.
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

        ``RooFit`` and ``spey`` normalise the log-likelihood differently:
        ``RooFit`` adds data dependent, parameter independent terms such as the
        category normalisation of a ``RooSimultaneous``. The backend calibrates
        that offset away once per dataset so that
        :func:`~spey.StatisticalModel.likelihood` returns the same value as any
        other ``spey`` backend for the same model.
    """

    name: str = "root.roofit"
    """Name of the backend"""
    version: str = __version__
    """Version of the backend"""
    author: str = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: str = ">=0.2.0"
    """Spey version required for the backend"""
    arXiv: List[str] = ["physics/0306116", "1009.1003"]
    """arXiv references: the RooFit toolkit and the RooStats project"""

    __slots__ = ["_interpreter", "_model", "_signal_yields", "_minimum_poi", "_template"]

    def __init__(
        self,
        workspace: Union[Any, str, WorkspaceInterpreter],
        signal_yields: Optional[Dict[str, Union[float, Sequence[float]]]] = None,
        model_config_name: str = "ModelConfig",
        data_name: Optional[str] = None,
        workspace_name: Optional[str] = None,
        poi_name: Optional[str] = None,
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
        self._signal_yields: Dict[str, List[float]] = interpreter.signal_per_channel

        self._model = WorkspaceData(
            interpreter.patch,
            model_config_name=model_config_name,
            data_name=data_name,
            poi_name=poi_name,
            strict_cache_flush=strict_cache_flush,
        )
        self._minimum_poi: Optional[float] = None
        self._template: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # ------------------------------------------------------------------ #
    # Accessors                                                           #
    # ------------------------------------------------------------------ #

    @property
    def model(self) -> WorkspaceData:
        """
        The flat ``RooFit`` model view backing the likelihood.

        Returns:
            ~spey_roofit.data.WorkspaceData:
            Container exposing channels, bins, parameters and the ``RooFit``
            negative log-likelihood.
        """
        return self._model

    @property
    def interpreter(self) -> WorkspaceInterpreter:
        """
        The interpreter that produced the patched workspace.

        Returns:
            ~spey_roofit.helper_functions.WorkspaceInterpreter:
            Interpreter holding the original workspace and the injected signal.
        """
        return self._interpreter

    @property
    def is_alive(self) -> bool:
        r"""
        Whether the model has a signal at all.

        Returns:
            ``bool``:
            ``True`` if at least one bin has a non-zero signal contribution,
            i.e. if :math:`\lambda_i(\mu=1)>\lambda_i(\mu=0)` somewhere.
        """
        return bool(np.any(self._signal_template() > 0.0))

    # ------------------------------------------------------------------ #
    # BackendBase contract                                                #
    # ------------------------------------------------------------------ #

    def config(
        self, allow_negative_signal: bool = True, poi_upper_bound: float = 10.0
    ) -> ModelConfig:
        r"""
        Model configuration.

        The parameter vector is :math:`(\mu,\theta_1,\dots,\theta_n)` where
        :math:`\mu` is the parameter of interest of the ``ModelConfig`` and
        :math:`\theta` are the floating nuisance parameters sorted by name.
        Initial values are the ones stored in the workspace — the post-fit
        values for a workspace published after a fit — and the bounds of the
        nuisance parameters are the ranges of the corresponding
        ``RooRealVar`` objects.

        Args:
            allow_negative_signal (``bool``, default ``True``): if ``True``,
              :math:`\hat\mu` is allowed to be negative, down to
              :attr:`~spey.base.model_config.ModelConfig.minimum_poi`.
            poi_upper_bound (``float``, default ``10.0``): upper bound for the
              parameter of interest, :math:`\mu`.

        Returns:
            ~spey.base.ModelConfig:
            Position of the POI in the parameter list, its minimum value, and
            the suggested initial values and bounds of every parameter.
        """
        minimum_poi = self.minimum_poi
        lower = minimum_poi if allow_negative_signal else 0.0
        if not np.isfinite(lower):
            lower = -poi_upper_bound

        initial = list(self._model.nominal_parameters)
        initial[0] = float(min(max(initial[0], lower), poi_upper_bound))

        bounds = [(float(lower), float(poi_upper_bound))]
        for parameter in self._model.nuisance_parameters:
            bounds.append((float(parameter.getMin()), float(parameter.getMax())))

        # RooFit silently clamps a variable to its range, therefore the POI has
        # to be able to hold every value the optimiser may propose.
        self._model.poi.setRange(float(lower), float(poi_upper_bound))

        return ModelConfig(
            poi_index=0,
            minimum_poi=float(minimum_poi),
            suggested_init=[float(value) for value in initial],
            suggested_bounds=bounds,
            parameter_names=self._model.parameter_names,
        )

    def get_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Callable[[np.ndarray], float]:
        r"""
        Build the function computing :math:`\log\mathcal{L}(\mu,\theta)`.

        Args:
            expected (~spey.ExpectationType): which dataset the likelihood is
              built for. Ignored when ``data`` is given.

              * :obj:`~spey.ExpectationType.observed`: the observed dataset of
                the workspace (default).
              * :obj:`~spey.ExpectationType.aposteriori`: same as ``observed``;
                the expected values are obtained by ``spey`` from the Asimov
                data generated with the post-fit nuisance parameters.
              * :obj:`~spey.ExpectationType.apriori`: the background-only
                Asimov dataset, i.e. the expectation at :math:`\mu=0` with the
                nuisance parameters at the values stored in the workspace.

            data (``Union[List[float], np.ndarray]``, default ``None``): data
              vector to fit, bin counts followed by auxiliary measurements. It
              takes precedence over ``expected`` and is how ``spey`` passes
              Asimov data back into the backend.

        Returns:
            ``Callable[[np.ndarray], float]``:
            Function mapping the parameter vector :math:`(\mu,\theta)` onto
            :math:`\log\mathcal{L}(\mu,\theta)`.
        """
        return self._model.logpdf_func(self._resolve_data(expected, data))

    def expected_data(self, pars: List[float]) -> List[float]:
        r"""
        Asimov data at a given parameter point.

        The bin counts are the expected yields :math:`\lambda_i(\mu,\theta)` of
        the ``RooFit`` model and the auxiliary measurements are set to the
        values the constraint terms compare the nuisance parameters against,
        which is the definition used by
        ``RooStats::AsymptoticCalculator::MakeAsimovData``.

        Args:
            pars (``List[float]``): parameter of interest, :math:`\mu`, and
              nuisance parameters, :math:`\theta`.

        Returns:
            ``List[float]``:
            Expected bin counts followed by the expected auxiliary data.
        """
        return self._model.expected_data(pars).tolist()

    def get_sampler(self, pars: np.ndarray) -> Callable[[int], np.ndarray]:
        r"""
        Build a pseudo-data generator, which enables ``spey``'s toy calculator.

        Bin counts are drawn from :math:`{\rm Pois}(\lambda_i(\mu,\theta))` and
        the auxiliary measurements are generated by ``RooFit`` from the
        constraint pdfs themselves, so that any constraint shape supported by
        ``RooFit`` is sampled correctly.

        Args:
            pars (``np.ndarray``): parameter values :math:`(\mu,\theta)` the
              pseudo-data are generated at.

        Returns:
            ``Callable[[int], np.ndarray]``:
            Function taking the number of samples and returning an array of
            shape ``(number_of_samples, nbins + naux)``, or ``(number_of_samples,
            nbins)`` when ``include_auxiliary`` is ``False``.
        """
        self._model.set_parameters(pars)
        rates = self._model.expected_yields()
        constraints = list(self._model.constraints)

        def sampler(number_of_samples: int, include_auxiliary: bool = True) -> np.ndarray:
            """
            Draw pseudo-data from the statistical model.

            Args:
                number_of_samples (``int``): number of samples to draw.
                include_auxiliary (``bool``, default ``True``): whether to
                  append the auxiliary measurements drawn from the constraint
                  terms.

            Returns:
                ``np.ndarray``:
                Generated samples.
            """
            main = np.random.poisson(
                np.clip(rates, 0.0, None), size=(number_of_samples, rates.size)
            ).astype(np.float64)
            if not include_auxiliary or not constraints:
                return main

            auxiliary = np.empty((number_of_samples, len(constraints)), dtype=np.float64)
            for index, constraint in enumerate(constraints):
                generated = constraint.pdf.generate(
                    constraint.norm_set, number_of_samples
                )
                name = constraint.global_observable.GetName()
                auxiliary[:, index] = [
                    generated.get(entry).find(name).getVal()
                    for entry in range(number_of_samples)
                ]
            return np.hstack([main, auxiliary])

        return sampler

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @property
    def minimum_poi(self) -> float:
        r"""
        Smallest value the parameter of interest may take.

        The bound keeps every expected yield positive,

        .. math::

            \mu_{\rm min} = -\min_i\left(
                \frac{\lambda_i(\mu=0)}{\lambda_i(\mu=1)-\lambda_i(\mu=0)}
            \right)\ ,\quad i\in\{\text{bins with signal}\}\ ,

        evaluated with the nuisance parameters at the values stored in the
        workspace.

        Returns:
            ``float``:
            The minimum value of :math:`\mu`, or ``-inf`` if the model has no
            signal at all.
        """
        if self._minimum_poi is None:
            background, signal = self._background_and_signal()
            mask = signal > 0.0
            self._minimum_poi = (
                -float(np.min(background[mask] / signal[mask]))
                if np.any(mask)
                else -np.inf
            )
            log.debug("Minimum POI set to %s", self._minimum_poi)
        return self._minimum_poi

    def _background_and_signal(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Expected background and signal templates at nominal nuisances.

        The result is cached: the templates only depend on the workspace, not
        on the parameter point the likelihood is currently evaluated at.

        Returns:
            ``Tuple[np.ndarray, np.ndarray]``:
            :math:`\lambda_i(\mu=0)` and
            :math:`\lambda_i(\mu=1)-\lambda_i(\mu=0)`.
        """
        if self._template is not None:
            return self._template
        nominal = self._model.nominal_parameters
        at_zero = nominal.copy()
        at_zero[0] = 0.0
        at_one = nominal.copy()
        at_one[0] = 1.0

        self._model.set_parameters(at_zero)
        background = self._model.expected_yields()
        self._model.set_parameters(at_one)
        signal = self._model.expected_yields() - background
        self._model.set_parameters(nominal)
        self._template = (background, signal)
        return self._template

    def _signal_template(self) -> np.ndarray:
        r"""
        Signal yields per bin at :math:`\mu=1`.

        Returns:
            ``np.ndarray``:
            :math:`\lambda_i(\mu=1)-\lambda_i(\mu=0)`.
        """
        return self._background_and_signal()[1]

    def _resolve_data(
        self,
        expected: ExpectationType,
        data: Optional[Union[List[float], np.ndarray]],
    ) -> np.ndarray:
        """
        Pick the data vector a likelihood should be built for.

        Args:
            expected (~spey.ExpectationType): requested expectation type.
            data (``Union[List[float], np.ndarray]``): explicit data vector,
              which always wins.

        Returns:
            ``np.ndarray``:
            The data vector.
        """
        if data is not None:
            return np.asarray(data, dtype=np.float64).ravel()
        if expected == ExpectationType.apriori:
            background_only = self._model.nominal_parameters
            background_only[0] = 0.0
            return self._model.expected_data(background_only)
        return self._model.observed_data
