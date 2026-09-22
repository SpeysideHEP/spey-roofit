r"""
Generic RooFit model container used by the ``spey`` interface.

This module translates an arbitrary ``RooFit`` workspace — as produced by
``HistFactory``, by ``RooJSONFactoryWSTool`` from an HS3 ``JSON`` file, or by
hand — into the flat, array based description that ``spey`` expects from a
backend:

* a single ordered vector of **parameters**, :math:`(\mu, \theta_1, \dots)`,
  built from the ``RooStats::ModelConfig`` parameter of interest and the
  floating nuisance parameters;
* a single ordered vector of **data**, built from the per-bin observed counts
  of every channel followed by the auxiliary measurements, i.e. the values of
  the global observables.

The log-likelihood is evaluated by ``RooFit`` itself through
``RooAbsPdf::createNLL``. ``RooFit`` and ``spey`` do not use the same additive
normalisation for the likelihood: ``RooFit`` adds terms that depend only on the
data (the category normalisation of a ``RooSimultaneous``, bin volumes, ...),
whereas ``spey`` — like ``pyhf`` — uses

.. math::

    \log\mathcal{L}(\mu,\theta) =
        \sum_{i\in\rm bins}\log{\rm Pois}(n_i\vert\lambda_i(\mu,\theta)) +
        \sum_{c\in\rm constraints}\log p_c(a_c\vert\theta)\ .

:class:`~spey_roofit.data.WorkspaceData` therefore *calibrates* the ``RooFit``
negative log-likelihood against the expression above once per dataset. The
calibration constant is validated at two independent parameter points, so a
genuine parameter dependence — which would invalidate the procedure — is
detected and reported rather than silently absorbed.

References
~~~~~~~~~~
* RooFit manual, https://root.cern/manual/roofit/
* G. Cowan, K. Cranmer, E. Gross, O. Vitells, :xref:`1007.1727`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import gammaln

from .manager import manager

__all__ = [
    "WorkspaceData",
    "ChannelInfo",
    "ConstraintInfo",
    "resolve_channels",
    "read_counts",
]


def __dir__():
    return __all__


log = logging.getLogger("Spey")

# pylint: disable=too-many-instance-attributes, too-many-locals, invalid-name

_EXCLUDED_OBSERVABLES = frozenset({"weightVar"})
"""Names of bookkeeping variables that ``HistFactory`` stores in the observable set."""


@dataclass
class ChannelInfo:
    r"""
    Description of a single channel (``RooSimultaneous`` category) of the model.

    Args:
        name (``str``): label of the channel, i.e. the state of the index
          category, or the name of the pdf for a non-simultaneous model.
        pdf (``ROOT.RooAbsPdf``): extended pdf describing the channel.
        observables (``List[ROOT.RooRealVar]``): binned observables of the
          channel, slowest-varying first.
        observable_set (``ROOT.RooArgSet``): the same observables as a
          ``RooArgSet``, cached because ``RooFit`` forbids temporaries in
          ``getVal``.
        shape (``Tuple[int, ...]``): number of bins of each observable.
    """

    name: str
    pdf: Any
    observables: List[Any]
    observable_set: Any
    shape: Tuple[int, ...]

    @property
    def nbins(self) -> int:
        """
        Total number of bins of the channel.

        Returns:
            ``int``:
            Product of the number of bins of every observable.
        """
        return int(np.prod(self.shape)) if self.shape else 1


@dataclass
class ConstraintInfo:
    r"""
    Description of a single constraint term :math:`p_c(a_c\vert\theta)`.

    Args:
        pdf (``ROOT.RooAbsPdf``): the constraint pdf, e.g. a ``RooGaussian``
          or a ``RooPoisson``.
        global_observable (``ROOT.RooRealVar``): the auxiliary measurement
          :math:`a_c` constrained by the term.
        expectation (``ROOT.RooAbsReal``): the quantity the auxiliary
          measurement is compared against, e.g. :math:`\theta` for
          :math:`{\rm Gauss}(a\vert\theta,1)`. Setting the global observable to
          the value of this expression is what defines the Asimov value of
          :math:`a_c`.
        norm_set (``ROOT.RooArgSet``): normalisation set used when evaluating
          the constraint, i.e. ``{global_observable}``.
    """

    pdf: Any
    global_observable: Any
    expectation: Any
    norm_set: Any


@dataclass
class _NLLRecord:
    """Internal container pairing a ``RooFit`` NLL with its calibration."""

    nll: Any
    dataset: Any
    offset: float
    global_observables: np.ndarray
    residual: float = 0.0
    keep: List[Any] = field(default_factory=list)


class WorkspaceData:
    r"""
    Flat, ``spey`` friendly view of a ``RooFit`` workspace.

    The class resolves the structure of a ``RooFit`` model once — channels,
    bins, constraint terms, floating parameters — and then exposes it through
    plain ``NumPy`` arrays. It is deliberately agnostic about how the workspace
    was produced so that it can be shared between the ``RooFit`` and the
    HS3-to-``RooFit`` ``spey`` plug-ins.

    Args:
        workspace (``ROOT.RooWorkspace``): the workspace holding the model.
        model_config_name (``str``, default ``"ModelConfig"``): name of the
          ``RooStats::ModelConfig`` object stored in the workspace.
        data_name (``str``, default ``None``): name of the observed dataset. If
          ``None`` the first dataset whose name does not look like Asimov data
          is used, preferring ``"obsData"``.
        poi_name (``str``, default ``None``): name of the parameter of
          interest. If ``None`` the first entry of the model configuration's
          ``ParametersOfInterest`` is used.
        strict_cache_flush (``bool``, default ``True``): if ``True`` the
          ``RooFit`` value cache of the negative log-likelihood is invalidated
          before every evaluation. ``RooFit`` keeps the result of the last
          evaluation and, depending on the ROOT version and on whether constant
          term optimisation is active, may return a stale value after the
          parameters have been changed from Python. The flush removes that
          failure mode at a negligible cost.
        calibration_tolerance (``float``, default ``1e-6``): largest relative
          deviation between the calibration constants obtained at two different
          parameter points before a warning is issued.
        max_cached_likelihoods (``int``, default ``16``): how many ``RooFit``
          negative log-likelihood objects to keep alive. A new one is built for
          every distinct data vector, and ``spey`` produces one Asimov dataset
          per tested value of the parameter of interest, so the cache is bounded
          to keep the memory footprint of an upper-limit scan flat. Evicting an
          entry does not invalidate a likelihood function already handed out.

    Raises:
        ``ValueError``: if the workspace does not contain the requested model
          configuration, if no observed dataset can be found, or if the model
          declares no parameter of interest.

    Example:

    .. code-block:: python3

        >>> from spey_roofit import load_workspace
        >>> from spey_roofit.data import WorkspaceData
        >>> ws = load_workspace("sPlusBFit_postFit.root")
        >>> model = WorkspaceData(ws)
        >>> model.parameter_names[:2]
        ['mu_SIG', 'alpha_EG_RESOLUTION_ALL']
    """

    __slots__ = [
        "workspace",
        "model_config",
        "pdf",
        "data",
        "index_category",
        "channels",
        "constraints",
        "_constraint_terms",
        "poi",
        "nuisance_parameters",
        "global_observables",
        "_parameter_objects",
        "_global_observable_objects",
        "_nominal_parameters",
        "_nominal_global_observables",
        "_current_global_observables",
        "_nll_cache",
        "_row_template",
        "_weight_variable",
        "_dataset_variables",
        "_dataset_counter",
        "_observed_counts",
        "_strict_cache_flush",
        "_calibration_tolerance",
        "_max_cached_likelihoods",
    ]

    def __init__(
        self,
        workspace: Any,
        model_config_name: str = "ModelConfig",
        data_name: Optional[str] = None,
        poi_name: Optional[str] = None,
        strict_cache_flush: bool = True,
        calibration_tolerance: float = 1e-6,
        max_cached_likelihoods: int = 16,
    ):
        ROOT = manager.ROOT

        self._strict_cache_flush = bool(strict_cache_flush)
        self._calibration_tolerance = float(calibration_tolerance)
        self._max_cached_likelihoods = max(1, int(max_cached_likelihoods))
        self._nll_cache: Dict[bytes, _NLLRecord] = {}
        self._dataset_counter: int = 0

        self.workspace = workspace
        self.model_config = workspace.obj(model_config_name)
        if not self.model_config:
            available = [obj.GetName() for obj in workspace.allGenericObjects()]
            raise ValueError(
                f"Workspace does not contain a ModelConfig named "
                f"'{model_config_name}'. Available objects: {', '.join(available)}."
            )

        self.pdf = self.model_config.GetPdf()
        self.data = self._resolve_data(data_name)

        self.index_category = (
            self.pdf.indexCat() if self.pdf.InheritsFrom("RooSimultaneous") else None
        )

        self.channels: List[ChannelInfo] = self._resolve_channels()
        self.constraints: List[ConstraintInfo] = self._resolve_constraints()

        self.poi = self._resolve_poi(poi_name)
        self.nuisance_parameters = self._resolve_nuisance_parameters()
        # Resolved once the floating parameters are known: a constraint term is
        # only relevant if the parameter it constrains actually floats.
        self._constraint_terms = self._resolve_constraint_terms()
        self._parameter_objects = [self.poi] + list(self.nuisance_parameters)

        self.global_observables = [cst.global_observable for cst in self.constraints]
        self._global_observable_objects = list(self.global_observables)

        self._nominal_parameters = np.array(
            [par.getVal() for par in self._parameter_objects], dtype=np.float64
        )
        self._nominal_global_observables = np.array(
            [gob.getVal() for gob in self._global_observable_objects], dtype=np.float64
        )
        self._current_global_observables = self._nominal_global_observables.copy()

        # bookkeeping objects used to build datasets from a flat count vector
        self._weight_variable = ROOT.RooRealVar("_spey_weight", "event weight", 1.0)
        row = ROOT.RooArgSet()
        for channel in self.channels:
            for obs in channel.observables:
                row.add(obs)
        if self.index_category is not None:
            row.add(self.index_category)
        self._row_template = row
        dataset_vars = ROOT.RooArgSet(row)
        dataset_vars.add(self._weight_variable)
        self._dataset_variables = dataset_vars
        manager.keep_alive(self._weight_variable, self._row_template, dataset_vars)

        self._observed_counts = self._read_counts(self.data)

    # ------------------------------------------------------------------ #
    # Resolution of the workspace structure                               #
    # ------------------------------------------------------------------ #

    def _resolve_data(self, data_name: Optional[str]) -> Any:
        """
        Locate the observed dataset inside the workspace.

        Args:
            data_name (``str``): explicit dataset name, or ``None`` to search.

        Raises:
            ``ValueError``: if the requested dataset does not exist or if the
              workspace holds no dataset at all.

        Returns:
            ``ROOT.RooAbsData``:
            The observed dataset.
        """
        if data_name is not None:
            dataset = self.workspace.data(data_name)
            if not dataset:
                available = [dat.GetName() for dat in self.workspace.allData()]
                raise ValueError(
                    f"Workspace does not contain a dataset named '{data_name}'. "
                    f"Available datasets: {', '.join(available)}."
                )
            return dataset

        datasets = list(self.workspace.allData())
        if len(datasets) == 0:
            raise ValueError("Workspace does not contain any dataset.")
        for dataset in datasets:
            if dataset.GetName() == "obsData":
                return dataset
        for dataset in datasets:
            if "asimov" not in dataset.GetName().lower():
                log.debug("Using '%s' as observed dataset.", dataset.GetName())
                return dataset
        log.warning(
            "Only Asimov-like datasets found, using '%s' as observed data.",
            datasets[0].GetName(),
        )
        return datasets[0]

    def _resolve_channels(self) -> List[ChannelInfo]:
        """
        Split the model into channels and resolve the binning of each of them.

        Raises:
            ``ValueError``: if a channel has no binned observable.

        Returns:
            ``List[~spey_roofit.data.ChannelInfo]``:
            One entry per category of the simultaneous pdf.
        """
        return resolve_channels(self.model_config, self.pdf, self.index_category)

    def _resolve_constraints(self) -> List[ConstraintInfo]:
        r"""
        Identify every constraint term of the model.

        A constraint is any pdf component that has exactly one global
        observable among its direct servers. For each of them the *expectation*
        — the quantity the auxiliary measurement is compared against — is
        resolved from the ``RooFit`` proxies (``x``/``mean`` for
        :math:`{\rm Gauss}` and :math:`{\rm Pois}`, ``x``/``median`` for
        :math:`{\rm LogNormal}`) with a generic server scan as fallback.

        Returns:
            ``List[~spey_roofit.data.ConstraintInfo]``:
            Constraint descriptions ordered by the name of the global
            observable, so that the auxiliary part of the data vector has a
            reproducible layout.
        """
        ROOT = manager.ROOT

        declared = self.model_config.GetGlobalObservables()
        if not declared:
            return []
        global_observables = {gob.GetName(): gob for gob in declared}

        found: Dict[str, ConstraintInfo] = {}
        for component in self.pdf.getComponents():
            if not component.InheritsFrom("RooAbsPdf"):
                continue
            servers = list(component.servers())
            matches = [srv for srv in servers if srv.GetName() in global_observables]
            if len(matches) != 1:
                continue
            gob = global_observables[matches[0].GetName()]
            if gob.GetName() in found:
                continue
            expectation = self._resolve_expectation(component, gob, servers)
            if expectation is None:
                log.warning(
                    "Could not determine the Asimov value of global observable '%s' "
                    "from constraint '%s'; it will be kept at its nominal value.",
                    gob.GetName(),
                    component.GetName(),
                )
                expectation = gob
            norm_set = ROOT.RooArgSet(gob)
            manager.keep_alive(norm_set)
            found[gob.GetName()] = ConstraintInfo(
                pdf=component,
                global_observable=gob,
                expectation=expectation,
                norm_set=norm_set,
            )

        missing = set(global_observables) - set(found)
        if missing:
            log.warning(
                "%d global observable(s) are not attached to an identifiable "
                "constraint term and are excluded from the data vector: %s",
                len(missing),
                ", ".join(sorted(missing)[:5]) + (", ..." if len(missing) > 5 else ""),
            )
        return [found[name] for name in sorted(found)]

    @staticmethod
    def _resolve_expectation(
        constraint: Any, global_observable: Any, servers: List[Any]
    ) -> Optional[Any]:
        r"""
        Find the expression a constraint compares its auxiliary measurement to.

        Args:
            constraint (``ROOT.RooAbsPdf``): the constraint pdf.
            global_observable (``ROOT.RooRealVar``): auxiliary measurement.
            servers (``List[ROOT.RooAbsArg]``): direct servers of the constraint.

        Returns:
            ``ROOT.RooAbsReal`` or ``None``:
            The expectation expression, or ``None`` if it cannot be resolved.
        """
        name = global_observable.GetName()
        for centre in ("getMean", "getMedian"):
            if not (hasattr(constraint, "getX") and hasattr(constraint, centre)):
                continue
            x_arg = constraint.getX()
            centre_arg = getattr(constraint, centre)()
            if x_arg.GetName() == name:
                return centre_arg
            if centre_arg.GetName() == name:
                return x_arg

        candidates = [
            srv
            for srv in servers
            if srv.GetName() != name
            and srv.InheritsFrom("RooAbsReal")
            and not srv.InheritsFrom("RooConstVar")
        ]
        if not candidates:
            return None
        floating = [srv for srv in candidates if not srv.isConstant()]
        return floating[0] if floating else candidates[0]

    def _resolve_constraint_terms(self) -> List[Any]:
        r"""
        Collect every constraint factor of the model and its normalisation set.

        :meth:`_resolve_constraints` only keeps the terms attached to a
        declared global observable, because those are the ones that define the
        auxiliary data vector. The likelihood, however, also contains
        constraints on parameters for which the workspace declares no global
        observable — a floating luminosity is the typical example. Those terms
        are picked up here so that :meth:`reference_logpdf` reproduces the full
        ``RooFit`` likelihood.

        A constraint is any factor of a channel's ``RooProdPdf`` that does not
        depend on the observables of that channel.

        Returns:
            ``List[Tuple[ROOT.RooAbsPdf, ROOT.RooArgSet]]``:
            Pairs of constraint pdf and the set it has to be normalised over.
        """
        ROOT = manager.ROOT

        declared = self.model_config.GetGlobalObservables()
        global_names = {gob.GetName() for gob in declared} if declared else set()

        terms: Dict[str, Any] = {
            constraint.pdf.GetName(): (constraint.pdf, constraint.norm_set)
            for constraint in self.constraints
        }

        def visit(factor: Any, observables: Any) -> None:
            """Register ``factor`` as a constraint unless it is the main term."""
            if factor.GetName() in terms:
                return
            if factor.dependsOn(observables):
                return
            if factor.InheritsFrom("RooProdPdf"):
                for nested in factor.pdfList():
                    visit(nested, observables)
                return
            norm_set = ROOT.RooArgSet()
            for server in factor.servers():
                if server.GetName() in global_names:
                    norm_set.add(server)
            if norm_set.getSize() == 0:
                for server in factor.servers():
                    if server.InheritsFrom("RooRealVar") and not server.isConstant():
                        norm_set.add(server)
            if norm_set.getSize() == 0:
                log.debug(
                    "Ignoring pdf factor '%s': no observable to normalise over.",
                    factor.GetName(),
                )
                return
            manager.keep_alive(norm_set)
            terms[factor.GetName()] = (factor, norm_set)

        for channel in self.channels:
            if not channel.pdf.InheritsFrom("RooProdPdf"):
                continue
            for factor in channel.pdf.pdfList():
                visit(factor, channel.observable_set)
        return list(terms.values())

    def _resolve_poi(self, poi_name: Optional[str]) -> Any:
        """
        Resolve the parameter of interest.

        Args:
            poi_name (``str``): explicit name, or ``None`` to take the first
              entry of the model configuration.

        Raises:
            ``ValueError``: if the model declares no parameter of interest, or
              if ``poi_name`` is not a variable of the workspace.

        Returns:
            ``ROOT.RooRealVar``:
            The parameter of interest.
        """
        if poi_name is not None:
            poi = self.workspace.var(poi_name)
            if not poi:
                raise ValueError(f"'{poi_name}' is not a variable of the workspace.")
            poi.setConstant(False)
            return poi

        declared = self.model_config.GetParametersOfInterest()
        pois = list(declared) if declared else []
        if len(pois) == 0:
            raise ValueError(
                "The model configuration does not declare a parameter of interest. "
                "This is expected for a background-only workspace: inject a signal "
                "with spey_roofit.WorkspaceInterpreter, or pass `poi_name` "
                "explicitly."
            )
        if len(pois) > 1:
            log.warning(
                "Model declares %d parameters of interest, using '%s'.",
                len(pois),
                pois[0].GetName(),
            )
        pois[0].setConstant(False)
        return pois[0]

    def _resolve_nuisance_parameters(self) -> List[Any]:
        """
        Collect the floating nuisance parameters of the model.

        The set is taken from the parameters the negative log-likelihood
        actually depends on, intersected with the nuisance parameters declared
        in the model configuration when that set is available. Constant
        parameters are excluded: they are not seen by the optimiser.

        Returns:
            ``List[ROOT.RooRealVar]``:
            Floating nuisance parameters ordered by name.
        """
        declared = self.model_config.GetNuisanceParameters()
        candidates = list(declared) if declared else []
        if not candidates:
            observables = {obs.GetName() for obs in self.model_config.GetObservables()}
            globals_ = {
                gob.GetName() for gob in (self.model_config.GetGlobalObservables() or [])
            }
            candidates = [
                par
                for par in self.pdf.getParameters(self.data)
                if par.GetName() not in observables and par.GetName() not in globals_
            ]
        nuisances = [
            par
            for par in candidates
            if par.InheritsFrom("RooRealVar")
            and not par.isConstant()
            and par.GetName() != self.poi.GetName()
        ]
        return sorted(nuisances, key=lambda par: par.GetName())

    # ------------------------------------------------------------------ #
    # Layout                                                              #
    # ------------------------------------------------------------------ #

    @property
    def parameter_names(self) -> List[str]:
        r"""
        Names of the model parameters in the order ``spey`` expects them.

        Returns:
            ``List[str]``:
            ``[poi, theta_1, ..., theta_n]``.
        """
        return [par.GetName() for par in self._parameter_objects]

    @property
    def global_observable_names(self) -> List[str]:
        """
        Names of the global observables in auxiliary-data order.

        Returns:
            ``List[str]``:
            Global observable names, sorted alphabetically.
        """
        return [gob.GetName() for gob in self._global_observable_objects]

    @property
    def bin_map(self) -> Dict[str, int]:
        """
        Number of bins of each channel.

        Returns:
            ``Dict[str, int]``:
            Mapping from channel name to bin count.
        """
        return {channel.name: channel.nbins for channel in self.channels}

    @property
    def channel_names(self) -> List[str]:
        """
        Names of the channels in data-vector order.

        Returns:
            ``List[str]``:
            Channel names.
        """
        return [channel.name for channel in self.channels]

    @property
    def nbins(self) -> int:
        """
        Total number of bins summed over all channels.

        Returns:
            ``int``:
            Length of the main part of the data vector.
        """
        return sum(channel.nbins for channel in self.channels)

    @property
    def naux(self) -> int:
        """
        Number of auxiliary measurements.

        Returns:
            ``int``:
            Length of the auxiliary part of the data vector.
        """
        return len(self._global_observable_objects)

    @property
    def npar(self) -> int:
        """
        Number of floating parameters.

        Returns:
            ``int``:
            ``1 + len(nuisance_parameters)``.
        """
        return len(self._parameter_objects)

    @property
    def nominal_parameters(self) -> np.ndarray:
        r"""
        Parameter values found in the workspace when it was loaded.

        Returns:
            ``np.ndarray``:
            Snapshot of :math:`(\mu, \theta)` taken at construction time.
        """
        return self._nominal_parameters.copy()

    @property
    def observed_counts(self) -> np.ndarray:
        """
        Observed bin counts of the workspace's dataset.

        Returns:
            ``np.ndarray``:
            Main part of the observed data vector.
        """
        return self._observed_counts.copy()

    @property
    def observed_data(self) -> np.ndarray:
        """
        Full observed data vector, main counts followed by auxiliary data.

        Returns:
            ``np.ndarray``:
            Concatenation of the observed counts and of the nominal values of
            the global observables.
        """
        return np.concatenate([self._observed_counts, self._nominal_global_observables])

    def split(self, data: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a flat data vector into its main and auxiliary parts.

        Args:
            data (``Sequence[float]``): data vector of length
              ``nbins + naux``, or of length ``nbins`` in which case the
              auxiliary part is taken to be the nominal one.

        Raises:
            ``ValueError``: if the length of ``data`` matches neither layout.

        Returns:
            ``Tuple[np.ndarray, np.ndarray]``:
            Bin counts and auxiliary measurements.
        """
        values = np.asarray(data, dtype=np.float64).ravel()
        if values.size == self.nbins + self.naux:
            return values[: self.nbins], values[self.nbins :]
        if values.size == self.nbins:
            return values, self._nominal_global_observables.copy()
        raise ValueError(
            f"Expected a data vector of length {self.nbins + self.naux} "
            f"({self.nbins} bins + {self.naux} auxiliary measurements) "
            f"but got {values.size}."
        )

    # ------------------------------------------------------------------ #
    # Parameter handling                                                  #
    # ------------------------------------------------------------------ #

    def set_parameters(self, pars: Sequence[float]) -> None:
        r"""
        Write a parameter vector into the ``RooFit`` variables of the model.

        Args:
            pars (``Sequence[float]``): values of :math:`(\mu, \theta)` in the
              order given by :attr:`parameter_names`.

        Raises:
            ``ValueError``: if the length of ``pars`` does not match the number
              of floating parameters.
        """
        values = np.asarray(pars, dtype=np.float64).ravel()
        if values.size != len(self._parameter_objects):
            raise ValueError(
                f"Expected {len(self._parameter_objects)} parameters "
                f"but got {values.size}."
            )
        for par, value in zip(self._parameter_objects, values):
            value = float(value)
            # RooFit clamps a variable to its range without raising, which would
            # silently return the likelihood of a different parameter point.
            if value < par.getMin() or value > par.getMax():
                log.debug(
                    "Widening the range of '%s' to accommodate %g.",
                    par.GetName(),
                    value,
                )
                par.setRange(min(par.getMin(), value), max(par.getMax(), value))
            par.setVal(value)

    def reset_parameters(self) -> None:
        """Restore the parameter values the workspace was loaded with."""
        self.set_parameters(self._nominal_parameters)
        self.set_global_observables(self._nominal_global_observables)

    def set_global_observables(self, values: Sequence[float]) -> None:
        """
        Write auxiliary measurements into the global observables.

        The values currently held by the ``RooFit`` variables are cached, so
        that alternating between the observed and the Asimov likelihood only
        touches ``RooFit`` when the auxiliary data actually change.

        Args:
            values (``Sequence[float]``): auxiliary measurements in the order
              given by :attr:`global_observable_names`.

        Raises:
            ``ValueError``: if the length of ``values`` does not match the
              number of global observables.
        """
        requested = np.asarray(values, dtype=np.float64).ravel()
        if requested.size != len(self._global_observable_objects):
            raise ValueError(
                f"Expected {len(self._global_observable_objects)} auxiliary "
                f"measurements but got {requested.size}."
            )
        if np.array_equal(requested, self._current_global_observables):
            return
        for gob, value in zip(self._global_observable_objects, requested):
            gob.setVal(float(value))
        self._current_global_observables = requested.copy()

    # ------------------------------------------------------------------ #
    # Expectation values                                                  #
    # ------------------------------------------------------------------ #

    def expected_yields(self) -> np.ndarray:
        r"""
        Expected bin contents :math:`\lambda_i` at the current parameter values.

        For every channel the expectation is evaluated the way ``RooFit`` and
        ``RooStats::AsymptoticCalculator`` define it,

        .. math::

            \lambda_i = N_{\rm exp}\cdot{\rm pdf}(x_i)\cdot V_i\ ,

        with :math:`N_{\rm exp}` the number of events the extended pdf expects,
        :math:`x_i` the centre of bin :math:`i` and :math:`V_i` its volume.

        Returns:
            ``np.ndarray``:
            Expected counts of every bin of every channel.
        """
        yields = np.empty(self.nbins, dtype=np.float64)
        cursor = 0
        for channel in self.channels:
            expected_events = channel.pdf.expectedEvents(channel.observable_set)
            for index in range(channel.nbins):
                volume = self._set_bin(channel, index)
                yields[cursor] = (
                    expected_events * channel.pdf.getVal(channel.observable_set) * volume
                )
                cursor += 1
        return yields

    def expected_global_observables(self) -> np.ndarray:
        r"""
        Asimov values of the auxiliary measurements at the current parameters.

        Each auxiliary measurement is set to the value of the expression the
        corresponding constraint compares it against, e.g. :math:`a=\theta` for
        :math:`{\rm Gauss}(a\vert\theta,1)`. This is the definition used by
        ``RooStats::AsymptoticCalculator::MakeAsimovData``.

        Returns:
            ``np.ndarray``:
            Asimov auxiliary data.
        """
        return np.array(
            [cst.expectation.getVal() for cst in self.constraints], dtype=np.float64
        )

    def expected_data(self, pars: Sequence[float]) -> np.ndarray:
        r"""
        Asimov data generated at the parameter point ``pars``.

        Args:
            pars (``Sequence[float]``): values of :math:`(\mu, \theta)`.

        Returns:
            ``np.ndarray``:
            Concatenation of the expected bin contents and of the Asimov
            auxiliary measurements.
        """
        self.set_parameters(pars)
        return np.concatenate(
            [self.expected_yields(), self.expected_global_observables()]
        )

    @staticmethod
    def _set_bin(channel: ChannelInfo, flat_index: int) -> float:
        """
        Move the observables of a channel to a given bin.

        Args:
            channel (``~spey_roofit.data.ChannelInfo``): channel to act on.
            flat_index (``int``): row-major bin index, the first observable
              varying slowest.

        Returns:
            ``float``:
            Volume of the selected bin.
        """
        volume = 1.0
        remainder = flat_index
        for observable, nbins in zip(
            reversed(channel.observables), reversed(channel.shape)
        ):
            remainder, index = divmod(remainder, nbins)
            observable.setBin(int(index))
            volume *= observable.getBinWidth(int(index))
        return volume

    # ------------------------------------------------------------------ #
    # Datasets                                                            #
    # ------------------------------------------------------------------ #

    def make_dataset(self, counts: Sequence[float], name: Optional[str] = None) -> Any:
        """
        Build a weighted ``RooDataSet`` from a vector of bin counts.

        Args:
            counts (``Sequence[float]``): one entry per bin of every channel,
              in the order defined by :attr:`channel_names` and by the binning
              of each channel.
            name (``str``, default ``None``): name of the dataset. A unique
              name is generated when omitted.

        Raises:
            ``ValueError``: if the length of ``counts`` does not match the
              total number of bins.

        Returns:
            ``ROOT.RooDataSet``:
            Dataset holding one weighted entry per bin.
        """
        ROOT = manager.ROOT

        values = np.asarray(counts, dtype=np.float64).ravel()
        if values.size != self.nbins:
            raise ValueError(f"Expected {self.nbins} bin counts but got {values.size}.")

        if name is None:
            self._dataset_counter += 1
            name = f"spey_data_{self._dataset_counter}"

        dataset = ROOT.RooDataSet(
            name,
            name,
            self._dataset_variables,
            ROOT.RooFit.WeightVar(self._weight_variable),
        )
        cursor = 0
        for channel in self.channels:
            if self.index_category is not None:
                self.index_category.setLabel(channel.name)
            for index in range(channel.nbins):
                self._set_bin(channel, index)
                dataset.add(self._row_template, float(values[cursor]))
                cursor += 1
        return dataset

    def _read_counts(self, dataset: Any) -> np.ndarray:
        """
        Read the bin counts of a ``RooFit`` dataset into a flat vector.

        Args:
            dataset (``ROOT.RooAbsData``): dataset to read.

        Returns:
            ``np.ndarray``:
            Bin counts in data-vector order.
        """
        return read_counts(dataset, self.channels, self.index_category)

    # ------------------------------------------------------------------ #
    # Likelihood                                                          #
    # ------------------------------------------------------------------ #

    def reference_logpdf(self, counts: Sequence[float]) -> float:
        r"""
        Evaluate :math:`\log\mathcal{L}` in ``spey``'s normalisation directly.

        This is the expression ``spey`` and ``pyhf`` use,

        .. math::

            \log\mathcal{L}(\mu,\theta) =
                \sum_i\left[n_i\log\lambda_i - \lambda_i - \log\Gamma(n_i+1)\right]
                + \sum_c \log p_c(a_c\vert\theta)\ ,

        evaluated from the expected yields and the constraint pdfs of the
        workspace at the *current* parameter values. It is used to calibrate
        the additive offset of the ``RooFit`` negative log-likelihood and, in
        the test suite, as an independent cross-check of it.

        Args:
            counts (``Sequence[float]``): observed counts, one per bin.

        Returns:
            ``float``:
            Log-likelihood of ``counts`` at the current parameter values.
        """
        values = np.asarray(counts, dtype=np.float64).ravel()
        rates = np.clip(self.expected_yields(), 1e-20, None)
        main = float(np.sum(values * np.log(rates) - rates - gammaln(values + 1.0)))
        constraint = float(
            np.sum(
                [
                    np.log(term.getVal(norm_set))
                    for term, norm_set in self._constraint_terms
                ]
            )
        )
        return main + constraint

    def _make_nll(self, data: Sequence[float]) -> _NLLRecord:
        """
        Build — and calibrate — the ``RooFit`` NLL for a given data vector.

        Args:
            data (``Sequence[float]``): full data vector, bins followed by
              auxiliary measurements.

        Returns:
            ``~spey_roofit.data._NLLRecord``:
            The ``RooFit`` NLL together with the additive constant that maps it
            onto ``spey``'s log-likelihood normalisation.
        """
        ROOT = manager.ROOT

        counts, auxiliary = self.split(data)
        dataset = self.make_dataset(counts)

        self.set_global_observables(auxiliary)
        global_set = ROOT.RooArgSet()
        for gob in self._global_observable_objects:
            global_set.add(gob)

        nll = self.pdf.createNLL(
            dataset, ROOT.RooFit.GlobalObservables(global_set), ROOT.RooFit.Offset(False)
        )
        # Constant-term optimisation caches sub-expressions that depend only on
        # parameters flagged constant. spey changes such parameters between
        # calls, therefore the optimisation is switched off for good.
        nll.constOptimizeTestStatistic(ROOT.RooAbsArg.DeActivate)

        offsets = []
        probes = [self._nominal_parameters, self._probe_parameters()]
        for probe in probes:
            self.set_parameters(probe)
            offsets.append(nll.getVal() + self.reference_logpdf(counts))
        self.set_parameters(self._nominal_parameters)

        offset = float(offsets[0])
        scale = max(abs(offset), 1.0)
        residual = float(abs(offsets[0] - offsets[1]) / scale)
        if residual > self._calibration_tolerance:
            log.warning(
                "The offset between the RooFit and the spey log-likelihood "
                "normalisation is not constant (relative drift %.3e). Absolute "
                "likelihood values may not follow spey's convention; likelihood "
                "ratios are unaffected.",
                residual,
            )
        return _NLLRecord(
            nll=nll,
            dataset=dataset,
            offset=offset,
            global_observables=auxiliary.copy(),
            residual=residual,
            keep=[global_set],
        )

    def _probe_parameters(self) -> np.ndarray:
        r"""
        Build a second parameter point used to validate the NLL calibration.

        The point is obtained by nudging every parameter away from its nominal
        value while staying inside its bounds.

        Returns:
            ``np.ndarray``:
            Parameter vector different from :attr:`nominal_parameters`.
        """
        probe = self._nominal_parameters.copy()
        for index, par in enumerate(self._parameter_objects):
            low, high = par.getMin(), par.getMax()
            span = high - low
            shift = 0.05 * span if np.isfinite(span) and span > 0 else 0.1
            candidate = probe[index] + shift
            if candidate >= high:
                candidate = probe[index] - shift
            probe[index] = min(max(candidate, low), high)
        return probe

    def _record(self, data: Sequence[float]) -> _NLLRecord:
        """
        Return the cached NLL record for a data vector, building it if needed.

        Args:
            data (``Sequence[float]``): full data vector.

        Returns:
            ``~spey_roofit.data._NLLRecord``:
            Cached record.
        """
        values = np.asarray(data, dtype=np.float64).ravel()
        key = values.tobytes()
        record = self._nll_cache.get(key)
        if record is None:
            record = self._make_nll(values)
            while len(self._nll_cache) >= self._max_cached_likelihoods:
                self._nll_cache.pop(next(iter(self._nll_cache)))
            self._nll_cache[key] = record
        return record

    def logpdf_func(
        self, data: Optional[Sequence[float]] = None
    ) -> Callable[[np.ndarray], float]:
        r"""
        Build the callable ``spey`` uses to evaluate the log-likelihood.

        Args:
            data (``Sequence[float]``, default ``None``): data vector to fit.
              The observed data of the workspace are used when omitted.

        Returns:
            ``Callable[[np.ndarray], float]``:
            Function mapping :math:`(\mu,\theta)` onto
            :math:`\log\mathcal{L}(\mu,\theta)`.
        """
        record = self._record(self.observed_data if data is None else data)
        flush = self._strict_cache_flush

        def logpdf(pars: np.ndarray) -> float:
            """Compute the log-likelihood at the given parameter point."""
            self.set_global_observables(record.global_observables)
            self.set_parameters(pars)
            if flush:
                record.nll.setValueDirty()
            value = record.nll.getVal()
            if not np.isfinite(value):
                # RooFit flags parameter points with a vanishing or negative
                # yield; they are outside the support of the likelihood.
                return -np.inf
            return record.offset - value

        return logpdf

    def flush(self) -> None:
        """
        Invalidate every ``RooFit`` value cache held by this model.

        ``RooFit`` only recomputes a quantity when one of its servers has been
        flagged dirty. Changing a parameter from Python normally does that, but
        constant-term optimisation and cached normalisation integrals can keep
        a stale result alive. Calling this method forces the next evaluation to
        be recomputed from scratch.
        """
        ROOT = manager.ROOT
        for record in self._nll_cache.values():
            record.nll.constOptimizeTestStatistic(ROOT.RooAbsArg.DeActivate)
            record.nll.setValueDirty()
            record.nll.setShapeDirty()
        self._current_global_observables = np.full(
            len(self._global_observable_objects), np.nan
        )

    def clear_cache(self) -> None:
        """Drop every cached negative log-likelihood and dataset."""
        self._nll_cache.clear()
        self._current_global_observables = np.full(
            len(self._global_observable_objects), np.nan
        )


# ---------------------------------------------------------------------------
# Module level helpers, shared with the workspace interpreter
# ---------------------------------------------------------------------------


def resolve_channels(
    model_config: Any, pdf: Any, index_category: Optional[Any] = None
) -> List[ChannelInfo]:
    """
    Split a ``RooFit`` model into channels and resolve the binning of each one.

    Args:
        model_config (``ROOT.RooStats.ModelConfig``): model configuration whose
          observable set defines which variables are data.
        pdf (``ROOT.RooAbsPdf``): top level pdf, simultaneous or not.
        index_category (``ROOT.RooAbsCategory``, default ``None``): index
          category of the simultaneous pdf. Resolved from ``pdf`` when omitted.

    Raises:
        ``ValueError``: if a channel does not depend on any declared observable.

    Returns:
        ``List[~spey_roofit.data.ChannelInfo]``:
        One entry per category of the simultaneous pdf, ordered by category
        index, or a single entry for a non-simultaneous model.
    """
    ROOT = manager.ROOT

    if index_category is None and pdf.InheritsFrom("RooSimultaneous"):
        index_category = pdf.indexCat()

    model_observables = [
        obs
        for obs in model_config.GetObservables()
        if obs.GetName() not in _EXCLUDED_OBSERVABLES
    ]

    if index_category is None:
        pdf_map = [(pdf.GetName(), pdf)]
    else:
        states = sorted(
            ((str(state.first), int(state.second)) for state in index_category),
            key=lambda item: item[1],
        )
        pdf_map = [(name, pdf.getPdf(name)) for name, _ in states]

    channels: List[ChannelInfo] = []
    for name, channel_pdf in pdf_map:
        observables = [
            obs
            for obs in model_observables
            if obs.InheritsFrom("RooAbsRealLValue") and channel_pdf.dependsOn(obs)
        ]
        if len(observables) == 0:
            raise ValueError(
                f"Channel '{name}' does not depend on any of the observables "
                "declared in the ModelConfig."
            )
        observable_set = ROOT.RooArgSet()
        for obs in observables:
            observable_set.add(obs)
        manager.keep_alive(observable_set)
        channels.append(
            ChannelInfo(
                name=name,
                pdf=channel_pdf,
                observables=observables,
                observable_set=observable_set,
                shape=tuple(obs.getBins() for obs in observables),
            )
        )
    return channels


def read_counts(
    dataset: Any, channels: List[ChannelInfo], index_category: Optional[Any] = None
) -> np.ndarray:
    """
    Read the bin counts of a ``RooFit`` dataset into a flat vector.

    Args:
        dataset (``ROOT.RooAbsData``): dataset to read, weighted or not.
        channels (``List[~spey_roofit.data.ChannelInfo]``): channel description
          defining the layout of the output.
        index_category (``ROOT.RooAbsCategory``, default ``None``): index
          category used to assign an entry to a channel. ``None`` for a model
          with a single channel.

    Returns:
        ``np.ndarray``:
        Bin counts, channels concatenated in the order they are given.
    """
    offsets: Dict[str, int] = {}
    cursor = 0
    for channel in channels:
        offsets[channel.name] = cursor
        cursor += channel.nbins
    channel_by_name = {channel.name: channel for channel in channels}

    counts = np.zeros(cursor, dtype=np.float64)
    category_name = index_category.GetName() if index_category is not None else None
    for entry in range(dataset.numEntries()):
        row = dataset.get(entry)
        if category_name is not None:
            channel_name = str(row.find(category_name).getLabel())
        else:
            channel_name = channels[0].name
        channel = channel_by_name.get(channel_name)
        if channel is None:
            log.warning("Dataset refers to unknown channel '%s'.", channel_name)
            continue
        flat_index = 0
        for observable in channel.observables:
            value = row.find(observable.GetName()).getVal()
            flat_index = flat_index * observable.getBins() + int(
                observable.getBinning().binNumber(value)
            )
        counts[offsets[channel.name] + flat_index] += dataset.weight()
    return counts
