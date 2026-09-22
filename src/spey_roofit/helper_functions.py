r"""
Inspection and signal-editing utilities for ``RooFit`` workspaces.

A ``RooFit`` workspace is a graph of ``RooAbsArg`` objects, which makes it
awkward to answer the questions a reinterpretation study actually asks: *which
regions does this analysis have, how many bins does each of them have, what is
the expected background, and how do I put my own signal in?*

:class:`~spey_roofit.helper_functions.WorkspaceInterpreter` answers those
questions and produces a new workspace with a user supplied signal attached to
the parameter of interest, without requiring any knowledge of ``RooFit``
internals. It is the ``RooFit`` counterpart of the ``spey-pyhf`` and
``spey-hs3`` interpreters.

The interpreter assumes the ``HistFactory`` layout that both ``HistFactory``
itself and ``RooJSONFactoryWSTool`` — the HS3 reader — produce:

.. math::

    {\rm simPdf} = \prod_{c\in\rm channels}
        \left[\underbrace{\sum_j \nu_j^c(\theta)\,f_j^c}_{\texttt{RooRealSumPdf}}
        \cdot \prod_k p_k(a_k\vert\theta)\right]

Signal injection adds one term :math:`\mu\,s^c` to the sum of a channel, which
leaves every other component of the workspace untouched.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from .data import ChannelInfo, read_counts, resolve_channels
from .manager import manager

__all__ = ["WorkspaceInterpreter", "load_workspace"]


def __dir__():
    return __all__


log = logging.getLogger("Spey")

# pylint: disable=invalid-name, too-many-instance-attributes, too-many-locals

_SAMPLE_DECORATIONS = (
    "_overallSyst_x_StatUncert",
    "_overallSyst_x_HistSyst",
    "_overallSyst_x_Exp",
    "_overallSyst",
    "_StatUncert",
    "_HistSyst",
    "_Exp",
    "_shapes",
)
"""Suffixes ``HistFactory`` appends to the sample functions of a channel."""

_SIGNAL_REGION_HINTS = ("sr", "signal")
_CONTROL_REGION_HINTS = ("cr", "control")
_VALIDATION_REGION_HINTS = ("vr", "validation")


def load_workspace(path: str, workspace_name: Optional[str] = None) -> Any:
    """
    Read a ``RooWorkspace`` from a ``ROOT`` file.

    Args:
        path (``str``): path to the ``.root`` file.
        workspace_name (``str``, default ``None``): key of the workspace inside
          the file. The first — and normally only — ``RooWorkspace`` of the
          file is used when omitted.

    Raises:
        ``IOError``: if the file cannot be opened.
        ``ValueError``: if the file contains no workspace, or not the requested
          one.

    Returns:
        ``ROOT.RooWorkspace``:
        The workspace. The underlying ``TFile`` is kept alive for the lifetime
        of the process.

    Example:

    .. code-block:: python3

        >>> from spey_roofit import load_workspace
        >>> workspace = load_workspace("bOnlyFit_postFit.root")
    """
    ROOT = manager.ROOT

    handle = ROOT.TFile.Open(path)
    if not handle or handle.IsZombie():
        raise IOError(f"Cannot open ROOT file '{path}'.")
    manager.keep_alive(handle)

    keys = [
        key.GetName()
        for key in handle.GetListOfKeys()
        if key.GetClassName() == "RooWorkspace"
    ]
    if workspace_name is None:
        if len(keys) == 0:
            raise ValueError(f"'{path}' does not contain a RooWorkspace.")
        if len(keys) > 1:
            log.warning(
                "File contains %d workspaces (%s), using '%s'.",
                len(keys),
                ", ".join(keys),
                keys[0],
            )
        workspace_name = keys[0]
    elif workspace_name not in keys:
        raise ValueError(
            f"'{path}' does not contain a RooWorkspace named '{workspace_name}'. "
            f"Available workspaces: {', '.join(keys) if keys else 'none'}."
        )

    workspace = handle.Get(workspace_name)
    manager.keep_alive(workspace)
    return workspace


def _ws_import(workspace: Any, obj: Any, *commands: Any) -> None:
    """
    Import an object into a workspace.

    ``import`` is a Python keyword, therefore ``RooWorkspace::import`` is not
    directly reachable. Recent ``PyROOT`` versions expose it as ``Import``;
    this helper works with either spelling.

    Args:
        workspace (``ROOT.RooWorkspace``): destination workspace.
        obj (``Any``): object to import.
        commands (``Any``): ``RooCmdArg`` options forwarded to ``RooFit``.
    """
    importer = getattr(workspace, "Import", None)
    if importer is None:  # pragma: no cover - older PyROOT
        importer = getattr(workspace, "import")
    importer(obj, *commands)


class WorkspaceInterpreter:
    r"""
    Inspect a ``RooFit`` workspace and attach a user defined signal to it.

    The interpreter never modifies the workspace it is given. Signal yields are
    recorded per channel and materialised on demand by :attr:`patch`, which
    returns a **new** workspace in which every requested channel carries an
    extra sample :math:`\mu\,s^c_i` and whose ``ModelConfig`` declares
    :math:`\mu` as the parameter of interest.

    Args:
        workspace (``ROOT.RooWorkspace`` or ``str``): the workspace, or the
          path to a ``.root`` file containing one.
        model_config_name (``str``, default ``"ModelConfig"``): name of the
          ``RooStats::ModelConfig`` object stored in the workspace.
        data_name (``str``, default ``None``): name of the observed dataset.
          Resolved automatically when omitted, preferring ``"obsData"``.
        workspace_name (``str``, default ``None``): key of the workspace inside
          the file, only used when ``workspace`` is a path.

    Raises:
        ``ValueError``: if the workspace does not contain the requested model
          configuration.

    Example:

    .. code-block:: python3

        >>> from spey_roofit import WorkspaceInterpreter
        >>> interpreter = WorkspaceInterpreter("bOnlyFit_postFit.root")
        >>> interpreter.summary(max_channels=5)
        >>> interpreter.inject_signal("SR_200_cuts", [12.0])
        >>> patched = interpreter.patch

    .. note::

        Channels are reported in the order of the index category of the
        simultaneous pdf, which is the order used for the flat data vectors of
        :class:`~spey_roofit.data.WorkspaceData`.
    """

    __slots__ = [
        "_workspace",
        "_model_config",
        "_pdf",
        "_data",
        "_index_category",
        "_channels",
        "_signal",
        "_strip_signal",
        "_poi_name",
        "_patch_counter",
    ]

    def __init__(
        self,
        workspace: Union[Any, str],
        model_config_name: str = "ModelConfig",
        data_name: Optional[str] = None,
        workspace_name: Optional[str] = None,
    ):
        if isinstance(workspace, str):
            workspace = load_workspace(workspace, workspace_name)

        self._workspace = workspace
        self._model_config = workspace.obj(model_config_name)
        if not self._model_config:
            available = [obj.GetName() for obj in workspace.allGenericObjects()]
            raise ValueError(
                f"Workspace does not contain a ModelConfig named "
                f"'{model_config_name}'. Available objects: {', '.join(available)}."
            )
        self._pdf = self._model_config.GetPdf()
        self._index_category = (
            self._pdf.indexCat() if self._pdf.InheritsFrom("RooSimultaneous") else None
        )
        self._channels = resolve_channels(
            self._model_config, self._pdf, self._index_category
        )
        self._data = self._find_data(data_name)

        self._signal: Dict[str, np.ndarray] = {}
        self._strip_signal: bool = False
        self._poi_name: Optional[str] = None
        self._patch_counter: int = 0

    def __repr__(self) -> str:
        return (
            f"WorkspaceInterpreter(channels={len(self._channels)}, "
            f"bins={sum(ch.nbins for ch in self._channels)}, "
            f"poi={self.poi_names if self.poi_names else 'none'}, "
            f"injected={list(self._signal)})"
        )

    def __getitem__(self, item: str) -> List[float]:
        """Signal yields injected into channel ``item``."""
        return list(self._signal[item])

    def _find_data(self, data_name: Optional[str]) -> Optional[Any]:
        """
        Resolve the observed dataset of the workspace.

        Args:
            data_name (``str``): explicit dataset name or ``None``.

        Returns:
            ``ROOT.RooAbsData`` or ``None``:
            The observed dataset, or ``None`` if the workspace holds none.
        """
        if data_name is not None:
            return self._workspace.data(data_name)
        datasets = list(self._workspace.allData())
        for dataset in datasets:
            if dataset.GetName() == "obsData":
                return dataset
        for dataset in datasets:
            if "asimov" not in dataset.GetName().lower():
                return dataset
        return datasets[0] if datasets else None

    # ------------------------------------------------------------------ #
    # Inspection                                                          #
    # ------------------------------------------------------------------ #

    @property
    def workspace(self) -> Any:
        """
        The unmodified workspace the interpreter was built from.

        Returns:
            ``ROOT.RooWorkspace``:
            Input workspace.
        """
        return self._workspace

    @property
    def channels(self) -> List[str]:
        """
        Names of the channels, i.e. the analysis regions.

        Returns:
            ``List[str]``:
            Channel names in index-category order.
        """
        return [channel.name for channel in self._channels]

    @property
    def bin_map(self) -> Dict[str, int]:
        """
        Number of bins of every channel.

        Returns:
            ``Dict[str, int]``:
            Mapping from channel name to bin count. This is the shape the
            signal yields passed to :meth:`inject_signal` must have.
        """
        return {channel.name: channel.nbins for channel in self._channels}

    @property
    def poi_names(self) -> List[str]:
        """
        Parameters of interest declared by the model configuration.

        Returns:
            ``List[str]``:
            Names of the parameters of interest, empty for a background-only
            workspace.
        """
        declared = self._model_config.GetParametersOfInterest()
        return [poi.GetName() for poi in declared] if declared else []

    @property
    def nuisance_parameters(self) -> List[str]:
        """
        Nuisance parameters declared by the model configuration.

        Returns:
            ``List[str]``:
            Nuisance parameter names, sorted alphabetically.
        """
        declared = self._model_config.GetNuisanceParameters()
        return sorted(par.GetName() for par in declared) if declared else []

    @property
    def global_observables(self) -> List[str]:
        """
        Global observables, i.e. the auxiliary measurements, of the model.

        Returns:
            ``List[str]``:
            Global observable names, sorted alphabetically.
        """
        declared = self._model_config.GetGlobalObservables()
        return sorted(gob.GetName() for gob in declared) if declared else []

    @property
    def observed_data(self) -> Dict[str, List[float]]:
        """
        Observed counts of every channel.

        Returns:
            ``Dict[str, List[float]]``:
            Mapping from channel name to the observed counts of its bins.
        """
        if self._data is None:
            return {}
        counts = read_counts(self._data, self._channels, self._index_category)
        return self._unflatten(counts)

    @property
    def samples(self) -> Dict[str, List[str]]:
        """
        Names of the samples contributing to every channel.

        Returns:
            ``Dict[str, List[str]]``:
            Mapping from channel name to sample names, with the ``HistFactory``
            decorations stripped from the underlying function names.
        """
        return {
            channel.name: [
                self._sample_name(func.GetName(), channel.name)
                for func, _ in self._sample_terms(channel)
            ]
            for channel in self._channels
        }

    @staticmethod
    def _is_signal(func: Any, coefficient: Any, pois: Optional[Any]) -> bool:
        """
        Decide whether a sample of a channel is scaled by a parameter of interest.

        ``HistFactory`` has used two layouts over time: older versions put the
        normalisation factors inside the sample *function* and use the
        coefficients only for the bin widths, newer ones put them into the
        *coefficient*. Both are covered by testing the two together.

        Args:
            func (``ROOT.RooAbsReal``): sample shape function.
            coefficient (``ROOT.RooAbsReal``): coefficient of the sample.
            pois (``ROOT.RooArgSet``): parameters of interest, or ``None``.

        Returns:
            ``bool``:
            ``True`` if the sample depends on a parameter of interest.
        """
        if pois is None:
            return False
        return bool(func.dependsOn(pois) or coefficient.dependsOn(pois))

    @property
    def signal_samples(self) -> Dict[str, List[str]]:
        """
        Samples that are scaled by a parameter of interest.

        Returns:
            ``Dict[str, List[str]]``:
            Mapping from channel name to the names of its signal samples. Empty
            lists for a background-only workspace.
        """
        pois = self._poi_arg_set()
        result: Dict[str, List[str]] = {}
        for channel in self._channels:
            names = []
            for func, coefficient in self._sample_terms(channel):
                if self._is_signal(func, coefficient, pois):
                    names.append(self._sample_name(func.GetName(), channel.name))
            result[channel.name] = names
        return result

    def sample_yields(
        self, channel: Optional[str] = None
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Expected yields of every sample, evaluated at the current parameters.

        Args:
            channel (``str``, default ``None``): restrict the output to a
              single channel. All channels are returned when omitted.

        Raises:
            ``KeyError``: if ``channel`` is not a channel of the workspace.

        Returns:
            ``Dict[str, Dict[str, List[float]]]``:
            Mapping ``channel -> sample -> per-bin yields``.
        """
        selection = self._select_channels(channel)
        result: Dict[str, Dict[str, List[float]]] = {}
        for info in selection:
            per_sample: Dict[str, List[float]] = {}
            for func, coefficient in self._sample_terms(info):
                name = self._sample_name(func.GetName(), info.name)
                per_sample[name] = self._evaluate_sample(info, func, coefficient)
            result[info.name] = per_sample
        return result

    @property
    def expected_background_yields(self) -> Dict[str, List[float]]:
        """
        Expected background, i.e. the sum of every non-signal sample.

        Returns:
            ``Dict[str, List[float]]``:
            Mapping from channel name to per-bin background yields at the
            parameter values currently stored in the workspace.
        """
        pois = self._poi_arg_set()
        result: Dict[str, List[float]] = {}
        for info in self._channels:
            total = np.zeros(info.nbins, dtype=np.float64)
            for func, coefficient in self._sample_terms(info):
                if self._is_signal(func, coefficient, pois):
                    continue
                total += np.array(self._evaluate_sample(info, func, coefficient))
            result[info.name] = total.tolist()
        return result

    @property
    def expected_signal_yields(self) -> Dict[str, List[float]]:
        r"""
        Expected signal already present in the workspace.

        The yields are evaluated at the parameter values stored in the
        workspace, i.e. at the current value of :math:`\mu` and not necessarily
        at :math:`\mu=1`.

        Returns:
            ``Dict[str, List[float]]``:
            Mapping from channel name to per-bin signal yields. Zeros for a
            background-only workspace.
        """
        pois = self._poi_arg_set()
        result: Dict[str, List[float]] = {}
        for info in self._channels:
            total = np.zeros(info.nbins, dtype=np.float64)
            for func, coefficient in self._sample_terms(info):
                if self._is_signal(func, coefficient, pois):
                    total += np.array(self._evaluate_sample(info, func, coefficient))
            result[info.name] = total.tolist()
        return result

    def guess_channel_type(self, channel_name: str) -> str:
        """
        Guess whether a channel is a signal, control or validation region.

        The guess is based on the channel name only: a name containing ``SR``
        or ``signal`` is taken to be a signal region, ``CR`` or ``control`` a
        control region and ``VR`` or ``validation`` a validation region.

        Args:
            channel_name (``str``): name of the channel.

        Returns:
            ``str``:
            One of ``"SR"``, ``"CR"``, ``"VR"`` or ``"__unknown__"``.
        """
        lowered = channel_name.lower()
        for hints, label in (
            (_CONTROL_REGION_HINTS, "CR"),
            (_VALIDATION_REGION_HINTS, "VR"),
            (_SIGNAL_REGION_HINTS, "SR"),
        ):
            if any(hint in lowered for hint in hints):
                return label
        return "__unknown__"

    def guess_signal_regions(self) -> List[str]:
        """
        Channels whose name suggests they are signal regions.

        Returns:
            ``List[str]``:
            Channel names classified as ``"SR"`` by :meth:`guess_channel_type`.
        """
        return [name for name in self.channels if self.guess_channel_type(name) == "SR"]

    def guess_control_regions(self) -> List[str]:
        """
        Channels whose name suggests they are control or validation regions.

        Returns:
            ``List[str]``:
            Channel names classified as ``"CR"`` or ``"VR"``.
        """
        return [
            name
            for name in self.channels
            if self.guess_channel_type(name) in ("CR", "VR")
        ]

    def summary(
        self,
        max_channels: Optional[int] = None,
        show_samples: bool = True,
        print_out: bool = True,
    ) -> str:
        """
        Human readable overview of the region structure of the workspace.

        Args:
            max_channels (``int``, default ``None``): only describe the first
              ``max_channels`` channels. All of them are shown when omitted.
            show_samples (``bool``, default ``True``): list the samples of each
              channel and their expected yields.
            print_out (``bool``, default ``True``): print the summary in
              addition to returning it.

        Returns:
            ``str``:
            The summary text.
        """
        background = self.expected_background_yields
        signal = self.expected_signal_yields
        observed = self.observed_data
        signal_samples = self.signal_samples
        yields = self.sample_yields() if show_samples else {}

        lines = [
            f"RooFit workspace '{self._workspace.GetName()}'",
            f"  channels            : {len(self._channels)}",
            f"  bins                : {sum(ch.nbins for ch in self._channels)}",
            f"  parameters of interest : "
            f"{', '.join(self.poi_names) if self.poi_names else 'none (background only)'}",
            f"  nuisance parameters : {len(self.nuisance_parameters)}",
            f"  global observables  : {len(self.global_observables)}",
            "",
        ]
        selection = (
            self._channels if max_channels is None else self._channels[:max_channels]
        )
        for info in selection:
            kind = self.guess_channel_type(info.name)
            lines.append(
                f"  [{kind}] {info.name} ({info.nbins} bin"
                f"{'s' if info.nbins > 1 else ''})"
            )
            lines.append(f"      observed   : {self._fmt(observed.get(info.name, []))}")
            lines.append(f"      background : {self._fmt(background[info.name])}")
            if any(value != 0.0 for value in signal[info.name]):
                lines.append(f"      signal     : {self._fmt(signal[info.name])}")
            if info.name in self._signal:
                lines.append(f"      injected   : {self._fmt(self._signal[info.name])}")
            if show_samples:
                for name, values in yields[info.name].items():
                    tag = " (signal)" if name in signal_samples[info.name] else ""
                    lines.append(f"        - {name}{tag}: {self._fmt(values)}")
        if max_channels is not None and len(self._channels) > max_channels:
            lines.append(f"  ... {len(self._channels) - max_channels} more channel(s)")

        text = "\n".join(lines)
        if print_out:
            print(text)
        return text

    # ------------------------------------------------------------------ #
    # Signal editing                                                      #
    # ------------------------------------------------------------------ #

    def inject_signal(
        self, channel: str, yields: Union[float, Sequence[float]]
    ) -> "WorkspaceInterpreter":
        r"""
        Record signal yields to be added to a channel.

        The signal enters the likelihood as one extra term :math:`\mu\,s_i` in
        the sum over samples of the channel, i.e. it is scaled by the parameter
        of interest and is not affected by any nuisance parameter.

        Args:
            channel (``str``): name of the channel, see :attr:`channels`.
            yields (``float`` or ``Sequence[float]``): signal yields, one per
              bin of the channel. A scalar is accepted for a single bin channel.

        Raises:
            ``ValueError``: if ``channel`` is unknown, if the number of yields
              does not match the number of bins of the channel, or if the
              channel already has signal injected.

        Returns:
            ~spey_roofit.helper_functions.WorkspaceInterpreter:
            ``self``, so that calls can be chained.
        """
        info = self._channel(channel)
        values = np.atleast_1d(np.asarray(yields, dtype=np.float64)).ravel()
        if values.size != info.nbins:
            raise ValueError(
                f"Channel '{channel}' has {info.nbins} bin(s) but "
                f"{values.size} signal yield(s) were given."
            )
        if channel in self._signal:
            raise ValueError(
                f"Channel '{channel}' already has an injected signal. "
                "Call `reset_signal()` first."
            )
        self._signal[channel] = values
        return self

    def inject_signals(
        self, signal_yields: Dict[str, Union[float, Sequence[float]]]
    ) -> "WorkspaceInterpreter":
        """
        Record signal yields for several channels at once.

        Args:
            signal_yields (``Dict[str, Sequence[float]]``): mapping from channel
              name to per-bin signal yields.

        Raises:
            ``ValueError``: propagated from :meth:`inject_signal`.

        Returns:
            ~spey_roofit.helper_functions.WorkspaceInterpreter:
            ``self``, so that calls can be chained.
        """
        for channel, yields in signal_yields.items():
            self.inject_signal(channel, yields)
        return self

    def reset_signal(self) -> "WorkspaceInterpreter":
        """
        Forget every injected signal.

        Returns:
            ~spey_roofit.helper_functions.WorkspaceInterpreter:
            ``self``, so that calls can be chained.
        """
        self._signal.clear()
        return self

    def strip_signal(self, strip: bool = True) -> "WorkspaceInterpreter":
        """
        Drop the signal samples already present in the workspace.

        Use this when starting from a signal-plus-background workspace but
        wanting to study a different signal hypothesis: the existing samples
        that are scaled by a parameter of interest are removed from the patched
        workspace, leaving a background-only model to inject into.

        Args:
            strip (``bool``, default ``True``): whether to remove the existing
              signal samples.

        Returns:
            ~spey_roofit.helper_functions.WorkspaceInterpreter:
            ``self``, so that calls can be chained.
        """
        self._strip_signal = bool(strip)
        return self

    @property
    def signal_per_channel(self) -> Dict[str, List[float]]:
        """
        Signal yields recorded so far.

        Returns:
            ``Dict[str, List[float]]``:
            Mapping from channel name to the per-bin yields passed to
            :meth:`inject_signal`.
        """
        return {name: values.tolist() for name, values in self._signal.items()}

    @property
    def is_alive(self) -> bool:
        """
        Whether the patched model has a non-vanishing signal.

        Returns:
            ``bool``:
            ``True`` if at least one injected or pre-existing signal yield is
            non-zero.
        """
        if any(np.any(values != 0.0) for values in self._signal.values()):
            return True
        if self._strip_signal:
            return False
        return any(
            any(value != 0.0 for value in values)
            for values in self.expected_signal_yields.values()
        )

    @property
    def patch(self) -> Any:
        r"""
        Build a new workspace containing the injected signal.

        The simultaneous pdf is rebuilt channel by channel: the
        ``RooRealSumPdf`` of every channel that receives a signal is replaced by
        one that carries the extra term :math:`\mu\,s_i`, every other component
        — constraints, nuisance parameters, observables, datasets — is reused
        unchanged. The ``ModelConfig`` of the new workspace declares
        :math:`\mu` as its parameter of interest.

        When nothing has been injected and no signal has to be stripped, the
        input workspace is returned as it is.

        Raises:
            ``ValueError``: if a channel that should receive a signal does not
              have the expected ``RooRealSumPdf`` structure.

        Returns:
            ``ROOT.RooWorkspace``:
            The patched workspace.

        .. note::

            The returned workspace is a new object; the workspace passed to the
            constructor is never modified. Parameter values are carried over —
            the variables are cloned as they stand — but ``RooWorkspace``
            snapshots are not, since ``spey`` reads the parameter values from
            the variables themselves.
        """
        if not self._signal and not self._strip_signal:
            return self._workspace
        return self._build_patch()

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _build_patch(self) -> Any:
        """
        Materialise the recorded signal into a new workspace.

        Returns:
            ``ROOT.RooWorkspace``:
            Workspace with the signal attached, see :attr:`patch`.
        """
        ROOT = manager.ROOT

        self._patch_counter += 1
        suffix = f"_spey{self._patch_counter}" if self._patch_counter > 1 else "_spey"

        poi = self._resolve_or_create_poi()
        pois = self._poi_arg_set()

        target = ROOT.RooWorkspace(
            f"{self._workspace.GetName()}{suffix}",
            f"{self._workspace.GetTitle()} (spey patched)",
        )
        keep: List[Any] = [poi]

        if self._index_category is not None:
            simultaneous = ROOT.RooSimultaneous(
                f"{self._pdf.GetName()}{suffix}", "", self._index_category
            )
        else:
            simultaneous = None

        for info in self._channels:
            channel_pdf = self._rebuild_channel(info, poi, pois, suffix, keep)
            if simultaneous is None:
                simultaneous = channel_pdf
            else:
                simultaneous.addPdf(channel_pdf, info.name)
        keep.append(simultaneous)
        manager.keep_alive(*keep)

        _ws_import(target, simultaneous, ROOT.RooFit.RecycleConflictNodes())
        for dataset in self._workspace.allData():
            _ws_import(target, dataset)

        model_config = self._rebuild_model_config(target, simultaneous, poi)
        _ws_import(target, model_config)
        manager.keep_alive(target, model_config)
        return target

    def _rebuild_channel(
        self, info: ChannelInfo, poi: Any, pois: Any, suffix: str, keep: List[Any]
    ) -> Any:
        """
        Rebuild the pdf of a single channel with the requested signal edits.

        Args:
            info (``~spey_roofit.data.ChannelInfo``): channel to rebuild.
            poi (``ROOT.RooRealVar``): parameter of interest scaling the signal.
            pois (``ROOT.RooArgSet``): all parameters of interest, used to
              recognise existing signal samples.
            suffix (``str``): suffix appended to every newly created object.
            keep (``List[Any]``): list collecting objects that must outlive the
              call.

        Raises:
            ``ValueError``: if the channel does not follow the
              ``RooRealSumPdf`` layout but needs to be edited.

        Returns:
            ``ROOT.RooAbsPdf``:
            The rebuilt — or, if nothing changed, the original — channel pdf.
        """
        ROOT = manager.ROOT

        needs_signal = info.name in self._signal
        if not needs_signal and not self._strip_signal:
            return info.pdf

        components = self._product_components(info.pdf)
        sums = [comp for comp in components if comp.InheritsFrom("RooRealSumPdf")]
        if len(sums) != 1:
            if needs_signal:
                raise ValueError(
                    f"Channel '{info.name}' is not built from a single "
                    "RooRealSumPdf; spey-roofit cannot inject a signal into it."
                )
            return info.pdf

        old_sum = sums[0]
        functions = ROOT.RooArgList()
        coefficients = ROOT.RooArgList()
        for func, coefficient in zip(old_sum.funcList(), old_sum.coefList()):
            if self._strip_signal and self._is_signal(func, coefficient, pois):
                continue
            functions.add(func)
            coefficients.add(coefficient)

        if needs_signal:
            # The shape holds the signal *density*, so that the contribution to
            # a bin is shape * coefficient * bin volume = signal yield for any
            # binning and for either HistFactory layout.
            shape = self._signal_shape(info, suffix, keep)
            product = ROOT.RooProduct(
                f"spey_signal_{info.name}{suffix}",
                f"signal yield of {info.name}",
                ROOT.RooArgList(poi, shape),
            )
            coefficient = ROOT.RooRealVar(
                f"spey_signal_coef_{info.name}{suffix}", "", 1.0
            )
            coefficient.setConstant(True)
            functions.add(product)
            coefficients.add(coefficient)
            keep += [product, coefficient]

        if functions.getSize() == 0:
            raise ValueError(
                f"Stripping the signal would leave channel '{info.name}' empty."
            )

        new_sum = ROOT.RooRealSumPdf(
            f"{old_sum.GetName()}{suffix}",
            old_sum.GetTitle(),
            functions,
            coefficients,
            True,
        )
        keep += [functions, coefficients, new_sum]

        if len(components) == 1:
            return new_sum

        new_components = ROOT.RooArgList()
        for comp in components:
            new_components.add(new_sum if comp is old_sum else comp)
        product_pdf = ROOT.RooProdPdf(
            f"{info.pdf.GetName()}{suffix}", info.pdf.GetTitle(), new_components
        )
        keep += [new_components, product_pdf]
        return product_pdf

    def _signal_shape(self, info: ChannelInfo, suffix: str, keep: List[Any]) -> Any:
        """
        Build the ``RooHistFunc`` holding the signal yields of a channel.

        Args:
            info (``~spey_roofit.data.ChannelInfo``): channel the signal
              belongs to.
            suffix (``str``): suffix appended to the created objects.
            keep (``List[Any]``): list collecting objects that must outlive the
              call.

        Raises:
            ``NotImplementedError``: if the channel has more than one
              observable.

        Returns:
            ``ROOT.RooHistFunc``:
            Piecewise constant function returning the signal yield of each bin.
        """
        ROOT = manager.ROOT

        if len(info.observables) != 1:
            raise NotImplementedError(
                f"Channel '{info.name}' has {len(info.observables)} observables; "
                "signal injection currently supports one-dimensional channels."
            )
        observable = info.observables[0]
        values = self._signal[info.name]

        binning = observable.getBinning()
        edges = np.array(
            [binning.binLow(index) for index in range(observable.getBins())]
            + [binning.binHigh(observable.getBins() - 1)],
            dtype=np.float64,
        )
        histogram = ROOT.TH1D(
            f"spey_signal_hist_{info.name}{suffix}",
            "",
            observable.getBins(),
            edges,
        )
        histogram.SetDirectory(0)
        for index, value in enumerate(values):
            histogram.SetBinContent(
                index + 1, float(value) / observable.getBinWidth(index)
            )

        observable_list = ROOT.RooArgList(observable)
        observable_set = ROOT.RooArgSet(observable)
        data_hist = ROOT.RooDataHist(
            f"spey_signal_dh_{info.name}{suffix}", "", observable_list, histogram
        )
        shape = ROOT.RooHistFunc(
            f"spey_signal_shape_{info.name}{suffix}", "", observable_set, data_hist, 0
        )
        keep += [histogram, observable_list, observable_set, data_hist, shape]
        return shape

    def _rebuild_model_config(self, target: Any, pdf: Any, poi: Any) -> Any:
        """
        Build the ``ModelConfig`` of the patched workspace.

        Args:
            target (``ROOT.RooWorkspace``): the patched workspace.
            pdf (``ROOT.RooAbsPdf``): the rebuilt top level pdf.
            poi (``ROOT.RooRealVar``): parameter of interest.

        Returns:
            ``ROOT.RooStats.ModelConfig``:
            Model configuration pointing at the objects of ``target``.
        """
        ROOT = manager.ROOT

        model_config = ROOT.RooStats.ModelConfig("ModelConfig", target)
        model_config.SetPdf(target.pdf(pdf.GetName()))

        poi_set = ROOT.RooArgSet()
        target_poi = target.var(poi.GetName())
        if target_poi:
            poi_set.add(target_poi)
        model_config.SetParametersOfInterest(poi_set)

        for setter, source in (
            (
                model_config.SetNuisanceParameters,
                self._model_config.GetNuisanceParameters(),
            ),
            (
                model_config.SetGlobalObservables,
                self._model_config.GetGlobalObservables(),
            ),
            (model_config.SetObservables, self._model_config.GetObservables()),
        ):
            collection = ROOT.RooArgSet()
            for arg in source or []:
                resolved = target.var(arg.GetName()) or target.cat(arg.GetName())
                if resolved:
                    collection.add(resolved)
            setter(collection)
        return model_config

    def _resolve_or_create_poi(self) -> Any:
        """
        Return the parameter of interest, creating it if the model has none.

        Returns:
            ``ROOT.RooRealVar``:
            The parameter of interest the injected signal is scaled by.
        """
        ROOT = manager.ROOT

        declared = self._model_config.GetParametersOfInterest()
        if declared and len(declared) > 0:
            poi = list(declared)[0]
            self._poi_name = poi.GetName()
            return poi

        name = self._poi_name or "mu"
        existing = self._workspace.var(name)
        if existing:
            self._poi_name = name
            return existing
        poi = ROOT.RooRealVar(name, "signal strength", 1.0, -10.0, 10.0)
        self._poi_name = name
        log.debug("Workspace declares no POI, creating '%s'.", name)
        return poi

    def _poi_arg_set(self) -> Optional[Any]:
        """
        Parameters of interest as a ``RooArgSet``.

        Returns:
            ``ROOT.RooArgSet`` or ``None``:
            The set of parameters of interest, or ``None`` if the model has
            none.
        """
        declared = self._model_config.GetParametersOfInterest()
        if not declared or len(declared) == 0:
            return None
        return declared

    @staticmethod
    def _product_components(pdf: Any) -> List[Any]:
        """
        Direct pdf components of a channel.

        Args:
            pdf (``ROOT.RooAbsPdf``): channel pdf.

        Returns:
            ``List[ROOT.RooAbsPdf]``:
            The factors of a ``RooProdPdf``, or the pdf itself.
        """
        if pdf.InheritsFrom("RooProdPdf"):
            return list(pdf.pdfList())
        return [pdf]

    def _sample_terms(self, info: ChannelInfo) -> List[Any]:
        """
        Sample functions and coefficients of a channel.

        Args:
            info (``~spey_roofit.data.ChannelInfo``): channel to inspect.

        Returns:
            ``List[Tuple[ROOT.RooAbsReal, ROOT.RooAbsReal]]``:
            Pairs of sample shape function and its coefficient. Empty if the
            channel does not follow the ``RooRealSumPdf`` layout.
        """
        sums = [
            comp
            for comp in self._product_components(info.pdf)
            if comp.InheritsFrom("RooRealSumPdf")
        ]
        if len(sums) != 1:
            return []
        functions = list(sums[0].funcList())
        coefficients = list(sums[0].coefList())
        if len(functions) != len(coefficients):
            log.warning(
                "Channel '%s' has %d sample functions but %d coefficients; "
                "the extra ones are ignored.",
                info.name,
                len(functions),
                len(coefficients),
            )
        return list(zip(functions, coefficients))

    def _evaluate_sample(
        self, info: ChannelInfo, func: Any, coefficient: Any
    ) -> List[float]:
        """
        Evaluate one sample of a channel bin by bin.

        Args:
            info (``~spey_roofit.data.ChannelInfo``): channel the sample
              belongs to.
            func (``ROOT.RooAbsReal``): sample shape function.
            coefficient (``ROOT.RooAbsReal``): coefficient the function is
              multiplied by inside the ``RooRealSumPdf``.

        Returns:
            ``List[float]``:
            Expected yield of the sample in every bin of the channel.
        """
        values = []
        for index in range(info.nbins):
            volume = self._set_bin(info, index)
            values.append(
                func.getVal(info.observable_set) * coefficient.getVal() * volume
            )
        return values

    @staticmethod
    def _set_bin(info: ChannelInfo, flat_index: int) -> float:
        """
        Move the observables of a channel to a given bin.

        Args:
            info (``~spey_roofit.data.ChannelInfo``): channel to act on.
            flat_index (``int``): row-major bin index.

        Returns:
            ``float``:
            Volume of the selected bin.
        """
        volume = 1.0
        remainder = flat_index
        for observable, nbins in zip(reversed(info.observables), reversed(info.shape)):
            remainder, index = divmod(remainder, nbins)
            observable.setBin(int(index))
            volume *= observable.getBinWidth(int(index))
        return volume

    @staticmethod
    def _sample_name(raw: str, channel: str) -> str:
        """
        Strip the ``HistFactory`` decorations from a sample function name.

        Args:
            raw (``str``): name of the ``RooFit`` function, e.g.
              ``"L_x_ttbar_SR_overallSyst_x_StatUncert"``.
            channel (``str``): name of the channel the sample belongs to.

        Returns:
            ``str``:
            The sample name, e.g. ``"ttbar"``. The raw name is returned when
            stripping leaves nothing behind.
        """
        name = raw
        if name.startswith("L_x_"):
            name = name[len("L_x_") :]
        for decoration in _SAMPLE_DECORATIONS:
            if name.endswith(decoration):
                name = name[: -len(decoration)]
                break
        if name.endswith(f"_{channel}"):
            name = name[: -len(channel) - 1]
        return name or raw

    def _channel(self, name: str) -> ChannelInfo:
        """
        Look up a channel by name.

        Args:
            name (``str``): channel name.

        Raises:
            ``ValueError``: if the channel does not exist.

        Returns:
            ``~spey_roofit.data.ChannelInfo``:
            The channel description.
        """
        for info in self._channels:
            if info.name == name:
                return info
        raise ValueError(
            f"'{name}' is not a channel of this workspace. "
            f"Available channels: {', '.join(self.channels)}."
        )

    def _select_channels(self, channel: Optional[str]) -> List[ChannelInfo]:
        """
        Resolve a channel selection.

        Args:
            channel (``str``): channel name, or ``None`` for all channels.

        Returns:
            ``List[~spey_roofit.data.ChannelInfo]``:
            The selected channels.
        """
        if channel is None:
            return list(self._channels)
        return [self._channel(channel)]

    def _unflatten(self, values: np.ndarray) -> Dict[str, List[float]]:
        """
        Split a flat per-bin vector into a per-channel dictionary.

        Args:
            values (``np.ndarray``): one entry per bin of every channel.

        Returns:
            ``Dict[str, List[float]]``:
            Mapping from channel name to its bins.
        """
        result: Dict[str, List[float]] = {}
        cursor = 0
        for info in self._channels:
            result[info.name] = values[cursor : cursor + info.nbins].tolist()
            cursor += info.nbins
        return result

    @staticmethod
    def _fmt(values: Iterable[float]) -> str:
        """
        Format a short list of yields for the summary table.

        Args:
            values (``Iterable[float]``): numbers to format.

        Returns:
            ``str``:
            Comma separated values, truncated after six entries.
        """
        values = list(values)
        shown = ", ".join(f"{value:.4g}" for value in values[:6])
        return f"[{shown}{', ...' if len(values) > 6 else ''}]"
