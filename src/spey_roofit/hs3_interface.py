r"""
``spey`` plug-in for HS3 likelihood documents.

The `HEP Statistics Serialization Standard
<https://hep-statistics-serialization-standard.github.io/>`_ describes a binned
likelihood as a ``JSON`` document, independently of the software that evaluates
it. This backend compiles such a document into a ``RooFit`` workspace with
``RooJSONFactoryWSTool`` and then hands it to exactly the same machinery that
:class:`~spey_roofit.interface.RooFitInterface` uses, so an analysis published
as HS3 and the same analysis published as a ``.root`` workspace give the same
numbers by construction.

The likelihood is the usual binned one,

.. math::

    \mathcal{L}(\mu,\theta) =
        \prod_{i\in\rm bins}{\rm Pois}\left(n_i\vert\lambda_i(\mu,\theta)\right)
        \cdot\prod_{c\in\rm constraints} p_c(a_c\vert\theta)\ ,

with :math:`\lambda_i` evaluated by ``RooFit`` from the compiled model and the
auxiliary measurements :math:`a_c` carried along in the data vector, so that
Asimov datasets follow the profiled nuisance parameters as required by
:xref:`1007.1727`.

The region structure of the document and the injection of a signal are handled
by the interpreter of `spey-hs3 <https://github.com/SpeysideHEP/spey-hs3>`_,
which works on the document itself — the user never has to touch ``RooFit``.

References
~~~~~~~~~~
* HS3 specification, https://hep-statistics-serialization-standard.github.io/
* RooFit manual, https://root.cern/manual/roofit/
* G. Cowan, K. Cranmer, E. Gross, O. Vitells, :xref:`1007.1727`
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._version import __version__
from .base import RooFitBackendBase
from .data import WorkspaceData
from .hs3 import compile_workspace, ensure_parameter, load_document

__all__ = ["HS3Interface"]


def __dir__():
    return __all__


log = logging.getLogger("Spey")

# pylint: disable=invalid-name, too-many-arguments, too-many-instance-attributes
# pylint: disable=import-outside-toplevel

#: Name given to a signal sample injected through the simplified interface.
DEFAULT_SAMPLE_NAME = "spey_signal"

#: Name given to the parameter of interest created for a background-only model.
DEFAULT_POI_NAME = "mu"


#: Interpreter members the backend relies on. ``spey-hs3`` 0.0.2 has neither
#: ``poi_parameters`` nor ``signal_samples`` nor ``strip_signal``, so the
#: requirement is checked against the class rather than against the version the
#: distribution metadata happens to advertise.
_REQUIRED_INTERPRETER_MEMBERS = (
    "bin_map",
    "inject_signals",
    "patch",
    "poi_parameters",
    "signal_samples",
    "strip_signal",
)


def _interpreter_class():
    """
    Import the HS3 interpreter of ``spey-hs3``.

    Raises:
        ``ImportError``: if ``spey-hs3`` is not installed, or if the installed
          version is too old to provide everything the backend needs.

    Returns:
        ``type``:
        ``spey_hs3.helper_functions.WorkspaceInterpreter``.
    """
    try:
        from spey_hs3.helper_functions import WorkspaceInterpreter
    except ImportError as err:  # pragma: no cover - environment dependent
        raise ImportError(
            "The HS3 backend uses the workspace interpreter of spey-hs3. "
            "Install it with `pip install spey-hs3`."
        ) from err

    missing = [
        member
        for member in _REQUIRED_INTERPRETER_MEMBERS
        if not hasattr(WorkspaceInterpreter, member)
    ]
    if missing:  # pragma: no cover - environment dependent
        raise ImportError(
            f"The installed spey-hs3 is too old for the HS3 backend: its "
            f"WorkspaceInterpreter has no {', '.join(missing)}. Install a "
            f"newer release, or the development version with "
            f"`pip install git+https://github.com/SpeysideHEP/spey-hs3`."
        )
    return WorkspaceInterpreter


def normalise_signal_yields(
    signal_yields: Dict[str, Union[Sequence[float], Dict[str, Any]]],
    sample_name: str = DEFAULT_SAMPLE_NAME,
) -> Dict[str, Dict[str, Any]]:
    r"""
    Bring user supplied signal yields into the shape ``spey-hs3`` expects.

    Two spellings are accepted. The short one names only the distribution,

    .. code-block:: python3

        {"model_SR": [3.0, 5.0, 2.0]}

    and the explicit one also names the sample, optionally with per-bin Monte
    Carlo errors,

    .. code-block:: python3

        {"model_SR": {"my_signal": {"contents": [3.0, 5.0], "errors": [0.1, 0.2]}}}

    Args:
        signal_yields (``Dict[str, Union[Sequence[float], Dict[str, Any]]]``):
          per-distribution signal specification in either spelling.
        sample_name (``str``, default ``"spey_signal"``): name given to the
          sample created by the short spelling.

    Raises:
        ``ValueError``: if a distribution maps to something that is neither a
          sequence of yields nor a mapping of samples.

    Returns:
        ``Dict[str, Dict[str, Any]]``:
        Mapping ``{distribution: {sample: specification}}``.
    """
    normalised: Dict[str, Dict[str, Any]] = {}
    for distribution, specification in signal_yields.items():
        if isinstance(specification, dict):
            if "contents" in specification:
                normalised[distribution] = {sample_name: specification}
            else:
                normalised[distribution] = dict(specification)
        elif isinstance(specification, (list, tuple, np.ndarray)):
            normalised[distribution] = {
                sample_name: [float(value) for value in np.atleast_1d(specification)]
            }
        else:
            raise ValueError(
                f"Signal yields for '{distribution}' have to be a sequence of "
                f"per-bin yields or a mapping of sample names, got "
                f"{type(specification).__name__}."
            )
    return normalised


class HS3Interface(RooFitBackendBase):
    r"""
    ``spey`` backend for an HS3 likelihood document.

    The document is compiled into a ``RooFit`` workspace and evaluated by
    ``RooFit``. It may already contain a signal — in which case its parameter
    of interest is used as it is — or be background-only, in which case
    ``signal_yields`` adds a signal sample scaled by a freshly created
    parameter of interest.

    Args:
        workspace (``Union[str, os.PathLike, Dict, spey_hs3.WorkspaceInterpreter]``):
          the model. A ``str`` is read as a path to a ``.json`` file, or as the
          document itself when it is ``JSON``. An interpreter is used as it is,
          together with whatever mutations have already been queued on it.
        signal_yields (``Dict[str, Union[Sequence[float], Dict]]``, default
          ``None``): per-distribution signal yields to inject, with one entry
          per bin. Use
          :attr:`~spey_hs3.helper_functions.WorkspaceInterpreter.bin_map` to
          discover the expected shape. Nothing is injected when omitted.
        poi_name (``str``, default ``None``): name of the parameter of
          interest. The one declared by the document is used when omitted, and
          ``"mu"`` is created when the document declares none.
        sample_name (``str``, default ``"spey_signal"``): name given to an
          injected signal sample when ``signal_yields`` does not name one.
        poi_bounds (``Tuple[float, float]``, default ``(-10.0, 10.0)``): range
          given to a parameter of interest that the document does not declare.
        model_config_name (``str``, default ``None``): name of the
          ``RooStats::ModelConfig`` to use. The reader names it after the HS3
          analysis, so it is resolved automatically when omitted.
        data_name (``str``, default ``None``): name of the observed dataset.
          Resolved automatically when omitted, preferring the dataset whose
          name does not look like Asimov data.
        strip_existing_signal (``bool``, default ``False``): remove the samples
          that are already scaled by a parameter of interest before injecting
          ``signal_yields``. Use this to reinterpret a published
          signal-plus-background document with a different signal.
        strict_cache_flush (``bool``, default ``True``): invalidate the
          ``RooFit`` value cache before every likelihood evaluation.
        repair (``bool``, default ``True``): restore constraint keys that the
          writer of the document may have omitted, see
          :func:`~spey_roofit.hs3.repair_document`.

    Raises:
        ``ValueError``: if the document defines no analysis, no dataset, or no
          parameter of interest and no signal was injected.
        ``ImportError``: if ``PyROOT`` or ``spey-hs3`` is not installed.

    Example:

    .. code-block:: python3
        :linenos:

        >>> import spey
        >>> from spey_hs3.helper_functions import WorkspaceInterpreter

        >>> interpreter = WorkspaceInterpreter(
        ...     json.load(open("workspace_HS3.json"))
        ... )
        >>> interpreter.bin_map          # which regions and how many bins each

        >>> wrapper = spey.get_backend("root.hs3")
        >>> statistical_model = wrapper(
        ...     workspace="workspace_HS3.json",
        ...     signal_yields={"model_SR": [3.0, 5.0, 2.0]},
        ...     analysis="atlas-topq-2018-24",
        ... )
        >>> statistical_model.exclusion_confidence_level()

    .. note::

        The compiled workspace is kept alive for as long as the backend exists
        and is reachable through :attr:`workspace`; the document it was built
        from, with every mutation applied, is available as :attr:`document`.
    """

    name: str = "root.hs3"
    """Name of the backend"""
    version: str = __version__
    """Version of the backend"""
    author: str = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: str = ">=0.2.0"
    """Spey version required for the backend"""
    arXiv: List[str] = ["physics/0306116", "1009.1003"]
    """arXiv references: the RooFit toolkit and the RooStats project"""

    __slots__ = ["_interpreter", "_document", "_workspace", "_signal_yields"]

    def __init__(
        self,
        workspace: Union[str, Dict[str, Any], Any],
        signal_yields: Optional[Dict[str, Union[Sequence[float], Dict[str, Any]]]] = None,
        poi_name: Optional[str] = None,
        sample_name: str = DEFAULT_SAMPLE_NAME,
        poi_bounds: Tuple[float, float] = (-10.0, 10.0),
        model_config_name: Optional[str] = None,
        data_name: Optional[str] = None,
        strip_existing_signal: bool = False,
        strict_cache_flush: bool = True,
        repair: bool = True,
    ):
        WorkspaceInterpreter = _interpreter_class()

        if isinstance(workspace, WorkspaceInterpreter):
            interpreter = workspace
        else:
            interpreter = WorkspaceInterpreter(load_document(workspace))

        if strip_existing_signal:
            interpreter.strip_signal()

        normalised: Dict[str, Dict[str, Any]] = {}
        if signal_yields:
            normalised = normalise_signal_yields(signal_yields, sample_name)
            declared = interpreter.poi_parameters
            poi = poi_name or (declared[0] if declared else DEFAULT_POI_NAME)
            interpreter.inject_signals(normalised, poi_name=poi)
        else:
            poi = poi_name

        self._interpreter = interpreter
        self._signal_yields = normalised

        document = _document_of(interpreter)
        if poi is not None:
            ensure_parameter(document, poi, value=1.0, bounds=poi_bounds)
        self._document: Dict[str, Any] = document

        self._workspace, model_config = compile_workspace(
            document,
            model_config_name=model_config_name,
            repair=repair,
        )
        self._model = WorkspaceData(
            self._workspace,
            model_config_name=model_config.GetName(),
            data_name=data_name,
            poi_name=poi,
            strict_cache_flush=strict_cache_flush,
        )
        self._minimum_poi: Optional[float] = None
        self._template: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # ------------------------------------------------------------------ #
    # Accessors                                                           #
    # ------------------------------------------------------------------ #

    @property
    def interpreter(self) -> Any:
        """
        The HS3 interpreter that produced the compiled document.

        Returns:
            ``spey_hs3.helper_functions.WorkspaceInterpreter``:
            Interpreter holding the original document and the injected signal.
        """
        return self._interpreter

    @property
    def document(self) -> Dict[str, Any]:
        """
        The HS3 document the workspace was compiled from.

        Returns:
            ``Dict[str, Any]``:
            The document with every mutation of :attr:`interpreter` applied.
        """
        return self._document

    @property
    def workspace(self) -> Any:
        """
        The compiled ``RooFit`` workspace.

        Returns:
            ``ROOT.RooWorkspace``:
            Workspace holding the model, its data and its model configuration.
        """
        return self._workspace


def _has_mutations(interpreter: Any) -> bool:
    """
    Whether an interpreter has anything queued for its patch.

    Args:
        interpreter (``spey_hs3.helper_functions.WorkspaceInterpreter``): the
          interpreter to inspect.

    Returns:
        ``bool``:
        ``True`` if at least one mutation is pending.
    """
    return any(
        bool(getattr(interpreter, attribute, None))
        for attribute in (
            "_signal_dict",
            "_to_remove",
            "_samples_to_remove",
            "_modifier_strip",
        )
    )


def _document_of(interpreter: Any) -> Dict[str, Any]:
    """
    Extract the document an interpreter describes, with its mutations applied.

    ``spey-hs3`` refuses to produce a patch when nothing has been queued and
    exposes no public accessor for the untouched document, so the two cases are
    separated here. The result is always a copy that is safe to modify.

    Args:
        interpreter (``spey_hs3.helper_functions.WorkspaceInterpreter``): the
          interpreter to read.

    Raises:
        ``TypeError``: if ``interpreter`` does not hold an HS3 document.

    Returns:
        ``Dict[str, Any]``:
        The document, deep-copied.
    """
    if _has_mutations(interpreter):
        # ``patch`` already returns a deep copy.
        return interpreter.patch

    document = getattr(interpreter, "_workspace", None)
    if not isinstance(document, dict):
        raise TypeError(
            "The interpreter does not expose the HS3 document it was built "
            "from; pass the document itself instead."
        )
    return copy.deepcopy(document)
