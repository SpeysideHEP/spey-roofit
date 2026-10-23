r"""
Compilation of HS3 documents into ``RooFit`` workspaces.

The `HEP Statistics Serialization Standard
<https://hep-statistics-serialization-standard.github.io/>`_ (HS3) describes a
binned likelihood as a plain ``JSON`` document. ``ROOT`` ships a reader for it,
``RooJSONFactoryWSTool``, which turns such a document into an ordinary
``RooWorkspace``; this module wraps that reader so that the result is directly
usable by the rest of the plug-in.

Two corrections are applied on the way, both of which are needed for documents
produced by current ``ROOT`` versions:

* the ``RooStats::ModelConfig`` built by the reader only declares the global
  observables that appear as free-standing distributions in the document, and
  not the ones belonging to the constraint terms that the ``histfactory_dist``
  importer creates on the fly. :func:`complete_model_config` recovers them, so
  that the auxiliary measurements enter the data vector as they must;
* ``ROOT`` 6.36 writes Gaussian constraints without the ``x`` key that the
  standard requires, which makes the document unreadable by ``ROOT`` itself.
  :func:`repair_document` restores the key where the constrained parameter can
  be identified unambiguously.

References
~~~~~~~~~~
* HS3 specification, https://hep-statistics-serialization-standard.github.io/
* C. Burgard et al., *RooFit and HistFactory JSON interface*,
  https://root.cern/manual/roofit/
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

from .manager import manager

__all__ = [
    "compile_workspace",
    "complete_model_config",
    "ensure_default_domain",
    "ensure_parameter",
    "load_document",
    "repair_document",
    "resolve_model_config",
    "save_workspace",
]


def __dir__():
    return __all__


log = logging.getLogger("Spey")

# pylint: disable=invalid-name

#: Anything :func:`load_document` accepts: a parsed document, a path, or JSON text.
DocumentLike = Union[str, bytes, os.PathLike, dict[str, Any]]

#: Distribution types whose ``x`` key :func:`repair_document` is able to restore.
_CONSTRAINT_TYPES = ("gaussian_dist", "poisson_dist", "lognormal_dist")

#: Suffixes ``HistFactory`` appends to the name of a constrained parameter.
_CONSTRAINT_SUFFIXES = ("Constraint", "_constraint", "_Constraint")

#: Prefixes ``HistFactory`` prepends to the name of a global observable,
#: longest first so that ``nominalLumi`` is not stripped down to ``inalLumi``.
_GLOBAL_PREFIXES = ("nominal_", "nominal", "nom_", "nom")


@contextmanager
def _quiet():
    """
    Silence the output the HS3 reader writes straight to ``stdout``.

    ``RooJSONFactoryWSTool`` prints one line per auto-created constraint term
    through the C++ standard output — a debug print left in ``ROOT`` 6.36, and
    several thousand lines for a large workspace. It bypasses both
    :mod:`logging` and ``RooMsgService``, so reaching it means redirecting the
    file descriptor itself for the duration of the import. The redirection is
    skipped when the descriptor cannot be duplicated, as happens when standard
    output has been closed.

    Yields:
        ``None``
    """
    try:
        sys.stdout.flush()
        saved = os.dup(1)
        devnull = os.open(os.devnull, os.O_WRONLY)
    except (OSError, ValueError):  # pragma: no cover - environment dependent
        yield
        return
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def load_document(source: DocumentLike) -> dict[str, Any]:
    """
    Read an HS3 document.

    Args:
        source (``Union[str, bytes, os.PathLike, Dict]``): a mapping, which is
          returned unchanged, the path of a ``.json`` file, or a ``str``
          holding the document itself.

    Raises:
        ``ValueError``: if ``source`` is neither a mapping, an existing file,
          nor a valid ``JSON`` string.
        ``TypeError``: if ``source`` has an unsupported type.

    Returns:
        ``Dict[str, Any]``:
        The parsed document.
    """
    if isinstance(source, dict):
        return source
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    if isinstance(source, os.PathLike):
        source = os.fspath(source)
    if not isinstance(source, str):
        raise TypeError(
            f"An HS3 document has to be a mapping, a path or a JSON string, "
            f"got {type(source).__name__}."
        )

    if os.path.isfile(source):
        with open(source, encoding="utf-8") as handle:
            return json.load(handle)

    stripped = source.lstrip()
    if stripped.startswith("{"):
        try:
            return json.loads(source)
        except json.JSONDecodeError as err:
            raise ValueError(f"Could not parse the HS3 document: {err}") from err
    raise ValueError(f"'{source}' is not an existing file and is not JSON.")


def _declared_parameters(document: dict[str, Any]) -> set:
    """
    Collect every parameter name the document declares.

    Args:
        document (``Dict[str, Any]``): an HS3 document.

    Returns:
        ``set``:
        Names appearing in the ``domains`` axes or in the ``parameter_points``.
    """
    names = set()
    for domain in document.get("domains", []) or []:
        for axis in domain.get("axes", []) or []:
            if "name" in axis:
                names.add(axis["name"])
    for point in document.get("parameter_points", []) or []:
        for parameter in point.get("parameters", []) or []:
            if "name" in parameter:
                names.add(parameter["name"])
    return names


def _strip(name: str, affixes: tuple[str, ...], prefix: bool = False) -> str | None:
    """
    Remove the first matching affix from ``name``.

    Args:
        name (``str``): the name to strip.
        affixes (``Tuple[str, ...]``): candidate affixes, tried in order.
        prefix (``bool``, default ``False``): strip a prefix instead of a suffix.

    Returns:
        ``str`` or ``None``:
        The stripped name, or ``None`` if no affix matched.
    """
    for affix in affixes:
        if prefix and name.startswith(affix) and len(name) > len(affix):
            return name[len(affix) :]
        if not prefix and name.endswith(affix) and len(name) > len(affix):
            return name[: -len(affix)]
    return None


def repair_document(
    document: dict[str, Any], inplace: bool = False
) -> tuple[dict[str, Any], list[str]]:
    r"""
    Restore the ``x`` key of constraint distributions that are missing it.

    ``ROOT`` 6.36 exports the Gaussian constraints of a ``HistFactory`` model
    without naming the parameter they constrain, which violates the HS3
    specification and makes the document unreadable — ``ROOT`` aborts with
    ``node '<name>' is missing key 'x'``. The constrained parameter is
    recovered from the name of the distribution, ``<parameter>Constraint``, or
    failing that from the name of its centre, ``nom_<parameter>``, and only
    accepted when the resulting name is a parameter the document declares.

    Args:
        document (``Dict[str, Any]``): an HS3 document.
        inplace (``bool``, default ``False``): modify ``document`` itself
          instead of a copy.

    Raises:
        ``ValueError``: if a constraint is missing its ``x`` key and the
          constrained parameter cannot be identified.

    Returns:
        ``Tuple[Dict[str, Any], List[str]]``:
        The repaired document and the names of the distributions that were
        repaired.

    .. versionadded:: 0.0.2
    """
    if not inplace:
        document = json.loads(json.dumps(document))

    declared = _declared_parameters(document)
    repaired: list[str] = []
    for distribution in document.get("distributions", []) or []:
        if distribution.get("type") not in _CONSTRAINT_TYPES:
            continue
        if "x" in distribution:
            continue
        name = distribution.get("name", "<unnamed>")

        candidates: list[str] = []

        def propose(value: str | None) -> None:
            """Add ``value`` and its capitalised form to the candidate list."""
            if not value:
                return
            for variant in (value, value[0].upper() + value[1:]):
                if variant not in candidates:
                    candidates.append(variant)

        # ``alpha_fooConstraint`` constrains ``alpha_foo``.
        propose(_strip(name, _CONSTRAINT_SUFFIXES))
        # ``nom_alpha_foo`` is the auxiliary measurement of ``alpha_foo``, and
        # ``nominalLumi`` that of ``Lumi``.
        centre = distribution.get("mean", distribution.get("median"))
        if isinstance(centre, str):
            propose(_strip(centre, _GLOBAL_PREFIXES, prefix=True))

        resolved = next((par for par in candidates if par in declared), None)
        if resolved is None:
            raise ValueError(
                f"Constraint '{name}' does not declare the parameter it "
                f"constrains (HS3 key 'x') and it cannot be inferred"
                + (f"; tried {', '.join(candidates)}." if candidates else ".")
            )
        distribution["x"] = resolved
        repaired.append(name)

    ensure_default_domain(document)

    if repaired:
        log.warning(
            "%d constraint(s) of the HS3 document do not declare the parameter "
            "they constrain; it has been inferred from their name. This is a "
            "known defect of the HS3 writer of ROOT 6.36. Affected: %s",
            len(repaired),
            ", ".join(repaired[:3]) + (", ..." if len(repaired) > 3 else ""),
        )
    return document, repaired


@contextmanager
def _captured_stderr():
    """
    Collect what ``ROOT`` writes to the standard error descriptor.

    Yields:
        ``Callable[[], List[str]]``:
        Function returning the captured lines; call it after the block.
    """
    import tempfile

    captured: list[str] = []
    sys.stderr.flush()
    with tempfile.TemporaryFile(mode="w+") as sink:
        try:
            saved = os.dup(2)
        except OSError:  # pragma: no cover - environment dependent
            yield lambda: captured
            return
        os.dup2(sink.fileno(), 2)
        try:
            yield lambda: captured
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            sink.seek(0)
            captured.extend(sink.read().splitlines())


#: File formats :func:`save_workspace` writes, by name and by file suffix.
_SAVE_FORMATS = {".root": "root", ".json": "hs3"}


def save_workspace(
    workspace: Any, path: str | os.PathLike, fmt: str | None = None
) -> str:
    """
    Write a ``RooFit`` workspace to a ``ROOT`` file or to an HS3 document.

    The file is written through a ``TFile`` and checked on disk afterwards,
    rather than trusting ``RooWorkspace::writeToFile``, whose return value has
    meant success in some ``ROOT`` versions and failure in others.

    An HS3 export from ``ROOT`` 6.36 misses the ``x`` key of the Gaussian
    constraints, which makes the document unreadable; it is restored with
    :func:`repair_document` before the file is finished, so that the result
    follows the standard.

    Args:
        workspace (``ROOT.RooWorkspace``): the workspace to write.
        path (``Union[str, os.PathLike]``): destination. An existing file is
          overwritten.
        fmt (``str``, default ``None``): ``"root"`` or ``"hs3"``. Taken from the
          suffix of ``path`` (``.root`` or ``.json``) when omitted.

    Raises:
        ``ValueError``: if the format is unknown or cannot be told from the
          suffix.
        ``OSError``: if the file cannot be written.

    Returns:
        ``str``:
        ``path``, as given.
    """
    path = os.fspath(path)
    if fmt is None:
        fmt = _SAVE_FORMATS.get(os.path.splitext(path)[1].lower())
    fmt = {"json": "hs3"}.get(fmt, fmt)
    if fmt not in ("root", "hs3"):
        raise ValueError(
            f"Cannot tell the format of '{path}': use a .root or .json suffix, or "
            f"pass fmt='root' or fmt='hs3'."
        )
    ROOT = manager.ROOT

    if os.path.exists(path):
        os.remove(path)
    if fmt == "root":
        handle = ROOT.TFile.Open(path, "RECREATE")
        if not handle or handle.IsZombie():
            raise OSError(f"Cannot open '{path}' for writing.")
        workspace.Write()
        handle.Close()
    else:
        with _captured_stderr() as messages:
            with _quiet():
                ROOT.RooJSONFactoryWSTool(workspace).exportJSON(path)
        # the exporter reports a class it cannot serialise on ``stderr`` and
        # still returns success, leaving a document ``ROOT`` cannot read back
        failure = [line for line in messages() if "unable to export" in line]
        if failure:
            if os.path.isfile(path):
                os.remove(path)
            raise OSError(
                f"The workspace cannot be written as HS3: {failure[0].strip()}. Save "
                f"it as a ROOT file instead."
            )
        if os.path.isfile(path):
            document, _ = repair_document(load_document(path))
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(document, stream, indent=2)

    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise OSError(f"Cannot write the workspace to '{path}'.")
    log.info("Workspace saved to '%s'.", path)
    return path


#: Name of the domain the ``ROOT`` reader falls back to for any parameter that
#: the analysis does not otherwise give a range for.
DEFAULT_DOMAIN = "default_domain"


def ensure_default_domain(document: dict[str, Any]) -> bool:
    """
    Make sure the document carries a domain named ``default_domain``.

    The ``ROOT`` reader looks up that exact name whenever a parameter is not
    covered by the domains its analysis lists, and **aborts the process** if no
    such domain exists — the failure is a C++ ``std::runtime_error`` raised
    below the Python layer, so it cannot be caught and it takes the interpreter
    down with it. A document that names its domain anything else is perfectly
    valid HS3, so rather than rejecting it, the union of every declared domain
    is added under the expected name. For a document that already has one this
    is a no-op, and for one written by ``ROOT`` the added ranges are the ones
    already in force.

    Args:
        document (``Dict[str, Any]``): the HS3 document, modified in place.

    Returns:
        ``bool``:
        ``True`` if a fallback domain had to be added.

    .. versionadded:: 0.0.2
    """
    domains = document.setdefault("domains", [])
    if any(domain.get("name") == DEFAULT_DOMAIN for domain in domains):
        return False

    axes: dict[str, dict[str, Any]] = {}
    for domain in domains:
        for axis in domain.get("axes", []) or []:
            name = axis.get("name")
            if name is not None and name not in axes:
                axes[name] = dict(axis)

    domains.append(
        {
            "name": DEFAULT_DOMAIN,
            "type": "product_domain",
            "axes": list(axes.values()),
        }
    )
    log.debug(
        "Added the fallback domain '%s' with %d axes; the ROOT reader requires "
        "it by name.",
        DEFAULT_DOMAIN,
        len(axes),
    )
    return True


def resolve_model_config(workspace: Any, model_config_name: str | None = None) -> Any:
    """
    Locate the ``RooStats::ModelConfig`` of a compiled workspace.

    The reader names the model configuration after the HS3 analysis it comes
    from, unless the document carries a ``ROOT`` specific hint asking for a
    different name. The name is therefore resolved rather than assumed.

    Args:
        workspace (``ROOT.RooWorkspace``): the workspace to search.
        model_config_name (``str``, default ``None``): explicit name. The only
          model configuration of the workspace is used when omitted.

    Raises:
        ``ValueError``: if the requested model configuration does not exist, or
          if the workspace holds none at all.

    Returns:
        ``ROOT.RooStats::ModelConfig``:
        The model configuration.
    """
    available = [
        obj
        for obj in workspace.allGenericObjects()
        if obj.ClassName() == "RooStats::ModelConfig"
    ]
    names = [obj.GetName() for obj in available]

    if model_config_name is not None:
        for obj in available:
            if obj.GetName() == model_config_name:
                return obj
        raise ValueError(
            f"The compiled workspace does not contain a ModelConfig named "
            f"'{model_config_name}'. Available: {', '.join(names) or 'none'}."
        )

    if not available:
        raise ValueError(
            "The HS3 document does not define an analysis, so the compiled "
            "workspace holds no ModelConfig. Add an 'analyses' section naming "
            "the likelihood and its parameters of interest."
        )
    if len(available) > 1:
        # The reader emits one model configuration per analysis. They share the
        # same pdf and differ only by the dataset, so the observed one is the
        # right default.
        for obj in available:
            if "asimov" not in obj.GetName().lower():
                log.debug("Using ModelConfig '%s'.", obj.GetName())
                return obj
        log.warning(
            "The document declares %d analyses and all of them look like "
            "Asimov analyses; using '%s'.",
            len(available),
            names[0],
        )
    return available[0]


def complete_model_config(workspace: Any, model_config: Any) -> list[str]:
    r"""
    Declare the global observables the HS3 reader left out.

    ``RooJSONFactoryWSTool`` creates the constraint terms of a
    ``histfactory_dist`` on the fly but does not add their auxiliary
    measurements to the model configuration, so a model read from HS3 typically
    declares far fewer global observables than it has constraint terms. Every
    such measurement is recovered here: it is the constant variable of a
    constraint factor that is neither a nuisance parameter, nor a parameter of
    interest, nor an observable.

    Args:
        workspace (``ROOT.RooWorkspace``): workspace holding the model.
        model_config (``ROOT.RooStats::ModelConfig``): configuration to
          complete. It is modified in place.

    Returns:
        ``List[str]``:
        Names of the global observables that have been added.

    .. attention::

        The model configuration is only completed, never pruned: global
        observables it already declares are kept as they are.
    """
    ROOT = manager.ROOT

    pdf = model_config.GetPdf()
    if not pdf:
        return []

    def names_of(collection) -> set:
        """Return the names held by a possibly null ``RooArgSet``."""
        return {arg.GetName() for arg in collection} if collection else set()

    nuisance = names_of(model_config.GetNuisanceParameters())
    pois = names_of(model_config.GetParametersOfInterest())
    observable_names = names_of(model_config.GetObservables())
    declared = names_of(model_config.GetGlobalObservables())

    observables = ROOT.RooArgSet()
    for observable in model_config.GetObservables() or []:
        observables.add(observable)
    if pdf.InheritsFrom("RooSimultaneous"):
        observables.add(pdf.indexCat())

    def is_candidate(variable: Any) -> bool:
        """Whether ``variable`` can be an auxiliary measurement."""
        name = variable.GetName()
        return (
            variable.InheritsFrom("RooRealVar")
            and variable.isConstant()
            and name not in nuisance
            and name not in pois
            and name not in observable_names
        )

    found: dict[str, Any] = {}
    for component in pdf.getComponents():
        if not component.InheritsFrom("RooAbsPdf"):
            continue
        if component.dependsOn(observables):
            # The main term of a channel, not a constraint.
            continue

        # The proxies of the standard constraint shapes identify the auxiliary
        # measurement unambiguously; the generic scan is only a fallback.
        proxies = []
        for accessor in ("getX", "getMean", "getMedian"):
            if hasattr(component, accessor):
                try:
                    proxies.append(getattr(component, accessor)())
                except (TypeError, ReferenceError):  # pragma: no cover - defensive
                    continue
        candidates = [arg for arg in proxies if is_candidate(arg)]
        if not candidates:
            candidates = [var for var in component.getVariables() if is_candidate(var)]

        if len(candidates) != 1:
            log.debug(
                "Constraint '%s' has %d candidate global observables; skipped.",
                component.GetName(),
                len(candidates),
            )
            continue
        found[candidates[0].GetName()] = candidates[0]

    missing = sorted(set(found) - declared)
    if not missing:
        return []

    complete = ROOT.RooArgSet()
    for observable in model_config.GetGlobalObservables() or []:
        complete.add(observable)
    for name in missing:
        complete.add(found[name])
    manager.keep_alive(complete)
    model_config.SetGlobalObservables(complete)

    log.debug(
        "Declared %d global observable(s) the HS3 reader left out: %s",
        len(missing),
        ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else ""),
    )
    return missing


def compile_workspace(
    document: DocumentLike,
    workspace_name: str = "hs3_workspace",
    model_config_name: str | None = None,
    repair: bool = True,
    complete: bool = True,
) -> tuple[Any, Any]:
    """
    Build a ``RooFit`` workspace from an HS3 document.

    Args:
        document (``Union[str, bytes, os.PathLike, Dict]``): the document, as a
          mapping, a file path or a ``JSON`` string.
        workspace_name (``str``, default ``"hs3_workspace"``): name given to
          the ``RooWorkspace``.
        model_config_name (``str``, default ``None``): name of the model
          configuration to use, resolved automatically when omitted.
        repair (``bool``, default ``True``): restore constraint keys that the
          writer of the document may have omitted, see
          :func:`repair_document`. The fallback domain of
          :func:`ensure_default_domain` is added either way, because its
          absence aborts the process rather than raising.
        complete (``bool``, default ``True``): declare the global observables
          the reader leaves out, see :func:`complete_model_config`.

    Raises:
        ``ValueError``: if the document cannot be read, if it is rejected by
          ``ROOT``, or if it defines no analysis.
        ``ImportError``: if ``PyROOT`` is not installed.

    Returns:
        ``Tuple[ROOT.RooWorkspace, ROOT.RooStats::ModelConfig]``:
        The compiled workspace and its model configuration.

    Example:

    .. code-block:: python3

        >>> from spey_roofit.hs3 import compile_workspace
        >>> workspace, model_config = compile_workspace("workspace_HS3.json")
        >>> model_config.GetPdf().GetName()

    .. note::

        The returned workspace owns every object of the model; keep a reference
        to it for as long as the model is in use.
    """
    ROOT = manager.ROOT

    parsed = load_document(document)
    if repair:
        parsed, _ = repair_document(parsed)
    elif not any(
        domain.get("name") == DEFAULT_DOMAIN for domain in parsed.get("domains", []) or []
    ):
        # `ensure_default_domain` below has to modify the document, and it runs
        # whatever `repair` says: without it the reader takes the process down.
        parsed = json.loads(json.dumps(parsed))
    ensure_default_domain(parsed)

    workspace = ROOT.RooWorkspace(workspace_name)
    tool = ROOT.RooJSONFactoryWSTool(workspace)
    try:
        with _quiet():
            tool.importJSONfromString(json.dumps(parsed))
    except Exception as err:  # pragma: no cover - depends on the document
        raise ValueError(f"ROOT could not read the HS3 document: {err}") from err

    model_config = resolve_model_config(workspace, model_config_name)
    if complete:
        complete_model_config(workspace, model_config)
    manager.keep_alive(workspace)
    return workspace, model_config


def ensure_parameter(
    document: dict[str, Any],
    name: str,
    value: float = 1.0,
    bounds: tuple[float, float] = (-10.0, 10.0),
) -> bool:
    r"""
    Make sure a parameter is declared with a value and a range.

    A parameter that is only referenced by a modifier is still created by the
    HS3 reader, but with whatever default range ``RooFit`` happens to pick.
    Since ``RooFit`` silently clamps a variable to its range, a parameter of
    interest introduced by signal injection has to be given an explicit domain
    before the document is compiled.

    Args:
        document (``Dict[str, Any]``): the HS3 document, modified in place.
        name (``str``): name of the parameter.
        value (``float``, default ``1.0``): initial value, used only when the
          parameter is not declared yet.
        bounds (``Tuple[float, float]``, default ``(-10.0, 10.0)``): range
          given to the parameter when it has no domain.

    Returns:
        ``bool``:
        ``True`` if the document had to be extended, ``False`` if the parameter
        was already fully declared.

    .. versionadded:: 0.0.2
    """
    changed = False

    domains = document.setdefault("domains", [])
    if not any(
        axis.get("name") == name
        for domain in domains
        for axis in domain.get("axes", []) or []
    ):
        if not domains:
            domains.append(
                {"name": "default_domain", "type": "product_domain", "axes": []}
            )
        domains[0].setdefault("axes", []).append(
            {"name": name, "min": float(bounds[0]), "max": float(bounds[1])}
        )
        changed = True

    points = document.setdefault("parameter_points", [])
    if not any(
        parameter.get("name") == name
        for point in points
        for parameter in point.get("parameters", []) or []
    ):
        if not points:
            points.append({"name": "default_values", "parameters": []})
        points[0].setdefault("parameters", []).append(
            {"name": name, "value": float(value)}
        )
        changed = True

    if changed:
        log.debug("Declared parameter '%s' in the HS3 document.", name)
    return changed
