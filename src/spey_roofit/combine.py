r"""
``CMS Combine`` datacards, compiled by ``combine`` itself.

`Combine <https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/>`_ is the
statistical tool of the CMS collaboration, and a *datacard* is what CMS
publishes when it publishes a likelihood. ``combine`` does not evaluate a
datacard directly: ``text2workspace.py`` first turns it into a ``RooWorkspace``,
and every statistical method then works on that workspace.

This module reimplements none of that. It drives ``combine``'s own
``DatacardParser`` and ``ShapeBuilder`` in process, so the workspace is exactly
the one ``text2workspace.py`` writes — same interpolation, same constraint
terms, same physics models — and hands it to the machinery that already backs
:class:`~spey_roofit.interface.RooFitInterface` and
:class:`~spey_roofit.hs3_interface.HS3Interface`. What the module adds is the
part ``combine`` has no interface for:

* :class:`DatacardInterpreter` reads the card *before* it is compiled and says
  what the model is — which channels exist, how many bins each has, which
  samples they contain and what every one of them is expected to yield — so
  that a published likelihood can be understood without knowing ``combine`` or
  ``RooFit``;
* :func:`patch_datacard` attaches a signal to it, or removes the one it already
  has, by rewriting the card. ``combine`` then compiles the result, so the
  reinterpreted model is built by the same code as the published one.

The likelihood is therefore ``combine``'s,

.. math::

    \mathcal{L}(\mu,\theta) =
        \prod_{c\in\rm channels}\prod_{i\in\rm bins}
            {\rm Pois}\left(n_{ci}\vert\lambda_{ci}(\mu,\theta)\right)
        \cdot\prod_{e} p_e(a_e\vert\theta)\ ,\qquad
    \lambda_{ci} = \sum_{p} M_{cp}(\mu)\,N_{cp}(\theta)\,\omega_{cpi}(\theta)\ ,

with the log-normal normalisation factors :math:`N` and the vertically morphed
templates :math:`\omega` that ``combine`` builds.

.. note::

    ``combine`` builds a counting experiment as a product of ``RooPoisson``
    terms over plain variables rather than as a binned model, which is not a
    shape ``spey`` can read. A card without ``shapes`` directives is therefore
    given one-bin templates first, which turns it into a binned model of the
    very same product of Poisson terms.

.. note::

    ``combine`` normally writes its Gaussian constraints as
    ``SimpleGaussianConstraint``, which drops the :math:`1/\sqrt{2\pi}` of
    every term and so shifts :math:`\log\mathcal{L}` by a constant. The plug-in
    asks ``combine`` for plain ``RooGaussian`` terms — its ``--no-optimize-pdfs``
    — which changes no likelihood ratio.

References
~~~~~~~~~~
* CMS Collaboration, *The CMS statistical analysis and combination tool:
  Combine*, :xref:`2404.06614`
* Combine manual, https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import shutil
import tempfile
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np

from .manager import manager

__all__ = [
    "DatacardInterpreter",
    "combine_modules",
    "read_source",
    "resolve_shapes",
    "combine_options",
    "compile_datacard",
    "parse_datacard",
    "patch_datacard",
]


def __dir__():
    return __all__


log = logging.getLogger("Spey")

# pylint: disable=invalid-name, too-many-locals, too-many-branches
# pylint: disable=too-many-statements, too-many-instance-attributes
# pylint: disable=import-outside-toplevel

#: Physics model ``combine`` uses when none is requested: every signal process
#: is scaled by :math:`r`, every background by one.
DEFAULT_PHYSICS_MODEL = "HiggsAnalysis.CombinedLimit.PhysicsModel:defaultModel"

#: ``combine``'s two hypothesis model, whose parameter of interest interpolates
#: between two signal processes. :math:`x=1` selects the one whose name carries
#: the ``altSignal`` marker, ``"ALT"`` by default.
TWO_HYPOTHESIS_MODEL = "HiggsAnalysis.CombinedLimit.HiggsJPC:twoHypothesisHiggs"

#: The same model with the two hypotheses exchanged, so that :math:`x=1`
#: selects the process *without* the marker. This is the convention of
#: CMS-EXO-24-011, and the one ``spey`` needs when the marked process is the
#: background-like hypothesis. See :mod:`spey_roofit.physics_models`.
SWAPPED_TWO_HYPOTHESIS_MODEL = "spey_roofit.physics_models:twoHypothesisHiggs"

#: Name given to a signal sample attached through the interpreter.
DEFAULT_SAMPLE_NAME = "spey_signal"

#: Name of the observed dataset ``combine`` writes.
DATA_NAME = "data_obs"

#: Directives that carry a row of the systematics table, one entry per column
#: of the ``process`` lines. Every other directive attaches to a parameter.
_TABLE_TYPES = frozenset(
    {
        "lnN",
        "lnU",
        "gmM",
        "gmN",
        "trG",
        "unif",
        "dFD",
        "dFD2",
        "shape",
        "shapeN",
        "shapeU",
        "shape?",
        "shapeN?",
        "shapeU?",
        "shape2",
        "shapeN2",
    }
)


@contextlib.contextmanager
def _inside(directory: str):
    """
    Run a block with the working directory set to ``directory``.

    ``combine`` writes a ``combine_logger.out`` next to wherever it is run
    from. Building inside a temporary directory keeps it out of the user's, and
    it is removed together with that directory. Every template path has been
    made absolute by then, so nothing else depends on where the process stands.

    Args:
        directory (``str``): the directory to work in.

    Yields:
        ``None``
    """
    previous = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def _quiet():
    """
    Silence the output ``combine`` writes straight to ``stdout``.

    ``ModelBuilder`` reports what it is doing with plain ``print`` calls — one
    table per ``autoMCStats`` channel, one line per process of a physics model
    — and ``ROOT`` writes its own messages straight to the C++ standard output.
    Both are silenced: :obj:`sys.stdout` is redirected for the ``print`` calls,
    which in a notebook kernel never reach a file descriptor at all, and file
    descriptor 1 for everything that comes from C++. The second redirection is
    skipped when the descriptor cannot be duplicated, as happens when standard
    output has been closed.

    Yields:
        ``None``
    """
    with contextlib.redirect_stdout(io.StringIO()):
        try:
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


def combine_modules() -> SimpleNamespace:
    """
    Import the ``python`` modules of ``CMS Combine``.

    Raises:
        ``ImportError``: if ``combine`` is not installed.

    Returns:
        ``types.SimpleNamespace``:
        The ``DatacardParser``, ``ModelTools``, ``PhysicsModel`` and
        ``ShapeTools`` modules of ``HiggsAnalysis.CombinedLimit``.
    """
    manager.ROOT  # combine's modules import ROOT, so configure it first
    try:
        from HiggsAnalysis.CombinedLimit import (
            DatacardParser,
            ModelTools,
            PhysicsModel,
            ShapeTools,
        )
    except ImportError as err:  # pragma: no cover - environment dependent
        raise ImportError(
            "The CMS Combine backend drives combine itself. Install it with "
            "`conda install -c conda-forge cms-combine`."
        ) from err
    return SimpleNamespace(
        parser=DatacardParser,
        tools=ModelTools,
        physics=PhysicsModel,
        shapes=ShapeTools,
    )


def combine_options(**overrides: Any) -> Any:
    r"""
    Build the option object ``combine`` passes around internally.

    Every switch of ``text2workspace.py`` is available here under the name the
    tool gives it, so a card that needs one of them can still be compiled:
    ``combine_options(noCheckNorm=True)`` is ``--X-no-check-norm``.

    Three of them are set differently from ``combine``. ``allowNoSignal`` and
    ``allowNoBackground`` are on, because a background-only card is a normal
    input for this plug-in — the signal is attached to it afterwards. And
    ``noOptimizePdf`` is on, so that the Gaussian constraints are written as
    plain ``RooGaussian`` terms rather than as ``SimpleGaussianConstraint``:
    the latter drops the :math:`1/\sqrt{2\pi}` of every term, which would put
    the log-likelihood in a different normalisation from every other ``spey``
    backend. It changes no likelihood ratio either way.

    Args:
        overrides (``Any``): values to set on the object after ``combine``'s
          own defaults have been applied.

    Returns:
        ``optparse.Values``:
        The options, with ``combine``'s defaults and the requested overrides.
    """
    from optparse import OptionParser

    modules = combine_modules()
    parser = OptionParser()
    modules.parser.addDatacardParserOptions(parser)
    options, _ = parser.parse_args([])
    # `text2workspace.py` adds these two itself rather than through
    # addDatacardParserOptions, so they have to be set by hand
    options.physModel = DEFAULT_PHYSICS_MODEL
    options.physOpt = []
    options.bin = True
    options.mass = 0.0
    options.fileName = ""
    options.baseDir = "."
    options.out = None
    # a background-only card is a perfectly good input here: the signal is
    # attached afterwards, and combine would otherwise refuse to read it
    options.allowNoSignal = True
    options.allowNoBackground = True
    # combine writes its Gaussian constraints as SimpleGaussianConstraint,
    # which drops the 1/sqrt(2*pi) of every term; asking for plain RooGaussian
    # terms instead puts the log-likelihood in spey's normalisation
    options.noOptimizePdf = True
    for key, value in overrides.items():
        setattr(options, key, value)
    return options


# ---------------------------------------------------------------------- #
# Reading                                                                 #
# ---------------------------------------------------------------------- #


def read_source(
    source: str | os.PathLike, directory: str | None = None
) -> tuple[str, str]:
    """
    Read a datacard given as a path or as text.

    Args:
        source (``Union[str, os.PathLike]``): path of the card, or its text.
        directory (``str``, default ``None``): directory the relative paths of
          the ``shapes`` lines are resolved against. The directory of the card
          is used for a path, the working directory for text.

    Returns:
        ``Tuple[str, str]``:
        The text of the card and the directory to resolve against.
    """
    text = str(source)
    if len(text) < 4096 and os.path.isfile(text):
        path = os.path.abspath(text)
        with open(path, encoding="utf-8") as stream:
            text = stream.read()
        base = directory or os.path.dirname(path)
    else:
        base = directory or os.getcwd()
    return resolve_shapes(text, base), base


def resolve_shapes(text: str, directory: str) -> str:
    """
    Make the template paths of the ``shapes`` directives absolute.

    ``combine`` resolves a relative path against the directory of the datacard
    and nothing else, so a card that is rewritten — to attach a signal, say —
    or read from its text would lose track of its templates. Published cards
    also routinely carry the absolute path of the machine they were produced
    on; when that path does not exist, the file is looked for next to the card
    under the same base name, which is where the published tarballs put it.

    Args:
        text (``str``): the datacard.
        directory (``str``): directory the paths are resolved against.

    Returns:
        ``str``:
        The card, with every template path that could be resolved replaced by
        an absolute one.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        tokens = line.split()
        if tokens[:1] != ["shapes"] or len(tokens) < 5:
            continue
        candidate = tokens[3]
        if os.path.isfile(candidate):
            tokens[3] = os.path.abspath(candidate)
        elif os.path.isfile(os.path.join(directory, candidate)):
            tokens[3] = os.path.abspath(os.path.join(directory, candidate))
        else:
            fallback = os.path.join(directory, os.path.basename(candidate))
            if not os.path.isfile(fallback):
                continue
            log.debug(
                "The template file '%s' does not exist; using '%s' instead.",
                candidate,
                fallback,
            )
            tokens[3] = os.path.abspath(fallback)
        lines[index] = " ".join(tokens)
    return "\n".join(lines) + "\n"


def parse_datacard(
    source: str | os.PathLike,
    mass: int | float = 0.0,
    directory: str | None = None,
    options: Any | None = None,
) -> tuple[Any, Any]:
    """
    Parse a datacard with ``combine``'s own reader.

    Args:
        source (``Union[str, os.PathLike]``): path of the ``.txt`` datacard, or
          its text.
        mass (``Union[int, float]``, default ``0.0``): value substituted for
          ``$MASS``, i.e. the ``-m`` option of ``text2workspace.py``.
        directory (``str``, default ``None``): directory the relative paths of
          the ``shapes`` lines are resolved against.
        options (``optparse.Values``, default ``None``): options for
          ``combine``, see :func:`combine_options`.

    Raises:
        ``ImportError``: if ``combine`` is not installed.
        ``RuntimeError``: propagated from ``combine`` when the card is
          malformed.

    Returns:
        ``Tuple[Datacard, optparse.Values]``:
        ``combine``'s parsed card and the options it was parsed with, whose
        ``fileName`` and ``baseDir`` point at it.
    """
    modules = combine_modules()
    settings = options if options is not None else combine_options()
    text, base = read_source(source, directory)

    # combine reads the card from a file and takes `baseDir` from where that
    # file sits, so a card given as text has to be written out first; every
    # template path is absolute by now, so the temporary location is harmless
    handle, path = tempfile.mkstemp(prefix="spey_datacard_", suffix=".txt")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(text)

    settings.fileName = path
    settings.baseDir = base
    settings.mass = float(mass)
    try:
        with open(path, encoding="utf-8") as stream, _quiet():
            card = modules.parser.parseCard(stream, settings)
    finally:
        os.unlink(path)
    return card, settings


def compile_datacard(
    source: str | os.PathLike,
    mass: int | float = 0.0,
    physics_model: str | None = None,
    physics_options: Sequence[str] = (),
    directory: str | None = None,
    options: Any | None = None,
) -> tuple[Any, Any]:
    r"""
    Compile a datacard into a ``RooFit`` workspace, with ``combine``.

    This is ``text2workspace.py``, driven in process: the card is read by
    ``combine``'s ``DatacardParser`` and built by its ``ShapeBuilder``, so the
    workspace is the one ``combine`` would have written.

    Args:
        source (``Union[str, os.PathLike]``): path of the ``.txt`` datacard, or
          its text.
        mass (``Union[int, float]``, default ``0.0``): value of the ``-m``
          option, substituted for ``$MASS``.
        physics_model (``str``, default ``None``): the ``-P`` option, in the
          form ``"module:object"``. ``combine``'s default model — every signal
          process scaled by :math:`r` — is used when omitted. A card such as
          CMS-EXO-24-011 asks for
          :data:`~spey_roofit.combine.TWO_HYPOTHESIS_MODEL` instead.
        physics_options (``Sequence[str]``, default ``()``): the ``--PO``
          options of the physics model.
        directory (``str``, default ``None``): directory the relative paths of
          the ``shapes`` lines are resolved against.
        options (``optparse.Values``, default ``None``): further options for
          ``combine``, see :func:`combine_options`.

    Raises:
        ``ImportError``: if ``combine`` is not installed.
        ``RuntimeError``: propagated from ``combine`` when the card cannot be
          compiled.

    Returns:
        ``Tuple[ROOT.RooWorkspace, ROOT.RooStats.ModelConfig]``:
        The workspace and its model configuration.
    """
    settings = options if options is not None else combine_options()
    if physics_model is not None:
        settings.physModel = physics_model
    settings.physOpt = list(physics_options)

    card, settings = parse_datacard(
        source, mass=mass, directory=directory, options=settings
    )
    if not card.hasShapes:
        # combine builds a counting experiment as a product of RooPoisson terms
        # over plain variables, which is not a binned shape spey can read; one
        # bin templates give the same product of Poisson terms as a binned model
        text, _ = read_source(source, directory)
        text, base = _counting_to_shapes(card, text)
        card, settings = parse_datacard(text, mass=mass, directory=base, options=settings)

    builder = combine_modules().shapes.ShapeBuilder(card, settings)
    builder.setPhysics(_physics_model(settings))

    workdir = tempfile.mkdtemp(prefix="spey_combine_")
    settings.out = os.path.join(workdir, "workspace.root")
    try:
        with _inside(workdir), _quiet():
            builder.doModel()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    workspace = builder.out
    model_config = workspace.obj("ModelConfig")
    if not model_config:  # pragma: no cover - combine always writes one
        raise RuntimeError("combine did not write a ModelConfig into the workspace.")
    _complete_model_config(workspace, model_config)
    _widen_global_observables(model_config)
    manager.keep_alive(workspace, builder)
    return workspace, model_config


def _complete_model_config(workspace: Any, model_config: Any) -> None:
    """
    Declare every floating parameter of the model as a nuisance parameter.

    ``combine`` keeps its ``nuisances`` set for the parameters that carry a
    constraint term, and leaves a ``rateParam`` or a ``flatParam`` out of it
    even though the likelihood depends on it. ``spey`` reads the set to decide
    what to profile, so anything that floats has to be in it — otherwise the
    parameter would silently stay at its nominal value.

    Args:
        workspace (``ROOT.RooWorkspace``): the compiled workspace.
        model_config (``ROOT.RooStats.ModelConfig``): its model configuration.
    """
    ROOT = manager.ROOT

    declared = model_config.GetNuisanceParameters()
    nuisances = ROOT.RooArgSet(declared) if declared else ROOT.RooArgSet()
    reserved = {
        parameter.GetName()
        for group in (
            model_config.GetParametersOfInterest(),
            model_config.GetObservables(),
            model_config.GetGlobalObservables(),
        )
        if group
        for parameter in group
    }

    added = []
    for parameter in model_config.GetPdf().getParameters(model_config.GetObservables()):
        if not parameter.InheritsFrom("RooRealVar") or parameter.isConstant():
            continue
        if parameter.GetName() in reserved or nuisances.find(parameter.GetName()):
            continue
        nuisances.add(parameter)
        added.append(parameter.GetName())
    if added:
        log.debug(
            "Adding %s to the nuisance parameters: they float but combine does "
            "not list them.",
            ", ".join(sorted(added)),
        )
        model_config.SetNuisanceParameters(nuisances)


def _widen_global_observables(model_config: Any, factor: float = 12.0) -> None:
    r"""
    Give every auxiliary measurement room around its constraint.

    The likelihood normalises a constraint term over its global observable, so
    a range that cuts into the tail of the constraint makes that normalisation
    depend on the nuisance parameter. ``combine`` never notices — it does not
    normalise its constraints at all — but it gives a ``shape`` parameter the
    range :math:`[-4,4]`, which at :math:`	heta=-1.5` already costs six parts
    in a thousand of the Gaussian.

    Args:
        model_config (``ROOT.RooStats.ModelConfig``): the model configuration.
        factor (``float``, default ``12.0``): how much wider to make the range.
    """
    declared = model_config.GetGlobalObservables()
    if not declared:
        return
    for observable in declared:
        if not observable.InheritsFrom("RooRealVar"):
            continue
        centre = observable.getVal()
        half = max(observable.getMax() - centre, centre - observable.getMin())
        if half <= 0.0:
            continue
        lower = centre - factor * half
        if observable.getMin() >= 0.0:
            # a Poisson sideband measurement cannot go negative
            lower = max(lower, 0.0)
        observable.setRange(lower, centre + factor * half)


def _physics_model(options: Any) -> Any:
    """
    Import and configure the physics model named by the options.

    Args:
        options (``optparse.Values``): the options, whose ``physModel`` is of
          the form ``"module:object"``.

    Raises:
        ``RuntimeError``: if the module or the object cannot be found, or if
          the object is not a ``combine`` physics model.

    Returns:
        ``PhysicsModelBase``:
        The model, with its ``--PO`` options applied.
    """
    import importlib
    import sys

    modules = combine_modules()
    module_name, _, object_name = options.physModel.partition(":")
    try:
        importlib.import_module(module_name)
    except ImportError as err:
        raise RuntimeError(f"Physics model module '{module_name}' not found.") from err
    model = getattr(sys.modules[module_name], object_name, None)
    if model is None or not isinstance(model, modules.physics.PhysicsModelBase):
        raise RuntimeError(
            f"'{object_name}' in '{module_name}' is not a combine physics model."
        )
    model.setPhysicsOptions(list(options.physOpt))
    return model


# ---------------------------------------------------------------------- #
# Rewriting                                                               #
# ---------------------------------------------------------------------- #


def _write_histograms(path: str, histograms: dict[str, Sequence[float]]) -> None:
    r"""
    Write a flat ``ROOT`` file of one-dimensional templates.

    The bin uncertainties are set to zero. A ``TH1`` that is filled with
    ``SetBinContent`` alone reports :math:`\sqrt{n}` instead, which
    ``autoMCStats`` would read as a Monte Carlo uncertainty of a yield the user
    gave exactly.

    Args:
        path (``str``): path of the file to write.
        histograms (``Dict[str, Sequence[float]]``): bin contents, keyed by the
          name each histogram gets in the file.
    """
    ROOT = manager.ROOT

    handle = ROOT.TFile.Open(path, "RECREATE")
    for name, values in histograms.items():
        histogram = ROOT.TH1D(name, name, len(values), 0.0, float(len(values)))
        histogram.SetDirectory(0)
        histogram.Sumw2()
        for index, value in enumerate(values):
            histogram.SetBinContent(index + 1, float(value))
            histogram.SetBinError(index + 1, 0.0)
        histogram.Write()
    handle.Close()


def _strip_comment(line: str) -> str:
    """Drop a trailing comment, the way ``combine`` does in a systematics line."""
    position = line.find("#")
    return line[:position].rstrip() if position > 0 else line.rstrip()


class _Card:
    """
    Line oriented view of a datacard, for rewriting it.

    The card is kept as text and only the lines that have to change are
    touched, so that every directive the plug-in does not know about survives
    verbatim into the card ``combine`` is finally given.

    Args:
        text (``str``): the datacard.

    Raises:
        ``ValueError``: if the text has no ``rate`` line, i.e. is not a
          datacard at all.
    """

    def __init__(self, text: str):
        self.lines: list[str] = text.splitlines()
        rate = self._find("rate")
        if rate is None:
            raise ValueError("The text does not look like a datacard: no 'rate' line.")
        self.rate: int = rate
        self.process_ids: int = self._find("process", before=self.rate)
        self.process_names: int = self._find("process", before=self.process_ids)
        self.columns: int = self._find("bin", before=self.process_names)

    def _find(self, keyword: str, before: int | None = None) -> int | None:
        """
        Index of the last line starting with ``keyword`` before a position.

        Args:
            keyword (``str``): the first token to look for.
            before (``int``, default ``None``): only consider lines before this
              index. The whole card is searched when omitted.

        Returns:
            ``int`` or ``None``:
            The index, or ``None`` if there is no such line.
        """
        limit = len(self.lines) if before is None else before
        for index in range(limit - 1, -1, -1):
            tokens = self.lines[index].split()
            if tokens and tokens[0] == keyword:
                return index
        return None

    @property
    def table_rows(self) -> list[int]:
        """
        Indices of the systematics lines that carry a table.

        Returns:
            ``List[int]``:
            One index per line whose type has one entry per column.
        """
        rows = []
        for index in range(self.rate + 1, len(self.lines)):
            tokens = _strip_comment(self.lines[index]).split()
            if len(tokens) > 1 and tokens[1] in _TABLE_TYPES:
                rows.append(index)
        return rows

    def entries(self, index: int) -> tuple[list[str], list[str]]:
        """
        Split a line into its leading keywords and its per-column entries.

        Args:
            index (``int``): index of the line.

        Returns:
            ``Tuple[List[str], List[str]]``:
            The tokens that come before the table and the table itself. A
            ``gmN`` line carries its sideband count among the keywords.
        """
        tokens = _strip_comment(self.lines[index]).split()
        if index in (self.columns, self.process_names, self.process_ids, self.rate):
            return tokens[:1], tokens[1:]
        keep = 3 if tokens[1] == "gmN" else 2
        return tokens[:keep], tokens[keep:]

    def set_entries(
        self, index: int, keywords: Sequence[str], entries: Sequence[str]
    ) -> None:
        """
        Write a line back from its keywords and its per-column entries.

        Args:
            index (``int``): index of the line.
            keywords (``Sequence[str]``): the leading tokens.
            entries (``Sequence[str]``): the table.
        """
        self.lines[index] = " ".join(list(keywords) + list(entries))

    @property
    def text(self) -> str:
        """
        The card as text.

        Returns:
            ``str``:
            The lines, newline terminated.
        """
        return "\n".join(self.lines) + "\n"


def patch_datacard(
    source: str | os.PathLike,
    injections: dict[str, Sequence[float]] | None = None,
    strip_signal: bool = False,
    sample_name: str = DEFAULT_SAMPLE_NAME,
    directory: str | None = None,
    workdir: str | None = None,
) -> tuple[str, str]:
    r"""
    Attach a signal to a datacard, or remove the one it already has.

    The card is rewritten rather than the workspace: only the lines that have
    to change are touched, and ``combine`` then compiles the result with the
    same code that built the published model. A signal is added as one extra
    process per channel, scaled by the parameter of interest and affected by no
    nuisance parameter.

    Args:
        source (``Union[str, os.PathLike]``): path of the datacard, or its text.
        injections (``Dict[str, Sequence[float]]``, default ``None``):
          per-channel signal yields, one entry per bin of the channel.
        strip_signal (``bool``, default ``False``): remove every process the
          card declares with an index :math:`\leq 0`.
        sample_name (``str``, default ``"spey_signal"``): name of the process
          created for ``injections``.
        directory (``str``, default ``None``): directory the relative paths of
          the ``shapes`` lines are resolved against.
        workdir (``str``, default ``None``): directory the templates of the
          injected signal are written to. A temporary one is used when omitted.

    Raises:
        ``ValueError``: if the card already has a process called
          ``sample_name``, if a channel of ``injections`` is not a channel of
          the card, or if stripping would leave a channel empty.

    Returns:
        ``Tuple[str, str]``:
        The patched card and the directory its relative paths resolve against.
    """
    prepared = {
        channel: np.atleast_1d(np.asarray(values, dtype=np.float64)).ravel()
        for channel, values in (injections or {}).items()
    }
    text, base = read_source(source, directory)
    card, _ = parse_datacard(text, directory=base)
    if prepared and sample_name in card.processes:
        raise ValueError(
            f"The datacard already has a process called '{sample_name}'; give "
            f"the injected signal another name."
        )
    unknown = set(prepared) - set(card.bins)
    if unknown:
        raise ValueError(
            f"{', '.join(sorted(unknown))} are not channels of the datacard. "
            f"Available channels: {', '.join(card.bins)}."
        )

    view = _Card(text)
    _relax_counts(view)
    if strip_signal:
        _strip_columns(view, card)
    if prepared:
        workdir = workdir or tempfile.mkdtemp(prefix="spey_signal_")
        _add_columns(view, card, prepared, sample_name, workdir)
        base = workdir
    return view.text, base


def _relax_counts(view: _Card) -> None:
    """
    Turn the ``imax``/``jmax``/``kmax`` declarations into wildcards.

    Adding or removing a process would otherwise make the card disagree with
    its own header.

    Args:
        view (``_Card``): the card being rewritten.
    """
    for index, line in enumerate(view.lines):
        tokens = line.split()
        if tokens and tokens[0] in ("imax", "jmax", "kmax"):
            view.lines[index] = f"{tokens[0]} *"


def _strip_columns(view: _Card, card: Any) -> None:
    """
    Remove every column of the table that belongs to a signal process.

    Args:
        view (``_Card``): the card being rewritten.
        card (``Datacard``): the parsed card, which says what is signal.

    Raises:
        ``ValueError``: if a channel would be left without any process.
    """
    _, processes = view.entries(view.process_names)
    _, channels = view.entries(view.columns)
    keep = [
        index
        for index, process in enumerate(processes)
        if not card.isSignal.get(process, False)
    ]
    if not keep:
        raise ValueError("Stripping the signal would leave the datacard empty.")
    missing = set(channels) - {channels[index] for index in keep}
    if missing:
        raise ValueError(
            f"Stripping the signal would leave {', '.join(sorted(missing))} "
            f"without any process."
        )

    for index in [view.columns, view.process_names, view.process_ids, view.rate]:
        keywords, entries = view.entries(index)
        view.set_entries(index, keywords, [entries[position] for position in keep])
    for index in view.table_rows:
        keywords, entries = view.entries(index)
        view.set_entries(index, keywords, [entries[position] for position in keep])


def _add_columns(
    view: _Card,
    card: Any,
    injections: dict[str, np.ndarray],
    sample_name: str,
    workdir: str,
) -> None:
    """
    Append one column per injected channel to the table.

    Args:
        view (``_Card``): the card being rewritten.
        card (``Datacard``): the parsed card.
        injections (``Dict[str, np.ndarray]``): per-channel signal yields.
        sample_name (``str``): name of the process to create.
        workdir (``str``): directory the templates are written to.
    """
    _, identifiers = view.entries(view.process_ids)
    fresh = str(min([int(value) for value in identifiers] + [1]) - 1)

    if card.hasShapes:
        rootfile = os.path.join(workdir, f"{sample_name}.root")
        _write_histograms(
            rootfile,
            {
                f"{channel}_{sample_name}": values
                for channel, values in injections.items()
            },
        )
        anchor = max(
            (
                index
                for index in range(view.columns)
                if view.lines[index].split()[:1] == ["shapes"]
            ),
            default=view.columns - 1,
        )
        # the entry has to name the channel: combine looks a process up inside
        # the map of its own channel first, so a generic `shapes * <channel>`
        # line of the card would otherwise win over a `shapes <process> *` one
        for offset, channel in enumerate(sorted(injections)):
            view.lines.insert(
                anchor + 1 + offset,
                f"shapes {sample_name} {channel} {rootfile} {channel}_{sample_name}",
            )
        view.__init__(view.text)  # pylint: disable=unnecessary-dunder-call

    added = sorted(injections)
    rates = [
        "-1" if card.hasShapes else "%.10g" % float(np.sum(injections[channel]))
        for channel in added
    ]
    for index, extra in (
        (view.columns, added),
        (view.process_names, [sample_name] * len(added)),
        (view.process_ids, [fresh] * len(added)),
        (view.rate, rates),
    ):
        keywords, entries = view.entries(index)
        view.set_entries(index, keywords, list(entries) + extra)
    for index in view.table_rows:
        keywords, entries = view.entries(index)
        view.set_entries(index, keywords, list(entries) + ["-"] * len(added))


def _counting_to_shapes(card: Any, text: str) -> tuple[str, str]:
    """
    Give a counting datacard one-bin templates.

    ``combine`` builds a card without ``shapes`` directives as a product of
    ``RooPoisson`` terms over plain variables. That is the same likelihood as
    the binned model it builds from templates, but it is not a shape
    :class:`~spey_roofit.data.WorkspaceData` can read, so the card is given one
    bin per channel and compiled as a shape card.

    Args:
        card (``Datacard``): the parsed counting card.
        text (``str``): the card, as text.

    Returns:
        ``Tuple[str, str]``:
        The card with ``shapes`` directives and the directory holding the
        templates.
    """
    workdir = tempfile.mkdtemp(prefix="spey_counting_")
    rootfile = os.path.join(workdir, "templates.root")

    histograms: dict[str, Sequence[float]] = {}
    for channel in card.bins:
        histograms[f"{channel}_{DATA_NAME}"] = [float(card.obs.get(channel, 0.0))]
        for process, rate in card.exp[channel].items():
            histograms[f"{channel}_{process}"] = [float(rate)]
    _write_histograms(rootfile, histograms)

    view = _Card(text)
    view.lines.insert(view.columns, f"shapes * * {rootfile} $CHANNEL_$PROCESS")
    log.debug("Counting card given one bin templates in %s", rootfile)
    return view.text, workdir


# ---------------------------------------------------------------------- #
# Interpreter                                                             #
# ---------------------------------------------------------------------- #


class DatacardInterpreter:
    r"""
    Read a ``CMS Combine`` datacard and attach a signal to it.

    The interpreter is the part of the plug-in a user is expected to touch. It
    answers what the channels are, how many bins each of them has, which
    samples they contain and what each of them is expected to yield — all read
    from the card by ``combine``'s own parser, before anything is compiled —
    and it attaches a signal to a published background model. The compilation
    itself is ``combine``'s.

    Args:
        datacard (``Union[str, os.PathLike]``): path of the ``.txt`` datacard,
          or its text.
        mass (``Union[int, float]``, default ``0.0``): value substituted for
          ``$MASS``, i.e. the ``-m`` option of ``text2workspace.py``.
        directory (``str``, default ``None``): directory the relative paths of
          the ``shapes`` lines are resolved against.
        options (``optparse.Values``, default ``None``): options for
          ``combine``, see :func:`combine_options`.

    Raises:
        ``ImportError``: if ``combine`` or ``PyROOT`` is not installed.
        ``RuntimeError``: propagated from ``combine`` when the card is
          malformed.

    Example:

    .. code-block:: python3
        :linenos:

        >>> from spey_roofit import DatacardInterpreter

        >>> interpreter = DatacardInterpreter("chi_datacard.txt")
        >>> interpreter.bin_map
        {'bin0': 12, 'bin1': 12}
        >>> interpreter.summary(max_channels=2)
        >>> interpreter.inject_signals({"bin0": [1.0, 2.0, 3.0]})
    """

    __slots__ = [
        "_card",
        "_options",
        "_text",
        "_directory",
        "_builder",
        "_signal",
        "_strip",
        "_mass",
    ]

    def __init__(
        self,
        datacard: str | os.PathLike,
        mass: int | float = 0.0,
        directory: str | None = None,
        options: Any | None = None,
    ):
        text, base = read_source(datacard, directory)
        self._directory: str = base
        self._text: str = text
        self._mass: float = float(mass)
        self._card, self._options = parse_datacard(
            text, mass=mass, directory=self._directory, options=options
        )
        self._builder: Any | None = None
        self._signal: dict[str, np.ndarray] = {}
        self._strip: bool = False

    def __repr__(self) -> str:
        return (
            f"DatacardInterpreter({len(self.channels)} channels, "
            f"{sum(self.bin_map.values())} bins, "
            f"{len(self._card.systs)} systematics)"
        )

    def __getitem__(self, item: str) -> list[float]:
        return self.expected_background_yields[item]

    # ------------------------------------------------------------------ #
    # Accessors                                                           #
    # ------------------------------------------------------------------ #

    @property
    def datacard(self) -> Any:
        """
        The card as ``combine`` parsed it.

        Returns:
            ``HiggsAnalysis.CombinedLimit.Datacard.Datacard``:
            ``combine``'s own container, with ``bins``, ``processes``,
            ``isSignal``, ``obs``, ``exp``, ``systs`` and ``shapeMap``.
        """
        return self._card

    @property
    def text(self) -> str:
        """
        The datacard, as text.

        Returns:
            ``str``:
            The card the interpreter was built from, before any mutation.
        """
        return self._text

    @property
    def directory(self) -> str:
        """
        Directory the ``shapes`` directives are resolved against.

        Returns:
            ``str``:
            The directory holding the card, or the one given to the constructor.
        """
        return self._directory

    @property
    def channels(self) -> list[str]:
        """
        Names of the channels.

        Returns:
            ``List[str]``:
            Channel names, in the order of the datacard.
        """
        return list(self._card.bins)

    @property
    def processes(self) -> list[str]:
        """
        Names of the processes.

        Returns:
            ``List[str]``:
            Process names, in the order of the datacard.
        """
        return list(self._card.processes)

    @property
    def signal_processes(self) -> list[str]:
        """
        Processes the card declares with an index :math:`\\leq 0`.

        Returns:
            ``List[str]``:
            Names of the signal processes.
        """
        return [name for name in self._card.processes if self._card.isSignal.get(name)]

    @property
    def nuisance_parameters(self) -> list[str]:
        """
        Names of the nuisance parameters the card declares.

        Returns:
            ``List[str]``:
            One name per systematics line, in declaration order.
        """
        return [syst[0] for syst in self._card.systs]

    @property
    def bin_map(self) -> dict[str, int]:
        """
        Number of bins of every channel.

        This is the shape ``signal_yields`` has to follow.

        .. note::

            ``combine`` pads every channel of the compiled workspace to the
            longest template it finds, so a model built from channels of
            different lengths has more bins per channel than this reports. The
            padded bins hold no data and no expectation.

        Returns:
            ``Dict[str, int]``:
            Mapping from channel name to number of bins.
        """
        return {channel: self._nbins(channel) for channel in self.channels}

    @property
    def samples(self) -> dict[str, list[str]]:
        """
        Samples of every channel.

        Returns:
            ``Dict[str, List[str]]``:
            Mapping from channel name to the processes contributing to it,
            including any signal attached with :meth:`inject_signals`.
        """
        out: dict[str, list[str]] = {}
        for channel in self.channels:
            names = [
                process
                for process in self._card.processes
                if process in self._card.exp[channel]
                and self._card.exp[channel][process] != 0.0
                and not (self._strip and self._card.isSignal.get(process))
            ]
            if channel in self._signal:
                names.append(DEFAULT_SAMPLE_NAME)
            out[channel] = names
        return out

    @property
    def signal_samples(self) -> dict[str, list[str]]:
        """
        Samples that are scaled by the parameter of interest.

        Returns:
            ``Dict[str, List[str]]``:
            Mapping from channel name to the names of its signal samples.
        """
        return {
            channel: [
                process
                for process in processes
                if self._card.isSignal.get(process) or process == DEFAULT_SAMPLE_NAME
            ]
            for channel, processes in self.samples.items()
        }

    @property
    def observed_data(self) -> dict[str, list[float]]:
        """
        Observed counts of every channel.

        Returns:
            ``Dict[str, List[float]]``:
            Mapping from channel name to per-bin observed counts.
        """
        return {channel: self._template(channel, DATA_NAME) for channel in self.channels}

    def sample_yields(
        self, channel: str | None = None, sample: str | None = None
    ) -> dict[str, dict[str, list[float]]]:
        """
        Expected yields of every sample at the nominal nuisance parameters.

        Args:
            channel (``str``, default ``None``): restrict the result to one
              channel. All of them are returned when omitted.
            sample (``str``, default ``None``): restrict the result to one
              sample. All of them are returned when omitted.

        Raises:
            ``ValueError``: if ``channel`` is not a channel of the card.

        Returns:
            ``Dict[str, Dict[str, List[float]]]``:
            Mapping ``{channel: {sample: per-bin yields}}``.
        """
        channels = self.channels if channel is None else [self._check_channel(channel)]
        samples = self.samples
        out: dict[str, dict[str, list[float]]] = {}
        for name in channels:
            entries: dict[str, list[float]] = {}
            for process in samples[name]:
                if process == DEFAULT_SAMPLE_NAME and name in self._signal:
                    entries[process] = self._signal[name].tolist()
                else:
                    entries[process] = self._template(name, process)
            if sample is not None:
                entries = {key: value for key, value in entries.items() if key == sample}
            out[name] = entries
        return out

    @property
    def expected_background_yields(self) -> dict[str, list[float]]:
        r"""
        Background yields, i.e. the expectation at :math:`\mu=0`.

        Returns:
            ``Dict[str, List[float]]``:
            Mapping from channel name to per-bin background yields.
        """
        return self._sum_samples(signal=False)

    @property
    def expected_signal_yields(self) -> dict[str, list[float]]:
        r"""
        Signal yields, i.e. what the parameter of interest scales.

        Returns:
            ``Dict[str, List[float]]``:
            Mapping from channel name to per-bin signal yields.
        """
        return self._sum_samples(signal=True)

    @property
    def is_alive(self) -> bool:
        """
        Whether the model has a signal at all.

        Returns:
            ``bool``:
            ``True`` if at least one bin of one channel has a signal sample
            with a non-zero yield.
        """
        return any(
            any(value > 0.0 for value in values)
            for values in self.expected_signal_yields.values()
        )

    def summary(
        self,
        max_channels: int | None = None,
        show_samples: bool = True,
        print_out: bool = True,
    ) -> str:
        """
        Human readable overview of the model the datacard describes.

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
        card = self._card
        background = self.expected_background_yields
        signal = self.expected_signal_yields
        observed = self.observed_data
        signal_samples = self.signal_samples
        yields = self.sample_yields() if show_samples else {}

        kinds: dict[str, int] = {}
        for syst in card.systs:
            kinds[syst[2]] = kinds.get(syst[2], 0) + 1
        breakdown = ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items()))

        lines = [
            f"CMS Combine datacard ({'shape' if card.hasShapes else 'counting'})",
            f"  channels            : {len(card.bins)}",
            f"  bins                : {sum(self.bin_map.values())}",
            f"  processes           : {', '.join(card.processes)}",
            f"  signal processes    : {', '.join(self.signal_processes) or 'none'}",
            f"  nuisance parameters : {len(card.systs)}"
            + (f" ({breakdown})" if breakdown else ""),
            f"  rate parameters     : {len(card.rateParams)}",
            f"  autoMCStats         : "
            f"{', '.join(card.binParFlags) if card.binParFlags else 'none'}",
            "",
        ]
        selection = card.bins if max_channels is None else card.bins[:max_channels]
        for name in selection:
            nbins = self.bin_map[name]
            lines.append(f"  {name} ({nbins} bin{'s' if nbins > 1 else ''})")
            lines.append(f"      observed   : {self._fmt(observed[name])}")
            lines.append(f"      background : {self._fmt(background[name])}")
            if any(value != 0.0 for value in signal[name]):
                lines.append(f"      signal     : {self._fmt(signal[name])}")
            if name in self._signal:
                lines.append(f"      injected   : {self._fmt(self._signal[name])}")
            if show_samples:
                for sample, values in yields[name].items():
                    tag = " (signal)" if sample in signal_samples[name] else ""
                    lines.append(f"        - {sample}{tag}: {self._fmt(values)}")
        if max_channels is not None and len(card.bins) > max_channels:
            lines.append(f"  ... {len(card.bins) - max_channels} more channel(s)")

        text = "\n".join(lines)
        if print_out:
            print(text)
        return text

    # ------------------------------------------------------------------ #
    # Signal editing                                                      #
    # ------------------------------------------------------------------ #

    def inject_signal(
        self, channel: str, yields: float | Sequence[float]
    ) -> DatacardInterpreter:
        r"""
        Record signal yields to be added to a channel.

        The signal enters the model as one extra process of the datacard,
        scaled by the parameter of interest and affected by no nuisance
        parameter.

        Args:
            channel (``str``): name of the channel, see :attr:`channels`.
            yields (``Union[float, Sequence[float]]``): signal yields, one per
              bin of the channel. A scalar is accepted for a one-bin channel.

        Raises:
            ``ValueError``: if ``channel`` is unknown, if the number of yields
              does not match the number of bins, or if the channel already has
              an injected signal.

        Returns:
            ~spey_roofit.combine.DatacardInterpreter:
            ``self``, so that calls can be chained.
        """
        name = self._check_channel(channel)
        values = np.atleast_1d(np.asarray(yields, dtype=np.float64)).ravel()
        expected = self._nbins(name)
        if values.size != expected:
            raise ValueError(
                f"Channel '{name}' has {expected} bin(s) but {values.size} "
                f"signal yield(s) were given."
            )
        if name in self._signal:
            raise ValueError(
                f"Channel '{name}' already has an injected signal. "
                "Call `reset_signal()` first."
            )
        self._signal[name] = values
        return self

    def inject_signals(
        self, signal_yields: dict[str, float | Sequence[float]]
    ) -> DatacardInterpreter:
        """
        Record signal yields for several channels at once.

        Args:
            signal_yields (``Dict[str, Union[float, Sequence[float]]]``):
              mapping from channel name to per-bin signal yields.

        Raises:
            ``ValueError``: propagated from :meth:`inject_signal`.

        Returns:
            ~spey_roofit.combine.DatacardInterpreter:
            ``self``, so that calls can be chained.
        """
        for channel, yields in signal_yields.items():
            self.inject_signal(channel, yields)
        return self

    def reset_signal(self) -> DatacardInterpreter:
        """
        Forget every injected signal.

        Returns:
            ~spey_roofit.combine.DatacardInterpreter:
            ``self``, so that calls can be chained.
        """
        self._signal.clear()
        return self

    def strip_signal(self, strip: bool = True) -> DatacardInterpreter:
        """
        Drop the signal processes the datacard already declares.

        Use this to reinterpret a published card with a different signal: every
        process declared with an index :math:`\\leq 0` is removed from the card
        ``combine`` is given, leaving a background-only model to inject into.

        Args:
            strip (``bool``, default ``True``): whether to remove the declared
              signal processes.

        Returns:
            ~spey_roofit.combine.DatacardInterpreter:
            ``self``, so that calls can be chained.
        """
        self._strip = bool(strip)
        return self

    @property
    def signal_per_channel(self) -> dict[str, list[float]]:
        """
        Signal yields recorded so far.

        Returns:
            ``Dict[str, List[float]]``:
            Mapping from channel name to the injected per-bin yields.
        """
        return {channel: values.tolist() for channel, values in self._signal.items()}

    # ------------------------------------------------------------------ #
    # Compilation                                                         #
    # ------------------------------------------------------------------ #

    @property
    def patch(self) -> Any:
        """
        The compiled ``RooFit`` workspace, with every mutation applied.

        Returns:
            ``ROOT.RooWorkspace``:
            Workspace holding the model, its data and its model configuration.
        """
        return self.workspace()[0]

    def save_patch(self, path: str | os.PathLike) -> str:
        """
        Write the patched datacard to a file.

        Combined with :meth:`strip_signal`, this turns a card that declares a
        signal into a background-only one; combined with :meth:`inject_signal`,
        it stores a reinterpretation. The card is the one ``combine`` is given:
        the ``shapes`` directives of the original point at absolute paths, so
        the saved card keeps working wherever it is moved to, and the
        templates of an injected signal are copied next to it.

        Args:
            path (``Union[str, os.PathLike]``): destination of the card, e.g.
              ``"background_only.txt"``. An existing file is overwritten.

        Returns:
            ``str``:
            The path the card was written to.

        Example:

        .. code-block:: python3

            >>> interpreter = DatacardInterpreter("chi_datacard.txt")
            >>> interpreter.strip_signal().save_patch("background_only.txt")
        """
        path = os.path.abspath(str(path))
        text, base = patch_datacard(
            self._text,
            injections=self._signal,
            strip_signal=self._strip,
            directory=self._directory,
        )
        target = os.path.dirname(path)
        os.makedirs(target, exist_ok=True)

        lines = text.splitlines()
        for index, line in enumerate(lines):
            tokens = line.split()
            if tokens[:1] != ["shapes"] or len(tokens) < 5:
                continue
            # only the templates of an injected signal live in the work directory
            if base == self._directory or os.path.dirname(tokens[3]) != base:
                continue
            destination = os.path.join(target, os.path.basename(tokens[3]))
            if os.path.abspath(tokens[3]) != destination:
                shutil.copyfile(tokens[3], destination)
            tokens[3] = os.path.basename(tokens[3])
            lines[index] = " ".join(tokens)

        with open(path, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
        log.info("Patched datacard saved to '%s'.", path)
        return path

    def workspace(self, **kwargs: Any) -> tuple[Any, Any]:
        """
        Compile the model into a ``RooFit`` workspace, with ``combine``.

        Args:
            kwargs (``Any``): forwarded to
              :func:`~spey_roofit.combine.compile_datacard`.

        Returns:
            ``Tuple[ROOT.RooWorkspace, ROOT.RooStats.ModelConfig]``:
            The workspace and its model configuration.
        """
        if self._signal or self._strip:
            text, directory = patch_datacard(
                self._text,
                injections=self._signal,
                strip_signal=self._strip,
                directory=self._directory,
            )
        else:
            text, directory = self._text, self._directory
        kwargs.setdefault("mass", self._mass)
        kwargs.setdefault("directory", directory)
        return compile_datacard(text, **kwargs)

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _shape_builder(self) -> Any | None:
        """
        ``combine``'s template reader for this card.

        Returns:
            ``ShapeBuilder`` or ``None``:
            The builder, or ``None`` for a counting card, which has no
            templates to read.
        """
        if not self._card.hasShapes:
            return None
        if self._builder is None:
            builder = combine_modules().shapes.ShapeBuilder(self._card, self._options)
            # resolves the `rate -1` entries against the template integrals and
            # checks them against the ones the card states, as combine does
            with _quiet():
                builder.prepareAllShapes()
            self._builder = builder
        return self._builder

    def _template(self, channel: str, process: str) -> list[float]:
        """
        Per-bin yields of one process of one channel, padded to the channel.

        Args:
            channel (``str``): name of the channel.
            process (``str``): name of the process, or ``"data_obs"``.

        Returns:
            ``List[float]``:
            One entry per bin of the channel.
        """
        values = self._raw_template(channel, process)
        nbins = self._nbins(channel)
        return values + [0.0] * (nbins - len(values))

    def _raw_template(self, channel: str, process: str) -> list[float]:
        """
        Per-bin yields of one process of one channel, as the card gives them.

        The shape comes from the histogram and the normalisation from the
        ``rate`` line, which is what ``combine`` does as well.

        Args:
            channel (``str``): name of the channel.
            process (``str``): name of the process, or ``"data_obs"``.

        Returns:
            ``List[float]``:
            One entry per bin of the template.
        """
        builder = self._shape_builder()
        if builder is None:
            if process == DATA_NAME:
                return [float(self._card.obs.get(channel, 0.0))]
            return [float(self._card.exp[channel].get(process, 0.0))]

        shape = builder.getShape(channel, process)
        values = np.array(
            [shape.GetBinContent(index + 1) for index in range(shape.GetNbinsX())],
            dtype=np.float64,
        )
        if process == DATA_NAME:
            return values.tolist()
        total = float(np.sum(values))
        rate = float(self._card.exp[channel].get(process, 0.0))
        if total == 0.0:
            return [0.0] * values.size
        return (rate * values / total).tolist()

    def _nbins(self, channel: str) -> int:
        """
        Number of bins of one channel.

        Published cards do mix binnings inside a channel — the tail of a
        distribution is rebinned for the processes that matter but not for a
        dummy one — so the channel is as long as its longest template, which is
        how ``combine`` builds it too.

        Args:
            channel (``str``): name of the channel.

        Returns:
            ``int``:
            The length of the longest template of the channel.
        """
        lengths = [len(self._raw_template(channel, DATA_NAME))]
        for process, rate in self._card.exp[channel].items():
            if rate != 0.0:
                lengths.append(len(self._raw_template(channel, process)))
        return max(lengths)

    def _sum_samples(self, signal: bool) -> dict[str, list[float]]:
        """
        Add up the samples of every channel.

        Args:
            signal (``bool``): whether to add the signal samples or the others.

        Returns:
            ``Dict[str, List[float]]``:
            Mapping from channel name to per-bin yields.
        """
        selected = self.signal_samples
        out: dict[str, list[float]] = {}
        for channel, samples in self.sample_yields().items():
            total = np.zeros(self._nbins(channel))
            for name, values in samples.items():
                if (name in selected[channel]) == signal:
                    total = total + np.asarray(values, dtype=np.float64)
            out[channel] = total.tolist()
        return out

    def _check_channel(self, channel: str) -> str:
        """
        Validate a channel name.

        Args:
            channel (``str``): the name to check.

        Raises:
            ``ValueError``: if the datacard has no such channel.

        Returns:
            ``str``:
            The name, unchanged.
        """
        if channel not in self._card.bins:
            raise ValueError(
                f"'{channel}' is not a channel of the datacard. Available "
                f"channels: {', '.join(self.channels)}."
            )
        return channel

    @staticmethod
    def _fmt(values: Sequence[float]) -> str:
        """
        Format a short list of yields.

        Args:
            values (``Sequence[float]``): the values to format.

        Returns:
            ``str``:
            At most six of them, with an ellipsis if there are more.
        """
        items = [f"{value:.4g}" for value in list(values)[:6]]
        if len(list(values)) > 6:
            items.append("...")
        return "[" + ", ".join(items) + "]"
