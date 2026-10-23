r"""
Lazy :mod:`ROOT` manager for the ``spey-roofit`` plug-in.

``PyROOT`` is an expensive import and is not required until a workspace is
actually loaded.  This module therefore defers the import until the first
access and centralises everything that has to be configured globally once
``ROOT`` is available:

* batch mode, so that no graphics context is created;
* the verbosity of ``RooMsgService``, which is extremely chatty by default;
* a keep-alive registry for objects that are owned by Python but referenced
  from C++ (``PyROOT`` would otherwise garbage-collect them while ``RooFit``
  still holds raw pointers).

References
~~~~~~~~~~
* RooFit manual: https://root.cern/manual/roofit/
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = ["RootManager", "manager"]


def __dir__():
    return __all__


log = logging.getLogger("Spey")

# pylint: disable=import-outside-toplevel


class RootManager:
    r"""
    Lazy accessor and global configuration holder for :mod:`ROOT`.

    The manager imports ``ROOT`` on first use, switches it to batch mode and
    silences ``RooFit``'s message service down to ``level``.  It additionally
    keeps a reference to every ``RooFit`` object that has been created from
    Python and handed over to a C++ container, which prevents ``PyROOT`` from
    collecting objects that ``RooFit`` still points at.

    Args:
        verbosity (``str``, default ``"ERROR"``): initial ``RooMsgService``
          kill-below level. Accepted values are ``"DEBUG"``, ``"INFO"``,
          ``"PROGRESS"``, ``"WARNING"``, ``"ERROR"`` and ``"FATAL"``.

    Example:

    .. code-block:: python3

        >>> from spey_roofit.manager import manager
        >>> ROOT = manager.ROOT
        >>> manager.silence("FATAL")
    """

    __slots__ = ["_root", "_verbosity", "_keep_alive", "_combine"]

    _LEVELS = ("DEBUG", "INFO", "PROGRESS", "WARNING", "ERROR", "FATAL")

    def __init__(self, verbosity: str = "ERROR"):
        self._root = None
        self._verbosity = verbosity
        self._keep_alive: list[Any] = []
        self._combine: Any = None

    @property
    def ROOT(self):  # pylint: disable=invalid-name
        """
        Accessor to the ``ROOT`` module.

        The module is imported on first access, put into batch mode and its
        ``RooFit`` message service is silenced.

        Raises:
            ``ImportError``: if ``PyROOT`` is not installed in the current
              environment.

        Returns:
            ``module``:
            The imported ``ROOT`` module.
        """
        if self._root is None:
            try:
                import ROOT  # noqa: N811
            except ImportError as err:  # pragma: no cover - environment dependent
                raise ImportError(
                    "spey-roofit requires PyROOT. Install ROOT e.g. via "
                    "`conda install -c conda-forge root`."
                ) from err

            ROOT.gROOT.SetBatch(True)
            # RooFit prints one line per evaluation error, e.g. whenever the
            # optimiser probes a parameter point with a vanishing yield. Those
            # points are handled by returning a non-finite likelihood, so the
            # errors are counted rather than printed.
            ROOT.RooAbsReal.setEvalErrorLoggingMode(ROOT.RooAbsReal.CountErrors)
            self._root = ROOT
            self.silence(self._verbosity)
        return self._root

    @property
    def is_available(self) -> bool:
        """
        Check whether ``PyROOT`` can be imported.

        Returns:
            ``bool``:
            ``True`` if ``import ROOT`` succeeds, ``False`` otherwise.
        """
        try:
            _ = self.ROOT
        except ImportError:
            return False
        return True

    @property
    def version(self) -> str:
        """
        ROOT version string, e.g. ``"6.36/04"``.

        Returns:
            ``str``:
            Value reported by ``TROOT::GetVersion``.
        """
        return str(self.ROOT.gROOT.GetVersion())

    @property
    def has_combine(self) -> bool:
        """
        Whether ``CMS Combine`` is available in this environment.

        The check loads ``libHiggsAnalysisCombinedLimit``, which is what makes
        a workspace written by ``text2workspace.py`` readable: it uses classes
        such as ``ProcessNormalization`` and ``FastVerticalInterpHistPdf2``
        that ship with ``combine`` rather than with ``ROOT``.

        Returns:
            ``bool``:
            ``True`` if the library could be loaded.
        """
        if self._combine is None:
            try:
                status = self.ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit")
            except Exception:  # pragma: no cover - environment dependent
                status = -1
            self._combine = status >= 0
            if not self._combine:
                log.debug(
                    "libHiggsAnalysisCombinedLimit could not be loaded; install "
                    "CMS Combine with `conda install -c conda-forge cms-combine` "
                    "to read the workspaces it writes."
                )
        return bool(self._combine)

    def silence(self, level: str = "ERROR") -> None:
        """
        Set the ``RooMsgService`` kill-below level.

        Args:
            level (``str``, default ``"ERROR"``): one of ``"DEBUG"``,
              ``"INFO"``, ``"PROGRESS"``, ``"WARNING"``, ``"ERROR"`` or
              ``"FATAL"``. Messages below this level are suppressed.

        Raises:
            ``ValueError``: if ``level`` is not a known ``RooFit`` message level.
        """
        level = level.upper()
        if level not in self._LEVELS:
            raise ValueError(
                f"Unknown RooFit message level '{level}'. "
                f"Available levels are {', '.join(self._LEVELS)}."
            )
        self._verbosity = level
        root = self._root
        if root is not None:
            service = root.RooMsgService.instance()
            service.setGlobalKillBelow(getattr(root.RooFit, level))
            service.setSilentMode(True)

    def keep_alive(self, *objects: Any) -> None:
        """
        Store references to ``RooFit`` objects created from Python.

        ``RooFit`` containers hold raw pointers, therefore any object created
        in Python and inserted into such a container has to outlive the Python
        scope in which it was defined.

        Args:
            objects (``Any``): objects to protect from garbage collection.
        """
        self._keep_alive.extend(objects)

    def release(self) -> None:
        """Drop all references held by :meth:`keep_alive`."""
        self._keep_alive.clear()


manager = RootManager()
"""Module level :class:`~spey_roofit.manager.RootManager` instance."""
