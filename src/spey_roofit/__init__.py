r"""
``spey`` plug-in for ``RooFit`` workspaces.

The package turns any binned ``RooFit`` model into a
:class:`~spey.StatisticalModel`, and provides the tooling needed to inspect the
region structure of a workspace and to attach a user defined signal to it
without touching ``RooFit`` directly.

.. code-block:: python3
    :linenos:

    >>> import spey
    >>> from spey_roofit import WorkspaceInterpreter

    >>> interpreter = WorkspaceInterpreter("bOnlyFit_postFit.root")
    >>> interpreter.bin_map            # which regions and how many bins each
    >>> interpreter.summary()          # full overview of the workspace

    >>> wrapper = spey.get_backend("root.roofit")
    >>> model = wrapper(
    ...     workspace="bOnlyFit_postFit.root",
    ...     signal_yields={"SR_200_cuts": [120.0]},
    ...     analysis="atlas-exot-2018-06",
    ... )
    >>> model.exclusion_confidence_level()
"""

from ._version import __version__
from .data import ChannelInfo, ConstraintInfo, WorkspaceData
from .helper_functions import WorkspaceInterpreter, load_workspace
from .interface import RooFitInterface
from .manager import manager

__all__ = [
    "RooFitInterface",
    "WorkspaceInterpreter",
    "WorkspaceData",
    "ChannelInfo",
    "ConstraintInfo",
    "load_workspace",
    "manager",
    "__version__",
]


def __dir__():
    return __all__
