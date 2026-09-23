r"""
``CMS Combine`` physics models contributed by the plug-in.

A *physics model* is how ``combine`` lets an analysis decide what its parameter
of interest scales; it is user code by design, selected with the ``-P`` option
of ``text2workspace.py``. This module contributes the one a two hypothesis
search needs when its processes are named the other way round from
``HiggsAnalysis.CombinedLimit.HiggsJPC``.

The stock ``twoHypothesisHiggs`` scales the process whose name contains the
``altSignal`` marker — ``"ALT"`` by default — by :math:`x`, and the other one
by :math:`1-x`, so that :math:`x=1` selects the *alternative* hypothesis. Some
published cards use the opposite convention: CMS-EXO-24-011 writes its two
signal processes as ``QCDADD<Λ>`` and ``QCDADD<Λ>_ALT``, where the first is
QCD+BSM and the second is QCD alone, and states that :math:`x=1` is QCD+BSM.
Its ``HiggsJPCmodified`` is the stock model with the two exchanged, which is
:data:`twoHypothesisHiggs` here.

The distinction matters to ``spey``: its background is the model at
:math:`\mu=0` and its signal the difference to :math:`\mu=1`, so picking the
wrong convention tests the wrong hypothesis and gives a negative signal.

References
~~~~~~~~~~
* Combine manual, physics models,
  https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part2/physicsmodels/
"""

from __future__ import annotations

from typing import Any

__all__ = ["SwappedTwoHypothesis", "twoHypothesisHiggs"]


def __dir__():
    return __all__


# pylint: disable=invalid-name, import-outside-toplevel


def _base_class() -> Any:
    """
    Import ``combine``'s two hypothesis model.

    Raises:
        ``ImportError``: if ``combine`` is not installed.

    Returns:
        ``type``:
        ``HiggsAnalysis.CombinedLimit.HiggsJPC.TwoHypotesisHiggs``.
    """
    try:
        from HiggsAnalysis.CombinedLimit.HiggsJPC import TwoHypotesisHiggs
    except ImportError as err:  # pragma: no cover - environment dependent
        raise ImportError(
            "The CMS Combine backend drives combine itself. Install it with "
            "`conda install -c conda-forge cms-combine`."
        ) from err
    return TwoHypotesisHiggs


try:
    _TwoHypotesisHiggs = _base_class()
except ImportError:  # pragma: no cover - environment dependent
    _TwoHypotesisHiggs = object


class SwappedTwoHypothesis(_TwoHypotesisHiggs):  # type: ignore[misc, valid-type]
    r"""
    ``combine``'s two hypothesis model, with the two hypotheses exchanged.

    Everything is inherited from
    ``HiggsAnalysis.CombinedLimit.HiggsJPC.TwoHypotesisHiggs`` — the parameter
    :math:`x\in[0,1]`, the optional floating signal strength, the ``--PO``
    options — except for which of the two signals :math:`x` scales:

    .. math::

        M_{cp}(x) = \begin{cases}
            x   & p\ \text{does not carry the}\ \texttt{altSignal}\ \text{marker}\ ,\\
            1-x & p\ \text{does}\ ,\\
            1   & p\ \text{is a background}\ .
        \end{cases}

    so that :math:`x=1` selects the primary hypothesis and :math:`x=0` the
    alternative one.
    """

    def doParametersOfInterest(self) -> None:
        """
        Create the parameter of interest, then exchange the two hypotheses.

        The base class builds both scale factors and records which one belongs
        to the alternative signal; only that mapping is inverted here, so every
        option of the stock model keeps working.
        """
        super().doParametersOfInterest()
        self.sigNorms = {True: self.sigNorms[False], False: self.sigNorms[True]}


twoHypothesisHiggs = SwappedTwoHypothesis()
"""
The model instance ``text2workspace.py`` expects behind ``-P``.

Select it with
``physics_model="spey_roofit.physics_models:twoHypothesisHiggs"``.
"""
