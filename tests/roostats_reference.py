r"""
The pure ``RooFit`` / ``RooStats`` reference for the published workspaces.

The validation notebooks compare ``spey``'s inference on a published likelihood
against the one an analyser would run without ``spey`` at all, starting from
the file the collaboration released:

* the exclusion is computed by ``RooStats::HypoTestInverter`` driving a
  ``RooStats::AsymptoticCalculator`` with the one-sided :math:`\tilde{q}_\mu`
  test statistic and :math:`\mathrm{CL}_s` — the configuration of
  ``StandardHypoTestInvDemo.C`` — with the expected bands taken from the
  Asimov dataset of the background-only hypothesis, its nuisance parameters
  fitted to the observed data;
* the profile likelihood :math:`-2\Delta\log\mathcal{L}(\mu)` is minimised by
  ``RooMinimizer`` (``Minuit2``) on the negative log-likelihood that
  ``RooAbsPdf::createNLL`` builds with the constraint terms and global
  observables of the ``ModelConfig``.

The reference runs in a **separate process** that never imports ``spey`` or
``spey-roofit``. That is not a formality: ``ROOT`` hands back the *same*
``RooWorkspace`` when a file that is already open is read again, so a
reference computed in the notebook's own process would share its parameters,
ranges and caches with the ``spey`` model of the same file.
:func:`run_reference` starts that process and returns its results;
:func:`spey_asymptotic_scan` and :func:`spey_profile_scan` compute the same
quantities with ``spey``, :func:`spey_likelihood_at` evaluates ``spey``'s
likelihood at the points ``RooFit`` fitted, and :func:`brazil_plot` and
:func:`chi2_plot` draw the two side by side.

References
~~~~~~~~~~
* G. Cowan, K. Cranmer, E. Gross, O. Vitells, :xref:`1007.1727`
* ``$ROOTSYS/tutorials/roostats/StandardHypoTestInvDemo.C``
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SIGMAS",
    "AsymptoticScan",
    "ProfileScan",
    "run_reference",
    "upper_limit",
    "spey_asymptotic_scan",
    "spey_profile_scan",
    "spey_likelihood_at",
    "summarise",
    "brazil_plot",
    "chi2_plot",
]

# Order of the expected bands everywhere in this module, as ``spey`` returns them.
SIGMAS = (-2, -1, 0, 1, 2)


# --------------------------------------------------------------------------- #
# Results                                                                      #
# --------------------------------------------------------------------------- #


def upper_limit(poi, cls, level: float = 0.05) -> float:
    r"""
    The first crossing of ``level`` from above, linearly interpolated.

    Args:
        poi (``Sequence[float]``): increasing values of the parameter of interest.
        cls (``Sequence[float]``): :math:`\mathrm{CL}_s` at each of them.
        level (``float``, default ``0.05``): the threshold.

    Returns:
        ``float``:
        The upper limit, or ``nan`` if the scan does not cross ``level``.
    """
    poi, cls = np.asarray(poi, dtype=float), np.asarray(cls, dtype=float)
    below = np.flatnonzero(cls < level)
    if below.size == 0 or below[0] == 0:
        return float("nan")
    i = int(below[0])
    return float(np.interp(level, [cls[i], cls[i - 1]], [poi[i], poi[i - 1]]))


@dataclass
class AsymptoticScan:
    r"""
    :math:`\mathrm{CL}_s` as a function of the parameter of interest.

    Args:
        poi (``np.ndarray``): the tested values.
        observed (``np.ndarray``): observed :math:`\mathrm{CL}_s`.
        expected (``np.ndarray``): expected :math:`\mathrm{CL}_s`, one row per
          tested value and one column per entry of :data:`SIGMAS`.
        seconds (``float``): wall-clock time the scan took.
        native_limits (``dict``): upper limits as the tool itself reports them,
          when it does: ``RooStats``' ``HypoTestInverterResult`` interpolates
          its own scan.
    """

    poi: np.ndarray
    observed: np.ndarray
    expected: np.ndarray
    seconds: float
    native_limits: dict = field(default_factory=dict)

    def upper_limits(self, level: float = 0.05) -> dict:
        r"""
        Observed and expected upper limits, read off the scan.

        Both workflows are put through the same interpolation, so that their
        limits differ only where their :math:`\mathrm{CL}_s` curves do.

        Args:
            level (``float``, default ``0.05``): :math:`\mathrm{CL}_s` threshold.

        Returns:
            ``dict``:
            ``"observed"``, and one entry per band keyed by :data:`SIGMAS`.
        """
        limits = {"observed": upper_limit(self.poi, self.observed, level)}
        for column, sigma in enumerate(SIGMAS):
            limits[sigma] = upper_limit(self.poi, self.expected[:, column], level)
        return limits


@dataclass
class ProfileScan:
    r"""
    The profile likelihood ratio on a grid.

    Args:
        poi (``np.ndarray``): the scanned values.
        chi2 (``np.ndarray``): :math:`-2\log[\mathcal{L}(\mu,\hat{\hat\theta}_\mu)
          /\mathcal{L}(\hat\mu,\hat\theta)]` at each of them.
        best_fit (``float``): :math:`\hat\mu`.
        seconds (``float``): wall-clock time the scan took.
        parameters (``list``): the fitted parameters, as ``{name: value}``, of
          the unconditional fit followed by those of every conditional fit.
    """

    poi: np.ndarray
    chi2: np.ndarray
    best_fit: float
    seconds: float
    parameters: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# The RooFit side, run in its own process                                      #
# --------------------------------------------------------------------------- #


def _root(strategy: int = 2, offset: bool = True):
    """
    Return :mod:`ROOT`, configured as the reference uses it.

    Args:
        strategy (``int``, default ``2``): ``Minuit2`` strategy.
        offset (``bool``, default ``True``): whether to offset the negative
          log-likelihood by its initial value.

    Returns:
        ``module``:
        The ``ROOT`` module.
    """
    import ROOT  # pylint: disable=import-outside-toplevel

    ROOT.gROOT.SetBatch(True)
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.FATAL)
    ROOT.Math.MinimizerOptions.SetDefaultMinimizer("Minuit2", "Migrad")
    ROOT.Math.MinimizerOptions.SetDefaultStrategy(strategy)
    ROOT.Math.MinimizerOptions.SetDefaultPrintLevel(-1)
    ROOT.RooStats.UseNLLOffset(offset)
    ROOT.RooStats.AsymptoticCalculator.SetPrintLevel(-1)
    return ROOT


def _open(request: dict):
    """
    Read the workspace, its model configuration and its observed data.

    Args:
        request (``dict``): see :func:`run_reference`.

    Returns:
        ``tuple``:
        ``(ROOT, file, model_config, data)``.
    """
    ROOT = _root(request.get("strategy", 2), request.get("offset", True))
    for library in request.get("libraries") or []:
        if ROOT.gSystem.Load(library) < 0:
            raise RuntimeError(f"cannot load {library}")
    handle = ROOT.TFile.Open(request["workspace"])
    name = request.get("workspace_name") or next(
        key.GetName()
        for key in handle.GetListOfKeys()
        if key.GetClassName() == "RooWorkspace"
    )
    workspace = handle.Get(name)
    config = workspace.obj(request.get("model_config") or "ModelConfig")
    data_name = request.get("data") or next(
        d.GetName() for d in workspace.allData() if "asimov" not in d.GetName().lower()
    )
    if request.get("poi_range"):
        config.GetParametersOfInterest().first().setRange(*request["poi_range"])
    return ROOT, handle, config, workspace.data(data_name)


def _asymptotic(ROOT, config, data, grid, strategy: int, offset: bool) -> dict:
    """``RooStats``' asymptotic CLs on a uniform grid."""
    poi = config.GetParametersOfInterest().first()
    start = time.time()
    # StandardHypoTestInvDemo.C's initial fit: the calculator starts every fit
    # from the values the parameters hold, and from the nominal values of a
    # freshly compiled combine workspace its background-only fits fail
    nll = config.GetPdf().createNLL(
        data,
        ROOT.RooFit.Constrain(config.GetNuisanceParameters()),
        ROOT.RooFit.GlobalObservables(config.GetGlobalObservables()),
        ROOT.RooFit.Offset("initial" if offset else "none"),
    )
    minimizer = ROOT.RooMinimizer(nll)
    minimizer.setPrintLevel(-1)
    minimizer.setStrategy(strategy)
    minimizer.minimize("Minuit2", "Migrad")

    poi.setVal(1.0)
    config.SetSnapshot(ROOT.RooArgSet(poi))
    background = config.Clone("B_only")
    poi.setVal(0.0)
    background.SetSnapshot(ROOT.RooArgSet(poi))

    calculator = ROOT.RooStats.AsymptoticCalculator(data, background, config)
    calculator.SetOneSided(True)
    calculator.SetQTilde(True)
    inverter = ROOT.RooStats.HypoTestInverter(calculator)
    inverter.SetConfidenceLevel(0.95)
    inverter.UseCLs(True)
    inverter.SetVerbose(False)
    inverter.SetFixedScan(len(grid), float(grid[0]), float(grid[-1]))
    result = inverter.GetInterval()

    size = result.ArraySize()
    expected = [
        [
            ROOT.RooStats.AsymptoticCalculator.GetExpectedPValues(
                result.GetResult(i).NullPValue(),
                result.GetResult(i).AlternatePValue(),
                float(sigma),
                True,
            )
            for sigma in SIGMAS
        ]
        for i in range(size)
    ]
    native = {"observed": result.UpperLimit()}
    for sigma in SIGMAS:
        native[str(sigma)] = result.GetExpectedUpperLimit(float(sigma))
    return {
        "poi": [result.GetXValue(i) for i in range(size)],
        "observed": [result.CLs(i) for i in range(size)],
        "expected": expected,
        "native_limits": native,
        "seconds": time.time() - start,
    }


def _profile(ROOT, config, data, grid, strategy: int, offset: bool) -> dict:
    """``RooMinimizer``'s profile likelihood ratio on a grid."""
    poi = config.GetParametersOfInterest().first()
    nuisances = config.GetNuisanceParameters()
    floating = ROOT.RooArgSet(poi)
    floating.add(nuisances)

    start = time.time()
    nll = config.GetPdf().createNLL(
        data,
        ROOT.RooFit.Constrain(nuisances),
        ROOT.RooFit.GlobalObservables(config.GetGlobalObservables()),
        ROOT.RooFit.Offset("initial" if offset else "none"),
    )
    minimizer = ROOT.RooMinimizer(nll)
    minimizer.setPrintLevel(-1)
    minimizer.setStrategy(strategy)

    def values():
        return {v.GetName(): v.getVal() for v in floating}

    poi.setConstant(False)
    minimizer.minimize("Minuit2", "Migrad")
    best_fit, minimum = poi.getVal(), nll.getVal()
    snapshot = floating.snapshot()
    parameters = [values()]

    chi2 = []
    for value in grid:
        # assign() restores the constant flags as well, so the parameter of
        # interest is fixed only after it
        floating.assign(snapshot)
        poi.setVal(float(value))
        poi.setConstant(True)
        minimizer.minimize("Minuit2", "Migrad")
        chi2.append(2.0 * (nll.getVal() - minimum))
        parameters.append(values())
    poi.setConstant(False)
    return {
        "poi": list(grid),
        "chi2": chi2,
        "best_fit": best_fit,
        "parameters": parameters,
        "seconds": time.time() - start,
    }


def _main(request_path: str, result_path: str) -> None:
    """Entry point of the reference process."""
    with open(request_path, encoding="utf-8") as handle:
        request = json.load(handle)
    result = {}
    # each calculation reads the file afresh, since a fit leaves its values
    # behind in the workspace
    if request.get("profile_grid"):
        ROOT, handle, config, data = _open(request)
        result["profile"] = _profile(
            ROOT,
            config,
            data,
            request["profile_grid"],
            request.get("strategy", 2),
            request.get("offset", True),
        )
        handle.Close()
    if request.get("cls_grid"):
        ROOT, handle, config, data = _open(request)
        result["asymptotic"] = _asymptotic(
            ROOT,
            config,
            data,
            request["cls_grid"],
            request.get("strategy", 2),
            request.get("offset", True),
        )
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle)


def run_reference(
    workspace: str,
    cls_grid=None,
    profile_grid=None,
    workspace_name: str | None = None,
    data: str | None = None,
    poi_range: tuple[float, float] | None = None,
    libraries=(),
    strategy: int = 2,
    offset: bool = True,
) -> tuple[AsymptoticScan | None, ProfileScan | None]:
    r"""
    Run the pure ``RooFit`` / ``RooStats`` reference in a separate process.

    Args:
        workspace (``str``): the ``.root`` file holding the workspace.
        cls_grid (``Sequence[float]``, default ``None``): equally spaced values
          of :math:`\mu` for the :math:`\mathrm{CL}_s` scan;
          ``HypoTestInverter``'s fixed scan only takes a uniform grid.
        profile_grid (``Sequence[float]``, default ``None``): values of
          :math:`\mu` for the profile likelihood scan.
        workspace_name (``str``, default ``None``): key of the workspace in the
          file; the first workspace in it when omitted.
        data (``str``, default ``None``): name of the observed dataset; the
          first one whose name does not mention Asimov when omitted.
        poi_range (``Tuple[float, float]``, default ``None``): range of the
          parameter of interest during the fits; the workspace's own when
          omitted.
        libraries (``Sequence[str]``, default ``()``): shared libraries the
          workspace's classes live in, e.g. ``libHiggsAnalysisCombinedLimit``.
        strategy (``int``, default ``2``): ``Minuit2`` strategy of every fit.
        offset (``bool``, default ``True``): offset the negative log-likelihood
          by its initial value. A ``HistFactory`` likelihood can carry a
          data-dependent constant of order :math:`10^7` — the bin volumes of
          its extended term — and without the offset ``Minuit`` cannot
          resolve the minimum under it.

    Raises:
        ``RuntimeError``: if the reference process fails.

    Returns:
        ``Tuple[AsymptoticScan, ProfileScan]``:
        The two scans; ``None`` for one that was not requested.
    """
    request = {
        "workspace": os.path.abspath(workspace),
        "workspace_name": workspace_name,
        "data": data,
        "poi_range": list(poi_range) if poi_range else None,
        "libraries": list(libraries),
        "strategy": int(strategy),
        "offset": bool(offset),
        "cls_grid": None if cls_grid is None else [float(x) for x in cls_grid],
        "profile_grid": (
            None if profile_grid is None else [float(x) for x in profile_grid]
        ),
    }
    with tempfile.TemporaryDirectory(prefix="roostats_reference_") as directory:
        request_path = os.path.join(directory, "request.json")
        result_path = os.path.join(directory, "result.json")
        with open(request_path, "w", encoding="utf-8") as handle:
            json.dump(request, handle)
        process = subprocess.run(
            [sys.executable, os.path.abspath(__file__), request_path, result_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0 or not os.path.exists(result_path):
            raise RuntimeError(f"the RooFit reference failed:\n{process.stderr[-3000:]}")
        with open(result_path, encoding="utf-8") as handle:
            result = json.load(handle)

    scan = profile = None
    if "asymptotic" in result:
        r = result["asymptotic"]
        scan = AsymptoticScan(
            poi=np.asarray(r["poi"]),
            observed=np.asarray(r["observed"]),
            expected=np.asarray(r["expected"]),
            seconds=r["seconds"],
            native_limits={
                key if key == "observed" else int(key): value
                for key, value in r["native_limits"].items()
            },
        )
    if "profile" in result:
        r = result["profile"]
        profile = ProfileScan(
            poi=np.asarray(r["poi"]),
            chi2=np.asarray(r["chi2"]),
            best_fit=r["best_fit"],
            seconds=r["seconds"],
            parameters=r["parameters"],
        )
    return scan, profile


# --------------------------------------------------------------------------- #
# The spey side                                                                #
# --------------------------------------------------------------------------- #


def spey_asymptotic_scan(statistical_model, poi_values, **kwargs) -> AsymptoticScan:
    r"""
    The scan of :func:`run_reference`, computed by ``spey``.

    ``spey``'s default test statistic for an exclusion is :math:`\tilde{q}_\mu`,
    and the expected values ``expected="all"`` returns are those of the post-fit
    Asimov dataset, :obj:`~spey.ExpectationType.aposteriori` — both as in
    ``RooStats``. ``spey`` reports :math:`1-\mathrm{CL}_s`, which is subtracted
    from one here.

    Args:
        statistical_model (~spey.StatisticalModel): the model.
        poi_values (``Sequence[float]``): values of :math:`\mu` to test.
        kwargs: forwarded to ``exclusion_confidence_level``, e.g. optimiser
          options.

    Returns:
        ~AsymptoticScan:
        The observed and expected :math:`\mathrm{CL}_s`.
    """
    poi_values = np.asarray(poi_values, dtype=float)
    start = time.time()
    observed, expected = [], []
    for value in poi_values:
        # ([observed], [five expected values])
        obs, exp = statistical_model.exclusion_confidence_level(
            poi_test=float(value), expected="all", **kwargs
        )
        observed.append(1.0 - float(obs[0]))
        expected.append([1.0 - float(v) for v in exp])
    return AsymptoticScan(
        poi=poi_values,
        observed=np.asarray(observed),
        expected=np.asarray(expected),
        seconds=time.time() - start,
    )


def spey_profile_scan(statistical_model, poi_values, **kwargs) -> ProfileScan:
    r"""
    The profile likelihood scan of :func:`run_reference`, computed by ``spey``.

    Args:
        statistical_model (~spey.StatisticalModel): the model.
        poi_values (``Sequence[float]``): values of :math:`\mu` to profile at.
        kwargs: forwarded to ``likelihood`` and ``maximize_likelihood``, e.g.
          optimiser options.

    Returns:
        ~ProfileScan:
        The profile likelihood ratio on the grid and the best fit.
    """
    start = time.time()
    best_fit, minimum = statistical_model.maximize_likelihood(**kwargs)
    chi2 = [
        2.0 * (statistical_model.likelihood(poi_test=float(value), **kwargs) - minimum)
        for value in poi_values
    ]
    return ProfileScan(
        poi=np.asarray(poi_values, dtype=float),
        chi2=np.asarray(chi2),
        best_fit=float(best_fit),
        seconds=time.time() - start,
    )


def spey_likelihood_at(statistical_model, profile: ProfileScan) -> np.ndarray:
    r"""
    ``spey``'s :math:`-2\Delta\log\mathcal{L}` at the points ``RooFit`` fitted.

    This separates the likelihood from the minimiser: if the two workflows
    evaluate the same function, ``spey``'s likelihood at the parameters
    ``RooFit`` found reproduces ``RooFit``'s curve, whatever either minimiser
    would have done.

    Args:
        statistical_model (~spey.StatisticalModel): the model.
        profile (~ProfileScan): the reference profile scan, with its parameters.

    Returns:
        ``np.ndarray``:
        The profile likelihood ratio on ``profile``'s grid.
    """
    backend = statistical_model.backend
    names = backend.model.parameter_names
    nominal = np.asarray(backend.model.nominal_parameters, dtype=float)
    logpdf = backend.get_logpdf_func()

    def at(values: dict) -> float:
        pars = nominal.copy()
        for index, name in enumerate(names):
            if name in values:
                pars[index] = values[name]
        return float(logpdf(pars))

    best = at(profile.parameters[0])
    return np.array([2.0 * (best - at(p)) for p in profile.parameters[1:]])


def summarise(
    reference_cls: AsymptoticScan,
    spey_cls: AsymptoticScan,
    reference_profile: ProfileScan,
    spey_profile: ProfileScan,
    spey_at_reference: np.ndarray,
) -> None:
    r"""
    Print the comparison of the two workflows.

    Args:
        reference_cls (~AsymptoticScan): ``RooStats``' :math:`\mathrm{CL}_s` scan.
        spey_cls (~AsymptoticScan): ``spey``'s, on the same grid.
        reference_profile (~ProfileScan): ``RooFit``'s profile likelihood scan.
        spey_profile (~ProfileScan): ``spey``'s, on the same grid.
        spey_at_reference (``np.ndarray``): ``spey``'s likelihood at
          ``RooFit``'s fitted points, see :func:`spey_likelihood_at`.
    """
    ours, theirs = reference_cls.upper_limits(), spey_cls.upper_limits()
    native = reference_cls.native_limits
    print(f"{'95% CL upper limit':<22} {'RooStats':>10} {'(its own)':>10} {'spey':>10}")
    for key, label in [("observed", "observed")] + [
        (sigma, f"expected {sigma:+d} sigma" if sigma else "expected median")
        for sigma in SIGMAS
    ]:
        own = native.get(key, float("nan"))
        print(f"{label:<22} {ours[key]:>10.4f} {own:>10.4f} {theirs[key]:>10.4f}")
    print("(limits read off the scan by linear interpolation; nan when the scan does")
    print(" not cross 0.05. 'its own' is HypoTestInverterResult's interpolation.)\n")

    print("largest |CLs(spey) - CLs(RooStats)| over the scan:")
    print(
        f"  observed          {np.max(np.abs(spey_cls.observed - reference_cls.observed)):.2e}"
    )
    for column, sigma in enumerate(SIGMAS):
        difference = np.max(
            np.abs(spey_cls.expected[:, column] - reference_cls.expected[:, column])
        )
        print(f"  expected {sigma:+d} sigma  {difference:.2e}")

    print(
        f"\nbest fit             RooFit {reference_profile.best_fit:.6f}"
        f"   spey {spey_profile.best_fit:.6f}"
    )
    print("largest |-2 Delta log L| difference over the profile scan:")
    print(
        f"  spey's own fits                       "
        f"{np.max(np.abs(spey_profile.chi2 - reference_profile.chi2)):.2e}"
    )
    print(
        f"  spey's likelihood at RooFit's points  "
        f"{np.max(np.abs(spey_at_reference - reference_profile.chi2)):.2e}"
    )
    print(
        f"\nwall-clock: RooStats {reference_cls.seconds:.0f} s + {reference_profile.seconds:.0f} s,"
        f" spey {spey_cls.seconds:.0f} s + {spey_profile.seconds:.0f} s"
    )


# --------------------------------------------------------------------------- #
# Plots                                                                        #
# --------------------------------------------------------------------------- #


def brazil_plot(
    reference: AsymptoticScan,
    spey_scan: AsymptoticScan,
    title: str,
    poi_label: str = r"$\mu$",
    log: bool = False,
):
    r"""
    Draw the two exclusions on top of each other, and their difference.

    The bands and the thick lines are ``RooStats``; ``spey`` is drawn as
    markers over them, so that agreement shows as markers sitting on lines.

    Args:
        reference (~AsymptoticScan): the pure ``RooStats`` result.
        spey_scan (~AsymptoticScan): the ``spey`` result on the same grid.
        title (``str``): figure title.
        poi_label (``str``): axis label of the parameter of interest.
        log (``bool``, default ``False``): logarithmic :math:`\mathrm{CL}_s` axis.

    Returns:
        ``matplotlib.figure.Figure``:
        The figure.
    """
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, (upper, lower) = plt.subplots(
        2,
        1,
        figsize=(7.0, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    x, band = reference.poi, reference.expected
    upper.fill_between(
        x, band[:, 0], band[:, 4], color="#FFD800", label=r"$\pm2\sigma$, RooStats"
    )
    upper.fill_between(
        x, band[:, 1], band[:, 3], color="#00B800", label=r"$\pm1\sigma$, RooStats"
    )
    upper.plot(x, band[:, 2], color="k", ls="--", lw=2.4, label="expected, RooStats")
    upper.plot(x, reference.observed, color="k", lw=2.4, label="observed, RooStats")
    for column in (0, 1, 3, 4):
        upper.plot(
            spey_scan.poi,
            spey_scan.expected[:, column],
            ls="none",
            marker="+",
            ms=7,
            color="#444444",
            label=r"$\pm1\sigma,\pm2\sigma$, spey" if column == 0 else None,
        )
    upper.plot(
        spey_scan.poi,
        spey_scan.expected[:, 2],
        ls="none",
        marker="o",
        ms=5,
        mfc="none",
        color="#d62728",
        label="expected, spey",
    )
    upper.plot(
        spey_scan.poi,
        spey_scan.observed,
        ls="none",
        marker="s",
        ms=4,
        color="#1f77b4",
        label="observed, spey",
    )
    upper.axhline(0.05, color="#c1272d", lw=1.0)
    upper.text(
        x[-1], 0.05, "95% CL", ha="right", va="bottom", color="#c1272d", fontsize=9
    )
    if log:
        upper.set_yscale("log")
    else:
        upper.set_ylim(0.0, 1.02)
    upper.set_ylabel(r"$\mathrm{CL}_s$")
    upper.set_title(title)
    upper.legend(frameon=False, ncol=2, fontsize=8)

    for column, colour, style, label in (
        (2, "#d62728", "--", "expected"),
        (0, "#bfa200", ":", r"$-2\sigma$"),
        (4, "#008a00", ":", r"$+2\sigma$"),
    ):
        lower.plot(
            x,
            spey_scan.expected[:, column] - band[:, column],
            color=colour,
            ls=style,
            lw=1.4,
            label=label,
        )
    lower.plot(
        x,
        spey_scan.observed - reference.observed,
        color="#1f77b4",
        lw=1.4,
        label="observed",
    )
    lower.axhline(0.0, color="#777777", lw=0.8)
    lower.set_xlabel(poi_label)
    lower.set_ylabel(r"spey $-$ RooStats")
    lower.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    lower.legend(frameon=False, fontsize=8, ncol=4)
    lower.set_xlim(x[0], x[-1])
    return fig


def chi2_plot(
    reference: ProfileScan,
    spey_scan: ProfileScan,
    title: str,
    poi_label: str = r"$\mu$",
    spey_at_reference: np.ndarray | None = None,
):
    r"""
    Draw the two profile likelihood scans on top of each other, and their difference.

    Args:
        reference (~ProfileScan): the pure ``RooFit`` result.
        spey_scan (~ProfileScan): the ``spey`` result on the same grid.
        title (``str``): figure title.
        poi_label (``str``): axis label of the parameter of interest.
        spey_at_reference (``np.ndarray``, default ``None``): ``spey``'s
          likelihood at ``RooFit``'s fitted parameters, see
          :func:`spey_likelihood_at`; drawn when given.

    Returns:
        ``matplotlib.figure.Figure``:
        The figure.
    """
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, (upper, lower) = plt.subplots(
        2,
        1,
        figsize=(7.0, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    upper.plot(reference.poi, reference.chi2, color="k", lw=2.4, label="RooFit + Minuit2")
    upper.plot(
        spey_scan.poi,
        spey_scan.chi2,
        ls="none",
        marker="o",
        ms=5,
        mfc="none",
        color="#d62728",
        label="spey",
    )
    if spey_at_reference is not None:
        upper.plot(
            reference.poi,
            spey_at_reference,
            ls="none",
            marker="x",
            ms=5,
            color="#1f77b4",
            label="spey's likelihood at RooFit's fitted points",
        )
    for level, text in ((1.0, r"$1\sigma$"), (3.84, "95% CL")):
        upper.axhline(level, color="#777777", lw=0.9, ls=":")
        upper.text(
            reference.poi[-1],
            level,
            text,
            ha="right",
            va="bottom",
            color="#777777",
            fontsize=9,
        )
    upper.set_ylabel(r"$-2\Delta\log\mathcal{L}$")
    upper.set_ylim(bottom=0.0)
    upper.set_title(title)
    upper.legend(frameon=False, fontsize=9)

    lower.plot(
        reference.poi,
        spey_scan.chi2 - reference.chi2,
        color="#d62728",
        lw=1.4,
        marker="o",
        ms=3,
        label="spey",
    )
    if spey_at_reference is not None:
        lower.plot(
            reference.poi,
            spey_at_reference - reference.chi2,
            color="#1f77b4",
            lw=1.4,
            marker="x",
            ms=4,
            label="spey's likelihood at RooFit's points",
        )
    lower.axhline(0.0, color="#777777", lw=0.8)
    lower.set_xlabel(poi_label)
    lower.set_ylabel(r"spey $-$ RooFit")
    lower.legend(frameon=False, fontsize=8)
    lower.set_xlim(reference.poi[0], reference.poi[-1])
    return fig


if __name__ == "__main__":
    _main(sys.argv[1], sys.argv[2])
