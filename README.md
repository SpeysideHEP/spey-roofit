# spey-roofit

[![arxiv](https://img.shields.io/static/v1?style=plastic&label=arXiv&message=2307.06996&color=brightgreen)](https://arxiv.org/abs/2307.06996)
[![DOI](https://img.shields.io/static/v1?style=plastic&label=DOI&message=10.21468/SciPostPhys.16.1.032&color=blue)](https://scipost.org/10.21468/SciPostPhys.16.1.032)

<img src="https://raw.githubusercontent.com/SpeysideHEP/spey/main/docs/img/spey-plug-in.png" alt="Spey logo" style="float: right; margin-right: 20px" align="right" width=250px/>

[`RooFit`](https://root.cern/manual/roofit/) plug-in for
[`spey`](https://spey.readthedocs.io).

`spey-roofit` turns any binned `RooFit` workspace into a `spey`
`StatisticalModel`, so that exclusion limits, significances, upper limits and
model combination are computed by `spey` exactly as they are for any other
likelihood prescription. It works with

* `HistFactory` workspaces published on HEPData as `.root` files,
* workspaces built by `RooJSONFactoryWSTool` from an HS3 `JSON` document,
* workspaces assembled by hand,

and it ships a **workspace interpreter** so that a user can inspect the region
structure of an analysis and inject their own signal without knowing anything
about `RooFit`.

## Installation

`ROOT` is only distributed through `conda-forge`, so the environment is best
created with `conda`:

```bash
conda env create -f environment.yml
conda activate spey-roofit
pip install -e .
```

`environment-lock.yml` pins the exact versions the shipped validation results
were produced with.

The ATLAS material used by the examples below, by
`notebooks/validation.ipynb` and by the `atlas`-marked tests is not committed —
it belongs to the HEPData record. Fetch it with:

```bash
./atlas-exot-2018-06/download.sh
```

See [atlas-exot-2018-06/README.md](atlas-exot-2018-06/README.md) for what it
downloads. Everything that does not need those files works without them.

## Quick start

### Look at what is in a workspace

```python
from spey_roofit import WorkspaceInterpreter

interpreter = WorkspaceInterpreter("atlas-exot-2018-06/bOnlyFit_postFit.root")

interpreter.channels                 # the analysis regions
interpreter.bin_map                  # how many bins each of them has
interpreter.observed_data            # observed counts per region
interpreter.expected_background_yields
interpreter.guess_signal_regions()   # regions whose name looks like a signal region
interpreter.summary(max_channels=5)  # everything at once
```

### Run inference

```python
import spey

model = spey.get_backend("root.roofit")(
    workspace="atlas-exot-2018-06/bOnlyFit_postFit.root",
    signal_yields={"SR_200_cuts": [1349.3], "SR_250_cuts": [630.6]},
    analysis="atlas-exot-2018-06",
)

model.maximize_likelihood()          # (muhat, -logL)
model.exclusion_confidence_level()   # 1 - CLs
model.poi_upper_limit()              # 95% CL upper limit on mu
model.significance()
```

A workspace that already contains a signal — such as
`sPlusBFit_postFit.root` — is used directly, without `signal_yields`:

```python
model = spey.get_backend("root.roofit")(
    workspace="atlas-exot-2018-06/sPlusBFit_postFit.root",
    analysis="atlas-exot-2018-06",
)
```

Passing `strip_existing_signal=True` removes the published signal first, so a
different signal hypothesis can be injected into the same workspace.

## How it works

| module                         | contents                                                                                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `spey_roofit.manager`          | lazy `PyROOT` import, batch mode, `RooMsgService` silencing, keep-alive registry                                                                 |
| `spey_roofit.data`             | `WorkspaceData`: flat, array-based view of a `RooFit` model — channels, bins, parameters, constraints, Asimov data and the calibrated likelihood |
| `spey_roofit.helper_functions` | `WorkspaceInterpreter`: inspection and signal injection                                                                                          |
| `spey_roofit.interface`        | `RooFitInterface`: the `spey` `BackendBase` implementation                                                                                       |

`WorkspaceData` is deliberately independent of how the workspace was produced,
so an HS3-to-`RooFit` plug-in can reuse it as it stands.

### Data and parameter layout

* the **parameter vector** is `[poi, theta_1, ..., theta_n]`, the parameter of
  interest of the `ModelConfig` followed by its floating nuisance parameters
  sorted by name;
* the **data vector** is the per-bin observed counts of every channel, in
  index-category order, followed by the auxiliary measurements, i.e. the global
  observables sorted by name.

Asimov datasets therefore carry the profiled auxiliary data, exactly as
`RooStats::AsymptoticCalculator::MakeAsimovData` and `pyhf` define them.

### Likelihood normalisation

`RooFit` and `spey` do not use the same additive normalisation for the
log-likelihood: `RooFit` includes data-dependent but parameter-independent
terms such as the category normalisation of a `RooSimultaneous`. The plug-in
calibrates that offset away once per dataset against

```
log L = sum_i [ n_i log(lambda_i) - lambda_i - log Gamma(n_i + 1) ]
      + sum_c log p_c(a_c | theta)
```

which is the convention `spey` and `pyhf` use. The calibration is validated at
two independent parameter points and a warning is issued if the offset turns
out not to be constant.

### Parameter caching

`RooFit` caches the result of the previous evaluation and only recomputes when
one of its servers is flagged dirty, which can return a stale likelihood after
the parameters have been set from Python. The plug-in switches constant-term
optimisation off, flushes the value cache before every evaluation
(`strict_cache_flush=True`, the default) and widens a variable's range rather
than letting `RooFit` silently clamp a value to it.

## Validation

`notebooks/validation.ipynb` compares the plug-in against

* `spey`'s own `default.uncorrelated_background` and
  `default.correlated_background` likelihoods, reproduced as `HistFactory`
  workspaces — the log-likelihoods agree to machine precision;
* `pyhf` on the same model — log-likelihood, Asimov data and `CLs` agree to
  machine precision;
* the published ATLAS full Run-2 monojet analysis
  ([arXiv:2102.10874](https://arxiv.org/abs/2102.10874),
  [HEPData ins1847779](https://www.hepdata.net/record/ins1847779)) — the
  observed counts and post-fit backgrounds of all 78 regions are reproduced,
  and the profile likelihood recovers the signal strength stored in the
  published post-fit workspace.

Run the test suite with

```bash
./atlas-exot-2018-06/download.sh   # once, for the ATLAS-based tests
pytest
```

The ATLAS tests skip themselves if the material has not been downloaded, so
`pytest` passes on a fresh clone either way. To skip them explicitly:

```bash
pytest -m "not atlas"
```

## Performance

`RooFit` provides no analytic gradient for a `HistFactory` likelihood, so
`spey` falls back to numerical differentiation, which costs one likelihood
evaluation per free parameter and per gradient. Measured on the ATLAS monojet
model (78 regions, 151 free parameters, Apple M-series laptop):

| operation                                      | time    |
| ---------------------------------------------- | ------- |
| one likelihood evaluation                      | 7 ms    |
| `maximize_likelihood` from the post-fit values | 0.3 s   |
| one conditional fit, `likelihood(poi_test)`    | 20 s    |
| `exclusion_confidence_level`                   | 2.5 min |
| `sigma_mu`                                     | 1.5 min |
| `poi_upper_limit`                              | 50 min  |

Models with fewer nuisance parameters are essentially instantaneous — the toy
models in `tests/` run a full hypothesis test in well under a second.

`RooFit`'s automatic-differentiation backend (`EvalBackend("codegen")`) would
remove the bottleneck, but it segfaults on this workspace with ROOT 6.36, so it
is not used.
