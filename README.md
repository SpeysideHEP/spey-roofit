# spey-roofit

[![arxiv](https://img.shields.io/static/v1?style=plastic&label=arXiv&message=2307.06996&color=brightgreen)](https://arxiv.org/abs/2307.06996)
[![DOI](https://img.shields.io/static/v1?style=plastic&label=DOI&message=10.21468/SciPostPhys.16.1.032&color=blue)](https://scipost.org/10.21468/SciPostPhys.16.1.032)

<img src="https://raw.githubusercontent.com/SpeysideHEP/spey/main/docs/img/spey-plug-in.png" alt="Spey logo" style="float: right; margin-right: 20px" align="right" width=250px/>

[`RooFit`](https://root.cern/manual/roofit/),
[HS3](https://hep-statistics-serialization-standard.github.io/) and
[CMS Combine](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/)
plug-ins for [`spey`](https://spey.readthedocs.io).

`spey-roofit` turns any binned `RooFit` workspace into a `spey`
`StatisticalModel`, so that exclusion limits, significances, upper limits and
model combination are computed by `spey` exactly as they are for any other
likelihood prescription. It provides three backends:

| backend | input |
| --- | --- |
| `root.roofit` | a `RooFit` workspace: a `HistFactory` `.root` file from HEPData, or one built in memory |
| `root.hs3` | an HS3 `JSON` document, compiled into a `RooFit` workspace by `RooJSONFactoryWSTool` |
| `root.cms-combine` | a `CMS Combine` datacard, compiled into a `RooFit` workspace by `combine` itself |

All three share the same likelihood, Asimov and sampling machinery, so an
analysis published in any of the three formats gives the same numbers. Each
ships an **interpreter** so that a user can inspect the region structure of an
analysis and inject their own signal without knowing anything about `RooFit` —
or about HS3, or about `combine`.

Neither the HS3 nor the `combine` model is reimplemented here: `ROOT` compiles
the one and `combine` the other, and the plug-in reads the result.

## Installation

`ROOT` and `combine` are only distributed through `conda-forge`, so the
environment is best created with `conda`:

```bash
conda env create -f environment.yml
conda activate spey-roofit
pip install -e .
```

`environment-lock.yml` pins the exact versions the shipped validation results
were produced with.

The published material used by the examples below, by the validation notebooks
and by the `atlas`-, `topq`- and `cms`-marked tests is not committed — it
belongs to the records that publish it. Fetch it with:

```bash
./atlas-exot-2018-06/download.sh    # monojet, RooFit workspaces
./atlas-topq-2018-24/download.sh    # ttbar+gamma, HS3 documents and pyhf twins
./cms-exo-24-011/download.sh        # dijet angular, CMS Combine datacards
```

See [atlas-exot-2018-06/README.md](atlas-exot-2018-06/README.md),
[atlas-topq-2018-24/README.md](atlas-topq-2018-24/README.md) and
[cms-exo-24-011/README.md](cms-exo-24-011/README.md) for what they download.
Everything that does not need those files works without them.

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

### Read an HS3 document instead

`root.hs3` takes the same arguments, but the workspace is an HS3 document —
a path, a `JSON` string, a `dict`, or a `spey-hs3` `WorkspaceInterpreter` — and
the regions are named after the HS3 distributions:

```python
from spey_hs3.helper_functions import WorkspaceInterpreter

from spey_roofit import load_document

path = "atlas-topq-2018-24/workspace_dilep_1NF_HS3.json"

interpreter = WorkspaceInterpreter(load_document(path))
interpreter.bin_map                  # {'model_SR_alt1': 10}
interpreter.signal_samples           # which samples the POI scales
interpreter.expected_background_yields

model = spey.get_backend("root.hs3")(
    workspace=path,
    signal_yields={"model_SR_alt1": [4.0, 12.0, 15.0, 12.0, 9.0, 6.0, 4.0, 3.0, 2.0, 1.0]},
    strip_existing_signal=True,
    poi_name="mu",
    analysis="atlas-topq-2018-24",
)

model.exclusion_confidence_level()
```

`spey-hs3` provides the interpreter, so it is a dependency of the package.
`ROOT` does the compilation, so nothing but `PyROOT` is needed to evaluate the
result.

### Read a CMS Combine datacard

`root.cms-combine` takes a datacard — a path, or its text — and hands it to
`combine`. `DatacardInterpreter` is what shows the model first:

```python
from spey_roofit import DatacardInterpreter
from spey_roofit.combine import SWAPPED_TWO_HYPOTHESIS_MODEL

card = "cms-exo-24-011/datacards/input/chi_datacard13TeV3_ADD12000_run2.txt"

interpreter = DatacardInterpreter(card)
interpreter.bin_map                  # {'bin0': 12, ..., 'bin7': 12}
interpreter.signal_processes         # ['QCDADD12000', 'QCDADD12000_ALT']
interpreter.summary(max_channels=2)  # channels, samples and their yields

model = spey.get_backend("root.cms-combine")(
    datacard=card,
    physics_model=SWAPPED_TWO_HYPOTHESIS_MODEL,   # the -P option of text2workspace.py
    analysis="cms-exo-24-011",
)

model.exclusion_confidence_level(poi_test=1.0)
```

A published card is reinterpreted the same way as a `RooFit` workspace, by
stripping the signal it declares and attaching one's own:

```python
model = spey.get_backend("root.cms-combine")(
    datacard=card,
    signal_yields={"bin7": [4.0, 3.0, 2.0, 1.0, 1.0] + [0.0] * 7},
    strip_existing_signal=True,
    analysis="reinterpretation",
)
```

The card is *rewritten* — one extra column per channel, the signal columns
removed — and handed back to `combine`, so the reinterpreted model is built by
exactly the same code as the published one.

## How it works

| module                         | contents                                                                                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `spey_roofit.manager`          | lazy `PyROOT` import, batch mode, `RooMsgService` silencing, keep-alive registry                                                                 |
| `spey_roofit.data`             | `WorkspaceData`: flat, array-based view of a `RooFit` model — channels, bins, parameters, constraints, Asimov data and the calibrated likelihood |
| `spey_roofit.helper_functions` | `WorkspaceInterpreter`: inspection and signal injection for `RooFit` workspaces                                                                  |
| `spey_roofit.base`             | `RooFitBackendBase`: the parts of the `spey` contract that do not depend on where the model came from                                            |
| `spey_roofit.interface`        | `RooFitInterface`, the `root.roofit` backend                                                                                                     |
| `spey_roofit.hs3`              | compilation of an HS3 document into a `RooFit` workspace, and the repairs that needs                                                             |
| `spey_roofit.hs3_interface`    | `HS3Interface`, the `root.hs3` backend                                                                                                           |
| `spey_roofit.combine`          | `DatacardInterpreter`, and the thin layer that drives `combine`'s own parser and model builder                                                   |
| `spey_roofit.combine_interface`| `CombineInterface`, the `root.cms-combine` backend                                                                                               |
| `spey_roofit.physics_models`   | a `combine` physics model the plug-in contributes, for two hypothesis cards                                                                      |

`WorkspaceData` is deliberately independent of how the workspace was produced,
which is what lets `root.hs3` and `root.cms-combine` reuse it unchanged: those
backends only add a compilation step, and every likelihood, Asimov and sampling
method is shared through `RooFitBackendBase`.

### Reading HS3

`RooJSONFactoryWSTool` does the actual work, but three corrections are needed
around it, all of them for documents that current `ROOT` versions produce or
demand:

* the reader creates the constraint terms of a `histfactory_dist` on the fly
  and does **not** declare their global observables, so the model configuration
  it builds is incomplete — for the ATLAS dilepton workspace it declares 1 of
  226. `complete_model_config` recovers them; without it the data vector would
  be missing its entire auxiliary half and every Asimov dataset would be wrong.
* `ROOT` 6.36 *writes* `HistFactory` Gaussian constraints without the `x` key
  the standard requires, producing documents its own reader rejects with
  `node '...Constraint' is missing key 'x'`. `repair_document` restores it from
  the naming convention, and says so in a warning.
* the reader looks up a domain named literally `default_domain` and aborts the
  process — with an uncatchable C++ exception — if the document names its
  domain anything else. `ensure_default_domain` adds one, as the union of what
  is already declared, so a perfectly valid document does not take the
  interpreter down.

### Driving `combine`

`text2workspace.py` is a thin script around `combine`'s `DatacardParser` and
`ShapeBuilder`; the plug-in runs the same two in process. The workspace that
comes back is `combine`'s own — `ProcessNormalization`,
`FastVerticalInterpHistPdf2`, a `RooSimultaneous` over the channels — and every
switch of the tool is reachable through `combine_options()`.

Four things are arranged around it, none of which touches the likelihood
`combine` builds:

* **plain Gaussian constraints.** `combine` writes them as
  `SimpleGaussianConstraint`, which drops the `1/sqrt(2 pi)` of every term and
  would put `log L` in a different normalisation from every other `spey`
  backend. The plug-in asks for `--no-optimize-pdfs`; no likelihood ratio
  changes either way.
* **room around the auxiliary measurements.** `spey` normalises a constraint
  over its auxiliary measurement, and `combine` gives a `shape` parameter the
  range `[-4, 4]`, which at `theta = -1.5` already cuts six parts in a thousand
  off the Gaussian. `combine` never notices, because it does not normalise its
  constraints at all.
* **a complete nuisance set.** `combine` lists only the *constrained*
  parameters in its `ModelConfig`, so a `rateParam` would float in the model
  but never be profiled. Everything that floats is added.
* **one-bin templates for a counting card.** `combine` builds a card without
  `shapes` directives as a product of `RooPoisson` terms over plain variables
  rather than as a binned model, which is the same likelihood but not a shape
  `WorkspaceData` can read.

Signal injection and stripping work on the *datacard*, not on the workspace:
the card is rewritten — a column per channel added or the signal columns
removed, every systematics row kept square — and `combine` compiles the result.

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

There is one notebook per backend.

[`atlas-exot-2018-06/validation.ipynb`](atlas-exot-2018-06/validation.ipynb)
covers `root.roofit`:

* `spey`'s own `default.uncorrelated_background` and
  `default.correlated_background` likelihoods, reproduced as `HistFactory`
  workspaces — the log-likelihoods agree to machine precision;
* `pyhf` on the same model, through the
  [`spey-pyhf`](https://github.com/SpeysideHEP/spey-pyhf) plug-in so that both
  sides are driven by the same `spey` interface — log-likelihood, Asimov
  data and `CLs` agree to machine precision;
* the distribution of the profile-likelihood test statistic over 2000 toys,
  against $\chi^2_1$ and between all three implementations of the model;
* expected and observed exclusion and the 95% CL upper limit, as a Brazilian
  plot drawn with each of the three;
* the published ATLAS full Run-2 monojet analysis
  ([arXiv:2102.10874](https://arxiv.org/abs/2102.10874),
  [HEPData ins1847779](https://www.hepdata.net/record/ins1847779)) — the
  observed counts and post-fit backgrounds of all 78 regions are reproduced,
  and the profile likelihood recovers the signal strength stored in the
  published post-fit workspace.

[`atlas-topq-2018-24/validation.ipynb`](atlas-topq-2018-24/validation.ipynb)
covers `root.hs3`:

* HS3 documents written by hand to be mathematically identical to
  `default.uncorrelated_background`, `default.correlated_background` and
  `default.poisson` — they agree to $10^{-12}$, to the exact Jacobian
  `0.5 log det(rho)`, and to $10^{-14}$ respectively;
* the Asimov dataset — both halves, including the auxiliary measurements,
  which follow the nuisance parameters as eq. (25) of
  [arXiv:1007.1727](https://arxiv.org/abs/1007.1727) requires;
* the distribution of the profile-likelihood test statistic over 2000 toys,
  against $\chi^2_1$ and against `spey`;
* expected and observed exclusion and the 95% CL upper limit, as a Brazilian
  plot;
* a `HistFactory` workspace exported to HS3 by `ROOT` and read back — the two
  backends agree **bit for bit**;
* the published ATLAS $t\bar{t}\gamma$ measurement
  ([arXiv:2403.09452](https://arxiv.org/abs/2403.09452),
  [HEPData ins2768921](https://www.hepdata.net/record/ins2768921)), whose six
  likelihoods HEPData publishes both as HS3 and as `pyhf` workspaces. The fit
  recovers the measured signal strength, and the expected yields agree with the
  `pyhf` twin — read through `spey-pyhf` — to machine precision.

[`cms-exo-24-011/validation.ipynb`](cms-exo-24-011/validation.ipynb) covers
`root.cms-combine`:

* datacards written by hand so that the model `combine` builds from them is one
  of `spey`'s own likelihoods — a counting card without systematics is
  `default.poisson`, and `autoMCStats` in its Gaussian mode is
  `default.uncorrelated_background`, because Barlow–Beeston-lite is the
  additive constraint that backend implements. `lnN` is *not*: it is
  multiplicative, and has no counterpart among `spey`'s defaults;
* `pyhf` on the same models, through `spey-pyhf`: a symmetric `lnN` is a
  `normsys` and `autoMCStats` is a `staterror`, both to the tolerance of the
  minimiser;
* `HistFactory` on the same model, through `root.roofit`: a `shape`
  uncertainty whose varied templates preserve the total yield is a `HistoSys`,
  and the two agree to machine precision;
* the Asimov dataset, both halves;
* the distribution of the profile-likelihood test statistic over 2000 toys and
  the expected exclusion as a Brazilian plot, with all three implementations;
* the published CMS dijet angular search
  ([CMS-EXO-24-011](https://repository.cern/records/wqfww-n3j76),
  [HEPData ins3136278](https://www.hepdata.net/record/ins3136278)) — a two
  hypothesis fit over eight mass bins with 149 shape uncertainties, whose
  structure and yields are reproduced from the card, whose $\hat{x}$ comes out
  compatible with the Standard Model, and whose limit on the ADD scale is
  recovered by scanning the twelve published mass points.

Two caveats are worth stating, because both are easy to mistake for bugs.

All `pyhf` comparisons go through the `spey-pyhf` plug-in rather than through
`pyhf` directly: both sides are then `spey` `StatisticalModel`s answering the
same calls, so the only thing that differs is how the likelihood is evaluated.
The plug-in selects the `jax` tensor backend when `jax` is importable; the
comparisons force `numpy`, which is `pyhf`'s own reference implementation and
has no platform-dependent precision.

**The two serialisations of the ATLAS likelihood are not the same model.** The
HS3 document asks for a Poisson constraint on the Monte-Carlo statistics
parameters, which is what `HistFactory` does, whereas `pyhf` always uses a
Gaussian one. The rate models are identical; the constraint models are not.

**`sigma_mu` is computed differently by the two backends.** `spey` uses the
analytic Hessian of the log-likelihood when a backend supplies one, and
`RooFit` does not supply one, so both `root` backends fall back to the
asymptotic $\sigma_A=|\mu-\mu^\prime|/\sqrt{q_{\mu,A}}$. The two estimators of
$\sigma_\mu$ agree only in the asymptotic limit; everything computed from the
likelihood itself agrees to optimiser precision.

### Which `spey` the numbers were produced with

The comparisons above were run against `spey` 0.2.8, the first version whose
Asimov datasets carry the profiled auxiliary measurements. Up to 0.2.7 the
built-in backends dropped the auxiliary half of the data vector, so the
`root` backends and the built-in ones disagreed on every post-fit Asimov
quantity — a few parts in a thousand on `CLs`, a few per cent on the upper
limit. Both validation notebooks and the Asimov assertions in the test suite
expect the fixed behaviour.

Run the test suite with

```bash
./atlas-exot-2018-06/download.sh   # once, for the atlas-marked tests
./atlas-topq-2018-24/download.sh   # once, for the topq-marked tests
./cms-exo-24-011/download.sh       # once, for the cms-marked tests
pytest
```

Those tests skip themselves if the material has not been downloaded, so
`pytest` passes on a fresh clone either way. To skip them explicitly:

```bash
pytest -m "not atlas and not topq and not cms"
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

The HS3 backend adds the compilation step, which is the only cost specific to
it; everything after that is the same `RooFit` model. On the ATLAS
$t\bar{t}\gamma$ workspaces:

| workspace | bins | free parameters | compile | one `likelihood(poi_test)` |
| --- | --- | --- | --- | --- |
| `dilep_1NF` | 10 | 226 | 2 s | 9 s |
| `ljets_1NF` | 21 | 249 | 2 s | 56 s |
| `combined_1NF` | 31 | 271 | 2 s | 83 s |

The `combine` backend adds its own compilation step and is otherwise the same
`RooFit` model again. `combine` builds its templates with
`FastVerticalInterpHistPdf2`, which caches the morphed histogram of a process
and only rebuilds it when one of its nuisance parameters moves, so a card with
a hundred shape uncertainties stays fast. On the CMS dijet angular card
(8 channels, 96 bins, 149 shape uncertainties):

| operation | time |
| --- | --- |
| parse the card, read its templates | 2 s |
| compile it with `combine` | 4 s |
| one likelihood evaluation | 0.2 ms |
| `maximize_likelihood` | 4 s |
| one conditional fit, `likelihood(poi_test)` | 2 s |
| `exclusion_confidence_level` | 18 s |

Models with fewer nuisance parameters are essentially instantaneous — the toy
models in `tests/` run a full hypothesis test in well under a second, and the
2000-toy test-statistic study of the HS3 notebook takes about two minutes
through both backends. The same study in the `combine` notebook takes about
eighty minutes, almost all of it in `spey-pyhf`: the three backends are run on
the same toys, and `pyhf` is by far the slowest of them on this model.

`RooFit`'s automatic-differentiation backend (`EvalBackend("codegen")`) would
remove the bottleneck, but it segfaults on this workspace with ROOT 6.36, so it
is not used.
