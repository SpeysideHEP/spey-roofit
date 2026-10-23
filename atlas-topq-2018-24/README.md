# Validation material for `root.hs3`

This directory holds the material used to validate the HS3 backend against a
published analysis:

> ATLAS Collaboration, *Measurements of inclusive and differential
> cross-sections of $t\bar{t}\gamma$ production in $pp$ collisions at
> $\sqrt{s}=13$ TeV with the ATLAS detector*,
> [arXiv:2403.09452](https://arxiv.org/abs/2403.09452),
> JHEP 10 (2024) 191,
> [HEPData ins2768921](https://www.hepdata.net/record/ins2768921).

**The files themselves are not committed.** They are tens of megabytes, they
belong to the HEPData record rather than to this repository, and they are
redistributed under HEPData's terms. Fetch them with:

```bash
./atlas-topq-2018-24/download.sh
```

Everything except `download.sh`, `validation.ipynb`, this README and
`.gitignore` is git-ignored, so the downloads never end up in a commit.

## What gets downloaded

The record publishes each fit **twice**, once as an HS3 document and once as a
`pyhf` workspace. That is what makes it valuable here: the two files describe
the same measurement in two independent serialisations, so `root.hs3` can be
checked against a completely different implementation of the same
`HistFactory` model. The `pyhf` side is read through the
[`spey-pyhf`](https://github.com/SpeysideHEP/spey-pyhf) plug-in, so both are
`spey` models answering the same calls.

| File | Source | Contents |
| --- | --- | --- |
| `workspace_ljets_1NF_HS3.json` | `r4` | single-lepton channel, inclusive $t\bar{t}\gamma$ fit |
| `workspace_ljets_2NF_HS3.json` | `r2` | single-lepton channel, production and decay separated |
| `workspace_dilep_1NF_HS3.json` | `r8` | dilepton channel, inclusive fit |
| `workspace_dilep_2NF_HS3.json` | `r6` | dilepton channel, production and decay separated |
| `workspace_combined_1NF_HS3.json` | `r12` | both channels combined, inclusive fit |
| `workspace_combined_2NF_HS3.json` | `r10` | both channels combined, production and decay separated |
| `workspace_<channel>_<fit>.json` | `r1`, `r3`, `r5`, `r7`, `r9`, `r11` | the same six likelihoods as `pyhf` workspaces |

The resources are addressed by their DOIs under `10.17182/hepdata.146899.v1`.
A resource DOI redirects to an HTML landing page rather than to the file, so
the script resolves the DOI and then requests the same URL with `view=true`;
that is why a bare `curl -OJL https://doi.org/...` gives you a web page instead
of a workspace.

Pass `--hs3-only` to skip the `pyhf` twins. The notebook and the tests still
run — the sections that compare the two serialisations are skipped.

## The two serialisations are *not* the same likelihood

This is worth stating plainly, because it is easy to mistake for a bug in one
of the tools. Compared bin by bin at the same parameter values:

* the **expected yields agree to machine precision** at the nominal nuisance
  parameters and for any value of the signal strength — the rate model, the
  data and the sample composition are identical;
* they differ by up to $10^{-5}$ *relative* once the nuisance parameters are
  displaced, because `ROOT` and `pyhf` implement the `HistFactory`
  interpolation of `normsys`/`histosys` slightly differently;
* the **constraint on the Monte-Carlo statistical uncertainty differs by
  construction**: the HS3 document asks for a Poisson constraint on the
  $\gamma$ parameters, which is what `HistFactory` does, whereas `pyhf`
  always uses a Gaussian one. This is a genuinely different likelihood, and it
  shifts $\log\mathcal{L}$ by an $\mathcal{O}(1)$ amount that *depends* on
  $\gamma$ rather than by a constant.

So the `pyhf` twin validates the rate model exactly and the constraint model
only qualitatively. The exact, number-by-number validation of the constraint
model is done in `tests/test_hs3.py` against `spey`'s own likelihoods and
against `root.roofit`, where both sides are known to be the same model.

## Which `spey` the numbers were produced with

`spey` 0.2.8, the first version whose Asimov datasets carry the profiled
auxiliary measurements. Against 0.2.7 or earlier the notebook still runs, but
sections 2.4, 3 and 4 show differences of a few parts in a thousand up to a few
per cent, all traceable to that one issue.

## Usage

```bash
./download.sh              # fetch whatever is missing (about 38 MB)
./download.sh --hs3-only   # HS3 documents only (about 11 MB)
./download.sh --check      # verify what is there, download nothing
./download.sh --force      # re-download everything
./download.sh --help
```

The script needs only `curl` and `python3`. It is idempotent: a file that is
already present and matches its checksum is left alone. Every file is verified
against a pinned `sha256` sum, so a truncated or replaced download is caught
rather than silently used, and a file that fails verification is downloaded
again.

## What uses it

* `validation.ipynb` — the second half: reading the published workspaces,
  recovering the measured signal strength, comparing against the `pyhf` twin,
  comparing `spey`'s Brazilian plot and profile likelihood against a pure
  `RooStats` / `RooFit` run on the dilepton document, and reinterpreting the dilepton channel with an injected signal.
* `tests/test_topq.py` — the same checks as a test suite.

Both degrade gracefully when the material is absent: the tests skip with
`pytest.skip` and the notebook prints a note and moves on, so a fresh clone
still runs everything that does not need these files.

```bash
pytest -m "not topq"   # skip them explicitly
```
