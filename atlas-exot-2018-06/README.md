# Validation material

This directory holds the material used to validate `spey-roofit` against a
published analysis:

> ATLAS Collaboration, *Search for new phenomena in events with an energetic jet
> and missing transverse momentum in $pp$ collisions at $\sqrt{s} = 13$ TeV with
> the ATLAS detector*,
> [arXiv:2102.10874](https://arxiv.org/abs/2102.10874),
> [HEPData ins1847779](https://www.hepdata.net/record/ins1847779).

**The files themselves are not committed.** They are several megabytes, they
belong to the HEPData record rather than to this repository, and they are
redistributed under HEPData's terms. Fetch them with:

```bash
./atlas-exot-2018-06/download.sh
```

Everything except `download.sh`, this README and `.gitignore` is git-ignored, so
the downloads never end up in a commit.

## What gets downloaded

| File | Source | Contents |
| --- | --- | --- |
| `bOnlyFit_postFit.root` | resource `r3` | background-only RooFit workspace, nuisance parameters at their post-fit values |
| `sPlusBFit_postFit.root` | resource `r2` | signal-plus-background RooFit workspace for the `DMA_1_2000_gq0p25` benchmark |
| `bkgonly_workspace.json` | resource `r1` | the same background-only model as a HistFactory JSON likelihood, extracted from the archive |
| `hepdata/Histogram_SR.json` | table 7 | published data and post-fit background of the signal region |
| `hepdata/Histogram_CR*.json` | tables 2-6 | the same for the five control regions |

The resources are addressed by their DOIs under `10.17182/hepdata.102093.v4`.
A resource DOI redirects to an HTML landing page rather than to the file, so the
script resolves the DOI and then requests the same URL with `view=true`; that is
why a bare `curl -OJL https://doi.org/...` gives you a web page instead of a
workspace.

The `values` of the HEPData tables are identical across every version of the
record, so pinning to `v4` only fixes the metadata, not the numbers.

## Usage

```bash
./download.sh           # fetch whatever is missing (about 10 MB)
./download.sh --check   # verify what is there, download nothing
./download.sh --force   # re-download everything
./download.sh --help
```

The script needs only `curl` and `python3`. It is idempotent: a file that is
already present and matches its checksum is left alone. The two workspaces and
the JSON likelihood are verified against pinned `sha256` sums, so a truncated or
replaced download is caught rather than silently used; a file that fails
verification is downloaded again. The HEPData tables are validated structurally
instead — thirteen bins, each carrying a data and a background column — because
a checksum would break the moment HEPData re-serialises a table without changing
a single number.

## What uses it

* `validation.ipynb` — sections 4 and 5: reproducing the published
  yields of all 78 regions, recovering the signal strength stored in the
  post-fit workspace, cross-checking the expected yields against `pyhf`,
  comparing `spey`'s Brazilian plot and profile likelihood against a pure
  `RooStats` / `RooFit` run on `sPlusBFit_postFit.root`, and injecting a signal
  into the background-only workspace.
* `tests/test_atlas.py` — the same checks as a test suite.

Both degrade gracefully when the material is absent: the tests skip with
`pytest.skip` and the notebook's `pyhf` cell prints a note and moves on, so a
fresh clone still runs everything that does not need these files.

```bash
pytest -m "not atlas"   # skip them explicitly
```
