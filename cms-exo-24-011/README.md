# Validation material for `root.cms-combine`

This directory holds the material used to validate the `CMS Combine` backend against a
published analysis:

> CMS Collaboration, *Measurement of dijet angular distributions and search for beyond
> the standard model physics in proton-proton collisions at $\sqrt{s}=13$ TeV*,
> CMS-EXO-24-011,
> [HEPData ins3136278](https://www.hepdata.net/record/ins3136278),
> statistical model at [10.17181/wqfww-n3j76](https://repository.cern/records/wqfww-n3j76).

**The files themselves are not committed.** The published archive is one gigabyte, it
belongs to the CERN Document Repository record rather than to this repository, and it is
redistributed under CC BY 4.0. Fetch it with:

```bash
./cms-exo-24-011/download.sh
```

Everything except `download.sh`, `validation.ipynb`, this README and `.gitignore` is
git-ignored, so the downloads never end up in a commit.

## What gets downloaded

The record publishes a single `v1.0.tar.gz` holding the datacards of every signal model
of the paper — ADD, ALP, contact interactions, dark matter mediators, quantum black
holes and SMEFT — together with the template files they refer to. The script downloads
it, checks it against the published `md5`, and unpacks only what is needed:

| Path | Contents |
| --- | --- |
| `datacards/input/` | the reference datacard, its template file and the systematics list |
| `datacards/ADD/` | one datacard per ADD mass point, $\Lambda$ from 9 to 20 TeV |
| `datacards/README.md`, `datacards/exo_combine_checks.sh` | the collaboration's own instructions and cross-check script |

That is about 130 MB unpacked. `--full` unpacks every signal model instead, and the
1 GB archive is deleted again unless `--keep` is given.

```bash
./download.sh           # fetch and unpack the datacards (about 130 MB unpacked)
./download.sh --full    # unpack every signal model as well (about 1 GB)
./download.sh --keep    # keep the downloaded archive
./download.sh --check   # verify what is there, download nothing
./download.sh --force   # download again
./download.sh --help
```

The script needs only `curl`, `tar` and `md5sum` (or `md5` on macOS). It is idempotent: a
directory that is already unpacked is left alone unless `--force` is given.

## The model this card describes

It is worth knowing three things about this card before reading the notebook, because all
three are unusual and all three are places where a reader can go wrong.

**It is a two hypothesis fit, not a signal-plus-background one.** Each of the eight dijet
mass bins carries two *signal* processes, `QCDADD<Λ>` (QCD plus the ADD contribution) and
`QCDADD<Λ>_ALT` (QCD alone), normalised to the same number of events, plus a dummy
background of $10^{-20}$ events that `combine` insists on. A parameter $x$ interpolates
between them, and the mass scale is excluded when the upper limit on $x$ drops below one.
The BSM signal is therefore a pure **shape**: `spey`'s signal template is large bin by bin
and integrates to nothing.

**Which hypothesis $x$ selects is a convention, and the two conventions are not
equivalent.** `combine` ships `HiggsAnalysis.CombinedLimit.HiggsJPC:twoHypothesisHiggs`,
which scales the process whose name carries the `ALT` marker by $x$ — QCD alone, here.
The record states the opposite, $x=1$ for QCD+BSM, and used a modified copy of the model
to get it. Since `spey` reads the model at $\mu=0$ as its background, the stock model
would test the wrong hypothesis and report a signal of the wrong sign. The plug-in
therefore contributes `spey_roofit.physics_models:twoHypothesisHiggs`, which is the stock
model with the two exchanged:

```python
import spey
from spey_roofit.combine import SWAPPED_TWO_HYPOTHESIS_MODEL

model = spey.get_backend("root.cms-combine")(
    datacard="datacards/input/chi_datacard13TeV3_ADD12000_run2.txt",
    physics_model=SWAPPED_TWO_HYPOTHESIS_MODEL,
    analysis="cms-exo-24-011",
)
```

**The card renames its own nuisance parameters, and the highest mass bin is ragged.** The
collaboration appends sixty `nuisance edit rename` lines — `jes1` becomes
`CMS_scale_j_AbsoluteScale` and so on — which rename the parameter but not the histograms
it morphs. And `chi7000_13000` is rebinned to five bins for the data and both signals
while the dummy background still has twelve, so `combine` pads the short templates. Both
are handled, and both are checked in `tests/test_cms.py`.

## Why this record and not a smaller one

The datacards of the ADD scan are the only part of the release the validation needs, but
they are the interesting part: twelve cards that differ only in the signal template, so
scanning them gives the exclusion as a function of the ADD scale and, where it crosses
95%, the limit on $\Lambda$. The notebook does exactly that; the numbers it prints are
the ones to compare against the paper.

## What uses it

* `validation.ipynb` — section 5: reading the published card, recovering $\hat{x}$,
  excluding the $\Lambda=12$ TeV hypothesis, scanning the mass points, and
  reinterpreting the card with an injected signal.
* `tests/test_cms.py` — the same checks as a test suite.

Both degrade gracefully when the material is absent: the tests skip with `pytest.skip`
and the notebook prints a note and moves on, so a fresh clone still runs everything that
does not need these files.

```bash
pytest -m "not cms"   # skip them explicitly
```

## Which versions the numbers were produced with

`combine` 10.5.0 from `conda-forge`, `ROOT` 6.36.04, and `spey` 0.2.8 — the first version
whose Asimov datasets carry the profiled auxiliary measurements. `environment-lock.yml`
in the repository root reproduces the environment exactly.
