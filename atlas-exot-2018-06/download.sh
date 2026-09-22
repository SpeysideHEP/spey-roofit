#!/usr/bin/env bash
#
# Download the ATLAS monojet validation material from HEPData.
#
# The files are large and belong to the HEPData record, not to this repository,
# so they are not committed here. This script fetches everything that
# `notebooks/validation.ipynb` and `tests/test_atlas.py` need:
#
#   bOnlyFit_postFit.root    background-only RooFit workspace      (resource r3)
#   sPlusBFit_postFit.root   signal+background RooFit workspace    (resource r2)
#   bkgonly_workspace.json   full likelihood in HistFactory JSON   (resource r1)
#   hepdata/Histogram_*.json published yields of the six regions   (tables 2-7)
#
# Record: https://www.hepdata.net/record/ins1847779
# Paper : ATLAS, arXiv:2102.10874
#
# Usage:
#   ./download.sh            download whatever is missing
#   ./download.sh --force    re-download everything
#   ./download.sh --check    verify what is already there, download nothing

set -euo pipefail

readonly RECORD_DOI="10.17182/hepdata.102093.v4"
readonly INSPIRE_ID="ins1847779"
readonly TABLE_VERSION=4

# The `values` of the HEPData tables are identical in every version of the
# record; pinning the version only keeps the metadata reproducible.

readonly HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TABLE_DIR="${HERE}/hepdata"

readonly REGIONS=(SR CR1mu0b CR1e0b CR1L1b CR2mu CR2e)

# sha256 of each resource, so a truncated or replaced download is caught.
readonly SHA_BONLY="89bd79bc849b5058c7bff0c3aea83d8bb71f42ea14f0aa387a0e1fc4fa69267f"
readonly SHA_SPLUSB="c8fa459edb0a4a92e80ca150d86709a277565add183ab45598870fdde7c4b732"
readonly SHA_JSON="1eebf1a8fa13e68646729d3ea83002182d9d2dcec9f1e9cf4bbb380d0e8abce2"

FORCE=0
CHECK_ONLY=0
for argument in "$@"; do
    case "${argument}" in
        --force) FORCE=1 ;;
        --check) CHECK_ONLY=1 ;;
        -h|--help) sed -n '3,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown option '${argument}'. See --help." >&2; exit 2 ;;
    esac
done

die() { echo "error: $*" >&2; exit 1; }
info() { echo "  $*"; }

for tool in curl python3; do
    command -v "${tool}" >/dev/null 2>&1 || die "'${tool}' is required but not installed."
done

# sha256sum on Linux, shasum on macOS.
sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

verify() {
    local path="$1" expected="$2" actual
    [[ -f "${path}" ]] || return 1
    actual="$(sha256_of "${path}")"
    [[ "${actual}" == "${expected}" ]] || {
        echo "    checksum mismatch for $(basename "${path}")" >&2
        echo "      expected ${expected}" >&2
        echo "      got      ${actual}" >&2
        return 2
    }
}

# A HEPData resource DOI redirects to an HTML landing page; the file itself is
# served from the same URL with `view=true`. Resolve the DOI first so that the
# resource IDs do not have to be hard-coded here.
resource_url() {
    local resolved
    resolved="$(curl -sSIL -o /dev/null -w '%{url_effective}' "https://doi.org/${RECORD_DOI}/$1")" \
        || die "cannot resolve DOI ${RECORD_DOI}/$1 - is the network up?"
    [[ "${resolved}" == *"hepdata.net/record/resource/"* ]] \
        || die "DOI ${RECORD_DOI}/$1 resolved to an unexpected URL: ${resolved}"
    echo "${resolved/landing_page=true/view=true}"
}

fetch() {
    local url="$1" destination="$2"
    curl -sSfL --retry 3 --retry-delay 2 -o "${destination}.part" "${url}" \
        || { rm -f "${destination}.part"; die "download failed: ${url}"; }
    mv "${destination}.part" "${destination}"
}

# ---------------------------------------------------------------------------
# RooFit workspaces
# ---------------------------------------------------------------------------

download_workspace() {
    local resource="$1" name="$2" expected="$3"
    local destination="${HERE}/${name}"

    if [[ ${FORCE} -eq 0 && -f "${destination}" ]]; then
        if verify "${destination}" "${expected}"; then
            info "${name} already present and verified"
            return 0
        fi
        [[ ${CHECK_ONLY} -eq 1 ]] && die "${name} is corrupted; re-run without --check"
        info "${name} failed verification, downloading again"
    fi

    [[ ${CHECK_ONLY} -eq 1 ]] && { echo "    ${name} is missing" >&2; return 1; }

    info "downloading ${name} (${RECORD_DOI}/${resource})"
    fetch "$(resource_url "${resource}")" "${destination}"
    verify "${destination}" "${expected}" || die "${name} does not match its expected checksum"
    info "${name} done"
}

# ---------------------------------------------------------------------------
# Full likelihood, shipped inside a zip archive
# ---------------------------------------------------------------------------

download_likelihood() {
    local destination="${HERE}/bkgonly_workspace.json"

    if [[ ${FORCE} -eq 0 && -f "${destination}" ]]; then
        if verify "${destination}" "${SHA_JSON}"; then
            info "bkgonly_workspace.json already present and verified"
            return 0
        fi
        [[ ${CHECK_ONLY} -eq 1 ]] && die "bkgonly_workspace.json is corrupted"
        info "bkgonly_workspace.json failed verification, downloading again"
    fi

    [[ ${CHECK_ONLY} -eq 1 ]] && { echo "    bkgonly_workspace.json is missing" >&2; return 1; }

    local archive
    archive="$(mktemp "${TMPDIR:-/tmp}/spey-roofit-likelihood.XXXXXX")"
    trap 'rm -f "${archive}"' RETURN

    info "downloading bkgonly_workspace.json (${RECORD_DOI}/r1)"
    fetch "$(resource_url r1)" "${archive}"

    # `unzip` is not installed everywhere; python3 is already a requirement.
    python3 - "${archive}" "${destination}" <<'PYEOF'
import sys
import zipfile

archive, destination = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(archive) as bundle:
    wanted = [
        name
        for name in bundle.namelist()
        if name.endswith("bkgonly_workspace.json") and not name.startswith("__MACOSX")
    ]
    if len(wanted) != 1:
        sys.exit(f"expected one bkgonly_workspace.json in the archive, found {wanted}")
    with bundle.open(wanted[0]) as source, open(destination, "wb") as target:
        target.write(source.read())
PYEOF

    verify "${destination}" "${SHA_JSON}" \
        || die "bkgonly_workspace.json does not match its expected checksum"
    info "bkgonly_workspace.json done"
}

# ---------------------------------------------------------------------------
# Published yields of the six regions
# ---------------------------------------------------------------------------

check_table() {
    python3 - "$1" <<'PYEOF'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    table = json.load(handle)
rows = table.get("values")
if not isinstance(rows, list) or len(rows) != 13:
    sys.exit(f"{sys.argv[1]}: expected 13 bins, found {len(rows) if rows else 0}")
if not all("y" in row and len(row["y"]) >= 2 for row in rows):
    sys.exit(f"{sys.argv[1]}: rows do not carry data and background columns")
PYEOF
}

download_tables() {
    mkdir -p "${TABLE_DIR}"
    local missing=0
    for region in "${REGIONS[@]}"; do
        local name="Histogram_${region}.json"
        local destination="${TABLE_DIR}/${name}"

        if [[ ${FORCE} -eq 0 && -f "${destination}" ]] && check_table "${destination}" 2>/dev/null; then
            info "${name} already present and verified"
            continue
        fi

        if [[ ${CHECK_ONLY} -eq 1 ]]; then
            echo "    ${name} is missing or invalid" >&2
            missing=1
            continue
        fi

        info "downloading ${name}"
        # The table name contains a space, hence the %20.
        fetch "https://www.hepdata.net/download/table/${INSPIRE_ID}/Histogram%20${region}/${TABLE_VERSION}/json" \
            "${destination}"
        check_table "${destination}" || die "${name} is not a valid HEPData table"
    done
    return "${missing}"
}

# ---------------------------------------------------------------------------

echo "ATLAS monojet validation material (HEPData ${INSPIRE_ID}, ${RECORD_DOI})"
echo "target: ${HERE}"

status=0
download_workspace r3 bOnlyFit_postFit.root "${SHA_BONLY}" || status=1
download_workspace r2 sPlusBFit_postFit.root "${SHA_SPLUSB}" || status=1
download_likelihood || status=1
download_tables || status=1

if [[ ${CHECK_ONLY} -eq 1 ]]; then
    if [[ ${status} -eq 0 ]]; then
        echo "all files present and verified"
    else
        echo "some files are missing; run ./download.sh to fetch them" >&2
    fi
    exit "${status}"
fi

echo "done. Everything notebooks/validation.ipynb needs is in place."
