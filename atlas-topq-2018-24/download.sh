#!/usr/bin/env bash
#
# Download the ATLAS ttbar+gamma validation material from HEPData.
#
# The files belong to the HEPData record rather than to this repository, so
# they are not committed here. This script fetches everything that
# `validation.ipynb` and `tests/test_topq.py` need:
#
#   workspace_<channel>_<fit>_HS3.json   likelihood in HS3 format
#   workspace_<channel>_<fit>.json       the same likelihood as a pyhf workspace
#
# with <channel> one of ljets, dilep, combined and <fit> one of 1NF, 2NF.
# The pyhf twins are what makes this record useful: the two files describe the
# same model in two different serialisations, so `root.hs3` can be checked
# against an entirely independent implementation.
#
# Record: https://www.hepdata.net/record/ins2768921
# Paper : ATLAS, arXiv:2403.09452, JHEP 10 (2024) 191
#
# Usage:
#   ./download.sh             download whatever is missing (about 38 MB)
#   ./download.sh --hs3-only  skip the pyhf twins (about 11 MB)
#   ./download.sh --force     re-download everything
#   ./download.sh --check     verify what is already there, download nothing

set -euo pipefail

readonly RECORD_DOI="10.17182/hepdata.146899.v1"

readonly HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# resource | file name | sha256
#
# The record stores the two serialisations of every fit next to each other:
# the odd resources are the pyhf workspaces, the even ones their HS3 twins.
readonly HS3_FILES=(
    "r2|workspace_ljets_2NF_HS3.json|e19673d3a29d3e87028a6531b6c19fe46763ff3c5de6cf4e86d3be7b8f7712c4"
    "r4|workspace_ljets_1NF_HS3.json|65bcedf498eb8b231994f1fbe89c7a67356cdb948415ac79d1cc5e01fd8b7fb0"
    "r6|workspace_dilep_2NF_HS3.json|a299961008aa6f111cebcb2a2513e4ffac924e749c6324b72065eab9f666d390"
    "r8|workspace_dilep_1NF_HS3.json|911588028e84be44c7fbf26075ace92b89f2776d5445f2cc9a715c0d09a4970b"
    "r10|workspace_combined_2NF_HS3.json|10530838cf3b61554420896d7f3434a65eb2e4c5ce455489dc49f4c8e73c66f4"
    "r12|workspace_combined_1NF_HS3.json|e06838847851619b643c3bba294ddeda38f1a1d308e1de576c3484e1a3ddc4db"
)

readonly PYHF_FILES=(
    "r1|workspace_ljets_2NF.json|94a2d2d29b6d0f698c216331d57df2c71d72fe2344851182337508d2acdebac9"
    "r3|workspace_ljets_1NF.json|9bcaac2fac520c26c7203aa76773a2e625eded6bf64c5618efcbaa97b9d9b414"
    "r5|workspace_dilep_2NF.json|b3cc3cd671c0035fb035de6434b72d47153fba8b7aa74bf03b0597cfbb6d9a3f"
    "r7|workspace_dilep_1NF.json|777b300aa74e038942bec8f467c864ea8f10860e30661bb36d6cfd903aca8ee3"
    "r9|workspace_combined_2NF.json|02cf134495fb1fbdf79130fa4e3bf6578fb6a97c1644fbd0fd7cebd4b1119fe4"
    "r11|workspace_combined_1NF.json|0a3112b0afc9fc7b324bd0ba3004718d5a572786bcb3263154b19496bf7f27d9"
)

FORCE=0
CHECK_ONLY=0
HS3_ONLY=0
for argument in "$@"; do
    case "${argument}" in
        --force) FORCE=1 ;;
        --check) CHECK_ONLY=1 ;;
        --hs3-only) HS3_ONLY=1 ;;
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

download_workspace() {
    local entry="$1"
    local resource="${entry%%|*}"
    local rest="${entry#*|}"
    local name="${rest%%|*}"
    local expected="${rest#*|}"
    local destination="${HERE}/${name}"

    if [[ ${FORCE} -eq 0 && -f "${destination}" ]]; then
        if verify "${destination}" "${expected}"; then
            info "${name} already present and verified"
            return 0
        fi
        [[ ${CHECK_ONLY} -eq 1 ]] && die "${name} is corrupted; re-run without --check"
        info "${name} failed verification, downloading again"
    fi

    if [[ ${CHECK_ONLY} -eq 1 ]]; then
        echo "    ${name} is missing" >&2
        return 1
    fi

    info "downloading ${name} (${RECORD_DOI}/${resource})"
    fetch "$(resource_url "${resource}")" "${destination}"
    verify "${destination}" "${expected}" \
        || die "${name} does not match its expected checksum"
}

# Every file of this record is a JSON document, so a structural check costs
# nothing and catches a landing page saved under the wrong name.
check_json() {
    python3 - "$1" <<'PYEOF'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    try:
        document = json.load(handle)
    except json.JSONDecodeError as error:
        sys.exit(f"{path}: not valid JSON ({error})")
if not isinstance(document, dict):
    sys.exit(f"{path}: expected a JSON object")
if not ({"distributions", "likelihoods"} <= set(document)  # HS3
        or {"channels", "measurements"} <= set(document)):  # pyhf
    sys.exit(f"{path}: neither an HS3 nor a pyhf workspace")
PYEOF
}

echo "ATLAS ttbar+gamma validation material (HEPData ins2768921, ${RECORD_DOI})"
echo "target: ${HERE}"

status=0
requested=("${HS3_FILES[@]}")
if [[ ${HS3_ONLY} -eq 0 ]]; then
    requested+=("${PYHF_FILES[@]}")
fi

for entry in "${requested[@]}"; do
    download_workspace "${entry}" || status=1
done

if [[ ${CHECK_ONLY} -eq 1 ]]; then
    if [[ ${status} -eq 0 ]]; then
        echo "all files present and verified"
    else
        echo "some files are missing; run ./download.sh to fetch them" >&2
    fi
    exit "${status}"
fi

for entry in "${requested[@]}"; do
    rest="${entry#*|}"
    check_json "${HERE}/${rest%%|*}" || die "${rest%%|*} is not a usable workspace"
done

echo "done. Everything validation.ipynb needs is in place."
