#!/usr/bin/env bash
#
# Download the CMS dijet angular statistical model.
#
# The material belongs to the CERN Document Repository record rather than to
# this repository, so it is not committed here. This script fetches the archive
# the CMS collaboration published, verifies it, and unpacks what
# `validation.ipynb` and `tests/test_cms.py` need:
#
#   datacards/input/    the reference datacard, its templates and the
#                       systematics list; this is the card the collaboration's
#                       own cross-check script runs on
#   datacards/ADD/      one datacard per ADD mass point, 9 to 20 TeV, each with
#                       its own template file
#
# The archive itself is 1.0 GB and holds every signal model of the paper; only
# a small part of it is unpacked unless `--full` is given, and the archive is
# deleted again unless `--keep` is.
#
# Record: https://repository.cern/records/wqfww-n3j76  (DOI 10.17181/wqfww-n3j76)
# Data  : https://www.hepdata.net/record/ins3136278
# Paper : CMS, CMS-EXO-24-011
#
# Usage:
#   ./download.sh          fetch and unpack the datacards (about 130 MB unpacked)
#   ./download.sh --full   unpack every signal model as well (about 1 GB)
#   ./download.sh --keep   keep the downloaded archive
#   ./download.sh --force  download again even if the datacards are there
#   ./download.sh --check  report what is already there, download nothing
#   ./download.sh --help

set -euo pipefail

readonly URL="https://repository.cern/api/records/wqfww-n3j76/files/v1.0.tar.gz/content"
readonly ARCHIVE="v1.0.tar.gz"
readonly MD5="e31fd7937e6e02f27e9ccea8c1a53bcc"
readonly SIZE="1026118161"

readonly HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# the card the collaboration's own checks use, and the one this repository does
readonly REFERENCE="datacards/input/chi_datacard13TeV3_ADD12000_run2.txt"

FORCE=0
CHECK_ONLY=0
FULL=0
KEEP=0
for argument in "$@"; do
    case "${argument}" in
        --force) FORCE=1 ;;
        --check) CHECK_ONLY=1 ;;
        --full)  FULL=1 ;;
        --keep)  KEEP=1 ;;
        --help|-h)
            sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option '${argument}'. Try --help." >&2
            exit 2
            ;;
    esac
done

cd "${HERE}"

md5_of() {
    # macOS ships `md5`, GNU coreutils `md5sum`; both are fine here.
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$1" | cut -d' ' -f1
    else
        md5 -q "$1"
    fi
}

report() {
    local count
    if [[ -f "${REFERENCE}" ]]; then
        count="$(find datacards -name '*.txt' 2>/dev/null | wc -l | tr -d ' ')"
        echo "OK      datacards unpacked (${count} cards)"
        return 0
    fi
    echo "MISSING datacards; run ./download.sh"
    return 1
}

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    report
    exit $?
fi

if [[ -f "${REFERENCE}" && "${FORCE}" -eq 0 && "${FULL}" -eq 0 ]]; then
    echo "The datacards are already there. Use --force to fetch them again."
    exit 0
fi

if [[ ! -f "${ARCHIVE}" || "${FORCE}" -eq 1 ]]; then
    echo "Downloading the statistical model (1.0 GB); this takes a while."
    curl -# -L --retry 3 -C - -o "${ARCHIVE}" "${URL}"
fi

actual_size="$(wc -c < "${ARCHIVE}" | tr -d ' ')"
if [[ "${actual_size}" != "${SIZE}" ]]; then
    echo "The archive is ${actual_size} bytes, expected ${SIZE}." >&2
    echo "The download was interrupted; run ./download.sh again to resume." >&2
    exit 1
fi

echo "Verifying the checksum."
actual_md5="$(md5_of "${ARCHIVE}")"
if [[ "${actual_md5}" != "${MD5}" ]]; then
    echo "Checksum mismatch: got ${actual_md5}, expected ${MD5}." >&2
    echo "Remove ${ARCHIVE} and try again." >&2
    exit 1
fi

if [[ "${FULL}" -eq 1 ]]; then
    echo "Unpacking everything."
    tar xzf "${ARCHIVE}"
else
    echo "Unpacking the datacards."
    tar xzf "${ARCHIVE}" \
        datacards/README.md \
        datacards/LICENSE \
        datacards/exo_combine_checks.sh \
        datacards/input \
        datacards/ADD
fi

if [[ "${KEEP}" -eq 0 ]]; then
    echo "Removing the archive; pass --keep to hold on to it."
    rm -f "${ARCHIVE}"
fi

report
