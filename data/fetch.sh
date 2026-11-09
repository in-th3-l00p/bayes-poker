#!/usr/bin/env bash
# Fetch the hand histories used in the implementation.
#
# Source: PHH dataset (Zenodo, DOI 10.5281/zenodo.10796885, MIT licence).
# The archive is a single ~20 GB zip; fetch_phh.py reads its index over HTTP
# range requests and extracts only the files we need into data/raw/.
#
# Usage:
#   data/fetch.sh [acpc|pluribus|handhq|all]     (default: acpc)
#
#   acpc      40 matches of ACPC 2011 heads-up limit Hold'em, 120,000 hands,
#             ~44 MB. Every hole card is revealed: main dataset (E3).
#   pluribus  10,000 six-max no-limit hands played by Pluribus, ~39 MB.
#   handhq    one 2009 online slice (human players, cards only at showdown),
#             used qualitatively for human-style patterns.
#   all       all of the above.
#
# Environment:
#   ACPC_FILES    number of ACPC matches to fetch (default 40)
#   HANDHQ_FILES  number of HandHQ files to fetch (default 50)

set -euo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")"

ACPC_FILES="${ACPC_FILES:-40}"
HANDHQ_FILES="${HANDHQ_FILES:-50}"

command -v python3 >/dev/null || { echo "error: python3 is required" >&2; exit 1; }

fetch() { python3 fetch_phh.py get "$@" --out raw; }

fetch_acpc()     { fetch "*acpc_2011_2p_limit*.phhs" --max-files "$ACPC_FILES"; }
fetch_pluribus() { fetch "data/pluribus/*"; }
fetch_handhq()   { fetch "data/handhq/PS-2009-07-01_2009-07-23_200NLH_OBFU/*" --max-files "$HANDHQ_FILES"; }

case "${1:-acpc}" in
  acpc)     fetch_acpc ;;
  pluribus) fetch_pluribus ;;
  handhq)   fetch_handhq ;;
  all)      fetch_acpc; fetch_pluribus; fetch_handhq ;;
  -h|--help) sed -n '2,24p' "$SELF" | sed 's/^# \{0,1\}//' ;;
  *) echo "unknown dataset: $1 (try --help)" >&2; exit 2 ;;
esac
