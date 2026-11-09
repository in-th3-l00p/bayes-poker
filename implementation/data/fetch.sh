#!/usr/bin/env bash
# Fetch the hand histories used in the implementation.
#
# Sources: the PHH dataset (Zenodo, DOI 10.5281/zenodo.10796885, MIT licence),
# a single ~20 GB zip whose index fetch_phh.py reads over HTTP range requests to
# extract only the matches we need; and the IRC Poker Database (one archive).
# Everything lands in data/raw/.
#
# Usage:
#   data/fetch.sh [acpc|irc|all]     (default: acpc)
#
#   acpc      ACPC 2011 heads-up limit Hold'em matches (3,000 hands, ~1.1 MB each).
#             Every hole card is revealed: main dataset (E3). The first 40 are the
#             paper's Table 2 sample. Writes raw/acpc_manifest.txt (archive order).
#   irc       IRC Poker Database (University of Alberta), 1995-2001 human
#             limit games, ~1 GB download. Heads-up hands are used to test
#             style changes in human play (cards only at showdown).
#   all       all of the above.
#
# Environment:
#   ACPC_FILES    number of ACPC matches to fetch (default 200)
#
# Re-running is safe: files already present with the right size are skipped.

set -euo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")"

ACPC_FILES="${ACPC_FILES:-200}"

command -v python3 >/dev/null || { echo "error: python3 is required" >&2; exit 1; }

fetch() { python3 fetch_phh.py get "$@" --out raw; }

fetch_acpc() {
  fetch "*acpc_2011_2p_limit*.phhs" --max-files "$ACPC_FILES"
  python3 fetch_phh.py list --grep "*acpc_2011_2p_limit*.phhs" --limit "$ACPC_FILES" 2>/dev/null \
    | awk '{print $2}' > raw/acpc_manifest.txt
}
fetch_irc() {
  mkdir -p raw/irc
  curl -L -C - --fail -o raw/irc/IRCdata.tgz https://poker.cs.ualberta.ca/IRC/IRCdata.tgz
}

case "${1:-acpc}" in
  acpc)     fetch_acpc ;;
  irc)      fetch_irc ;;
  all)      fetch_acpc; fetch_irc ;;
  -h|--help) sed -n '2,/^$/p' "$SELF" | sed 's/^# \{0,1\}//' ;;
  *) echo "unknown dataset: $1 (try --help)" >&2; exit 2 ;;
esac
