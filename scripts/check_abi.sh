#!/usr/bin/env bash
# Verify that libdbs exports exactly the symbols listed in the ABI manifest,
# and nothing that an earlier release exported has disappeared.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="${1:-$ROOT/build/libdbs.so}"
EXPECTED="${2:-$ROOT/abi/libdbs.symbols}"
BASELINE="${3:-$ROOT/abi/baseline-v0.5.symbols}"

if [[ ! -f "$LIB" ]]; then
  echo "usage: $0 path/to/libdbs.so [expected-symbols] [baseline-symbols]" >&2
  exit 2
fi

current="$(mktemp)"
trap 'rm -f "$current"' EXIT
nm -D --defined-only "$LIB" | awk '{print $3}' | sed 's/@.*//' | grep '^dbs_' | sort -u > "$current"

status=0
if ! diff -u <(sort -u "$EXPECTED") "$current"; then
  echo "error: exported symbols differ from $EXPECTED" >&2
  status=1
fi
while IFS= read -r sym; do
  [[ -z "$sym" ]] && continue
  if ! grep -qx "$sym" "$current"; then
    echo "error: symbol removed since the baseline release: $sym" >&2
    status=1
  fi
done < "$BASELINE"
[[ $status -eq 0 ]] && echo "ABI ok: $(wc -l < "$current") exported symbols match $(basename "$EXPECTED")"
exit $status
