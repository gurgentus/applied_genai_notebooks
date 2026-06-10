#!/usr/bin/env bash
# Export every notebook to a self-contained static HTML file (no kernel at runtime).
# Each export RUNS the notebook, so heavy training notebooks take longer.
# Usage: scripts/export_all.sh [per-notebook-timeout-seconds]
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=exports
TIMEOUT="${1:-1800}"   # default 30 min per notebook
mkdir -p "$OUT"
LOG="$OUT/_export_all.log"
: > "$LOG"

# `timeout` may be GNU coreutils (gtimeout on macOS via brew) or absent.
if command -v gtimeout >/dev/null 2>&1; then TO=gtimeout
elif command -v timeout  >/dev/null 2>&1; then TO=timeout
else TO=""; fi

declare -a OK=() FAIL=()

for f in notebooks/*.py; do
  base="$(basename "${f%.py}")"
  out="$OUT/$base.html"
  echo "=== exporting $base (timeout ${TIMEOUT}s) ===" | tee -a "$LOG"
  if [ -n "$TO" ]; then
    $TO "$TIMEOUT" uv run marimo export html "$f" -o "$out" --force >>"$LOG" 2>&1
  else
    uv run marimo export html "$f" -o "$out" --force >>"$LOG" 2>&1
  fi
  rc=$?
  if [ $rc -eq 0 ] && [ -s "$out" ]; then
    echo "OK    $base" | tee -a "$LOG"; OK+=("$base")
  else
    echo "FAIL  $base (exit $rc)" | tee -a "$LOG"; FAIL+=("$base")
    rm -f "$out"   # don't leave a half-written file that would be served
  fi
done

# Build an index page linking every successful export.
{
  echo '<!doctype html><html><head><meta charset="utf-8">'
  echo '<title>Applied GenAI — Notebooks</title>'
  echo '<style>body{font:16px system-ui,sans-serif;max-width:720px;margin:3rem auto;padding:0 1rem}'
  echo 'h1{font-size:1.4rem}a{display:block;padding:.5rem 0;text-decoration:none;color:#2563eb}'
  echo 'a:hover{text-decoration:underline}</style></head><body>'
  echo '<h1>Applied GenAI — Practical Notebooks</h1>'
  for base in "${OK[@]}"; do
    label="$(echo "$base" | sed 's/_/ /g')"
    echo "<a href=\"./$base.html\">$label</a>"
  done
  echo '</body></html>'
} > "$OUT/index.html"

echo ""                                  | tee -a "$LOG"
echo "EXPORTED OK (${#OK[@]}): ${OK[*]}" | tee -a "$LOG"
echo "FAILED     (${#FAIL[@]}): ${FAIL[*]}" | tee -a "$LOG"
[ ${#FAIL[@]} -eq 0 ]
