#!/usr/bin/env bash
# Regenerate exports/index.html from the HTML files actually present in exports/.
# Reflects reality: a notebook that wasn't exported simply isn't listed.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=exports

{
  echo '<!doctype html><html><head><meta charset="utf-8">'
  echo '<title>Applied GenAI — Notebooks</title>'
  echo '<style>body{font:16px system-ui,sans-serif;max-width:720px;margin:3rem auto;padding:0 1rem}'
  echo 'h1{font-size:1.4rem}a{display:block;padding:.5rem 0;text-decoration:none;color:#2563eb}'
  echo 'a:hover{text-decoration:underline}</style></head><body>'
  echo '<h1>Applied GenAI — Practical Notebooks</h1>'
  for f in $(ls -1 "$OUT"/*.html 2>/dev/null | sort -V); do
    base="$(basename "${f%.html}")"
    [ "$base" = "index" ] && continue
    label="$(echo "$base" | sed 's/_/ /g')"
    echo "<a href=\"./$base.html\">$label</a>"
  done
  echo '</body></html>'
} > "$OUT/index.html"

echo "index.html written with:"
grep -oE 'href="\./[^"]+"' "$OUT/index.html" | sed 's/href="\.\///;s/"//'
