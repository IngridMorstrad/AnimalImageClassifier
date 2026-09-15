#!/usr/bin/env bash
# Probe candidate weight/dataset URLs and print the real HTTP status of each.
# Usage: scripts/probe_urls.sh <urls-file>
# Each line of <urls-file> is "<label>|<url>". A ranged GET of the first KiB is used so the
# probe is cheap but still exercises the real asset (not just a redirect or a landing page).
set -uo pipefail

urls_file="${1:?usage: probe_urls.sh <urls-file>}"

while IFS='|' read -r label url; do
  [ -z "${label:-}" ] && continue
  case "$label" in \#*) continue ;; esac
  read -r code size ctype <<<"$(
    curl -sSL --max-time 25 -r 0-1023 \
      -o /dev/null \
      -w '%{http_code} %{size_download} %{content_type}' \
      "$url" 2>/dev/null
  )"
  printf '%-38s %-6s %-8s %-28s %s\n' "$label" "${code:-000}" "${size:-0}" "${ctype:--}" "$url"
done <"$urls_file"
