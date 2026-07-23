#!/usr/bin/env bash
# Render a Markdown deck to an Open-Athena-styled reveal.js HTML slideshow.
#
# Usage:
#   bash decks/render.sh decks/kelp.md      # explicit path
#   bash decks/render.sh kelp               # bare name (resolved under decks/)
#   bash decks/render.sh                    # renders every *.md in decks/
#
# Output: <deck>.html next to the source. Open it in a browser (needs internet
# for the reveal.js CDN + Google Fonts; the OA theme CSS is embedded).
#
# Deck conventions (see decks/kelp.md):
#   - Level-1 headers (#)  -> section-divider slides. Make them dark with:
#       # My Section {background-color="#1F1E1B" .oa-dark}
#   - Level-2 headers (##) -> content slides (beige).
#   - Title slide comes from the YAML metadata block at the top.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
THEME="$HERE/theme/oa.css"
REVEAL_URL="${REVEAL_URL:-https://cdn.jsdelivr.net/npm/reveal.js@5.1.0}"

render_one() {
    local in="$1"
    local out="${in%.md}.html"
    local srcdir
    srcdir="$(cd "$(dirname "$in")" && pwd)"
    pandoc "$in" \
        --from markdown \
        --to revealjs \
        --standalone \
        --embed-resources \
        --resource-path="$srcdir:$HERE" \
        --slide-level=2 \
        --css "$THEME" \
        --variable revealjs-url="$REVEAL_URL" \
        --variable theme=white \
        --variable width=1280 \
        --variable height=720 \
        --variable margin=0.045 \
        --variable transition=fade \
        --variable slideNumber=true \
        --variable hash=true \
        --output "$out"
    echo "wrote $out"
}

# No arg: render all decks. Arg: a path or a bare deck name.
if [ "$#" -eq 0 ]; then
    shopt -s nullglob
    decks=("$HERE"/*.md)
    [ "${#decks[@]}" -gt 0 ] || { echo "no *.md decks in $HERE" >&2; exit 1; }
    for d in "${decks[@]}"; do render_one "$d"; done
else
    in="$1"
    [ -f "$in" ] || in="$HERE/${1%.md}.md"
    [ -f "$in" ] || { echo "no such deck: $1" >&2; exit 1; }
    render_one "$in"
fi
