#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ACADEMIA="$(cd "${ROOT}/../.." && pwd)"
TECTONIC="${ACADEMIA}/Academia/tectonic"
cd "${ROOT}"
MAIN="icsee2026_poster"

if command -v latexmk >/dev/null 2>&1; then
	latexmk -pdf -interaction=nonstopmode -halt-on-error "${MAIN}.tex"
elif command -v pdflatex >/dev/null 2>&1; then
	pdflatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"
	pdflatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"
elif [[ -x "${TECTONIC}" ]]; then
	"${TECTONIC}" --reruns 2 "${MAIN}.tex"
else
	echo "No LaTeX engine found." >&2
	exit 1
fi

if command -v pdfinfo >/dev/null 2>&1; then
	echo "---"
	pdfinfo "${MAIN}.pdf" | grep -E 'Page size|Pages|File size'
fi
echo "Wrote ${ROOT}/${MAIN}.pdf"
