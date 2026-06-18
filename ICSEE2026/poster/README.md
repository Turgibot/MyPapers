# ICSEE 2026 Poster

Portrait poster (**90 cm × 125 cm**) for the accepted ICSEE paper.

## Recommended workflow (conference template)

1. Open **`Poster-Template-ASEMD2023-2.ppt`** in PowerPoint or LibreOffice Impress.
2. Follow **`ASEMD_TEMPLATE_FILLIN.md`** — copy-paste text and insert figures from the paths listed there.
3. Keep template fonts/colors (Arial, light-yellow background, dark-green text).
4. Replace ASEMD2023 header text with the ICSEE line from the fill-in guide ([icsee2026.org](https://www.icsee2026.org/): Jerusalem, **10--11 June 2026**).
5. Export to PDF at **90 × 125 cm** for printing.

## Optional LaTeX version (ASEMD-styled PDF)

A PDF matching the template typography and section layout:

```bash
./compile.sh
```

Output: `icsee2026_poster.pdf` (Abstract, II. Principle, III. Theoretical Modelling, V. Results, VI. Conclusion).

Uses `tectonic` from `Academia/tectonic` if no system LaTeX is installed.

## Sources

- Paper: `../icsee2026_eta_gnn_testing.tex`
- System report: `../../22997/Project_Report/`
- Figures: `../nodes.png`, `../mae_vs_time_plot_inverted.png`, `../../Academia/images/`
