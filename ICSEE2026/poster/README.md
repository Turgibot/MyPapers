# ICSEE 2026 Poster

Portrait poster (**90 cm × 125 cm**) for the accepted paper on the integrated Traffic-DSTG-Gen + TrafficLab platform.

## Build

```bash
./compile.sh
```

Output: `icsee2026_poster.pdf`

Uses `tectonic` from `Academia/tectonic` if no system LaTeX is installed.

## Sources

- Paper: `../icsee2026_eta_gnn_testing.tex`
- System details & figures: `../../22997/Project_Report/` and `../../Academia/images/`

## Layout (3 rows × 3 columns)

1. **Motivation** | **End-to-end workflow** | **Platform summary**
2. **Graph schema** | **Desktop UI** | **Web UI**
3. **Evaluation results** | **Example journey** | **Conclusions + QR codes**

Edit `icsee2026_poster.tex` to adjust text, swap figures, or add institutional logos.
