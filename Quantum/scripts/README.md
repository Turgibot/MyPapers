# Plotting Scripts for Chapter 3: Historical Foundations

This directory contains Python scripts to generate publication-quality figures for Chapter 3 (Historical Foundations: From Classical to Quantum).

## Requirements

- Python 3.6+
- numpy
- matplotlib

Install dependencies:
```bash
pip install numpy matplotlib
```

## Scripts

### 1. `plot_coulomb_law.py`
**Output:** `images/chapter03/coulomb_law_field.png`

Visualizes Coulomb's law showing the electric field around a positive point charge. The figure shows:
- Radial electric field lines emanating from the charge
- Field strength indicated by color intensity
- Mathematical equation annotation

**Usage:**
```bash
python3 plot_coulomb_law.py
```

### 2. `plot_ampere_law.py`
**Output:** `images/chapter03/ampere_law_field.png`

Visualizes Ampère's law showing the magnetic field around a current-carrying wire. The figure shows:
- Circular magnetic field lines around the wire
- Current direction indicator (out of page)
- Field strength indicated by color intensity
- Mathematical equation annotation

**Usage:**
```bash
python3 plot_ampere_law.py
```

### 3. `plot_faraday_law.py`
**Output:** `images/chapter03/faraday_law_induction.png`

Visualizes Faraday's law of electromagnetic induction. The figure shows:
- Left panel: Changing magnetic field (into page, increasing) and induced electric field
- Right panel: Coil with changing magnetic flux and induced current
- Mathematical equations for both integral and differential forms

**Usage:**
```bash
python3 plot_faraday_law.py
```

### 4. `plot_maxwell_equations.py`
**Output:** `images/chapter03/maxwell_equations_complete.png`

Visualizes all four Maxwell's equations in a 2×2 grid layout:
- Top-left: Gauss's law for electricity (electric field from charges)
- Top-right: Gauss's law for magnetism (magnetic field lines form closed loops)
- Bottom-left: Faraday's law (changing B creates E)
- Bottom-right: Ampère-Maxwell law (changing E creates B)

**Usage:**
```bash
python3 plot_maxwell_equations.py
```

### 5. `plot_field_operators.py`
**Output:** `images/chapter03/field_operators.png`

Visualizes vector field operators (divergence and curl) in a 2×2 grid:
- Top-left: Positive divergence (source field)
- Top-right: Negative divergence (sink field)
- Bottom-left: Non-zero curl (rotational field)
- Bottom-right: Zero divergence and curl (uniform field)

**Usage:**
```bash
python3 plot_field_operators.py
```

## Generating All Figures

To generate all figures at once:
```bash
cd scripts
python3 plot_coulomb_law.py
python3 plot_ampere_law.py
python3 plot_faraday_law.py
python3 plot_maxwell_equations.py
python3 plot_field_operators.py
```

Or use a loop:
```bash
for script in plot_*.py; do python3 "$script"; done
```

## Figure Specifications

All figures are generated with:
- **Resolution:** 300 DPI (publication quality)
- **Format:** PNG
- **Style:** Serif fonts, professional appearance
- **Backend:** Agg (non-interactive, suitable for headless servers)

## Notes

- All scripts use the `Agg` backend, so they can run without a display (suitable for servers)
- Mathematical notation uses simplified LaTeX (without `\boldsymbol`) for compatibility
- Colors are chosen for clarity and print compatibility
- All figures are saved to `../images/chapter03/` relative to the scripts directory

