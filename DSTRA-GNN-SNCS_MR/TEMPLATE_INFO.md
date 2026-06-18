# Springer Nature LaTeX Template Information

## Template Successfully Downloaded ✅

**Version**: December 2024  
**Source**: Official Springer Nature LaTeX Author Support  
**Size**: 880 KB (ZIP file)

## Template Structure

### Main Files
- `sn-jnl.cls` - Springer Nature LaTeX class file (copied to root directory)
- `sn-article-template/ssn-article.tex` - Complete example template
- `sn-article-template/user-manual.pdf` - Detailed user manual

### Bibliography Styles Available
Located in `sn-article-template/bst/`:
- `sn-basic.bst` - Basic Springer Nature Reference Style
- `sn-nature.bst` - Style for Nature Portfolio journals
- `sn-mathphys-ay.bst` - Math and Physical Sciences Author Year
- `sn-mathphys-num.bst` - Math and Physical Sciences Numbered
- `sn-aps.bst` - American Physical Society (APS) Reference Style
- `sn-vancouver-num.bst` - Vancouver Numbered
- `sn-vancouver-ay.bst` - Vancouver Author Year
- `sn-apacite.bst` - APA Reference Style
- `sn-chicago.bst` - Chicago-based Humanities Reference Style

## Key Template Features

### Document Class Options
The template supports multiple reference styles. For SN Computer Science, you can use:
```latex
\documentclass[pdflatex,sn-basic]{sn-jnl}  % Basic Springer Nature style
\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}  % Math/Physical Sciences Numbered
```

### Structured Abstract Support
The template includes commented-out example for structured abstract (lines 128-134):
```latex
\abstract{\textbf{Purpose:} ...
\textbf{Methods:} ...
\textbf{Results:} ...
\textbf{Conclusion:} ...}
```

### Required Sections Included
The template includes all required sections:
- **Declarations** section (line 551) with:
  - Funding
  - Conflict of interest/Competing interests
  - Ethics approval and consent to participate
  - Consent for publication
  - Data availability
  - Materials availability
  - Code availability
  - **Author contribution** (line 563)

### Author Format
The template uses a specific author format:
```latex
\author*[1,2]{\fnm{First} \sur{Author}}\email{email@example.com}
\affil*[1]{\orgdiv{Department}, \orgname{Organization}, 
           \orgaddress{\street{Street}, \city{City}, 
           \postcode{100190}, \state{State}, \country{Country}}}
```

### Appendices Support
The template includes `\begin{appendices}...\end{appendices}` environment for the required appendix explaining 30% new material.

## Important Notes

1. **Single File Submission**: The template documentation states:
   > "Please do not use \input{...} to include other tex files. Submit your LaTeX manuscript as one .tex document."

2. **Figures**: Should be attached separately, not embedded in the .tex file itself.

3. **Compilation**: Use `pdflatex` for compilation (default option in template).

4. **Bibliography**: Use `\bibliography{filename}` command (not `\bibliographystyle`).

## Next Steps

1. Review `sn-article-template/user-manual.pdf` for detailed instructions
2. Use `sn-article-template/sn-article.tex` as a reference
3. Convert your DSTRA-GNN paper to use `sn-jnl.cls` class
4. Ensure structured abstract format (Purpose/Methods/Results/Conclusion)
5. Add all required Declarations sections
6. Include appendix explaining 30% new material

