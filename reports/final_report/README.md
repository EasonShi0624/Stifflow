# Final LaTeX Report

This directory contains the source for the project final report:

`Stiffness-Modulated Harmonic Priors for Molecular Conformation Flow Matching: A Full-Database Evaluation`

## Contents

- `main.tex`: report source.
- `references.bib`: minimal bibliography based only on references already named in project docs.
- `figures/`: copied final closure figures.
- `tables/`: concise LaTeX tables generated from final summaries.
- `Makefile`: build helper.
- `compile_report.sh`: shell build helper.

## Compile

If LaTeX is available:

```bash
make
```

or:

```bash
./compile_report.sh
```

The expected output is:

```text
final_report.pdf
```

In the current environment, `latexmk`, `pdflatex`, `xelatex`, and `lualatex` were not found on `PATH`, so the source is ready but PDF compilation may need to run elsewhere.

## Claim Boundary

The report uses the final closure interpretation:

- `lambda=5.0` is validation/RMSD and AMR-P precision-favorable.
- `lambda=0.025` is best tested hybrid recall/balanced.
- `harmonic_prior` remains the AMR-R recall anchor in the harmonic-vs-hybrid lambda comparison.
- No tested hybrid lambda dominates harmonic on both precision and recall.
- Physical benchmark validation was not completed and is not claimed.
