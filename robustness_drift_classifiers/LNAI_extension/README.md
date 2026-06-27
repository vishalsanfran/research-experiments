# LNAI Extension — LaTeX Source

## Setup

This folder uses the standard Springer LNCS/LNAI template (`llncs` class).

**On Overleaf (recommended):**
1. New Project → Upload Project → zip this folder
2. Overleaf includes `llncs.cls` automatically via the LNCS template

**Local compilation:**
1. Download `llncs.cls` from https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines
2. Place it in this folder
3. `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`

## Files

| File | Purpose |
|------|---------|
| `main.tex` | Main LNAI paper |
| `difference_statement.tex` | Separate document for editors (submit with cover letter) |
| `refs.bib` | All references including self-citation and Zenodo DOI |

## Figures

Figures are referenced as `../figures/*.png` (one level up in the repo).
On Overleaf, upload the `figures/` folder alongside this one.

## Page estimate

~22 pages in LNCS format — within the 30-page limit.
