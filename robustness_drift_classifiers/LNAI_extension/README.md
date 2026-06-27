# LNAI Extension — LaTeX Source

## Setup

This folder uses the standard Springer LNCS/LNAI template (`llncs` class).

**On Overleaf (recommended):**
```
make zip
```
Then: Overleaf → New Project → Upload Project → select `overleaf_upload.zip`.

**Local compilation:**
```
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Files

| File | Purpose |
|------|---------|
| `main.tex` | Main LNAI paper |
| `difference_statement.tex` | Separate document for editors (submit with cover letter) |
| `refs.bib` | All references including self-citation and Zenodo DOI |
| `llncs.cls` | Springer LNCS document class (from official template package) |
| `splncs04.bst` | Springer BibTeX style (alphabetic sorting) |
| `Makefile` | `make zip` → builds `overleaf_upload.zip` for Overleaf upload |

## Figures

Figures are referenced as `../figures/*.png` (one level up in the repo).
On Overleaf, upload the `figures/` folder alongside this one.

## Page estimate

~18–20 pages in LNCS format — within the 30-page limit.
