FIG_DIR := ../figures
ZIP     := Beyond_Gradient_Attacks_LNAI_2026.zip
TMP     := _build_tmp

# Figures referenced in \includegraphics lines (auto-extracted)
FIGURES := $(shell grep -oh '[a-zA-Z0-9_-]*\.png' main.tex | grep -v '^fig1' | sort -u | sed 's|^|$(FIG_DIR)/|')

DRAFT_ZIP := Beyond_Gradient_Attacks_LNAI_2026_draft.zip
PDF       := main.pdf
PDF_FINAL := beyond_gradient_attacks_rajhans.pdf

.PHONY: zip zip-draft pdf clean

pdf:
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex
	@cp main.pdf $(PDF_FINAL)
	@echo "Done → $(PDF_FINAL)"

zip:
	@rm -rf $(TMP) $(ZIP)
	@mkdir -p $(TMP)/figures
	@cp main.tex refs.bib llncs.cls splncs04.bst $(TMP)/
	@cp $(FIGURES) $(TMP)/figures/
	@cd $(TMP) && zip -qr ../$(ZIP) .
	@rm -rf $(TMP)
	@echo "Created $(ZIP)  —  upload source zip to Springer/ICAART Primoris"

# Draft zip: figures shown as placeholder boxes, faster local compile
zip-draft:
	@rm -rf $(TMP) $(DRAFT_ZIP)
	@mkdir -p $(TMP)
	@sed 's/\\usepackage{graphicx}/\\usepackage[draft]{graphicx}/' main.tex > $(TMP)/main.tex
	@cp refs.bib llncs.cls splncs04.bst $(TMP)/
	@cd $(TMP) && zip -qr ../$(DRAFT_ZIP) .
	@rm -rf $(TMP)
	@echo "Created $(DRAFT_ZIP)  —  draft mode: figures shown as boxes, fast compile"

clean:
	rm -f $(ZIP) $(DRAFT_ZIP)
	rm -rf $(TMP)
