FIG_DIR := ../figures
ZIP     := Beyond_Gradient_Attacks_LNAI_2026.zip
TMP     := _overleaf_tmp

# Figures referenced in \includegraphics lines (auto-extracted)
FIGURES := $(shell grep -oh '[a-zA-Z0-9_-]*\.png' main.tex | grep -v '^fig1' | sort -u | sed 's|^|$(FIG_DIR)/|')

.PHONY: zip clean

zip:
	@rm -rf $(TMP) $(ZIP)
	@mkdir -p $(TMP)/figures
	@cp main.tex refs.bib difference_statement.tex llncs.cls splncs04.bst $(TMP)/
	@cp $(FIGURES) $(TMP)/figures/
	@cd $(TMP) && zip -qr ../$(ZIP) .
	@rm -rf $(TMP)
	@echo "Created $(ZIP)  —  upload to Overleaf via New Project → Upload Project"

clean:
	rm -f $(ZIP)
	rm -rf $(TMP)
