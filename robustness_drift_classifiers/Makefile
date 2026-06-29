PYTHON  := python3
VENV    := .venv
PIP     := $(VENV)/bin/pip
JUPYTER := $(VENV)/bin/jupyter
KAGGLE  := $(VENV)/bin/kaggle

.PHONY: venv data notebook clean help

help:
	@echo "Targets:"
	@echo "  make venv         — create .venv and install dependencies"
	@echo "  make data         — download Kaggle datasets (Phishing, UNSW-NB15, NF-ToN-IoT)"
	@echo "  make data-hikari  — download HIKARI-2021 from Zenodo (no API key required)"
	@echo "  make notebook     — launch Jupyter from this directory"
	@echo "  make clean        — remove .venv (data files kept)"

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Venv ready. Activate with: source $(VENV)/bin/activate"

data: $(VENV)/bin/activate
	@test -f $(KAGGLE) || $(PIP) install kaggle -q
	@test -f ~/.kaggle/kaggle.json || { \
	    echo "Missing ~/.kaggle/kaggle.json — get your token at kaggle.com/settings"; \
	    exit 1; }
	$(KAGGLE) datasets download shashwatwork/phishing-dataset-for-machine-learning -p . --unzip
	$(KAGGLE) datasets download mrwellsdavid/unsw-nb15 -p . --unzip
	$(KAGGLE) datasets download dhoogla/nftoniot -p . --unzip
	@echo "Datasets downloaded."

data-hikari: $(VENV)/bin/activate
	curl -L --progress-bar -o ALLFLOWMETER_HIKARI2021.csv.zip \
	    "https://zenodo.org/records/5199540/files/ALLFLOWMETER_HIKARI2021.csv.zip?download=1"
	unzip -o ALLFLOWMETER_HIKARI2021.csv.zip
	rm ALLFLOWMETER_HIKARI2021.csv.zip
	@echo "HIKARI-2021 downloaded: ALLFLOWMETER_HIKARI2021.csv"

notebook: $(VENV)/bin/activate
	$(JUPYTER) notebook

clean:
	rm -rf $(VENV)
	@echo "Removed .venv (data files and results kept)."
