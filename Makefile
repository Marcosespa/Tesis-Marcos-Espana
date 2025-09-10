PY?=python3
VENV?=.venv
PIP=$(VENV)/bin/pip
PYBIN=$(VENV)/bin/python

venv:
	python3 -m venv $(VENV) && $(PYBIN) -m pip install -r requeriments.txt || true

rag-extract:
	./scripts/rag/10_extract.sh data/raw

rag-chunk:
	./scripts/rag/20_chunk.sh

rag-index:
	./scripts/rag/30_index.sh

ft-prepare:
	./scripts/ft/10_prepare.sh

ft-train:
	./scripts/ft/20_train.sh

ft-eval:
	./scripts/ft/30_eval.sh
