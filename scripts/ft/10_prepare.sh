#!/usr/bin/env bash
set -euo pipefail
python - <<"PY"
print("Preparing FT datasets -> data/ft_datasets/{train,val}.jsonl (stub)")
PY
