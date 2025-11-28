#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Creating branch instr/himanshu (if not exists)"
git checkout -B instr/himanshu || true

# copy .env example if not present
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Copied .env.example -> .env. Please edit .env to add WANDB_API_KEY if you want wandb."
fi

# create venv and activate
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools wheel
# install editable lab-logger
pip install -e libs/lab_logger

# install toy example deps
pip install -r examples/train_toy/requirements.txt || true

# create DB
python create_db.py

# run the toy example once to create a sample manifest row
python examples/train_toy/run.py

echo "Bootstrap complete. If you want wandb visualisation set WANDB_API_KEY in your environment and re-run the example."
