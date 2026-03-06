#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="data/raw_train/gpt4_alpaca"
RAW_FILE="${RAW_DIR}/alpaca_gpt4_data.json"

mkdir -p "${RAW_DIR}"

echo "Downloading the gpt4-llm dataset..."
wget -P "${RAW_DIR}/" https://raw.githubusercontent.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/main/data/alpaca_gpt4_data.json

if [[ ! -f "${RAW_FILE}" ]]; then
  echo "ERROR: Expected raw dataset is missing: ${RAW_FILE}" >&2
  exit 1
fi

echo "Processing datasets..."
python scripts/reformat_datasets.py --raw_data_dir data/raw_train/ --output_dir data/processed/ --dataset gpt4_alpaca
