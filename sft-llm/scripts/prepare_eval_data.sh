#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/downloads
mkdir -p data/eval

download_file() {
    local url="$1"
    local out="$2"
    local tmp="${out}.tmp"
    rm -f "${tmp}"
    curl -fL --retry 3 --retry-delay 2 -o "${tmp}" "${url}"
    if [[ ! -s "${tmp}" ]]; then
        echo "Downloaded file is empty: ${url}" >&2
        exit 1
    fi
    mv "${tmp}" "${out}"
}

# MMLU dataset
download_file "https://people.eecs.berkeley.edu/~hendrycks/data.tar" "data/downloads/mmlu_data.tar"
mkdir -p data/downloads/mmlu_data
tar -xvf data/downloads/mmlu_data.tar -C data/downloads/mmlu_data
mv data/downloads/mmlu_data/data data/eval/mmlu && rm -r data/downloads/mmlu_data data/downloads/mmlu_data.tar


# Big-Bench-Hard dataset
download_file "https://github.com/suzgunmirac/BIG-Bench-Hard/archive/refs/heads/main.zip" "data/downloads/bbh_data.zip"
mkdir -p data/downloads/bbh
unzip data/downloads/bbh_data.zip -d data/downloads/bbh
mv data/downloads/bbh/BIG-Bench-Hard-main/ data/eval/bbh && rm -r data/downloads/bbh data/downloads/bbh_data.zip


# TyDiQA-GoldP dataset
mkdir -p data/eval/tydiqa
python scripts/export_tydiqa_from_hf.py --output_dir data/eval/tydiqa


# GSM dataset
mkdir -p data/eval/gsm
download_file "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl" "data/eval/gsm/test.jsonl"


# Codex HumanEval
mkdir -p data/eval/codex_humaneval
download_file "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz" "data/eval/codex_humaneval/HumanEval.jsonl.gz"
