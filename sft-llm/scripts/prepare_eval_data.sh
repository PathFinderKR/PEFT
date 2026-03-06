#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"

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
if [[ -d data/eval/mmlu/data/test && -d data/eval/mmlu/data/dev ]]; then
    echo "MMLU eval data already present; skipping."
else
    download_file "https://people.eecs.berkeley.edu/~hendrycks/data.tar" "data/downloads/mmlu_data.tar"
    mkdir -p data/downloads/mmlu_data
    tar -xvf data/downloads/mmlu_data.tar -C data/downloads/mmlu_data
    rm -rf data/eval/mmlu
    mv data/downloads/mmlu_data/data data/eval/mmlu
    rm -rf data/downloads/mmlu_data data/downloads/mmlu_data.tar
fi


# Big-Bench-Hard dataset
if [[ -d data/eval/bbh/bbh ]]; then
    echo "BBH eval data already present; skipping."
else
    download_file "https://github.com/suzgunmirac/BIG-Bench-Hard/archive/refs/heads/main.zip" "data/downloads/bbh_data.zip"
    mkdir -p data/downloads/bbh
    if command -v unzip >/dev/null 2>&1; then
        unzip data/downloads/bbh_data.zip -d data/downloads/bbh
    else
        "${PYTHON_BIN}" -m zipfile -e data/downloads/bbh_data.zip data/downloads/bbh
    fi
    rm -rf data/eval/bbh
    mv data/downloads/bbh/BIG-Bench-Hard-main/ data/eval/bbh
    rm -rf data/downloads/bbh data/downloads/bbh_data.zip
fi


# TyDiQA-GoldP dataset
mkdir -p data/eval/tydiqa
if [[ -s data/eval/tydiqa/tydiqa-goldp-v1.1-dev.json ]]; then
    echo "TyDiQA eval data already present; skipping."
else
    if ! "${PYTHON_BIN}" -c "import datasets" >/dev/null 2>&1; then
        echo "Missing Python package: datasets" >&2
        echo "Install it with: pip install datasets" >&2
        exit 1
    fi
    "${PYTHON_BIN}" scripts/export_tydiqa_from_hf.py --output_dir data/eval/tydiqa
fi


# GSM dataset
mkdir -p data/eval/gsm
if [[ -s data/eval/gsm/test.jsonl ]]; then
    echo "GSM eval data already present; skipping."
else
    download_file "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl" "data/eval/gsm/test.jsonl"
fi


# Codex HumanEval
mkdir -p data/eval/codex_humaneval
if [[ -s data/eval/codex_humaneval/HumanEval.jsonl.gz ]]; then
    echo "HumanEval data already present; skipping."
else
    download_file "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz" "data/eval/codex_humaneval/HumanEval.jsonl.gz"
fi
