#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B}"
TOKENIZER_NAME="${TOKENIZER_NAME:-meta-llama/Llama-3.2-1B}"
TRAIN_FILE="${TRAIN_FILE:-data/processed/gpt4_alpaca/gpt4_alpaca_data.jsonl}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20}"
FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-16}"
ADAPTER_BATCH_SIZE="${ADAPTER_BATCH_SIZE:-32}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
NUM_MACHINES="${NUM_MACHINES:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-auto}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-127.0.0.1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/pathfinder/projects/PEFT/results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODEL_KEY="${MODEL_KEY:-$(echo "${MODEL_NAME}" | tr '/:' '__')}"
RUN_ROOT="${OUTPUT_ROOT}/${MODEL_KEY}/${RUN_ID}"
export HF_HOME="${HF_HOME:-$(pwd)/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export PYTHONPATH="$(pwd)/peft/src${PYTHONPATH:+:${PYTHONPATH}}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HOME}/.cache/huggingface/token}"
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" && -f "${HF_TOKEN_FILE}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$(cat "${HF_TOKEN_FILE}")"
fi
if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi

mkdir -p "${RUN_ROOT}"/{checks,eval,hessian,training,tmp}
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"
echo "Run root: ${RUN_ROOT}"

check_python_module() {
  local module="$1"
  "${PYTHON_BIN}" -c "import ${module}" >/dev/null 2>&1
}

check_model_access() {
  if [[ -d "${MODEL_NAME}" || -f "${MODEL_NAME}" ]]; then
    return 0
  fi
  local access_log="${RUN_ROOT}/checks/model_access.log"
  set +e
  "${PYTHON_BIN}" - <<'PY' > "${access_log}" 2>&1
import os
import sys

from transformers import AutoConfig

model_name = os.environ["MODEL_NAME"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
try:
    AutoConfig.from_pretrained(model_name, token=token)
except Exception as exc:
    print(f"MODEL_ACCESS_CHECK_FAILED: {exc}")
    raise
print("MODEL_ACCESS_CHECK_OK")
PY
  local rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    echo "ERROR: Cannot load model config for MODEL_NAME=${MODEL_NAME}." >&2
    if command -v rg >/dev/null 2>&1; then
      if rg -qi "gated repo|401|unauthorized|access.*restricted|please log in" "${access_log}"; then
        echo "Hint: ${MODEL_NAME} is gated or requires authentication." >&2
        echo "Request access on Hugging Face and set HF_TOKEN (or run huggingface-cli login)." >&2
      fi
    fi
    echo "Details: ${access_log}" >&2
    return 1
  fi
}

if [[ ! "${FULL_BATCH_SIZE}" =~ ^[0-9]+$ || "${FULL_BATCH_SIZE}" -lt 1 ]]; then
  echo "ERROR: FULL_BATCH_SIZE must be a positive integer. got=${FULL_BATCH_SIZE}" >&2
  exit 1
fi
if [[ ! "${ADAPTER_BATCH_SIZE}" =~ ^[0-9]+$ || "${ADAPTER_BATCH_SIZE}" -lt 1 ]]; then
  echo "ERROR: ADAPTER_BATCH_SIZE must be a positive integer. got=${ADAPTER_BATCH_SIZE}" >&2
  exit 1
fi
if [[ ! "${GLOBAL_BATCH_SIZE}" =~ ^[0-9]+$ || "${GLOBAL_BATCH_SIZE}" -lt 1 ]]; then
  echo "ERROR: GLOBAL_BATCH_SIZE must be a positive integer. got=${GLOBAL_BATCH_SIZE}" >&2
  exit 1
fi

VISIBLE_CUDA_DEVICES="$("${PYTHON_BIN}" - <<'PY'
try:
    import torch
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
except Exception:
    n = 0
print(n)
PY
)"
if [[ ! "${VISIBLE_CUDA_DEVICES}" =~ ^[0-9]+$ ]]; then
  VISIBLE_CUDA_DEVICES=0
fi

if [[ "${NUM_PROCESSES}" == "auto" ]]; then
  NUM_PROCESSES="${VISIBLE_CUDA_DEVICES}"
fi
if [[ ! "${NUM_PROCESSES}" =~ ^[0-9]+$ || "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "ERROR: NUM_PROCESSES must be a positive integer or 'auto'. got=${NUM_PROCESSES}" >&2
  exit 1
fi
if [[ "${VISIBLE_CUDA_DEVICES}" -lt "${NUM_PROCESSES}" ]]; then
  echo "ERROR: Need at least ${NUM_PROCESSES} visible CUDA devices, but found ${VISIBLE_CUDA_DEVICES}." >&2
  exit 1
fi

missing_modules=()
for mod in accelerate transformers datasets peft; do
  if ! check_python_module "${mod}"; then
    missing_modules+=("${mod}")
  fi
done
if [[ ${#missing_modules[@]} -gt 0 ]]; then
  echo "ERROR: Missing Python modules in ${PYTHON_BIN}: ${missing_modules[*]}" >&2
  echo "Install dependencies first (example):" >&2
  echo "  ${PYTHON_BIN} -m pip install -r requirements.txt" >&2
  exit 1
fi

check_model_access

accelerate_cmd=(
  "${PYTHON_BIN}" -m accelerate.commands.launch
  --num_machines "${NUM_MACHINES}"
  --num_processes "${NUM_PROCESSES}"
  --machine_rank "${MACHINE_RANK}"
  --main_process_ip "${MAIN_PROCESS_IP}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  finetune/finetune.py
)
echo "Accelerate launch: num_machines=${NUM_MACHINES} num_processes=${NUM_PROCESSES} machine_rank=${MACHINE_RANK} main_process_ip=${MAIN_PROCESS_IP} main_process_port=${MAIN_PROCESS_PORT}"
echo "Batch config target: global=${GLOBAL_BATCH_SIZE} | full(per_device=${FULL_BATCH_SIZE}) | lora/sft(per_device=${ADAPTER_BATCH_SIZE})"

base_train_args=(
  --model_name_or_path "${MODEL_NAME}"
  --tokenizer_name "${TOKENIZER_NAME}"
  --train_file "${TRAIN_FILE}"
  --max_seq_length "${MAX_SEQ_LENGTH}"
  --preprocessing_num_workers 1
  --learning_rate 1e-5
  --lr_scheduler_type linear
  --warmup_ratio 0.0
  --weight_decay 0.0
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --logging_steps 1
  --gradient_checkpointing
  --compute_hessian_effective_rank
  --hessian_top_n 4
  --hessian_num_batches 1
  --hessian_max_iter 10
  --hessian_tol 1e-2
  --with_tracking
  --report_to wandb
  --tracker_project_name peft-hessian
)

method_args() {
  local method="$1"
  if [[ "${method}" == "full" ]]; then
    echo ""
  elif [[ "${method}" == "lora" ]]; then
    echo "--peft_tuner lora --lora_rank 8 --lora_alpha 16 --lora_dropout 0.1"
  elif [[ "${method}" == "sft" ]]; then
    echo "--peft_tuner sft --sft_num_tunable_weights 262144 --sft_selection rigl"
  else
    echo "unknown method: ${method}" >&2
    exit 1
  fi
}

PROBE_FATAL=0
PROBE_FATAL_REASON=""
FIND_MAX_BS_RESULT=0

floor_pow2() {
  local n="$1"
  if [[ "${n}" -lt 1 ]]; then
    echo 0
    return
  fi
  local p=1
  while [[ $((p * 2)) -le "${n}" ]]; do
    p=$((p * 2))
  done
  echo "${p}"
}

probe_one() {
  local method="$1"
  local bs="$2"
  local log_path="${RUN_ROOT}/checks/probe_${method}_bs${bs}.log"
  local out_dir="${RUN_ROOT}/tmp/probe_${method}_bs${bs}"
  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"
  local extra
  extra="$(method_args "${method}")"

  set +e
  "${accelerate_cmd[@]}" \
    --model_name_or_path "${MODEL_NAME}" \
    --tokenizer_name "${TOKENIZER_NAME}" \
    --train_file "${TRAIN_FILE}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --preprocessing_num_workers 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.0 \
    --weight_decay 0.0 \
    --gradient_checkpointing \
    --output_dir "${out_dir}" \
    --per_device_train_batch_size "${bs}" \
    --max_train_steps 1 \
    --wandb_run_name "probe-${method}-bs${bs}" \
    ${extra} \
    > "${log_path}" 2>&1
  local rc=$?
  set -e

  if [[ ${rc} -ne 0 ]]; then
    if command -v rg >/dev/null 2>&1; then
      if rg -qi "gated repo|401|unauthorized|access.*restricted|please log in|hf_raise_for_status|GatedRepoError" "${log_path}"; then
        PROBE_FATAL=1
        PROBE_FATAL_REASON="Model access/auth failure for method=${method}. Check Hugging Face token/permissions. See ${log_path}"
        return 2
      fi
    fi
    if command -v rg >/dev/null 2>&1; then
      if rg -qi "out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|RuntimeError: CUDA error" "${log_path}"; then
        return 1
      fi
    else
      if grep -Eqi "out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|RuntimeError: CUDA error" "${log_path}"; then
        return 1
      fi
    fi
    PROBE_FATAL=1
    PROBE_FATAL_REASON="Non-OOM probe failure for method=${method}, bs=${bs}. See ${log_path}"
    return 2
  fi
  if command -v rg >/dev/null 2>&1; then
    if rg -qi "out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|RuntimeError: CUDA error" "${log_path}"; then
      return 1
    fi
  else
    if grep -Eqi "out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|RuntimeError: CUDA error" "${log_path}"; then
      return 1
    fi
  fi
  return 0
}

find_max_bs() {
  local method="$1"
  local low=0
  local high=1
  local cap=32
  while [[ ${high} -le ${cap} ]]; do
    if probe_one "${method}" "${high}"; then
      low=${high}
      high=$((high * 2))
    else
      local rc=$?
      if [[ ${rc} -eq 2 ]]; then
        return 2
      fi
      break
    fi
  done
  if [[ ${low} -eq 0 ]]; then
    FIND_MAX_BS_RESULT=0
    return 0
  fi
  local left=$((low + 1))
  local right=$((high - 1))
  while [[ ${left} -le ${right} ]]; do
    local mid=$(((left + right) / 2))
    if probe_one "${method}" "${mid}"; then
      low=${mid}
      left=$((mid + 1))
    else
      local rc=$?
      if [[ ${rc} -eq 2 ]]; then
        return 2
      fi
      right=$((mid - 1))
    fi
  done
  FIND_MAX_BS_RESULT="$(floor_pow2 "${low}")"
  return 0
}

run_train() {
  local method="$1"
  local bs="$2"
  local ga="$3"
  local out_dir="${RUN_ROOT}/training/${method}"
  local log_path="${RUN_ROOT}/checks/train_${method}.log"
  local extra
  extra="$(method_args "${method}")"
  mkdir -p "${out_dir}"

  "${accelerate_cmd[@]}" \
    "${base_train_args[@]}" \
    --output_dir "${out_dir}" \
    --per_device_train_batch_size "${bs}" \
    --gradient_accumulation_steps "${ga}" \
    --wandb_run_name "train-${method}-${RUN_ID}" \
    ${extra} \
    > "${log_path}" 2>&1

  mkdir -p "${RUN_ROOT}/hessian/${method}"
  if [[ -f "${out_dir}/hessian_effective_rank.json" ]]; then
    cp "${out_dir}/hessian_effective_rank.json" "${RUN_ROOT}/hessian/${method}/effective_rank.json"
  fi
}

run_landscape() {
  local method="$1"
  local out_dir="${RUN_ROOT}/hessian/${method}/landscape3d"
  local log_path="${RUN_ROOT}/checks/landscape_${method}.log"
  mkdir -p "${out_dir}"
  local adapter_arg=()
  if [[ "${method}" != "full" ]]; then
    adapter_arg=(--adapter_path "${RUN_ROOT}/training/${method}")
  fi

  "${PYTHON_BIN}" scripts/plot_hessian_landscape_3d.py \
    --model_name_or_path "${MODEL_NAME}" \
    --tokenizer_name "${TOKENIZER_NAME}" \
    "${adapter_arg[@]}" \
    --train_file "${TRAIN_FILE}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --per_device_batch_size 1 \
    --preprocessing_num_workers 1 \
    --hessian_top_n 2 \
    --hessian_max_iter 6 \
    --hessian_tol 1e-2 \
    --hessian_num_batches 1 \
    --landscape_num_batches 1 \
    --grid_points 9 \
    --radius 0.05 \
    --output_dir "${out_dir}" \
    > "${log_path}" 2>&1
}

run_eval_suite() {
  local method="$1"
  local eval_root="${RUN_ROOT}/eval/${method}"
  local peft_args=()
  if [[ "${method}" != "full" ]]; then
    peft_args=(--peft_name_or_path "${RUN_ROOT}/training/${method}")
  fi
  mkdir -p "${eval_root}"

  "${PYTHON_BIN}" -m eval.gsm.run_eval \
    --data_dir data/eval/gsm \
    --max_num_examples 64 \
    --save_dir "${eval_root}/gsm" \
    --model_name_or_path "${MODEL_NAME}" \
    --tokenizer_name_or_path "${TOKENIZER_NAME}" \
    "${peft_args[@]}" \
    --eval_batch_size 1 \
    > "${RUN_ROOT}/checks/eval_gsm_${method}.log" 2>&1

  "${PYTHON_BIN}" -m eval.mmlu.run_eval \
    --data_dir data/eval/mmlu \
    --save_dir "${eval_root}/mmlu" \
    --model_name_or_path "${MODEL_NAME}" \
    --tokenizer_name_or_path "${TOKENIZER_NAME}" \
    "${peft_args[@]}" \
    --subjects abstract_algebra anatomy high_school_biology \
    --n_instances 20 \
    --ntrain 3 \
    --eval_batch_size 1 \
    > "${RUN_ROOT}/checks/eval_mmlu_${method}.log" 2>&1

  "${PYTHON_BIN}" -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir "${eval_root}/bbh" \
    --model_name_or_path "${MODEL_NAME}" \
    --tokenizer_name_or_path "${TOKENIZER_NAME}" \
    "${peft_args[@]}" \
    --max_num_examples_per_task 3 \
    --eval_batch_size 1 \
    > "${RUN_ROOT}/checks/eval_bbh_${method}.log" 2>&1

  "${PYTHON_BIN}" -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa \
    --save_dir "${eval_root}/tydiqa" \
    --model_name_or_path "${MODEL_NAME}" \
    --tokenizer_name_or_path "${TOKENIZER_NAME}" \
    "${peft_args[@]}" \
    --max_num_examples_per_lang 3 \
    --n_shot 0 \
    --eval_batch_size 1 \
    > "${RUN_ROOT}/checks/eval_tydiqa_${method}.log" 2>&1
}

echo "Preparing eval data ..."
if [[ -s data/eval/gsm/test.jsonl && -s data/eval/tydiqa/tydiqa-goldp-v1.1-dev.json ]]; then
  echo "Eval data already present; skipping download." | tee "${RUN_ROOT}/checks/prepare_eval_data.log"
else
  bash scripts/prepare_eval_data.sh > "${RUN_ROOT}/checks/prepare_eval_data.log" 2>&1
fi

declare -A MAX_BS
declare -A GRAD_ACCUM
MAX_BS["full"]="${FULL_BATCH_SIZE}"
MAX_BS["lora"]="${ADAPTER_BATCH_SIZE}"
MAX_BS["sft"]="${ADAPTER_BATCH_SIZE}"

for m in full lora sft; do
  denom=$((NUM_PROCESSES * MAX_BS[$m]))
  if (( GLOBAL_BATCH_SIZE % denom != 0 )); then
    echo "ERROR: GLOBAL_BATCH_SIZE (${GLOBAL_BATCH_SIZE}) must be divisible by num_processes*per_device for ${m} (${NUM_PROCESSES}*${MAX_BS[$m]}=${denom})." >&2
    exit 1
  fi
  GRAD_ACCUM["${m}"]=$((GLOBAL_BATCH_SIZE / denom))
  if [[ "${GRAD_ACCUM[$m]}" -lt 1 ]]; then
    echo "ERROR: Invalid gradient_accumulation_steps for ${m}: ${GRAD_ACCUM[$m]}" >&2
    exit 1
  fi
  echo "${m}=${MAX_BS[$m]} (grad_accum=${GRAD_ACCUM[$m]}, global=${GLOBAL_BATCH_SIZE})" | tee -a "${RUN_ROOT}/checks/max_batch_sizes.txt"
done

for m in full lora sft; do
  if [[ "${MAX_BS[$m]}" -lt 1 ]]; then
    echo "Skipping ${m}: no feasible batch size found" | tee -a "${RUN_ROOT}/checks/failed_methods.txt"
    continue
  fi
  echo "Training ${m} with bs=${MAX_BS[$m]} grad_accum=${GRAD_ACCUM[$m]} (global=${GLOBAL_BATCH_SIZE}) ..."
  run_train "${m}" "${MAX_BS[$m]}" "${GRAD_ACCUM[$m]}"
  echo "Hessian landscape ${m} ..."
  run_landscape "${m}"
  echo "Eval ${m} ..."
  run_eval_suite "${m}"
done

"${PYTHON_BIN}" - <<PY
import json, os
run_root = "${RUN_ROOT}"
summary = {"run_root": run_root, "methods": {}}
for m in ["full","lora","sft"]:
    ent = {}
    bs_file = os.path.join(run_root, "checks", "max_batch_sizes.txt")
    if os.path.exists(bs_file):
        with open(bs_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(m+"="):
                    ent["max_batch_size"] = int(line.strip().split("=")[1])
    h = os.path.join(run_root, "hessian", m, "effective_rank.json")
    if os.path.exists(h):
        with open(h, "r", encoding="utf-8") as f:
            ent["hessian_effective_rank"] = json.load(f).get("effective_rank")
    metrics = {}
    for k in ["gsm","mmlu","bbh","tydiqa"]:
        p = os.path.join(run_root, "eval", m, k, "metrics.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                metrics[k] = json.load(f)
    if metrics:
        ent["eval"] = metrics
    if ent:
        summary["methods"][m] = ent
with open(os.path.join(run_root, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PY

echo "Done: ${RUN_ROOT}"
