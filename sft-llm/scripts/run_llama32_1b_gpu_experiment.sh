#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B}"
TOKENIZER_NAME="${TOKENIZER_NAME:-meta-llama/Llama-3.2-1B}"
TRAIN_FILE="${TRAIN_FILE:-data/processed/gpt4_alpaca/gpt4_alpaca_data.jsonl}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/pathfinder/projects/PEFT/results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODEL_KEY="${MODEL_KEY:-$(echo "${MODEL_NAME}" | tr '/:' '__')}"
RUN_ROOT="${OUTPUT_ROOT}/${MODEL_KEY}/${RUN_ID}"
export HF_HOME="${HF_HOME:-/home/pathfinder/projects/PEFT/sft-llm/.hf_cache}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
if [[ -f /home/pathfinder/.cache/huggingface/token ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$(cat /home/pathfinder/.cache/huggingface/token)}"
  export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN}}"
fi

mkdir -p "${RUN_ROOT}"/{checks,eval,hessian,training,tmp}
echo "Run root: ${RUN_ROOT}"

accelerate_cmd=(accelerate launch --num_machines 1 --num_processes 1 finetune/finetune.py)

base_train_args=(
  --model_name_or_path "${MODEL_NAME}"
  --tokenizer_name "${TOKENIZER_NAME}"
  --train_file "${TRAIN_FILE}"
  --max_seq_length "${MAX_SEQ_LENGTH}"
  --preprocessing_num_workers 1
  --gradient_accumulation_steps 1
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
    --compute_hessian_effective_rank \
    --hessian_top_n 2 \
    --hessian_num_batches 1 \
    --hessian_max_iter 3 \
    --hessian_tol 1e-2 \
    --output_dir "${out_dir}" \
    --per_device_train_batch_size "${bs}" \
    --max_train_steps 1 \
    --wandb_run_name "probe-${method}-bs${bs}" \
    ${extra} \
    > "${log_path}" 2>&1
  local rc=$?
  set -e

  if [[ ${rc} -ne 0 ]]; then
    return 1
  fi
  if rg -qi "out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|RuntimeError: CUDA error" "${log_path}"; then
    return 1
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
      break
    fi
  done
  if [[ ${low} -eq 0 ]]; then
    echo 0
    return
  fi
  local left=$((low + 1))
  local right=$((high - 1))
  while [[ ${left} -le ${right} ]]; do
    local mid=$(((left + right) / 2))
    if probe_one "${method}" "${mid}"; then
      low=${mid}
      left=$((mid + 1))
    else
      right=$((mid - 1))
    fi
  done
  echo "${low}"
}

run_train() {
  local method="$1"
  local bs="$2"
  local out_dir="${RUN_ROOT}/training/${method}"
  local log_path="${RUN_ROOT}/checks/train_${method}.log"
  local extra
  extra="$(method_args "${method}")"
  mkdir -p "${out_dir}"

  "${accelerate_cmd[@]}" \
    "${base_train_args[@]}" \
    --output_dir "${out_dir}" \
    --per_device_train_batch_size "${bs}" \
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

  python scripts/plot_hessian_landscape_3d.py \
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

  python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm \
    --max_num_examples 64 \
    --save_dir "${eval_root}/gsm" \
    --model_name_or_path "${MODEL_NAME}" \
    --tokenizer_name_or_path "${TOKENIZER_NAME}" \
    "${peft_args[@]}" \
    --eval_batch_size 1 \
    > "${RUN_ROOT}/checks/eval_gsm_${method}.log" 2>&1

  python -m eval.mmlu.run_eval \
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

  python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir "${eval_root}/bbh" \
    --model_name_or_path "${MODEL_NAME}" \
    --tokenizer_name_or_path "${TOKENIZER_NAME}" \
    "${peft_args[@]}" \
    --max_num_examples_per_task 3 \
    --eval_batch_size 1 \
    > "${RUN_ROOT}/checks/eval_bbh_${method}.log" 2>&1

  python -m eval.tydiqa.run_eval \
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
for m in full lora sft; do
  echo "Probing max batch size for ${m} ..."
  MAX_BS["${m}"]="$(find_max_bs "${m}")"
  echo "${m}=${MAX_BS[$m]}" | tee -a "${RUN_ROOT}/checks/max_batch_sizes.txt"
done

for m in full lora sft; do
  if [[ "${MAX_BS[$m]}" -lt 1 ]]; then
    echo "Skipping ${m}: no feasible batch size found" | tee -a "${RUN_ROOT}/checks/failed_methods.txt"
    continue
  fi
  echo "Training ${m} with bs=${MAX_BS[$m]} ..."
  run_train "${m}" "${MAX_BS[$m]}"
  echo "Hessian landscape ${m} ..."
  run_landscape "${m}"
  echo "Eval ${m} ..."
  run_eval_suite "${m}"
done

python - <<PY
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
