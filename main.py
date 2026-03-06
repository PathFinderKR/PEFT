#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    default_train = repo_root / "sft-llm" / "data" / "processed" / "gpt4_alpaca" / "gpt4_alpaca_data.jsonl"
    default_tiny = repo_root / "sft-llm" / "data" / "processed" / "gpt4_alpaca" / "gpt4_alpaca_tiny.jsonl"

    parser = argparse.ArgumentParser(
        description=(
            "Run end-to-end GPU experiment for meta-llama/Llama-3.2-1B:\n"
            "1) Full FT / Sparse FT / LoRA training\n"
            "2) Hessian effective rank + loss curvature landscape\n"
            "3) Evaluation metrics on trained checkpoints"
        )
    )
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--tokenizer-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--train-file", default=str(default_train))
    parser.add_argument("--use-tiny-train-file", action="store_true", help=f"Use {default_tiny} for faster smoke runs.")
    parser.add_argument(
        "--prepare-train-data",
        dest="prepare_train_data",
        action="store_true",
        help="If train file is missing, run sft-llm/scripts/prepare_train_data.sh automatically.",
    )
    parser.add_argument(
        "--no-prepare-train-data",
        dest="prepare_train_data",
        action="store_false",
        help="Do not auto-run data preparation when the train file is missing.",
    )
    parser.set_defaults(prepare_train_data=True)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Accelerate processes (typically GPUs) to use. Default: auto-detect all visible GPUs.",
    )
    parser.add_argument("--num-machines", type=int, default=1, help="Number of machines for Accelerate launch.")
    parser.add_argument("--output-root", default=str(repo_root / "results"))
    parser.add_argument("--run-id", default=None, help="Optional run id. If omitted, script uses current timestamp.")
    parser.add_argument("--wandb-mode", default="offline", choices=["offline", "online", "disabled"])
    return parser.parse_args()


def ensure_train_file(train_file: Path, sft_root: Path, prepare_train_data: bool) -> Path:
    if train_file.exists():
        return train_file

    if not prepare_train_data:
        prep_cmd = "bash sft-llm/scripts/prepare_train_data.sh"
        raise FileNotFoundError(
            f"Missing train file: {train_file}\n"
            f"Prepare it first with: {prep_cmd}\n"
            "or rerun this script with --prepare-train-data."
        )

    prep_script = sft_root / "scripts" / "prepare_train_data.sh"
    if not prep_script.exists():
        raise FileNotFoundError(f"Missing script: {prep_script}")

    print(f"Train file not found. Preparing data with: bash {prep_script}")
    completed = subprocess.run(["bash", str(prep_script)], cwd=str(sft_root), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"prepare_train_data.sh failed with exit code {completed.returncode}")
    if not train_file.exists():
        raise FileNotFoundError(
            f"Train data preparation completed but file still missing: {train_file}"
        )
    return train_file


def resolve_num_processes(user_value: int | None) -> int | None:
    if user_value is not None:
        return user_value
    try:
        import torch  # type: ignore

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        num_gpus = 0

    # The downstream script fails fast on "auto" when no CUDA devices are visible.
    # Fall back to single-process launch so CPU-only environments can still run.
    if num_gpus < 1:
        return 1
    return None


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    sft_root = repo_root / "sft-llm"
    script_path = sft_root / "scripts" / "run_llama32_1b_gpu_experiment.sh"

    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    train_file = Path(args.train_file)
    if args.use_tiny_train_file:
        train_file = sft_root / "data" / "processed" / "gpt4_alpaca" / "gpt4_alpaca_tiny.jsonl"
    train_file = ensure_train_file(train_file, sft_root, args.prepare_train_data)

    env = os.environ.copy()
    env["MODEL_NAME"] = args.model_name
    env["TOKENIZER_NAME"] = args.tokenizer_name
    env["TRAIN_FILE"] = str(train_file)
    env["MAX_SEQ_LENGTH"] = str(args.max_seq_length)
    env["NUM_TRAIN_EPOCHS"] = str(args.num_train_epochs)
    env["OUTPUT_ROOT"] = args.output_root
    env["WANDB_MODE"] = args.wandb_mode
    env["NUM_MACHINES"] = str(args.num_machines)
    env["PYTHON_BIN"] = sys.executable
    resolved_num_processes = resolve_num_processes(args.num_processes)
    if resolved_num_processes is not None:
        env["NUM_PROCESSES"] = str(resolved_num_processes)
    if args.run_id:
        env["RUN_ID"] = args.run_id

    cmd = ["bash", str(script_path)]
    print("Running:", " ".join(cmd))
    print(f"MODEL_NAME={env['MODEL_NAME']}")
    print(f"TOKENIZER_NAME={env['TOKENIZER_NAME']}")
    print(f"TRAIN_FILE={env['TRAIN_FILE']}")
    print(f"MAX_SEQ_LENGTH={env['MAX_SEQ_LENGTH']}")
    print(f"NUM_TRAIN_EPOCHS={env['NUM_TRAIN_EPOCHS']}")
    print(f"OUTPUT_ROOT={env['OUTPUT_ROOT']}")
    print(f"WANDB_MODE={env['WANDB_MODE']}")
    print(f"NUM_MACHINES={env['NUM_MACHINES']}")
    print(f"NUM_PROCESSES={env.get('NUM_PROCESSES', 'auto')}")
    if args.num_processes is None and env.get("NUM_PROCESSES") == "1":
        print("No CUDA devices detected. Falling back to NUM_PROCESSES=1 (CPU/single-process mode).")

    completed = subprocess.run(cmd, cwd=str(sft_root), env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
