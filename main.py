#!/usr/bin/env python3
import argparse
import os
import subprocess
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
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-train-steps", type=int, default=20)
    parser.add_argument("--output-root", default=str(repo_root / "results"))
    parser.add_argument("--run-id", default=None, help="Optional run id. If omitted, script uses current timestamp.")
    parser.add_argument("--wandb-mode", default="offline", choices=["offline", "online", "disabled"])
    return parser.parse_args()


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
    if not train_file.exists():
        raise FileNotFoundError(f"Missing train file: {train_file}")

    env = os.environ.copy()
    env["MODEL_NAME"] = args.model_name
    env["TOKENIZER_NAME"] = args.tokenizer_name
    env["TRAIN_FILE"] = str(train_file)
    env["MAX_SEQ_LENGTH"] = str(args.max_seq_length)
    env["MAX_TRAIN_STEPS"] = str(args.max_train_steps)
    env["OUTPUT_ROOT"] = args.output_root
    env["WANDB_MODE"] = args.wandb_mode
    if args.run_id:
        env["RUN_ID"] = args.run_id

    cmd = ["bash", str(script_path)]
    print("Running:", " ".join(cmd))
    print(f"MODEL_NAME={env['MODEL_NAME']}")
    print(f"TOKENIZER_NAME={env['TOKENIZER_NAME']}")
    print(f"TRAIN_FILE={env['TRAIN_FILE']}")
    print(f"MAX_SEQ_LENGTH={env['MAX_SEQ_LENGTH']}")
    print(f"MAX_TRAIN_STEPS={env['MAX_TRAIN_STEPS']}")
    print(f"OUTPUT_ROOT={env['OUTPUT_ROOT']}")
    print(f"WANDB_MODE={env['WANDB_MODE']}")

    completed = subprocess.run(cmd, cwd=str(sft_root), env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
