#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import math
import os
import random
import sys
import tempfile
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GPT2Tokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    OPTForCausalLM,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot 3D Hessian curvature landscape over top-2 eigenvector directions.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default=None, help="Optional PEFT adapter path.")
    parser.add_argument("--train_file", type=str, required=True, help="json/jsonl train file.")
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hessian_top_n", type=int, default=2)
    parser.add_argument("--hessian_max_iter", type=int, default=30)
    parser.add_argument("--hessian_tol", type=float, default=1e-3)
    parser.add_argument("--hessian_num_batches", type=int, default=1)
    parser.add_argument("--landscape_num_batches", type=int, default=1)
    parser.add_argument("--grid_points", type=int, default=21)
    parser.add_argument("--radius", type=float, default=0.15, help="Perturbation radius along each Hessian direction.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_to_wandb", action="store_true", help="Upload 3D landscape outputs to Weights & Biases.")
    parser.add_argument("--wandb_project", type=str, default="peft-hessian")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode. Use offline in restricted environments.",
    )
    return parser.parse_args()


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    if not example["prompt"].endswith((" ", "\n", "\t")) and not example["completion"].startswith((" ", "\n", "\t")):
        text = example["prompt"] + " " + example["completion"]
    else:
        text = example["prompt"] + example["completion"]
    text = text + tokenizer.eos_token
    tok = tokenizer(text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = tok.input_ids
    labels = input_ids.clone()
    tok_prompt = tokenizer(example["prompt"], return_tensors="pt", max_length=max_seq_length, truncation=True)
    labels[:, : tok_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length):
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(msgs):
        out = ""
        for msg in msgs:
            if msg["role"] == "system":
                out += "<|system|>\n" + msg["content"].strip() + "\n"
            elif msg["role"] == "user":
                out += "<|user|>\n" + msg["content"].strip() + "\n"
            elif msg["role"] == "assistant":
                out += "<|assistant|>\n" + msg["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError(f"Invalid role: {msg['role']}")
        return out

    text = _concat_messages(messages).strip()
    tok = tokenizer(text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = tok.input_ids
    labels = input_ids.clone()

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            if i == 0:
                s = 0
            else:
                s = tokenizer(_concat_messages(messages[:i]), return_tensors="pt", max_length=max_seq_length, truncation=True).input_ids.shape[1]
            if i < len(messages) - 1 and messages[i + 1]["role"] == "assistant":
                msgs_so_far = _concat_messages(messages[: i + 1]) + "<|assistant|>\n"
            else:
                msgs_so_far = _concat_messages(messages[: i + 1])
            e = tokenizer(msgs_so_far, return_tensors="pt", max_length=max_seq_length, truncation=True).input_ids.shape[1]
            labels[:, s:e] = -100
            if e >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def _lm_batch_parser(batch, device):
    labels = batch["labels"].to(device)
    model_inputs = {}
    for k in ("input_ids", "attention_mask"):
        if k in batch:
            model_inputs[k] = batch[k].to(device)
    return model_inputs, labels


def _lm_criterion(outputs, labels):
    if hasattr(outputs, "loss") and outputs.loss is not None:
        return outputs.loss
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _load_config_with_rope_fallback(config_source):
    token = _get_hf_token()
    try:
        return AutoConfig.from_pretrained(config_source, token=token)
    except ValueError as e:
        if "rope_scaling" not in str(e):
            raise
        if os.path.isdir(config_source):
            cfg_path = os.path.join(config_source, "config.json")
        else:
            cfg_path = hf_hub_download(repo_id=config_source, filename="config.json", token=token)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        rope = cfg.get("rope_scaling", {})
        factor = float(rope.get("factor", 32.0)) if isinstance(rope, dict) else 32.0
        cfg["rope_scaling"] = {"type": "dynamic", "factor": factor}
        with tempfile.TemporaryDirectory(prefix="legacy_rope_cfg_") as td:
            patched = os.path.join(td, "config.json")
            with open(patched, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            return AutoConfig.from_pretrained(td, token=token)


def _get_hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text(encoding="utf-8").strip()
    return None


def _prepare_tokenizer_and_model(args, device):
    config = _load_config_with_rope_fallback(args.model_name_or_path)
    token = _get_hf_token()
    tok_name = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        token=token,
    )

    if isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
        tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>"})
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        tokenizer.add_special_tokens({"unk_token": "<unk>"})

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    emb_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > emb_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)

    model.to(device)
    model.eval()
    return model, tokenizer


def _build_train_dataset(args, tokenizer):
    raw = load_dataset("json", data_files={"train": args.train_file})
    cols = raw["train"].column_names
    if "prompt" in cols and "completion" in cols:
        encode_fn = partial(encode_with_prompt_completion_format, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    elif "messages" in cols:
        encode_fn = partial(encode_with_messages_format, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    else:
        raise ValueError("Need 'prompt'&'completion' or 'messages' columns.")

    ds = raw.map(
        encode_fn,
        batched=False,
        num_proc=args.preprocessing_num_workers,
        remove_columns=[c for c in cols if c not in ["input_ids", "labels", "attention_mask"]],
        desc="Tokenizing data for hessian landscape",
    )
    ds.set_format(type="pt")
    ds = ds.filter(lambda ex: (ex["labels"] != -100).any())
    return ds["train"]


def _take_batches(dataset, tokenizer, model, batch_size, num_batches):
    dl = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=batch_size,
    )
    out = []
    for i, batch in enumerate(dl):
        out.append(batch)
        if i + 1 >= num_batches:
            break
    return out


def _mean_loss(model, batches, device):
    losses = []
    with torch.no_grad():
        for batch in batches:
            inputs, labels = _lm_batch_parser(batch, device)
            out = model(**inputs, labels=labels, use_cache=False)
            losses.append(out.loss.detach().float().item())
    return float(np.mean(losses)) if losses else math.nan


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = _prepare_tokenizer_and_model(args, device)
    train_dataset = _build_train_dataset(args, tokenizer)

    pyhessian_dir = Path(__file__).resolve().parents[2] / "PyHessian"
    if not pyhessian_dir.exists():
        raise FileNotFoundError(f"PyHessian directory not found at {pyhessian_dir}")
    if str(pyhessian_dir) not in sys.path:
        sys.path.insert(0, str(pyhessian_dir))
    from pyhessian.hessian import hessian

    h_batches = _take_batches(
        dataset=train_dataset,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.per_device_batch_size,
        num_batches=max(1, args.hessian_num_batches),
    )
    l_batches = _take_batches(
        dataset=train_dataset,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.per_device_batch_size,
        num_batches=max(1, args.landscape_num_batches),
    )

    h_data = h_batches[0] if len(h_batches) == 1 else h_batches
    h_kwargs = {"data": h_data} if len(h_batches) == 1 else {"dataloader": h_data}
    hessian_comp = hessian(
        model=model.eval(),
        criterion=_lm_criterion,
        cuda=(device == "cuda"),
        batch_parser=_lm_batch_parser,
        **h_kwargs,
    )

    eigenvalues, eigenvectors = hessian_comp.eigenvalues(
        top_n=max(2, args.hessian_top_n),
        maxIter=args.hessian_max_iter,
        tol=args.hessian_tol,
    )
    v1 = eigenvectors[0]
    v2 = eigenvectors[1]

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    originals = [p.data.clone() for p in trainable_params]

    def set_perturb(alpha, beta):
        with torch.no_grad():
            for p, p0, d1, d2 in zip(trainable_params, originals, v1, v2):
                p.data.copy_(p0 + alpha * d1.to(p0.dtype) + beta * d2.to(p0.dtype))

    xs = np.linspace(-args.radius, args.radius, args.grid_points)
    ys = np.linspace(-args.radius, args.radius, args.grid_points)
    zz = np.zeros((args.grid_points, args.grid_points), dtype=np.float64)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            set_perturb(float(x), float(y))
            zz[i, j] = _mean_loss(model, l_batches, device)

    set_perturb(0.0, 0.0)

    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel("Direction 1 (top Hessian eigvec)")
    ax.set_ylabel("Direction 2 (top Hessian eigvec)")
    ax.set_zlabel("Loss")
    ax.set_title("Hessian Curvature 3D Landscape")
    fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.1, label="Loss")
    plt.tight_layout()
    png_path = os.path.join(args.output_dir, "hessian_landscape_3d.png")
    plt.savefig(png_path, dpi=220)
    plt.close(fig)

    npz_path = os.path.join(args.output_dir, "hessian_landscape_3d.npz")
    np.savez(npz_path, x=xs, y=ys, z=zz, eigenvalues=np.array(eigenvalues, dtype=np.float64))

    meta = {
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "train_file": args.train_file,
        "grid_points": args.grid_points,
        "radius": args.radius,
        "hessian_num_batches": args.hessian_num_batches,
        "landscape_num_batches": args.landscape_num_batches,
        "top_eigenvalues": [float(v) for v in eigenvalues[:2]],
        "png_path": png_path,
        "npz_path": npz_path,
    }
    with open(os.path.join(args.output_dir, "hessian_landscape_3d_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if args.log_to_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            dir=args.output_dir,
            config={
                "model_name_or_path": args.model_name_or_path,
                "adapter_path": args.adapter_path,
                "train_file": args.train_file,
                "grid_points": args.grid_points,
                "radius": args.radius,
                "hessian_num_batches": args.hessian_num_batches,
                "landscape_num_batches": args.landscape_num_batches,
            },
        )
        run.log(
            {
                "landscape/top_eigenvalue_1": meta["top_eigenvalues"][0],
                "landscape/top_eigenvalue_2": meta["top_eigenvalues"][1],
                "landscape/plot_3d": wandb.Image(png_path),
            }
        )

        artifact = wandb.Artifact("hessian-landscape-3d", type="analysis")
        artifact.add_file(png_path)
        artifact.add_file(npz_path)
        artifact.add_file(os.path.join(args.output_dir, "hessian_landscape_3d_meta.json"))
        run.log_artifact(artifact)
        run.finish()

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
