#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Hessian summary metrics from hessian_effective_rank.json.")
    parser.add_argument("--hessian_json", type=str, required=True, help="Path to hessian_effective_rank.json")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for figures. Defaults to the json parent directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.hessian_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir = args.out_dir if args.out_dir else os.path.dirname(os.path.abspath(args.hessian_json))
    os.makedirs(out_dir, exist_ok=True)

    eigs = np.array(data.get("abs_eigenvalues_sorted") or np.abs(np.array(data.get("eigenvalues", []))), dtype=np.float64)
    if eigs.size == 0:
        raise ValueError("No eigenvalues found in the Hessian json.")

    x = np.arange(1, eigs.size + 1)
    cume = np.cumsum(eigs) / (np.sum(eigs) + 1e-12)

    plt.figure(figsize=(7, 4))
    plt.plot(x, eigs, marker="o", linewidth=1.5)
    plt.title("Hessian Scree Plot (|eigenvalues|)")
    plt.xlabel("Index (sorted)")
    plt.ylabel("|eigenvalue|")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hessian_scree.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(x, cume, marker="o", linewidth=1.5)
    plt.axhline(0.8, linestyle="--", linewidth=1)
    plt.axhline(0.9, linestyle="--", linewidth=1)
    plt.title("Cumulative Curvature Explained")
    plt.xlabel("Top-k eigenvalues")
    plt.ylabel("Cumulative ratio")
    plt.ylim(0.0, 1.02)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hessian_cumulative.png"), dpi=180)
    plt.close()

    top_k = min(20, eigs.size)
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(top_k), eigs[:top_k])
    plt.title(f"Top-{top_k} |eigenvalues|")
    plt.xlabel("Rank")
    plt.ylabel("|eigenvalue|")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hessian_topk_bar.png"), dpi=180)
    plt.close()

    print(f"Saved plots to {out_dir}")
    print(
        json.dumps(
            {
                "effective_rank": data.get("effective_rank"),
                "effective_rank_normalized": data.get("effective_rank_normalized"),
                "explained_80_rank": data.get("explained_80_rank"),
                "explained_90_rank": data.get("explained_90_rank"),
                "condition_number_approx": data.get("condition_number_approx"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
