#!/usr/bin/env python3
import argparse
import json
import os

from datasets import load_dataset


def convert_split(split_name):
    ds = load_dataset("tydiqa", "secondary_task", split=split_name)
    data = []
    for row in ds:
        answers = row["answers"]
        qas = [
            {
                "id": row["id"],
                "question": row["question"],
                "answers": [
                    {"text": t, "answer_start": int(s)}
                    for t, s in zip(answers.get("text", []), answers.get("answer_start", []))
                ],
            }
        ]
        data.append(
            {
                "title": row.get("title", ""),
                "paragraphs": [{"context": row["context"], "qas": qas}],
            }
        )
    return {"version": "1.1", "data": data}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/eval/tydiqa")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dev = convert_split("validation")
    train = convert_split("train")

    dev_path = os.path.join(args.output_dir, "tydiqa-goldp-v1.1-dev.json")
    train_path = os.path.join(args.output_dir, "tydiqa-goldp-v1.1-train.json")

    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev, f, ensure_ascii=False)
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False)

    print(dev_path)
    print(train_path)


if __name__ == "__main__":
    main()
