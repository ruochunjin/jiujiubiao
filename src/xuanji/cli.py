"""Command-line interface:  python -m xuanji --dataset data.json [--task wtq]"""

from __future__ import annotations

import argparse
import json
import sys

from xuanji.eval import denotation_accuracy, load_dataset
from xuanji.pipeline import XuanJi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xuanji",
        description="XuanJi (璇玑): unified table/database question answering",
    )
    parser.add_argument("--dataset", required=True, help="Path to a WTQ/TabFact-style JSON dataset")
    parser.add_argument("--task", default="tqa", choices=["tqa", "tfv"])
    parser.add_argument("--mode", default="auto", choices=["auto", "solve", "refine"])
    parser.add_argument("--limit", type=int, default=-1, help="Only the first N examples")
    parser.add_argument("--output", default="", help="Write per-example results JSON here")
    parser.add_argument("--disambiguate", action="store_true", help="Enable the ambiguity stage")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data = load_dataset(args.dataset)
    if args.limit > 0:
        data = data[: args.limit]
    engine = XuanJi(mode=args.mode, task=args.task, disambiguate=args.disambiguate)
    predictions, labels, results = [], [], []
    for i, example in enumerate(data):
        question = example.get("question", example.get("statement", ""))
        table = example.get("table_text")
        result = engine.ask(
            question, table, task=args.task, caption=example.get("table_caption", "")
        )
        predictions.append(result.answer)
        labels.append(example.get("label"))
        results.append(
            {
                "id": example.get("id", i),
                "question": question,
                "answer": result.answer,
                "norm": result.norm,
                "label": example.get("label"),
                "mode": result.mode,
                "status": result.status,
            }
        )
        print(f"[{i + 1}/{len(data)}] {result.status} answer={result.answer!r}")
    accuracy = denotation_accuracy(predictions, labels, task=args.task)
    print(f"\nAccuracy: {accuracy:.4f} ({len(data)} examples)")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"accuracy": accuracy, "results": results}, handle, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
