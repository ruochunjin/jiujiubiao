"""Disambiguation golden evaluation on CLAMBSQL (test-driven driver).

For each sampled ambiguous item: the disambiguator must (1) flag ambiguity,
(2) ask a clarification, (3) rewrite the question so the GOLD target column
(from the benchmark's `clear_ambiguity` mapping) is referenced.

End-to-end coverage (rewritten question → solver answer vs gold SQL values)
is the headline metric; the intermediate rates localize failures.

Usage:
    XUANJI_BACKEND=anthropic .venv/Scripts/python.exe benchmarks/eval_disambig.py [--n 35] [--impl v1|v2]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def gold_terms(row: dict) -> list[str]:
    """Column names from the benchmark's clear_ambiguity mapping.

    The field is a JSON STRING (not a dict) in the vendored dataset.
    """

    mapping = row.get("clear_ambiguity") or "{}"
    if isinstance(mapping, str):
        try:
            mapping = json.loads(mapping)
        except json.JSONDecodeError:
            return []
    terms: list[str] = []
    for _term, tables in mapping.items():
        if isinstance(tables, dict):
            for _table, columns in tables.items():
                terms.extend(columns)
    return terms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=35)
    parser.add_argument("--impl", default="v2", choices=["v1", "v2"])
    args = parser.parse_args()

    import jiujiubiao
    from jiujiubiao.tools.db import Database

    data = json.loads(Path("benchmarks/data/clambsql/clambsql.json").read_text(encoding="utf-8"))
    rng = random.Random(20261009)
    by_type: dict[str, list[dict]] = {}
    for row in data:
        by_type.setdefault(row["ambig_type"], []).append(row)
    picked: list[dict] = []
    per_type = max(1, args.n // len(by_type))
    for ambig_type, rows in sorted(by_type.items()):
        rows = list(rows)
        rng.shuffle(rows)
        picked.extend(rows[:per_type])
    picked = picked[: args.n]
    print(f"disambig eval: {len(picked)} items (impl={args.impl})")

    db_cache: dict[str, Database] = {}
    report = []
    n_flag = n_asked = n_anchored = 0
    cov_direct_sum = cov_rewrite_sum = 0.0
    for row in picked:
        db_file = row["db_file"]
        if db_file not in db_cache:
            db_cache[db_file] = Database.from_sqlite(f"benchmarks/data/clambsql/database/{db_file}")
        db = db_cache[db_file]
        schema = db.schema_text()
        terms = gold_terms(row)
        clarify_answer = row.get("clarification_context") or ""

        from jiujiubiao.disambiguate import Disambiguator, DisambiguatorV2

        cls = DisambiguatorV2 if args.impl == "v2" else Disambiguator
        start = time.perf_counter()
        result = cls(
            jiujiubiao.AnthropicClient(),
            clarify=lambda _q, a=clarify_answer: a,
        ).run(row["question"], schema)
        elapsed = time.perf_counter() - start

        anchored = bool(terms) and any(
            term.casefold() in result.question.casefold() for term in terms
        )
        n_flag += result.ambiguous
        n_asked += bool(result.clarification)
        n_anchored += anchored
        report.append(
            {
                "id": f"{row['ambig_type']}-{row['index']}",
                "question": row["question"],
                "ambiguous": result.ambiguous,
                "clarification": result.clarification,
                "rewritten": result.question,
                "gold_terms": terms,
                "anchored": anchored,
                "seconds": round(elapsed, 1),
            }
        )
        print(
            f"{row['ambig_type']}-{row['index']}: ambiguous={result.ambiguous} "
            f"anchored={anchored} rewrite={result.question[:70]!r}",
            flush=True,
        )

    out = {
        "impl": args.impl,
        "n": len(picked),
        "flag_rate": n_flag / len(picked),
        "asked_rate": n_asked / len(picked),
        "anchor_rate": n_anchored / len(picked),
        "cases": report,
    }
    out_path = Path(f"benchmarks/disambig_eval_{args.impl}.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"\n{args.impl}: flag={out['flag_rate']:.2f} asked={out['asked_rate']:.2f} "
        f"anchor={out['anchor_rate']:.2f} -> {out_path}"
    )


if __name__ == "__main__":
    main()
