"""Boundary battery for the JiuJiuBiao pipeline (offline, FakeLLM).

Edge inputs that must never crash and must route correctly:
- empty / single-row / single-column / wide / long tables
- unicode content, markdown noise, duplicate headers
- refusal-shaped and empty LLM answers
- mode routing thresholds (long_table_rows boundary)
"""

from __future__ import annotations

import unittest

from jiujiubiao import FakeLLM, JiuJiuBiao
from jiujiubiao.table import Table

_ANSWER = "Let's think step by step. Row 0 has the answer.\nAnswer: Paris"


def _engine(**kwargs) -> JiuJiuBiao:
    llm = FakeLLM(rules=[(r".*", _ANSWER)])
    kwargs.setdefault("refiner_kwargs", {"n_samples": 3})
    kwargs.setdefault("solver_kwargs", {"sc": 2, "max_depth": 3})
    return JiuJiuBiao(llm=llm, **kwargs)


class TableEdgeTests(unittest.TestCase):
    def test_single_row_table(self) -> None:
        result = _engine().ask("Where?", [["Venue"], ["Paris"]])
        self.assertEqual(result.norm, "paris")

    def test_single_column_table(self) -> None:
        result = _engine().ask("Which city?", [["Paris"], ["London"]])
        self.assertEqual(result.norm, "paris")

    def test_wide_table(self) -> None:
        table = [[f"col{i}" for i in range(60)], ["x"] * 60]
        result = _engine().ask("Which col?", table)
        self.assertEqual(result.norm, "paris")

    def test_unicode_table(self) -> None:
        table = [["城市", "得分"], ["北京", "九"], ["巴黎", "十"]]
        result = _engine().ask("哪个城市得分最高?", table)
        self.assertEqual(result.norm, "paris")  # FakeLLM always answers Paris

    def test_markdown_table_input(self) -> None:
        md = "| Year | Venue |\n|---|---|\n| 2003 | Paris |"
        result = _engine().ask("Where?", md)
        self.assertEqual(result.norm, "paris")

    def test_duplicate_headers(self) -> None:
        table = [["A", "A", "B"], ["1", "2", "3"]]
        result = _engine().ask("What is A?", table)
        self.assertEqual(result.norm, "paris")


class AnswerEdgeTests(unittest.TestCase):
    def test_empty_llm_answer(self) -> None:
        engine = JiuJiuBiao(llm=FakeLLM(rules=[], default=""))
        result = engine.ask("Where?", [["Venue"], ["Paris"]])
        self.assertEqual(result.norm, "")
        self.assertEqual(result.status, "incomplete")

    def test_refusal_answer(self) -> None:
        engine = JiuJiuBiao(llm=FakeLLM(rules=[(r".*", "Answer: cannot be determined")]))
        result = engine.ask("Where?", [["Venue"], ["Paris"]])
        self.assertEqual(result.norm, "")
        self.assertEqual(result.status, "incomplete")

    def test_answer_with_markdown_noise(self) -> None:
        engine = JiuJiuBiao(llm=FakeLLM(rules=[(r".*", "**Answer:** `Paris`.")]))
        result = engine.ask("Where?", [["Venue"], ["Paris"]])
        self.assertIn("paris", result.norm)


class RoutingTests(unittest.TestCase):
    def test_mode_boundary(self) -> None:
        engine = _engine(long_table_rows=5)
        small = [["A", "B"]] + [[str(i), str(i)] for i in range(4)]
        long = [["A", "B"]] + [[str(i), str(i)] for i in range(6)]
        self.assertEqual(engine.ask("q?", small).mode, "refine")
        self.assertEqual(engine.ask("q?", long).mode, "solve")

    def test_db_routes_to_solve(self) -> None:
        import pandas as pd

        from jiujiubiao.tools.db import Database

        db = Database.from_dataframe(pd.DataFrame({"a": [1]}))
        engine = _engine()
        result = engine.ask("how many rows?", db=db)
        self.assertEqual(result.mode, "solve")


if __name__ == "__main__":
    unittest.main(verbosity=2)
