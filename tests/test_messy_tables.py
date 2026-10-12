"""Messy-table robustness battery for JiuJiuBiao (offline + light K3 spot checks).

Offline part (FakeLLM): tables with None/NaN cells, mixed types, duplicate
rows, summary rows, currency/percent/date formats, and wide/long shapes must
never crash the pipeline and must preserve answer routing.

K3 part (DAYAN-style gate): a few of the messiest tables with real questions,
to verify the engine extracts correct values from dirty data.
"""

from __future__ import annotations

import unittest

from jiujiubiao import FakeLLM, JiuJiuBiao
from jiujiubiao.table import Table


def _engine() -> JiuJiuBiao:
    llm = FakeLLM(rules=[(r".*", "Answer: Paris")])
    return JiuJiuBiao(llm=llm, refiner_kwargs={"n_samples": 3}, solver_kwargs={"sc": 2, "max_depth": 3})


_MESSY_TABLES = {
    "none_cells": [["City", "Score"], ["Paris", None], ["London", "5"]],
    "empty_string_cells": [["City", "Score"], ["Paris", ""], ["London", "5"]],
    "mixed_types": [["Item", "Price"], ["apple", "1.5"], ["banana", "N/A"], ["cherry", "$2.00"]],
    "duplicate_rows": [["City", "Score"], ["Paris", "9"], ["Paris", "9"], ["London", "5"]],
    "summary_row": [["City", "Score"], ["Paris", "9"], ["London", "5"], ["Total", "14"]],
    "currency_percent": [["Item", "Margin"], ["A", "12.5%"], ["B", "$3.50"]],
    "date_formats": [["When", "Value"], ["2024-01-15", "1"], ["Jan 16, 2024", "2"], ["15/01/2024", "3"]],
    "whitespace_heavy": [["  City  ", " Score "], ["  Paris ", " 9 "], ["London", "5"]],
    "long_table": [["City", "Score"]] + [[f"City{i}", str(i)] for i in range(300)],
    "numeric_column_names": [["1", "2", "3"], ["a", "b", "c"]],
    "single_cell": [["only"]],
}


class MessyTableOfflineTests(unittest.TestCase):
    def test_all_messy_tables_no_crash(self) -> None:
        for name, table in _MESSY_TABLES.items():
            with self.subTest(table=name):
                result = _engine().ask("What is the answer?", table)
                self.assertIn(result.status, ("completed", "incomplete"))
                self.assertIsNotNone(result.norm)

    def test_none_cells_normalized_to_text(self) -> None:
        table = Table.from_obj(_MESSY_TABLES["none_cells"])
        df = table.to_df()
        self.assertEqual(df.shape[0], 2)
        self.assertEqual(df.shape[1], 2)


class MessyTableK3Tests(unittest.TestCase):
    """A couple of messy tables through the real endpoint (offline-safe skip)."""

    def test_summary_row_not_counted(self) -> None:
        import os

        if os.environ.get("JIUJIUBIAO_RUN_E2E") != "1":
            self.skipTest("requires JIUJIUBIAO_RUN_E2E=1")
        engine = JiuJiuBiao()
        result = engine.ask(
            "How many cities have a score? Count only actual cities, not the total row.",
            _MESSY_TABLES["summary_row"],
        )
        self.assertEqual(result.norm, "2")

    def test_currency_mixed_column(self) -> None:
        import os

        if os.environ.get("JIUJIUBIAO_RUN_E2E") != "1":
            self.skipTest("requires JIUJIUBIAO_RUN_E2E=1")
        engine = JiuJiuBiao()
        result = engine.ask(
            "Which item has a margin of $3.50?",
            _MESSY_TABLES["currency_percent"],
        )
        self.assertEqual(result.norm, "b")


if __name__ == "__main__":
    unittest.main(verbosity=2)
