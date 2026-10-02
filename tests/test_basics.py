"""Offline tests: Table, normalization, tools, database backend."""

from __future__ import annotations

import unittest

import pandas as pd

from xuanji import FakeLLM
from xuanji.normalize import (
    extract_answer,
    extract_code,
    extract_json,
    majority_vote,
    normalize,
    parse_critic,
)
from xuanji.registry import DEFAULT_REGISTRY, ToolContext
from xuanji.table import Table
from xuanji.tools.db import Database

TABLE = Table(
    ["Year", "Venue", "Position"],
    [["2001", "Edmonton", "3rd"], ["2003", "Paris", "1st"], ["2005", "Helsinki", "5th"]],
    title="Results",
)


def make_ctx(df=None, db=None, llm=None, question="q", memory=None):
    return ToolContext(
        df=df, llm=llm or FakeLLM(), question=question,
        memory={"df": df} if memory is None and df is not None else (memory or {}),
        db=db,
    )


class TableTests(unittest.TestCase):
    def test_from_obj_and_markdown(self) -> None:
        table = Table.from_obj([["A", "B"], ["1", "2"]])
        self.assertEqual(table.header, ["A", "B"])
        md = table.to_markdown()
        self.assertIn("| row_id | A | B |", md)
        self.assertIn("| 0 | 1 | 2 |", md)

    def test_from_dataframe_and_text(self) -> None:
        table = Table.from_obj(pd.DataFrame({"x": [1, 2]}))
        self.assertEqual(len(table), 2)
        parsed = Table.from_text("a | b\n1 | 2\n3 | 4")
        self.assertEqual(parsed.rows, [["1", "2"], ["3", "4"]])

    def test_sub_table_keeps_original_ids(self) -> None:
        sub = TABLE.sub_table(columns=["Venue"], row_ids=[1])
        self.assertEqual(sub.header, ["Venue"])
        self.assertEqual(sub.rows, [["Paris"]])


class NormalizeTests(unittest.TestCase):
    def test_extract_and_normalize(self) -> None:
        self.assertEqual(extract_answer("reasoning...\nAnswer: Paris"), "Paris")
        self.assertEqual(normalize("The Paris.", "tqa"), "paris")
        self.assertEqual(normalize("1,200", "tqa"), "1200")
        self.assertEqual(normalize(["b", "a"], "tqa"), "a | b")
        self.assertEqual(normalize("entailed", "tfv"), "True")
        self.assertEqual(normalize("REFUTED", "tfv"), "False")

    def test_majority_vote_and_critic_parse(self) -> None:
        answer, payload, counts = majority_vote([("a", 1), ("b", 2), ("a", 3)])
        self.assertEqual(answer, "a")
        need, sug = parse_critic("Need refinement: Yes\nSuggestions: check the sum")
        self.assertTrue(need)
        self.assertIn("sum", sug)
        need, _ = parse_critic("Need refinement: No\nSuggestions: None")
        self.assertFalse(need)

    def test_extract_code_and_json(self) -> None:
        self.assertEqual(extract_code("text\n```python\nx = 1\n```"), "x = 1")
        self.assertEqual(extract_json('here {"columns": ["A"]}'), {"columns": ["A"]})


class TableToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = TABLE.to_df()

    def test_get_value_finds_cell(self) -> None:
        out = DEFAULT_REGISTRY.get("get_value").run(make_ctx(self.df), "Paris")
        self.assertIn("Venue", out)
        self.assertIn("row_indices", out)

    def test_get_value_reports_missing(self) -> None:
        out = DEFAULT_REGISTRY.get("get_value").run(make_ctx(self.df), "Nowhere")
        self.assertIn("No columns contain", out)

    def test_get_row(self) -> None:
        out = DEFAULT_REGISTRY.get("get_row").run(make_ctx(self.df), "1")
        self.assertIn("Paris", out)
        self.assertIn("out of range", DEFAULT_REGISTRY.get("get_row").run(make_ctx(self.df), "99"))

    def test_fuzzy_search(self) -> None:
        out = DEFAULT_REGISTRY.get("find_columns_containing_value_fuzzy").run(
            make_ctx(self.df), "paris france"
        )
        self.assertIn("Paris", out)


class DatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = Database.from_dataframe(
            pd.DataFrame({"venue": ["Paris", "London"], "wins": [5, 3]}), table_name="results"
        )

    def test_execute_sql(self) -> None:
        out = self.db.execute("SELECT venue FROM results WHERE wins > 4")
        self.assertIn("Paris", out)

    def test_sql_error_is_returned_not_raised(self) -> None:
        self.assertIn("SQL Error", self.db.execute("SELECT * FROM nowhere"))

    def test_schema_and_value_tools(self) -> None:
        ctx = make_ctx(db=self.db)
        self.assertIn("results", self.db.schema_text())
        fuzzy = DEFAULT_REGISTRY.get("find_schema_fuzzy").run(ctx, "Paris")
        self.assertIn("venue", fuzzy)
        check = DEFAULT_REGISTRY.get("is_value_in_column").run(ctx, "results.venue=Paris")
        self.assertIn("found", check)
        info = DEFAULT_REGISTRY.get("get_column_info").run(ctx, "results.wins")
        self.assertIn("INTEGER", info.upper())


if __name__ == "__main__":
    unittest.main(verbosity=2)
