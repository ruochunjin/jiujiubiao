"""Tests for the CORES-style process reward (offline, in-memory fixtures)."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest

from xuanjitu.training.reward import (
    compute_process_reward,
    parse_sections,
    schema_reward,
    structure_reward,
)

SCHEMA = {"results": ["venue", "wins"]}

GOLD_SQL = "SELECT venue FROM results WHERE wins > 4"
GOLD_PYTHON = "result = sql('SELECT venue FROM results WHERE wins > 4')"

GOOD_RESPONSE = f"""<nl_reasoning>Find the venue in the results table whose wins exceed 4.</nl_reasoning>
<python_reasoning>```python
{GOLD_PYTHON}
```</python_reasoning>
<sql>```sql
{GOLD_SQL}
```</sql>"""


def make_db() -> str:
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    conn = sqlite3.connect(handle.name)
    conn.execute("CREATE TABLE results (venue TEXT, wins INTEGER)")
    conn.executemany("INSERT INTO results VALUES (?, ?)", [("Paris", 5), ("London", 3)])
    conn.commit()
    conn.close()
    return handle.name


class ParseTests(unittest.TestCase):
    def test_parse_sections(self) -> None:
        sections = parse_sections(GOOD_RESPONSE)
        self.assertIsNotNone(sections)
        nl, py, sql = sections
        self.assertIn("venue", nl)
        self.assertIn("sql(", py)
        self.assertIn("SELECT venue", sql)

    def test_missing_section_fails_format_gate(self) -> None:
        self.assertIsNone(parse_sections("<nl_reasoning>only</nl_reasoning>"))
        reward = compute_process_reward(
            "no sections", GOLD_SQL, GOLD_PYTHON, SCHEMA, ":memory:"
        )
        self.assertFalse(reward.format_ok)
        self.assertEqual(reward.total, 0.0)


class RewardDimensionTests(unittest.TestCase):
    def test_schema_reward_perfect_match(self) -> None:
        score = schema_reward(
            "venue from results", GOLD_PYTHON, GOLD_SQL, GOLD_SQL, SCHEMA
        )
        self.assertGreater(score, 2.0)  # close to the 2.5 maximum

    def test_schema_reward_no_overlap(self) -> None:
        score = schema_reward("nothing relevant", "x = 1", "SELECT 1", GOLD_SQL, SCHEMA)
        self.assertLess(score, 0.5)

    def test_structure_reward_identical(self) -> None:
        score = structure_reward(GOLD_PYTHON, GOLD_PYTHON, GOLD_SQL, GOLD_SQL)
        self.assertAlmostEqual(score, 2.0)

    def test_structure_reward_different(self) -> None:
        score = structure_reward("import os", GOLD_PYTHON, "DELETE FROM x", GOLD_SQL)
        self.assertLess(score, 1.5)


class CompositeTests(unittest.TestCase):
    def test_perfect_response_scores_high(self) -> None:
        reward = compute_process_reward(
            GOOD_RESPONSE, GOLD_SQL, GOLD_PYTHON, SCHEMA, make_db()
        )
        self.assertTrue(reward.format_ok)
        self.assertGreater(reward.r_exec, 2.0)  # both run, both match gold
        self.assertGreater(reward.total, 0.6)
        self.assertLessEqual(reward.total, 1.0)

    def test_wrong_sql_lowers_execution_reward(self) -> None:
        bad = GOOD_RESPONSE.replace(
            "SELECT venue FROM results WHERE wins > 4",
            "SELECT venue FROM results WHERE wins > 100",
        ).replace(
            "sql('SELECT venue FROM results WHERE wins > 4')",
            "sql('SELECT venue FROM results WHERE wins > 100')",
            # NL text intentionally left intact: it carries the true schema intent
        )
        reward = compute_process_reward(bad, GOLD_SQL, GOLD_PYTHON, SCHEMA, make_db())
        perfect = compute_process_reward(GOOD_RESPONSE, GOLD_SQL, GOLD_PYTHON, SCHEMA, make_db())
        self.assertLess(reward.r_exec, perfect.r_exec)
        self.assertLess(reward.total, perfect.total)


if __name__ == "__main__":
    unittest.main(verbosity=2)
