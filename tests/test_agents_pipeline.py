"""Offline tests: solver loop, refiner loop, pipeline orchestration.

Everything runs on FakeLLM — no endpoint, no network.  The sandbox executes
real Python via subprocess (cloudpickle); no model is involved.
"""

from __future__ import annotations

import unittest

from jiujiubiao import FakeLLM, JiuJiuBiao
from jiujiubiao.agents.refiner import Refiner
from jiujiubiao.agents.solver import Solver
from jiujiubiao.execution import python_repl_ast
from jiujiubiao.table import Table
from jiujiubiao.tools.db import Database

TABLE = Table(
    ["Year", "Venue", "Position"],
    [["2001", "Edmonton", "3rd"], ["2003", "Paris", "1st"], ["2005", "Helsinki", "5th"]],
)


class ExecutionTests(unittest.TestCase):
    def test_sandbox_executes_and_returns_state(self) -> None:
        df = TABLE.to_df()
        observation, memory, success = python_repl_ast(
            "print(df['Venue'].tolist()[1])",
            custom_locals={"df": df},
            custom_globals={},
            memory={"df": df},
        )
        self.assertEqual(success, 1)
        self.assertIn("Paris", observation)

    def test_sandbox_error_preserves_state(self) -> None:
        df = TABLE.to_df()
        observation, memory, success = python_repl_ast(
            "raise ValueError('boom')",
            custom_locals={"df": df},
            custom_globals={},
            memory={"df": df},
        )
        self.assertEqual(success, 0)
        self.assertIn("ValueError", observation)


class SolverTests(unittest.TestCase):
    def test_solve_with_scripted_llm(self) -> None:
        llm = FakeLLM(
            rules=[
                # Step 2 (after the tool observation; note the template itself
                # mentions "Observation:", so anchor on real observation text).
                (r"The columns below contain the value", "Thought: Paris won in 2003\nFinal Answer: Paris"),
                # Step 1: look up the venue value.
                (r"Begin!.*Thought: $", "Thought: find the winning venue\nAction: get_value('Paris')"),
            ]
        )
        solver = Solver(llm, sc=1)
        result = solver.solve(TABLE.to_df(), "In which venue did the athlete win?")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, "Paris")
        self.assertEqual(result.tool_counts["get_value"], 1)

    def test_solver_incomplete_when_never_terminal(self) -> None:
        llm = FakeLLM(default="Thought: keep looking\nAction: get_row(0)")
        solver = Solver(llm, sc=1, max_depth=2, use_critic=False)
        result = solver.solve(TABLE.to_df(), "anything")
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.answer, "")

    def test_solver_rejects_bad_config(self) -> None:
        with self.assertRaises(ValueError):
            Solver(FakeLLM(), sc=0)


class RefinerTests(unittest.TestCase):
    def test_init_answer_accepted_when_critic_says_no(self) -> None:
        llm = FakeLLM(
            rules=[
                (r"strict reviewer", "Need refinement: No\nSuggestions: None"),
                (r"Now it's your turn", ["Let's think. Row 1 has Position 1st.\nAnswer: Paris"] * 3),
            ]
        )
        refiner = Refiner(llm, n_samples=3)
        result = refiner.run(TABLE, "In which venue did the athlete win the first place?")
        self.assertEqual(result.norm, "paris")
        self.assertEqual(result.stage, "init")
        self.assertEqual(result.iterations, 0)

    def test_refine_round_uses_python_tool(self) -> None:
        code = "```python\nanswer = int(df['Quantity'].iloc[0]) * int(df['Unit Cost'].iloc[0])\n```"
        table = Table(["Item", "Unit Cost", "Quantity"], [["Laptops", "700", "5"]])
        llm = FakeLLM(
            rules=[
                (r"strict code reviewer", "Need refinement: No\nSuggestions: None"),
                (r"expert Python data analyst", [code] * 2),
                (r"strict reviewer", "Need refinement: Yes\nSuggestions: compute it with code"),
                (r"Now it's your turn", ["The cost is 700*5.\nAnswer: 3500"] * 3),
            ]
        )
        refiner = Refiner(llm, n_samples=3, tools=["python"])
        result = refiner.run(table, "What is the total cost?")
        self.assertEqual(result.norm, "3500")
        self.assertTrue(result.history[-1]["stage"].startswith("refine"))


class PipelineTests(unittest.TestCase):
    def test_small_table_routes_to_refiner(self) -> None:
        llm = FakeLLM(
            rules=[
                (r"strict reviewer", "Need refinement: No\nSuggestions: None"),
                (r"Now it's your turn", ["Row 1 is the 1st place.\nAnswer: Paris"] * 3),
            ]
        )
        result = JiuJiuBiao(llm=llm, task="tqa", refiner_kwargs={"n_samples": 3}).ask(
            "In which venue did the athlete win the first place?", TABLE
        )
        self.assertEqual(result.mode, "refine")
        self.assertEqual(result.norm, "paris")

    def test_database_routes_to_solver(self) -> None:
        db = Database.from_dataframe(TABLE.to_df(), table_name="results")
        llm = FakeLLM(
            rules=[
                (r"Thought: $", "Thought: query the table\nFinal Answer: Paris"),
            ]
        )
        result = JiuJiuBiao(llm=llm, solver_kwargs={"sc": 1}).ask(
            "Which venue is in the table?", db=db
        )
        self.assertEqual(result.mode, "solve")
        self.assertEqual(result.answer, "Paris")

    def test_disambiguation_rewrites_question(self) -> None:
        llm = FakeLLM(
            rules=[
                (r"strict reviewer", "Need refinement: No\nSuggestions: None"),
                (r"unambiguous enough", "Ambiguous: Yes\nClarification question: Which year?"),
                (r"Rewrite the question", "In which venue did the athlete win in 2003?"),
                (r"Now it's your turn", ["Answer: Paris"] * 3),
            ]
        )
        engine = JiuJiuBiao(
            llm=llm, disambiguate=True, clarify=lambda q: "The 2003 edition.",
            refiner_kwargs={"n_samples": 3},
        )
        result = engine.ask("In which venue did the athlete win?", TABLE)
        self.assertIsNotNone(result.disambiguation)
        self.assertTrue(result.disambiguation.ambiguous)
        self.assertEqual(result.question, "In which venue did the athlete win in 2003?")
        self.assertEqual(result.norm, "paris")

    def test_disambiguation_passthrough_without_clarify(self) -> None:
        llm = FakeLLM(
            rules=[
                (r"strict reviewer", "Need refinement: No"),
                (r"unambiguous enough", "Ambiguous: Yes\nClarification question: Which year?"),
                (r"Now it's your turn", ["Answer: Paris"] * 3),
            ]
        )
        engine = JiuJiuBiao(llm=llm, disambiguate=True, refiner_kwargs={"n_samples": 3})
        result = engine.ask("In which venue did the athlete win?", TABLE)
        self.assertTrue(result.disambiguation.ambiguous)
        self.assertEqual(result.question, "In which venue did the athlete win?")


if __name__ == "__main__":
    unittest.main(verbosity=2)
