"""Unit tests for the answer-type consistency gate."""

from __future__ import annotations

import unittest

from xuanji import FakeLLM, XuanJi
from xuanji.pipeline import _expected_type, _is_bare_number, _type_mismatch_hint

_TABLE = [
    ["Team", "Goals"],
    ["San Marino", "1"],
    ["Raul United", "9"],
    ["Freedonia", "4"],
]


class TypeGateHelperTests(unittest.TestCase):
    def test_expected_type(self) -> None:
        self.assertEqual(_expected_type("who scored the least goals?"), "entity")
        self.assertEqual(_expected_type("Whose team won?"), "entity")
        self.assertEqual(_expected_type("how many models use diesel?"), "number")
        self.assertIsNone(_expected_type("what year did they win?"))

    def test_bare_number(self) -> None:
        self.assertTrue(_is_bare_number("1,234"))
        self.assertFalse(_is_bare_number("8 years"))

    def test_hint(self) -> None:
        self.assertIsNotNone(_type_mismatch_hint("who scored the least?", "8"))
        self.assertIsNone(_type_mismatch_hint("who scored the least?", "San Marino"))
        self.assertIsNotNone(_type_mismatch_hint("how many models?", "many models"))
        self.assertIsNone(_type_mismatch_hint("how many models?", "110"))


class TypeGateFlowTests(unittest.TestCase):
    def test_who_question_retries_numeric_answer(self) -> None:
        llm = FakeLLM(
            rules=[
                # First pass (no hint): engine returns a bare number.
                (r"^(?!.*not a bare number).*$", "Answer: 8"),
                # Guided retry (hint present): returns the entity.
                (r"not a bare number", "Answer: San Marino"),
            ]
        )
        engine = XuanJi(llm=llm, refiner_kwargs={"n_samples": 3})
        result = engine.ask("who scored the least international goals?", table=_TABLE)
        self.assertTrue(result.type_retry)
        self.assertEqual(result.norm, "san marino")

    def test_correct_first_answer_not_retried(self) -> None:
        llm = FakeLLM(rules=[(r".*", "Answer: San Marino")])
        engine = XuanJi(llm=llm, refiner_kwargs={"n_samples": 3})
        result = engine.ask("who scored the least international goals?", table=_TABLE)
        self.assertFalse(result.type_retry)
        self.assertEqual(result.norm, "san marino")

    def test_retry_rejected_if_still_mismatched(self) -> None:
        llm = FakeLLM(rules=[(r".*", "Answer: 8")])
        engine = XuanJi(llm=llm, refiner_kwargs={"n_samples": 3})
        result = engine.ask("who scored the least international goals?", table=_TABLE)
        # Retry happened but still numeric -> keep the FIRST outcome.
        self.assertFalse(result.type_retry)
        self.assertEqual(result.norm, "8")

    def test_number_question_no_entity_retry(self) -> None:
        llm = FakeLLM(rules=[(r".*", "Answer: 110")])
        engine = XuanJi(llm=llm, refiner_kwargs={"n_samples": 3})
        result = engine.ask("how many models utilize diesel?", table=_TABLE)
        self.assertFalse(result.type_retry)


if __name__ == "__main__":
    unittest.main()
