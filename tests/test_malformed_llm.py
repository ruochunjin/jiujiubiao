"""Malformed-LLM-output robustness battery (offline, FakeLLM).

The pipeline must never crash on degenerate model output: empty strings,
refusals, missing answer markers, duplicated markers, huge blobs, unicode
junk, or marker-only responses.  Every case must end in a defined status
(``completed``/``incomplete``) with ``norm`` either a string or None.
"""

from __future__ import annotations

import unittest

from jiujiubiao import FakeLLM, JiuJiuBiao

_TABLE = [
    ["City", "Score"],
    ["Paris", "9"],
    ["London", "5"],
]

_MALFORMED = {
    "empty": "",
    "whitespace_only": "   \n\t  ",
    "refusal": "I'm sorry, I cannot answer this question.",
    "no_marker": "The answer is obviously Paris, trust me.",
    "marker_only": "Answer:",
    "duplicated_markers": "Answer: Paris\nAnswer: London\nAnswer: 9",
    "huge_blob": "thinking... " * 5000 + "Answer: Paris",
    "unicode_junk": "答：巴黎 🗼\x00\x01� Answer: Paris",
    "markdown_table_dump": "| City | Score |\n|---|---|\n| Paris | 9 |\n" * 50,
    "answer_with_quotes": 'Answer: "Paris"',
}


class MalformedOutputTests(unittest.TestCase):
    def test_never_crashes_defined_status(self) -> None:
        for name, blob in _MALFORMED.items():
            with self.subTest(output=name):
                llm = FakeLLM(rules=[(r".*", blob)])
                engine = JiuJiuBiao(llm=llm, refiner_kwargs={"n_samples": 3},
                                    solver_kwargs={"sc": 2, "max_depth": 3})
                result = engine.ask("Which city has score 9?", _TABLE)
                self.assertIn(result.status, ("completed", "incomplete"))
                self.assertTrue(result.norm is None or isinstance(result.norm, str))

    def test_garbage_does_not_hang(self) -> None:
        """A pathological blob must still terminate within the step budget."""

        import time

        llm = FakeLLM(rules=[(r".*", _MALFORMED["huge_blob"])])
        engine = JiuJiuBiao(llm=llm, refiner_kwargs={"n_samples": 2},
                            solver_kwargs={"sc": 1, "max_depth": 2})
        start = time.perf_counter()
        engine.ask("Which city has score 9?", _TABLE)
        self.assertLess(time.perf_counter() - start, 30.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
