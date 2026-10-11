"""Unit tests for v0.3 additions: answer-set scoring, CSV table loading."""

from __future__ import annotations

import os
import tempfile
import unittest

from jiujiubiao.eval import denotation_accuracy
from jiujiubiao.table import Table


class DenotationAccuracyTests(unittest.TestCase):
    def test_multi_answer_set_equality(self) -> None:
        # nu-3074: case/order must not matter for multi-part answers.
        self.assertEqual(
            denotation_accuracy(
                ["surround stakes | warwick stakes"],
                [["Surround Stakes", "Warwick Stakes"]],
            ),
            1.0,
        )

    def test_extra_wrong_part_misses(self) -> None:
        # nu-4111: an extra incorrect part must NOT match a singleton gold.
        self.assertEqual(
            denotation_accuracy(["john briggs | kerry baily"], [["Kerry Baily"]]),
            0.0,
        )

    def test_single_part_prediction_in_alternatives(self) -> None:
        self.assertEqual(
            denotation_accuracy(["paris"], [["Paris", "Lyon"]]),
            1.0,
        )

    def test_numeric_normalization(self) -> None:
        self.assertEqual(denotation_accuracy(["1,200"], [["1200"]]), 1.0)

    def test_tfv_bool(self) -> None:
        self.assertEqual(
            denotation_accuracy(["yes"], [["True"]], task="tfv"), 1.0
        )
        self.assertEqual(
            denotation_accuracy(["no"], [["True"]], task="tfv"), 0.0
        )


class TableInputTests(unittest.TestCase):
    def test_from_csv_path_dispatch(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as handle:
            handle.write("Name,Points\nAda,10\nLin,20\n")
            path = handle.name
        try:
            table = Table.from_obj(path)
            self.assertEqual(table.header, ["Name", "Points"])
            self.assertEqual(table.rows, [["Ada", "10"], ["Lin", "20"]])
        finally:
            os.unlink(path)

    def test_text_content_not_confused_with_path(self) -> None:
        table = Table.from_obj("Name | Points\nAda | 10")
        self.assertEqual(table.header, ["Name", "Points"])
        self.assertEqual(table.rows, [["Ada", "10"]])

    def test_tsv(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as handle:
            handle.write("A\tB\n1\t2\n")
            path = handle.name
        try:
            table = Table.from_csv(path)
            self.assertEqual(table.rows, [["1", "2"]])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
