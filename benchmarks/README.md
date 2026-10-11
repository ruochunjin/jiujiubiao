# Benchmarks

JiuJiuBiao ships with loaders for the lab's benchmark assets.  The datasets
themselves are **not vendored** into this repository (size + licensing);
use the links below and point the CLI at the downloaded files.

| Benchmark | From | Content | Link |
| --- | --- | --- | --- |
| **CLAMBSQL** | CLEAR (ICDE 2025) | NL2SQL ambiguity benchmark with fine-grained ambiguity and clarification annotations | [github.com/mengzhang18/CLEAR](https://github.com/mengzhang18/CLEAR) |
| **WTQ-L** | TALON (EMNLP 2025) | 341 long tables (50–500+ rows) derived from WikiTableQuestions | [download](https://drive.google.com/drive/folders/1tWa79UzwV-e9-vvyIp23d5uav3koyx-h) |
| **BirdQA** | TALON (EMNLP 2025) | 314 single-table QA over BIRD-SQL tables (avg. 44k rows) | same drive folder |
| **ProcQA** | CORES (ACL 2026 Findings) | 12k execution-verified Python–SQL aligned complex-query pairs for training | [github.com/mengzhang18/CORES](https://github.com/mengzhang18/CORES) |

## Format

The CLI accepts the common JSON-array format (also used by WTQ-L/BirdQA):

```json
[{
  "id": "example-1",
  "table_caption": "Scores",
  "table_text": [["Name", "Points"], ["Ada", "10"], ["Lin", "20"]],
  "question": "How many points does Ada have?",
  "label": "10"
}]
```

- TabFact-style files use `statement` with a boolean label.
- BirdQA `table_text` may be a path to a JSON table file, resolved relative
  to the dataset file.

A tiny synthetic example set is at `examples/data/wtq_tiny.json`.
