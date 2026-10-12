# JiuJiuBiao 九九表

[![ci](https://github.com/ruochunjin/jiujiubiao/actions/workflows/ci.yml/badge.svg)](https://github.com/ruochunjin/jiujiubiao/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/jiujiubiao)](https://pypi.org/p/jiujiubiao)

> *三七——二十一。* — Every Chinese schoolchild learns the 九九表
> (multiplication table): a grid you interrogate and it answers, instantly.
> **JiuJiuBiao** does this for your data — one table, every question.

JiuJiuBiao is a **unified question-answering toolkit for tables and databases**.
One pipeline covers the full chain, each stage pluggable:

```text
question + table/database
  → [1] Disambiguation (optional, CLEAR-style): detect ambiguity, clarify, rewrite
  → [2] Solving — two interchangeable engines:
        • Solver: ReAct agent + self-consistency over a tool registry
          (long tables, messy formats, databases)           [TALON/RAISR family]
        • Refiner: init → self-criticism → multi-tool refinement →
          multi-tool consistency vote                        [TableCritic family]
  → [3] Output check: normalization, refusal detection, format gating
```

## Quickstart

```python
import jiujiubiao

engine = jiujiubiao.JiuJiuBiao()  # endpoint from env (see below)
result = engine.ask(
    "In which venue did the athlete win the first place?",
    table=[["Year", "Venue", "Position"],
           ["2001", "Edmonton", "3rd"],
           ["2003", "Paris", "1st"]],
)
print(result.answer)   # Paris
print(result.mode)     # "refine" (small table) — "solve" for long tables / DBs
```

Databases are first-class — the solver grounds itself with schema tools and
executes real SQL:

```python
from jiujiubiao import Database

db = Database.from_sqlite("concert_singer.db")   # or Database.from_dataframe(df)
result = engine.ask("How many singers are from France?", db=db)
```

**Offline development / tests** need no endpoint at all:

```python
from jiujiubiao import FakeLLM, JiuJiuBiao

llm = FakeLLM(rules=[(r"Now it's your turn", ["Answer: Paris"] * 3)])
engine = JiuJiuBiao(llm=llm, refiner_kwargs={"n_samples": 3})
```

## Endpoint configuration

No SDK dependency — plain HTTP to any OpenAI-compatible endpoint
(vLLM, Ollama, DeepSeek, OpenAI, ...) or an Anthropic-messages endpoint
(`JIUJIUBIAO_BACKEND=anthropic`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `JIUJIUBIAO_BACKEND` | `openai` | `openai` (chat/completions) or `anthropic` (messages API) |
| `JIUJIUBIAO_BASE_URL` / `OPENAI_BASE_URL` | `https://api.openai.com/v1` | endpoint |
| `JIUJIUBIAO_API_KEY` / `OPENAI_API_KEY` | — | API key |
| `JIUJIUBIAO_MODEL` / `OPENAI_MODEL` | — | model name (required) |
| `JIUJIUBIAO_TIMEOUT` / `JIUJIUBIAO_RETRIES` | `180` / `6` | request tuning (rate-limit-aware backoff) |

## Benchmarks (public, reproducible)

Seeded stratified samples, K3-class endpoint, `sc=3`, `mode=auto`:

| Benchmark | Sample | Accuracy | Notes |
| --- | --- | --- | --- |
| WikiTableQuestions (test) | 200 | **0.79** | larger sample; misses cluster on subtle counting and gold-format quirks |
| TabFact (validation) | 100 | **0.97** | balanced labels (51/49); misses are subtle "only X" claims |

Reproduce:

```bash
JIUJIUBIAO_BACKEND=anthropic python benchmarks/run_benchmark.py --benchmark wtq --n 100
```

## CLI

```bash
jiujiubiao --dataset data/examples/wtq.json --task tqa --limit 10 --output results.json
```

Dataset format: a JSON array of
`{"id", "table_caption", "table_text": [[header], [row]...], "question", "label"}`
(TabFact style: `statement` + boolean label).

## Extending

Register a solver tool and it joins the action space — prompt and dispatch
update automatically:

```python
from jiujiubiao import register_tool

@register_tool
def count_rows(ctx, arg):
    "count_rows() — Return the number of rows in the current table."
    return str(len(ctx.memory.get("df", ctx.df)))
```

Refine-stage tools (each with `generate` + `critic`) register via
`@register_refine_tool` in `jiujiubiao.agents.refine_tools`.

## Installation

```bash
pip install -e .            # pandas + cloudpickle only
pip install -e ".[fuzzy]"   # faster fuzzy matching (rapidfuzz)
pip install -e ".[dev]"     # tests + lint
```

## Training-side utilities

`jiujiubiao.training` carries the CORES composite process reward (format gate +
schema grounding + structure similarity + execution correctness against a
SQLite database), with everything injected — usable standalone or as a
verl/GRPO reward function:

```python
from jiujiubiao.training import compute_process_reward

reward = compute_process_reward(
    response, gold_sql, gold_python,
    schema={"results": ["venue", "wins"]}, db_path="data/results.db",
)
print(reward.total, reward.r_schema, reward.r_struct, reward.r_exec)
```

## Research lineage

JiuJiuBiao consolidates five peer-reviewed works from one lab:

- **CLEAR** (ICDE 2025) — parser-independent NL2SQL disambiguation → stage [1]
- **TALON** (EMNLP 2025) — multi-agent long-table exploration → solver
- **RAISR** (ICASSP 2026) — collaborative database agents, execution grounding,
  output checking → database tools, stage [3]
- **TableCritic** (CogSci 2025) — self-criticism + tool library → refiner
- **CORES** (ACL 2026 Findings) — code-oriented reasoning (Python as pivot);
  the `def()` code tool follows this paradigm

If you use JiuJiuBiao, please cite the relevant papers (see CITATION.cff):

```bibtex
@inproceedings{zhang2025clear,
  title={Clear: A parser-independent disambiguation framework for nl2sql},
  author={Zhang, Meng and Ma, Kexin and Xu, Liyang and Zhang, Kedi and Peng, Yuanxi and Jin, Ruochun},
  booktitle={2025 IEEE 41st International Conference on Data Engineering (ICDE)},
  pages={1--14},
  year={2025},
  organization={IEEE}
}

@inproceedings{jin2025talon,
  title={TALON: A multi-agent framework for long-table exploration and question answering},
  author={Jin, Ruochun and Wang, Xiyue and Wang, Dong and Zheng, Haoqi and Qi, Yunpeng and Yang, Silin and Zhang, Meng},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={27385--27401},
  year={2025}
}

@inproceedings{zhang2026reliable,
  title={Reliable Database Question Answering with Collaborative Agents},
  author={Zhang, Meng and Wang, Xiyue and Peng, Yuanxi and Yang, Wenjing and Yang, Silin and Jin, Ruochun},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={17142--17146},
  year={2026},
  organization={IEEE}
}

@inproceedings{jin2025tablecritic,
  title={TableCritic: Refine Table Reasoning via Self-Criticism and Tool Library},
  author={Jin, Ruochun and Wang, Dong and Wang, Xiyue and Zheng, Haoqi},
  booktitle={Proceedings of the Annual Meeting of the Cognitive Science Society},
  volume={47},
  year={2025}
}

@inproceedings{zhang2026cores,
  title={CORES: Code-Oriented Reasoning for Complex Text-to-SQL and Generalizable TableQA},
  author={Zhang, Meng and Jin, Ruochun and Peng, Yuanxi and Yang, Wenjing and Wang, Haotian and Sun, Liting and Hu, Kun and Yang, Silin and Ke-di, Zhang},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  pages={4827--4847},
  year={2026}
}
```

## License

Apache-2.0. Ported modules retain their original notices (solver prompts and
execution sandbox: Apache-2.0, Google Research; refiner: CC-BY, TableCritic).
