"""Offline quickstart: the full pipeline on a scripted FakeLLM.

Run from the repository root:  python examples/quickstart_offline.py

With a real endpoint, drop FakeLLM and set XUANJI_BASE_URL / XUANJI_MODEL /
XUANJI_API_KEY — the pipeline code is identical.
"""

import xuanji
from xuanji import FakeLLM

TABLE = [
    ["Year", "Venue", "Position"],
    ["2001", "Edmonton", "3rd"],
    ["2003", "Paris", "1st"],
    ["2005", "Helsinki", "5th"],
]

# The FakeLLM plays both roles: initial solver and strict reviewer.
llm = FakeLLM(
    rules=[
        (r"strict reviewer", "Need refinement: No\nSuggestions: None"),
        (
            r"Now it's your turn",
            [
                "Let's think step by step. Position '1st' is in row 1; its venue is Paris.\nAnswer: Paris",
                "Row with 1st place is 2003 Paris.\nAnswer: Paris",
                "The first place was in Paris, 2003.\nAnswer: Paris",
            ],
        ),
    ]
)

engine = xuanji.XuanJi(llm=llm, refiner_kwargs={"n_samples": 3})
result = engine.ask("In which venue did the athlete win the first place?", TABLE)

print("mode:   ", result.mode)
print("answer: ", result.answer)
print("norm:   ", result.norm)
print("status: ", result.status)
