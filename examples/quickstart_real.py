"""Real-endpoint quickstart (needs XUANJI_BASE_URL / XUANJI_MODEL / XUANJI_API_KEY).

Run from the repository root:  python examples/quickstart_real.py
"""

import xuanji
from xuanji import Database

# 1. Small table -> refiner path (self-criticism + tool library)
TABLE = [
    ["Year", "Venue", "Position"],
    ["2001", "Edmonton", "3rd"],
    ["2003", "Paris", "1st"],
    ["2005", "Helsinki", "5th"],
]
engine = xuanji.XuanJi()
result = engine.ask("In which venue did the athlete win the first place?", TABLE)
print("refine:", result.answer)

# 2. Long table -> solver path (agentic tool exploration, self-consistency)
LONG_TABLE = [["City", "Population"]] + [[f"City{i}", str(1000 * i)] for i in range(200)]
result = engine.ask("What is the population of City42?", LONG_TABLE)
print("solve:", result.answer, f"({result.steps} steps)")

# 3. Database -> solver with SQL tools
import pandas as pd

db = Database.from_dataframe(pd.DataFrame({"venue": ["Paris", "London"], "wins": [5, 3]}), "results")
result = engine.ask("How many wins does Paris have?", db=db)
print("db:", result.answer)
