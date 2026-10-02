"""Iterative self-criticism refiner (TableCritic Algorithm 1, CC-BY adapted).

Initialization → criticism → multi-tool refinement → multi-tool consistency
vote, iterated until the critic is satisfied (or the answer stabilizes).
Tools come from ``REFINE_TOOL_LIBRARY``; new ones register with
``@register_refine_tool``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from xuanji.agents.refine_tools import REFINE_TOOL_LIBRARY, Candidate, RefineContext
from xuanji.normalize import extract_answer, majority_vote, normalize, parse_critic
from xuanji.prompts import refine as P
from xuanji.table import Table

DEFAULT_TOOLSETS = {
    "tqa": ["python", "subtable"],
    "tfv": ["python", "subtable"],
}


@dataclass
class RefineResult:
    answer: str
    norm: str
    stage: str  # 'init' | 'refine-k'
    iterations: int
    history: list = field(default_factory=list)


class Refiner:
    """Self-criticism + tool-library refinement loop."""

    def __init__(
        self,
        llm,
        tools=None,
        n_samples: int = 5,
        max_iter: int = 3,
        use_critic: bool = True,
        max_table_rows: int | None = None,
        verbose: bool = False,
        vote_with_init: bool = False,
    ) -> None:
        self.llm = llm
        self.n = n_samples
        self.max_iter = max_iter
        self.use_critic = use_critic
        self.max_table_rows = max_table_rows
        self.verbose = verbose
        self.vote_with_init = vote_with_init
        self.toolsets = (
            tools
            if isinstance(tools, dict)
            else ({"tqa": tools, "tfv": tools} if tools is not None else DEFAULT_TOOLSETS)
        )
        self.pool = ThreadPoolExecutor(max_workers=4)

    def log(self, *args):
        if self.verbose:
            print(*args, flush=True)

    # ------------------------------------------------------------------ init
    def initialize(self, ctx: RefineContext):
        prompt = P.INIT.format(
            fewshot=P.FEWSHOT_INIT[ctx.task],
            table=ctx.table.to_markdown(self.max_table_rows),
            **ctx.fmt,
        )
        outs = self.llm.sample(prompt, self.n)
        cands = [
            Candidate("cot", a := extract_answer(o), normalize(a, ctx.task), o) for o in outs
        ]
        ctx.init_cands = cands
        return majority_vote([(c.norm, c) for c in cands])

    def critic_init(self, ctx, cand):
        prompt = P.CRITIC_INIT.format(
            table=ctx.table.to_markdown(self.max_table_rows),
            reasoning=cand.trace,
            answer=cand.answer or "None",
            **ctx.fmt,
        )
        out = self.llm.chat(prompt, temperature=0.0)
        need, sug = parse_critic(out)
        if not cand.norm:
            need = True
        return need, sug, out

    # ------------------------------------------------------------------ main
    def run(self, table, question: str, task: str = "tqa") -> RefineResult:
        assert task in ("tqa", "tfv")
        table = Table.from_obj(table)
        ctx = RefineContext(table, question, task)
        tools = [
            REFINE_TOOL_LIBRARY[t](self.llm, n=self.n, max_table_rows=self.max_table_rows)
            for t in self.toolsets[task]
        ]
        history = []

        # (1) Initialization
        y0, rep0, cnt0 = self.initialize(ctx)
        self.log(f"[init] answer={y0!r} votes={dict(cnt0)}")
        history.append(
            {"stage": "init", "answer": y0, "votes": dict(cnt0),
             "reasoning": rep0.trace if rep0 else ""}
        )

        # (2) Criticism on the initial answer
        if self.use_critic:
            need, sug, ctext = self.critic_init(ctx, rep0)
            history[-1]["critic"] = ctext
            self.log(f"[critic-init] need_refinement={need}")
            if not need:
                return RefineResult(rep0.answer, y0, "init", 0, history)
        else:
            sug = ""

        feedback = {t.name: None for t in tools}
        init_hint = (
            {"previous": f"(chain-of-thought) answer: {rep0.answer if rep0 else ''}",
             "suggestions": sug}
            if sug
            else None
        )
        prev_answer, y_t, y_rep, stage = y0, y0, rep0, "init"
        unchanged = 0

        for it in range(self.max_iter):
            # (3) Refinement: every tool yields n candidates
            futs = [self.pool.submit(t.generate, ctx, feedback[t.name] or init_hint) for t in tools]
            per_tool = {t.name: f.result() for t, f in zip(tools, futs)}
            all_cands = [c for t in tools for c in per_tool[t.name]]
            if self.vote_with_init:
                all_cands += getattr(ctx, "init_cands", [])

            # Multi-tool consistency vote
            y_new, rep, cnt = majority_vote([(c.norm, c) for c in all_cands])
            if not y_new:  # every tool failed -> fall back to the previous answer
                y_new, rep = y_t, y_rep
            self.log(f"[refine-{it + 1}] answer={y_new!r} votes={dict(cnt)}")
            rec = {
                "stage": f"refine-{it + 1}", "answer": y_new, "votes": dict(cnt),
                "winner_tool": rep.tool if rep else None,
                "per_tool": {k: [c.norm for c in v] for k, v in per_tool.items()},
            }
            history.append(rec)

            unchanged = unchanged + 1 if y_new == prev_answer else 0
            prev_answer, y_t, y_rep, stage = y_new, y_new, rep, f"refine-{it + 1}"

            if not self.use_critic:
                break

            # (2) Tool-specific criticism
            reps = {}
            for t in tools:
                cs = per_tool[t.name]
                agree = [c for c in cs if c.norm == y_new]
                valid = [c for c in cs if c.norm]
                reps[t.name] = (agree or valid or cs)[0]
            futs = {t.name: self.pool.submit(t.critic, ctx, reps[t.name]) for t in tools}
            crit = {k: f.result() for k, f in futs.items()}
            for t in tools:
                need_t, sug_t, _ = crit[t.name]
                c = reps[t.name]
                feedback[t.name] = {
                    "previous": (c.trace[-2500:] + f"\nResult: {c.answer or 'None'}"),
                    "suggestions": sug_t if need_t else (sug_t or "None"),
                }
            win_tool = rep.tool if rep and rep.tool in crit else tools[0].name
            need = crit[win_tool][0]
            rec["critic"] = {k: v[2] for k, v in crit.items()}
            rec["need_refinement"] = need
            self.log(f"[critic-{it + 1}] tool={win_tool} need_refinement={need}")
            if not need:
                break
            if unchanged >= 2:  # early stop: unchanged for two corrections
                self.log("[stop] answer unchanged for two consecutive corrections")
                break

        raw = y_rep.answer if y_rep is not None else y_t
        return RefineResult(raw, y_t, stage, len(history) - 1, history)


__all__ = ["DEFAULT_TOOLSETS", "RefineResult", "Refiner"]
