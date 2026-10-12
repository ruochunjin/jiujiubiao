"""Answer parsing / normalisation / voting utilities."""
import re
from collections import Counter

_TRUE = {"true", "yes", "entailed", "supported", "support", "correct", "1", "entails"}
_FALSE = {"false", "no", "refuted", "refutes", "refute", "incorrect", "0", "not supported"}


def extract_answer(text):
    """Get the content after the last 'Answer:' marker."""
    if not text:
        return ""
    m = re.findall(r"(?:final answer|answer)\s*[:：]\s*(.+)", text, flags=re.I)
    lines = text.strip().splitlines()
    if not m and not lines:
        # whitespace-only output has no last line to fall back to
        return ""
    ans = m[-1] if m else lines[-1]
    ans = ans.strip().strip("*").strip()
    ans = re.sub(r"^\\boxed\{(.*)\}$", r"\1", ans)
    return ans.strip().rstrip(".").strip().strip('"').strip()


def to_bool_str(a):
    s = str(a).strip().lower().strip(".").strip()
    s = re.sub(r"[*`\"']", "", s)
    if s in _TRUE or s.startswith("true") or s.startswith("entail"):
        return "True"
    if s in _FALSE or s.startswith("false") or s.startswith("refut"):
        return "False"
    return ""


def _num(s):
    t = re.sub(r"[,$€£¥%\s]", "", s)
    try:
        f = float(t)
        return f
    except ValueError:
        return None


def normalize(ans, task):
    """Canonical form used for voting & evaluation."""
    if ans is None:
        return ""
    if isinstance(ans, bool):
        return "True" if ans else "False"
    if task == "tfv":
        return to_bool_str(ans)
    if isinstance(ans, (list, tuple, set)):
        parts = [normalize(x, task) for x in ans]
        return " | ".join(sorted(p for p in parts if p))
    s = str(ans).strip()
    if "|" in s:
        parts = [normalize(p, task) for p in s.split("|")]
        return " | ".join(sorted(p for p in parts if p))
    s = s.lower().strip().strip(".").strip()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    # Answer-shaping residue: drop trailing parentheticals ("p4 (11/22 ps)")
    # and stray emphasis markers ("baltimore bullets*") from model outputs
    # (WTQ-100 miss attribution, 2026-10-10).
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)
    s = s.strip("*").strip()
    # Unify unicode dashes with the ASCII hyphen ("8–14" == "8-14").
    s = re.sub(r"[‐-―]", "-", s)
    s = re.sub(r"\s+", " ", s)
    f = _num(s)
    if f is not None:
        import math

        if not math.isfinite(f):
            return s  # inf/nan: keep the surface (BIRD OverflowError fix)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return f"{f:.4f}".rstrip("0").rstrip(".")
    return s


_REFUSAL = re.compile(r"(unanswerable|cannot be determined|not applicable|^n/?a$|not enough info|"
                      r"no answer|unknown|none of|not (found|available|provided|mentioned)|insufficient)")


def is_refusal(a):
    return bool(_REFUSAL.search(str(a or "").lower()))


def majority_vote(candidates):
    """candidates: list of (normalized_answer, payload). Returns (answer, payload, counter).
    Ties are broken by first appearance, which follows the tool order."""
    valid = [(a, p) for a, p in candidates
             if a not in ("", "none", "nan", "error") and not is_refusal(a)]
    if not valid:
        return "", (candidates[0][1] if candidates else None), Counter()
    cnt = Counter(a for a, _ in valid)
    best = max(cnt.values())
    for a, p in valid:
        if cnt[a] == best:
            return a, p, cnt


def parse_critic(text):
    """Return (need_refinement: bool, suggestions: str)."""
    need = True
    m = re.findall(r"need\s*refinement\s*[:：]\s*\**\s*(yes|no)", text or "", flags=re.I)
    if m:
        need = m[-1].lower() == "yes"
    elif re.search(r"answer is (right|correct)", text or "", re.I):
        need = False
    s = re.search(r"suggestions?\s*[:：](.*)", text or "", flags=re.I | re.S)
    sug = s.group(1).strip() if s else ""
    if sug.lower().strip(". ") in ("none", "n/a", ""):
        sug = ""
    return need, sug


def extract_code(text):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", flags=re.S)
    if m:
        return max(m, key=len).strip()
    return (text or "").strip()


def extract_json(text):
    import json
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    if not m:
        return {}
    raw = m.group(0)
    for cand in (raw, raw.replace("'", '"')):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return {}
