"""v2 three-turn forced-retrieve calibrated reward for PopQA (GRPO, EasyR1/VeRL).

Authoritative spec: docs/METHODS_v2_3turn.md. Baseline 2-turn reward
(popqa_multistep_score.py) is left fully intact; this is a NEW, separate scorer.

Trajectory (exact tags):
  T1 (no ctx):  <think> <answer>A1 <analysis> <confidence>c1
  T2 (no ctx):  <think> <action>ANSWER|RETRIEVE <analysis> <estimated_confidence>ec   (NO answer)
  T3 (after RETRIEVE; 1 passage appended): <think> <answer>A2 <analysis> <confidence>c2

Training FORCES retrieve (T3 always executed) but records the emitted T2 action.
Eval follows the emitted action: final = A1 if action==ANSWER else A2.

Reward (per rollout, bounded ~[0,1]):
  r = discount_factor; cA1 = popqa_correct(A1); cA2 = popqa_correct(A2); post-retrieval label y = cA2.
  oracle   = ANSWER if cA1 >= cA2 * r else RETRIEVE       (cost-optimal given both realized outcomes)
  act_rew  = 1[action == oracle]
  cal      = mean over present terms of { 1-(c1-cA1)^2, 1-(ec-y)^2, 1-(c2-y)^2 }   (Brier; ec & c2 vs y)
  core     = (w_a1*cA1 + w_act*act_rew + w_a2*cA2 + w_cal*cal) / (w_a1+w_act+w_a2+w_cal)   in [0,1]
  overall  = 0 if (bad format OR len(rl) > max_turns) else format_weight + (1-format_weight)*core

Returned dict is FLOATS ONLY (VeRL sums every returned key). c1/ec/c2/oracle/action diagnostics go to the
JSONL dump ($CTA_EVAL_DUMP / dump_path) only.
"""
import re, os, json as _json

CONF_RE     = re.compile(r"<confidence>\s*([0-9]*\.?[0-9]+)\s*</confidence>", re.I | re.S)
ECONF_RE    = re.compile(r"<estimated_confidence>\s*([0-9]*\.?[0-9]+)\s*</estimated_confidence>", re.I | re.S)
ANS_RE      = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)
ACT_RE      = re.compile(r"<action>\s*(RETRIEVE|ANSWER)\s*</action>", re.I | re.S)
ANALYSIS_RE = re.compile(r"<analysis>(.*?)</analysis>", re.I | re.S)
THINK_RE    = re.compile(r"<think>.*?</think>", re.I | re.S)


def _as_list(x):
    # array-safe: VeRL passes possible_answers/action_seqs/response_list as numpy object arrays;
    # `arr or []` raises "truth value ambiguous" for >1 element. Never use `or` on these.
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    try:
        return list(x)
    except TypeError:
        return [x]


def popqa_correct(pred, answers):
    """Full-alias substring match (same semantics as popqa_multistep_score.popqa_correct)."""
    if pred is None:
        return 0.0
    p = str(pred).strip(); pl = p.lower()
    for a in _as_list(answers):
        a = str(a)
        if a and (a in p or a.lower() in pl or a.capitalize() in p):
            return 1.0
    return 0.0


def _f01(m):
    if not m:
        return None
    try:
        return min(max(float(m.group(1)), 0.0), 1.0)
    except Exception:
        return None


def _strip_think(t):
    t = THINK_RE.sub(" ", str(t))
    return t.replace("<|im_end|>", "")


def _truncated(text):
    return ("<think>" in str(text)) and ("</think>" not in str(text))


def parse_t1(text):
    """T1: <answer>A1 <analysis> <confidence>c1 (no action)."""
    body = _strip_think(text)
    return {
        "answer": (lambda m: m.group(1).strip() if m else None)(ANS_RE.search(body)),
        "conf": _f01(CONF_RE.search(body)),
        "has_analysis": ANALYSIS_RE.search(body) is not None,
        "truncated": _truncated(text),
    }


def parse_t2(text):
    """T2: <action> <analysis> <estimated_confidence>ec (NO answer)."""
    body = _strip_think(text)
    act_m = ACT_RE.search(body)
    return {
        "action": act_m.group(1).upper() if act_m else None,
        "ec": _f01(ECONF_RE.search(body)),
        "has_analysis": ANALYSIS_RE.search(body) is not None,
        "answer": (lambda m: m.group(1).strip() if m else None)(ANS_RE.search(body)),
        "truncated": _truncated(text),
    }


def parse_t3(text):
    """T3: <answer>A2 <analysis> <confidence>c2."""
    body = _strip_think(text)
    return {
        "answer": (lambda m: m.group(1).strip() if m else None)(ANS_RE.search(body)),
        "conf": _f01(CONF_RE.search(body)),
        "has_analysis": ANALYSIS_RE.search(body) is not None,
        "truncated": _truncated(text),
    }


def t1_format_ok(p):
    return (p["answer"] is not None and p["has_analysis"] and p["conf"] is not None
            and not p["truncated"])


def t2_format_ok(p):
    return (p["action"] in ("RETRIEVE", "ANSWER") and p["has_analysis"]
            and p["ec"] is not None and not p["truncated"])


def t3_format_ok(p):
    return (p["answer"] is not None and p["has_analysis"] and p["conf"] is not None
            and not p["truncated"])


def _action_from_seq(action_seqs):
    """Extract the emitted T2 action (RETRIEVE/ANSWER) from the rollout's action_seq.

    action_seqs is the per-turn list of (action_type, content) tuples recorded by the rollout.
    T1 records ("CONTINUE", ...); T2 records the real ("ANSWER"|"RETRIEVE", ...). We scan for the
    first ANSWER/RETRIEVE token. Robust to nested numpy arrays.
    """
    import numpy as _np
    for a in _as_list(action_seqs):
        at = a
        # an action_seq element is typically (type, content); descend to the type token
        while isinstance(at, (list, tuple, _np.ndarray)):
            if len(at) == 0:
                at = None
                break
            at = at[0]
        tok = str(at).strip().upper()
        if tok in ("RETRIEVE", "ANSWER"):
            return tok
    return None


def count_retrieves(action_seqs):
    import numpy as _np
    n = 0
    for a in _as_list(action_seqs):
        at = a
        while isinstance(at, (list, tuple, _np.ndarray)):
            if len(at) == 0:
                at = None
                break
            at = at[0]
        if str(at).strip().upper() == "RETRIEVE":
            n += 1
    return n


def compute_score(reward_inputs, format_weight=0.1,
                  w_a1=1.0, w_act=1.0, w_a2=1.0, w_cal=1.0,
                  max_turns=3, dump_path=None, **kw):
    results = []
    diags = []
    for ri in reward_inputs:
        answers = ri.get("possible_answers", ri.get("gold_answers", []))
        r = float(ri.get("discount_factor", 1.0))
        rl = [str(x) for x in _as_list(ri.get("response_list"))]

        # action recorded by the rollout (preferred over re-parsing rl[1])
        action = _action_from_seq(ri.get("action_seqs", []))

        # Parse turns by index.
        t1 = parse_t1(rl[0]) if len(rl) >= 1 else parse_t1("")
        t2 = parse_t2(rl[1]) if len(rl) >= 2 else None
        t3 = parse_t3(rl[2]) if len(rl) >= 3 else None

        # Fall back to parsing the action from T2 text if action_seqs absent (e.g. unit tests).
        if action is None and t2 is not None:
            action = t2["action"]

        A1, c1 = t1["answer"], t1["conf"]
        ec = t2["ec"] if t2 is not None else None
        A2, c2 = (t3["answer"], t3["conf"]) if t3 is not None else (None, None)

        cA1 = popqa_correct(A1, answers)
        cA2 = popqa_correct(A2, answers) if A2 is not None else 0.0
        y = cA2  # post-retrieval label

        # Oracle action (cost-optimal given both realized outcomes). tie -> ANSWER.
        oracle = "ANSWER" if cA1 >= cA2 * r else "RETRIEVE"
        act_reward = 1.0 if (action is not None and action == oracle) else 0.0

        # Calibration (Brier 1 - error^2) over present terms. ec & c2 share label y.
        cal_terms = []
        if c1 is not None:
            cal_terms.append(1.0 - (c1 - cA1) ** 2)
        if ec is not None:
            cal_terms.append(1.0 - (ec - y) ** 2)
        if (c2 is not None) and (t3 is not None):
            cal_terms.append(1.0 - (c2 - y) ** 2)
        cal = sum(cal_terms) / len(cal_terms) if cal_terms else 0.0

        core = (w_a1 * cA1 + w_act * act_reward + w_a2 * cA2 + w_cal * cal) / (w_a1 + w_act + w_a2 + w_cal)

        # Format gate: present turns must parse. T3 only required when a T3 turn exists.
        fmt = t1_format_ok(t1)
        if t2 is not None:
            fmt = fmt and t2_format_ok(t2)
        else:
            fmt = False  # T2 is always required (decision turn)
        if t3 is not None:
            fmt = fmt and t3_format_ok(t3)
        trace_too_long = len(rl) > max_turns

        overall = 0.0 if (not fmt or trace_too_long) else (format_weight + (1.0 - format_weight) * core)

        # num_retrieves: a T3 turn exists IFF a retrieval happened (context was injected before T3).
        # This is robust for BOTH phases: forced-train always runs T3 -> num_ret=1 (even when the
        # emitted action was ANSWER); eval RETRIEVE runs T3 -> 1; eval ANSWER stops at T2 -> 0.
        # (action_seqs is NOT reliable for the count here because forced-train records the emitted
        # ANSWER token at T2 even though T3 was executed.)
        num_ret = 1 if (t3 is not None and A2 is not None) else 0
        # Eval semantics: final follows the emitted action (ANSWER -> A1, else A2).
        final_correct = cA1 if (action == "ANSWER") else cA2
        discounted = final_correct * (r ** num_ret)

        results.append({
            "overall": round(overall, 6),
            "discounted_reward": round(discounted, 6),
            "accuracy": float(final_correct),
            "correctness": float(final_correct),
            "num_retrieves": float(num_ret),
            "format_reward": 1.0 if fmt else 0.0,
            "oracle_match": float(act_reward),
            "cal_reward": round(cal, 6),
            "a1_correct": float(cA1),
            "a2_correct": float(cA2),
        })
        diags.append({
            "c1": c1, "ec": ec, "c2": c2,
            "action": action, "oracle": oracle,
            "cA1": cA1, "cA2": cA2, "y": y, "r": r,
            "brier_c1": (None if c1 is None else round((c1 - cA1) ** 2, 6)),
            "brier_ec": (None if ec is None else round((ec - y) ** 2, 6)),
            "brier_c2": (None if (c2 is None or t3 is None) else round((c2 - y) ** 2, 6)),
            "core": round(core, 6),
        })

    _dump = dump_path or os.environ.get("CTA_EVAL_DUMP")
    if _dump:
        with open(_dump, "a") as _f:
            for _ri, _res, _dg in zip(reward_inputs, results, diags):
                _rec = dict(_res); _rec.update(_dg)
                _rec["task_id"] = str(_ri.get("task_id", _ri.get("index", "")))
                _rec["_pred"] = str(_ri.get("pred_answers", ""))
                _rl = _ri.get("response_list")
                _rec["_responses"] = [str(x) for x in ([] if _rl is None else list(_rl))]
                _rec["_gold"] = str(_ri.get("possible_answers", _ri.get("gold_answers", "")))[:160]
                _rec["_df"] = _ri.get("discount_factor")
                _f.write(_json.dumps(_rec) + "\n")
    return results


if __name__ == "__main__":
    T1 = "<think>reason</think><answer>Paris</answer><analysis>sure</analysis><confidence>0.9</confidence>"
    T2 = "<think>decide</think><action>RETRIEVE</action><analysis>need it</analysis><estimated_confidence>0.8</estimated_confidence>"
    T3 = "<think>ctx</think><answer>Paris</answer><analysis>ctx confirms</analysis><confidence>0.95</confidence>"
    ri = {"possible_answers": ["Paris"], "discount_factor": 0.5,
          "action_seqs": [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]],
          "response_list": [T1, T2, T3], "pred_answers": "Paris"}
    out = compute_score([ri])[0]
    print(out)
