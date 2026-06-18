"""Multi-step RLCR reward (merged RL#1 RLCR + RL#2 CTA) for PopQA.

Two-turn TABC rollout:
  Turn 1 (no context): <think> <answer>A1 <analysis> <confidence>c1   <think> <action>ANSWER|RETRIEVE
                        [schema 2 only: <estimated_confidence>ec]
  Turn 2 (only if RETRIEVE; context appended; forced answer):
                        <think> <answer>A2 <analysis> <confidence>c2
  final = A1 if ANSWER else A2

Reward (per trajectory, bounded ~[0,1]):
  overall = 0 if (format invalid or trace too long)
            else format_weight + (1-format_weight) * core
  core    = convex( w_task*task + w_cal*cal [+ w_mse*mse] )
    task  = correct(final) * r^(#retrieves)
    cal   = mean over present confidences of RLCR reward (1 - Brier):
              c1 vs correct(A1)   [no-context calibration]
              c2 vs correct(final) [post-retrieval calibration, retrieve only]
    mse   = 1 - (ec - c2)^2       [schema 2, retrieve only; ec forecasts c2]

Decisions (locked 2026-06-17): MSE as a reward term (not a value-head loss);
ec does NOT control the action; ec MSE only supervised on retrieve rollouts.
"""
import re, string, os, json as _json

CONF_RE     = re.compile(r"<confidence>\s*([0-9]*\.?[0-9]+)\s*</confidence>", re.I | re.S)
ECONF_RE    = re.compile(r"<estimated_confidence>\s*([0-9]*\.?[0-9]+)\s*</estimated_confidence>", re.I | re.S)
ANS_RE      = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)
ACT_RE      = re.compile(r"<action>\s*(RETRIEVE|ANSWER)\s*</action>", re.I | re.S)
ANALYSIS_RE = re.compile(r"<analysis>(.*?)</analysis>", re.I | re.S)
THINK_RE    = re.compile(r"<think>.*?</think>", re.I | re.S)


def _as_list(x):
    # array-safe: VeRL passes possible_answers/action_seqs/response_list as numpy object
    # arrays; `arr or []` raises "truth value ambiguous" for >1 element. Never use `or` on these.
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    try:
        return list(x)
    except TypeError:
        return [x]


def popqa_correct(pred, answers):
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
    # remove ALL complete <think>...</think> spans (turn 1 has two think blocks),
    # so the answer/analysis/confidence between them survive.
    t = THINK_RE.sub(" ", str(t))
    return t.replace("<|im_end|>", "")


def parse_turn(text):
    truncated = ("<think>" in str(text)) and ("</think>" not in str(text))
    body = _strip_think(text)
    ans_m = ANS_RE.search(body)
    act_m = ACT_RE.search(body)
    return {
        "answer": ans_m.group(1).strip() if ans_m else None,
        "conf": _f01(CONF_RE.search(body)),
        "ec": _f01(ECONF_RE.search(body)),
        "action": act_m.group(1).upper() if act_m else None,
        "has_analysis": ANALYSIS_RE.search(body) is not None,
        "truncated": truncated,
    }


def turn1_format_ok(p, schema):
    ok = (p["answer"] is not None and p["has_analysis"] and p["conf"] is not None
          and p["action"] in ("RETRIEVE", "ANSWER") and not p["truncated"])
    if schema == 2:
        ok = ok and (p["ec"] is not None)
    return ok


def turn2_format_ok(p):
    return (p["answer"] is not None and p["has_analysis"] and p["conf"] is not None
            and not p["truncated"])


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


def compute_score(reward_inputs, format_weight=0.1, schema=1,
                  w_task=0.7, w_cal=0.3, w_mse=0.15, info_gain_weight=0.0,
                  log_dir=None, max_turns=2, dump_path=None, **kw):
    # info_gain_weight>0 (belief-RL inspired, arXiv 2602.12342): additive shaping that credits a
    # RETRIEVE only when it improves correctness (correct_final - correct_A1 in {-1,0,+1}); 0 by
    # default so schema1/schema2 baselines are unaffected.
    schema = int(schema)
    results = []
    diags = []  # None-able diagnostics: dumped, but NOT returned (VeRL sums every returned key)
    for ri in reward_inputs:
        answers = ri.get("possible_answers", ri.get("gold_answers", []))
        r = float(ri.get("discount_factor", 1.0))
        rl = [str(x) for x in _as_list(ri.get("response_list"))]
        num_retrieves = count_retrieves(ri.get("action_seqs", []))
        retrieved = num_retrieves > 0

        t1 = parse_turn(rl[0]) if len(rl) >= 1 else parse_turn("")
        t2 = parse_turn(rl[1]) if (retrieved and len(rl) >= 2) else None
        A1, c1, ec = t1["answer"], t1["conf"], t1["ec"]
        if retrieved and t2 is not None:
            A2, c2 = t2["answer"], t2["conf"]
            final = A2
        else:
            A2, c2 = None, None
            final = A1

        correct_A1 = popqa_correct(A1, answers)
        correct_final = popqa_correct(final, answers)

        fmt = turn1_format_ok(t1, schema) and ((not retrieved) or (t2 is not None and turn2_format_ok(t2)))
        trace_too_long = len(rl) > max_turns

        task = correct_final * (r ** num_retrieves)
        cal_terms = []
        if c1 is not None:
            cal_terms.append(1.0 - (c1 - correct_A1) ** 2)
        if retrieved and c2 is not None:
            cal_terms.append(1.0 - (c2 - correct_final) ** 2)
        cal = sum(cal_terms) / len(cal_terms) if cal_terms else 0.0

        mse_r = None
        if schema == 2 and retrieved and (ec is not None) and (c2 is not None):
            mse_r = 1.0 - (ec - c2) ** 2

        if schema == 2 and mse_r is not None:
            core = (w_task * task + w_cal * cal + w_mse * mse_r) / (w_task + w_cal + w_mse)
        else:
            core = (w_task * task + w_cal * cal) / (w_task + w_cal)

        overall = 0.0 if (not fmt or trace_too_long) else (format_weight + (1.0 - format_weight) * core)

        info_gain = (correct_final - correct_A1) if retrieved else 0.0
        if info_gain_weight and fmt and not trace_too_long:
            overall = overall + info_gain_weight * info_gain

        # Returned dict: FLOATS ONLY (VeRL reduces/sums every key across the batch).
        results.append({
            "overall": round(overall, 6),
            "discounted_reward": round(task, 6),
            "correctness": float(correct_final),
            "num_retrieves": float(num_retrieves),
            "format_reward": 1.0 if fmt else 0.0,
            "accuracy": float(correct_final),
            "retrieve_rate": 1.0 if retrieved else 0.0,
            "cal_reward": round(cal, 6),
            "mse_reward": (round(mse_r, 6) if mse_r is not None else 0.0),
            "info_gain": float(info_gain),
        })
        # Diagnostics (may be None) go ONLY into the dump for offline ECE/MSE.
        diags.append({
            "c1": c1, "c2": c2, "ec": ec, "correct_A1": correct_A1,
            "brier_c1": (None if c1 is None else round((c1 - correct_A1) ** 2, 6)),
            "brier_c2": (None if c2 is None else round((c2 - correct_final) ** 2, 6)),
            "mse_ec": (None if (ec is None or c2 is None) else round((ec - c2) ** 2, 6)),
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
    T1_ans = "<think>reason</think><answer>Paris</answer><analysis>I am sure</analysis><confidence>0.9</confidence><think>no need</think><action>ANSWER</action>"
    T1_ret = "<think>hmm</think><answer>Berlin</answer><analysis>unsure</analysis><confidence>0.2</confidence><think>retrieve will help</think><action>RETRIEVE</action><estimated_confidence>0.8</estimated_confidence>"
    T2     = "<think>using ctx</think><answer>Libreville</answer><analysis>ctx says so</analysis><confidence>0.85</confidence>"
    cases = [
        ("schema1 ANSWER correct", 1, {"possible_answers": ["Paris"], "discount_factor": 0.9,
            "action_seqs": [["ANSWER", "Paris"]], "response_list": [T1_ans], "pred_answers": "Paris"}),
        ("schema1 RETRIEVE", 1, {"possible_answers": ["Libreville"], "discount_factor": 0.5,
            "action_seqs": [["RETRIEVE", None], ["ANSWER", "Libreville"]],
            "response_list": [T1_ret, T2], "pred_answers": "Libreville"}),
        ("schema2 RETRIEVE +ec", 2, {"possible_answers": ["Libreville"], "discount_factor": 0.5,
            "action_seqs": [["RETRIEVE", None], ["ANSWER", "Libreville"]],
            "response_list": [T1_ret, T2], "pred_answers": "Libreville"}),
        ("schema2 missing ec (format fail)", 2, {"possible_answers": ["Paris"], "discount_factor": 0.9,
            "action_seqs": [["ANSWER", "Paris"]], "response_list": [T1_ans], "pred_answers": "Paris"}),
        ("bad format (no tags)", 1, {"possible_answers": ["Paris"], "discount_factor": 0.9,
            "action_seqs": [["ANSWER", "Paris"]], "response_list": ["ANSWER: Paris"], "pred_answers": "Paris"}),
    ]
    for name, sch, ri in cases:
        out = compute_score([ri], schema=sch)[0]
        print(f"\n## {name} (schema {sch})")
        print({k: out[k] for k in ("overall", "discounted_reward", "correctness", "num_retrieves",
                                   "format_reward", "cal_reward", "mse_reward", "retrieve_rate")})
        assert all(v is not None for v in out.values()), f"returned dict has None: {out}"
        rl = [str(x) for x in ri["response_list"]]
        print("  parsed t1:", {k: parse_turn(rl[0])[k] for k in ("answer", "conf", "ec", "action", "has_analysis")})
        if len(rl) > 1:
            print("  parsed t2:", {k: parse_turn(rl[1])[k] for k in ("answer", "conf", "has_analysis")})
