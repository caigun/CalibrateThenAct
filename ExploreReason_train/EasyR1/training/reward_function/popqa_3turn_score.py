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
                  w_a1=1.0, w_act=1.0, w_a2=1.0, w_cal=1.0, w_ec=1.0,
                  max_turns=3, dump_path=None, oracle_mode="rollout", **kw):
    """oracle_mode:
      "rollout" (default, UNCHANGED): per-rollout oracle = ANSWER iff cA1 >= cA2*r, with binary
        cA1,cA2 -> r-DEGENERATE (see RESULTS_v2_3turn.md WAVE-1). Kept for backward compatibility.
      "group" (cost-aware MC oracle): estimate p_A1=mean(cA1), p_A2=mean(cA2) over the GRPO rollout
        GROUP (same task_id, same r — all n rollouts of a prompt are present in this single
        compute_score call; verified against BatchFunctionRewardManager). Then a SINGLE group-level
        oracle_group = ANSWER iff p_A1 >= p_A2*r is genuinely r-sensitive because p_A1,p_A2 in [0,1].
        Each rollout's oracle_match = 1[emitted_action == oracle_group]. cA1/cA2 are
        action-independent (A1 from T1, A2 from forced T3) -> clean, non-circular estimates. Groups
        of size 1 fall back to per-rollout-like behavior (p == that rollout's binary correctness).
      "verbal" (verbalized-confidence oracle; ec-before-action variant): per-rollout oracle derived
        from the model's OWN continuous verbalized confidences: oracle = ANSWER iff c1 >= ec*r else
        RETRIEVE. Because c1,ec in [0,1] this threshold moves with r (r-sensitive, unlike the binary
        rollout oracle). oracle_match = 1[emitted_action == oracle]. If c1 or ec is None (format
        fail) -> oracle is None -> oracle_match = 0 (and the format gate zeroes overall anyway).
        Pairs with the popqa_3turn_ecfirst[_cost].jinja templates that emit ec BEFORE the action.
      "combined" (MC + verbal blend): compute BOTH the group-MC oracle (as in "group") and the
        verbal oracle, then oracle_match = 0.5*1[action==oracle_mc] + 0.5*1[action==oracle_verbal].
        Requires the group pass (reuses the group code).
      "precomp" (PRECOMPUTED-MC oracle): read OFFLINE-estimated per-question probabilities p_A1,
        p_A2 directly from the reward_input (baked dataset columns; see scripts/precompute_mc_oracle.py
        and data/rl_3turn_p1_precomp). oracle = ANSWER iff p_A1 >= p_A2*r else RETRIEVE; this is
        genuinely r-sensitive AND sharp (MC=10, 11 levels) — unlike "group" (coarse at n=4 live
        rollouts) and the r-degenerate "rollout". oracle_match = 1[emitted_action == oracle]. The
        p_A1/p_A2 used are the STATIC dataset columns, NOT this batch's realized cA1/cA2. If a column
        is missing it falls back to the rollout's own binary correctness (and a None-action -> 0).
    Brier (c1/ec/c2) and all other terms are identical across modes.
    """
    if oracle_mode not in ("rollout", "group", "verbal", "combined", "precomp"):
        raise ValueError(
            "oracle_mode must be 'rollout', 'group', 'verbal', 'combined' or 'precomp', "
            f"got {oracle_mode!r}")

    # ---- Pass 1: per-rollout parse + realized correctness (action-independent). ----
    rows = []
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
        rows.append({
            "ri": ri, "r": r, "rl": rl, "action": action,
            "t1": t1, "t2": t2, "t3": t3,
            "A1": A1, "A2": A2, "c1": c1, "ec": ec, "c2": c2,
            "cA1": cA1, "cA2": cA2,
            "task_id": str(ri.get("task_id", ri.get("index", ""))),
        })

    # ---- Group-mean (MC) probabilities per task_id (used by "group" and "combined"). ----
    group_p = {}
    if oracle_mode in ("group", "combined"):
        by_tid = {}
        for row in rows:
            by_tid.setdefault(row["task_id"], []).append(row)
        for tid, grp in by_tid.items():
            n = len(grp)
            p_A1 = sum(g["cA1"] for g in grp) / n
            p_A2 = sum(g["cA2"] for g in grp) / n
            r_g = grp[0]["r"]  # r is identical within a task_id group
            oracle_group = "ANSWER" if p_A1 >= p_A2 * r_g else "RETRIEVE"
            group_p[tid] = {"p_A1": p_A1, "p_A2": p_A2, "oracle_group": oracle_group,
                            "group_size": n}

    # ---- Pass 2: oracle, calibration, core, overall. ----
    results = []
    diags = []
    for row in rows:
        ri = row["ri"]; r = row["r"]; rl = row["rl"]; action = row["action"]
        t1 = row["t1"]; t2 = row["t2"]; t3 = row["t3"]
        A1 = row["A1"]; A2 = row["A2"]; c1 = row["c1"]; ec = row["ec"]; c2 = row["c2"]
        cA1 = row["cA1"]; cA2 = row["cA2"]
        y = cA2  # post-retrieval label

        # Verbal oracle (continuous c1/ec). r-sensitive. None if c1 or ec missing (format fail).
        oracle_verbal = None
        if c1 is not None and ec is not None:
            oracle_verbal = "ANSWER" if c1 >= ec * r else "RETRIEVE"

        oracle_mc = None  # group-MC oracle (set for group/combined)
        if oracle_mode in ("group", "combined"):
            gp = group_p[row["task_id"]]
            p_A1, p_A2 = gp["p_A1"], gp["p_A2"]
            oracle_mc = gp["oracle_group"]

        if oracle_mode == "group":
            oracle = oracle_mc
            act_reward = 1.0 if (action is not None and action == oracle) else 0.0
        elif oracle_mode == "verbal":
            # per-rollout binary correctness still reported as p_A1/p_A2 for the diag.
            p_A1, p_A2 = float(cA1), float(cA2)
            oracle = oracle_verbal
            act_reward = 1.0 if (action is not None and oracle is not None and action == oracle) else 0.0
        elif oracle_mode == "combined":
            # 0.5/0.5 blend of MC and verbal matches. A None verbal oracle contributes 0 to its half.
            mc_match = 1.0 if (action is not None and action == oracle_mc) else 0.0
            vb_match = 1.0 if (action is not None and oracle_verbal is not None
                              and action == oracle_verbal) else 0.0
            act_reward = 0.5 * mc_match + 0.5 * vb_match
            oracle = oracle_mc  # report the MC oracle as the headline "oracle" diag
        elif oracle_mode == "precomp":
            # PRECOMPUTED static per-question MC probabilities (baked dataset columns).
            # Fall back to this rollout's binary correctness if a column is absent / sentinel.
            # A negative value (e.g. -1.0 baked into eval rows with no precomp estimate) is
            # treated as "missing" -> fallback, so eval splits never get a degenerate p=0 oracle.
            def _pf(key, default):
                v = ri.get(key, None)
                if v is None:
                    return float(default)
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    return float(default)
                if fv < 0.0:
                    return float(default)
                return min(max(fv, 0.0), 1.0)
            p_A1 = _pf("p_A1", cA1)
            p_A2 = _pf("p_A2", cA2)
            oracle = "ANSWER" if p_A1 >= p_A2 * r else "RETRIEVE"
            act_reward = 1.0 if (action is not None and action == oracle) else 0.0
        else:
            # rollout (legacy): per-rollout oracle (cost-optimal given both realized outcomes). tie -> ANSWER.
            p_A1, p_A2 = float(cA1), float(cA2)
            oracle = "ANSWER" if cA1 >= cA2 * r else "RETRIEVE"
            act_reward = 1.0 if (action is not None and action == oracle) else 0.0

        # Calibration (Brier 1 - error^2) over present terms. ec & c2 share label y.
        # WEIGHTED mean: c1 and c2 Brier terms carry weight 1.0; the ec term carries
        # weight w_ec (RELATIVE up-weight, default 1.0). At w_ec=1.0 this reduces EXACTLY
        # to the plain mean of present terms (identical to the pre-w_ec behavior). When ec
        # is absent (no retrieval / format fail) its term + weight are simply not included.
        cal_num = 0.0
        cal_den = 0.0
        if c1 is not None:
            cal_num += 1.0 * (1.0 - (c1 - cA1) ** 2)
            cal_den += 1.0
        if ec is not None:
            cal_num += w_ec * (1.0 - (ec - y) ** 2)
            cal_den += w_ec
        if (c2 is not None) and (t3 is not None):
            cal_num += 1.0 * (1.0 - (c2 - y) ** 2)
            cal_den += 1.0
        cal = cal_num / cal_den if cal_den > 0.0 else 0.0

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
            "oracle_verbal": oracle_verbal,
            "oracle_mc": oracle_mc,
            "oracle_mode": oracle_mode,
            "p_A1": round(p_A1, 6), "p_A2": round(p_A2, 6),
            "oracle_group": (group_p[row["task_id"]]["oracle_group"]
                             if oracle_mode in ("group", "combined") else None),
            "group_size": (group_p[row["task_id"]]["group_size"]
                           if oracle_mode in ("group", "combined") else 1),
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
