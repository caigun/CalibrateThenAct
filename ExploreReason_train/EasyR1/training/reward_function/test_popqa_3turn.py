"""Pure-python unit tests for the v2 3-turn forced-retrieve reward + turn-aware parsing.

Run inside the overlay (no GPU):
  python3 -m pytest test_popqa_3turn.py -q
or stand-alone:
  python3 test_popqa_3turn.py
"""
import math
import popqa_3turn_score as S


def _T1(ans="Paris", conf="0.9", analysis="sure"):
    return f"<think>x</think><answer>{ans}</answer><analysis>{analysis}</analysis><confidence>{conf}</confidence>"


def _T2(action="RETRIEVE", ec="0.8", analysis="decide"):
    return f"<think>x</think><action>{action}</action><analysis>{analysis}</analysis><estimated_confidence>{ec}</estimated_confidence>"


def _T3(ans="Paris", conf="0.95", analysis="ctx"):
    return f"<think>x</think><answer>{ans}</answer><analysis>{analysis}</analysis><confidence>{conf}</confidence>"


def _ri(response_list, answers, r, action_seqs):
    return {"possible_answers": answers, "discount_factor": r,
            "action_seqs": action_seqs, "response_list": response_list,
            "pred_answers": ""}


def _score(ri, **kw):
    return S.compute_score([ri], **kw)[0]


def _ri_tid(response_list, answers, r, action_seqs, task_id):
    d = _ri(response_list, answers, r, action_seqs)
    d["task_id"] = task_id
    return d


def _approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ---------------------------------------------------------------- oracle logic
def test_oracle_cases():
    # (cA1, cA2, r) -> expected oracle. oracle = ANSWER if cA1 >= cA2*r else RETRIEVE.
    cases = [
        # cA1=1, cA2=1, r=0.5 -> 1 >= 0.5 -> ANSWER
        (1, 1, 0.5, "ANSWER"),
        # cA1=0, cA2=1, r=0.5 -> 0 >= 0.5 (False) -> RETRIEVE
        (0, 1, 0.5, "RETRIEVE"),
        # boundary tie: cA1=1, cA2=1, r=1.0 -> 1 >= 1 -> ANSWER
        (1, 1, 1.0, "ANSWER"),
        # cA1=1, cA2=0 -> 1 >= 0 -> ANSWER (retrieval would hurt)
        (1, 0, 0.5, "ANSWER"),
        # cA1=0, cA2=0 -> 0 >= 0 -> ANSWER (retrieval doesn't help)
        (0, 0, 0.5, "ANSWER"),
        # cA1=0, cA2=1, r=0.1 -> still RETRIEVE
        (0, 1, 0.1, "RETRIEVE"),
        # cA1=1, cA2=1, r=0.65 -> 1 >= 0.65 -> ANSWER
        (1, 1, 0.65, "ANSWER"),
    ]
    gold = ["Paris"]
    for cA1, cA2, r, expect in cases:
        a1 = "Paris" if cA1 else "Berlin"
        a2 = "Paris" if cA2 else "Berlin"
        rl = [_T1(ans=a1), _T2(action="RETRIEVE"), _T3(ans=a2)]
        ri = _ri(rl, gold, r, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", a2]])
        d = S.compute_score([ri], dump_path=None)
        # recompute oracle via diag (dump disabled) -> re-derive from rule for assertion clarity
        oracle = "ANSWER" if cA1 >= cA2 * r else "RETRIEVE"
        assert oracle == expect, f"{(cA1,cA2,r)} -> {oracle} != {expect}"
        # action emitted is RETRIEVE; oracle_match should be 1 iff oracle==RETRIEVE
        assert d[0]["oracle_match"] == (1.0 if oracle == "RETRIEVE" else 0.0)
    print("PASS test_oracle_cases")


# ---------------------------------------------------------------- brier values
def test_brier_values():
    # c1=0.8, cA1=1 -> 1-(0.8-1)^2 = 1-0.04 = 0.96
    # ec=0.3, y=cA2=1 -> 1-(0.3-1)^2 = 1-0.49 = 0.51
    # c2=0.9, y=1 -> 1-(0.9-1)^2 = 1-0.01 = 0.99
    rl = [_T1(ans="Paris", conf="0.8"), _T2(action="RETRIEVE", ec="0.3"), _T3(ans="Paris", conf="0.9")]
    ri = _ri(rl, ["Paris"], 0.5, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]])
    d = _score(ri)
    expected_cal = (0.96 + 0.51 + 0.99) / 3.0
    assert _approx(d["cal_reward"], round(expected_cal, 6)), (d["cal_reward"], expected_cal)
    print("PASS test_brier_values")


# ---------------------------------------------------------------- action==oracle reward
def test_action_oracle_reward():
    # cA1=0, cA2=1, r=0.5 -> oracle RETRIEVE. emit RETRIEVE -> match=1
    rl = [_T1(ans="Berlin"), _T2(action="RETRIEVE"), _T3(ans="Paris")]
    ri = _ri(rl, ["Paris"], 0.5, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]])
    assert _score(ri)["oracle_match"] == 1.0
    # emit ANSWER -> mismatch (but force-train still ran T3; action recorded ANSWER)
    rl2 = [_T1(ans="Berlin"), _T2(action="ANSWER"), _T3(ans="Paris")]
    ri2 = _ri(rl2, ["Paris"], 0.5, [["CONTINUE", None], ["ANSWER", None], ["ANSWER", "Paris"]])
    assert _score(ri2)["oracle_match"] == 0.0
    print("PASS test_action_oracle_reward")


# ---------------------------------------------------------------- format gate
def test_format_gate():
    # missing <confidence> in T1 -> overall 0
    bad_t1 = "<think>x</think><answer>Paris</answer><analysis>sure</analysis>"
    rl = [bad_t1, _T2(), _T3()]
    ri = _ri(rl, ["Paris"], 0.5, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]])
    assert _score(ri)["overall"] == 0.0
    assert _score(ri)["format_reward"] == 0.0

    # truncated think in T3 (<think> without </think>) -> overall 0
    trunc_t3 = "<think>x<answer>Paris</answer><analysis>c</analysis><confidence>0.9</confidence>"
    rl2 = [_T1(), _T2(), trunc_t3]
    ri2 = _ri(rl2, ["Paris"], 0.5, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]])
    assert _score(ri2)["overall"] == 0.0

    # missing <action> in T2 -> overall 0
    bad_t2 = "<think>x</think><analysis>c</analysis><estimated_confidence>0.5</estimated_confidence>"
    rl3 = [_T1(), bad_t2, _T3()]
    ri3 = _ri(rl3, ["Paris"], 0.5, [["CONTINUE", None], ["CONTINUE", None], ["ANSWER", "Paris"]])
    assert _score(ri3)["overall"] == 0.0

    # trace too long (len(rl) > max_turns=3) -> overall 0
    rl4 = [_T1(), _T2(), _T3(), _T3()]
    ri4 = _ri(rl4, ["Paris"], 0.5, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"], ["ANSWER", "Paris"]])
    assert _score(ri4, max_turns=3)["overall"] == 0.0
    print("PASS test_format_gate")


# ---------------------------------------------------------------- end-to-end overall
def test_end_to_end_overall():
    # cA1=1, cA2=1, r=0.5; action RETRIEVE.
    # oracle = ANSWER (1 >= 0.5) -> act_reward = 0 (emitted RETRIEVE).
    # cal: c1=0.9 vs cA1=1 -> 0.99 ; ec=0.8 vs y=1 -> 0.96 ; c2=0.95 vs y=1 -> 0.9975
    # cal = (0.99 + 0.96 + 0.9975)/3 = 0.9825
    # core = (1*1 + 1*0 + 1*1 + 1*0.9825)/4 = 2.9825/4 = 0.745625
    # overall = 0.1 + 0.9*0.745625 = 0.7710625
    rl = [_T1(ans="Paris", conf="0.9"), _T2(action="RETRIEVE", ec="0.8"), _T3(ans="Paris", conf="0.95")]
    ri = _ri(rl, ["Paris"], 0.5, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]])
    d = _score(ri)
    assert d["oracle_match"] == 0.0
    cal = (0.99 + 0.96 + 0.9975) / 3.0
    core = (1.0 + 0.0 + 1.0 + cal) / 4.0
    overall = 0.1 + 0.9 * core
    assert _approx(d["cal_reward"], round(cal, 6)), (d["cal_reward"], cal)
    assert _approx(d["overall"], round(overall, 6)), (d["overall"], overall)
    assert d["a1_correct"] == 1.0 and d["a2_correct"] == 1.0
    print("PASS test_end_to_end_overall")


# ---------------------------------------------------------------- eval: action-following discounted_reward
def test_eval_action_following():
    # Eval ANSWER: final=A1, num_ret=0 -> discounted = cA1 * r^0 = cA1
    rl = [_T1(ans="Paris", conf="0.9"), _T2(action="ANSWER", ec="0.5")]
    ri = _ri(rl, ["Paris"], 0.4, [["CONTINUE", None], ["ANSWER", None]])
    d = _score(ri)
    assert d["num_retrieves"] == 0.0
    assert _approx(d["discounted_reward"], 1.0)  # cA1=1, r^0=1
    assert d["accuracy"] == 1.0
    # but format gate: T3 absent and not required when not retrieved; T2 ANSWER present -> fmt ok
    assert d["format_reward"] == 1.0

    # Eval RETRIEVE: final=A2, num_ret=1 -> discounted = cA2 * r
    rl2 = [_T1(ans="Berlin"), _T2(action="RETRIEVE"), _T3(ans="Paris")]
    ri2 = _ri(rl2, ["Paris"], 0.4, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]])
    d2 = _score(ri2)
    assert d2["num_retrieves"] == 1.0
    assert _approx(d2["discounted_reward"], 1.0 * 0.4)
    print("PASS test_eval_action_following")


# ---------------------------------------------------------------- group-mean (MC) oracle: r-sensitivity
def _build_group(gold, r, task_id, cA1_list, cA2_list, action="RETRIEVE"):
    """Build a GRPO group of rollouts sharing task_id & r. Each rollout's A1/A2 are made correct or
    wrong per cA1_list[i]/cA2_list[i]. The emitted T2 action is `action` for every rollout."""
    group = []
    for cA1, cA2 in zip(cA1_list, cA2_list):
        a1 = gold if cA1 else "Berlin"
        a2 = gold if cA2 else "Berlin"
        rl = [_T1(ans=a1), _T2(action=action), _T3(ans=a2)]
        aseq = [["CONTINUE", None], [action, None], ["ANSWER", a2]]
        group.append(_ri_tid(rl, [gold], r, aseq, task_id))
    return group


def test_group_oracle_r_sensitivity():
    """Group with p_A1=0.5, p_A2=1.0. Oracle flips with r (this is the WAVE-2 fix headline):
       r=0.4 -> 0.5 >= 0.4*1.0 = 0.4  -> ANSWER
       r=0.65-> 0.5 <  0.65*1.0 = 0.65 -> RETRIEVE
    The per-rollout binary oracle CANNOT do this (it is r-degenerate)."""
    # 4 rollouts: cA1 = [1,1,0,0] -> p_A1=0.5 ; cA2 = [1,1,1,1] -> p_A2=1.0
    cA1_list = [1, 1, 0, 0]
    cA2_list = [1, 1, 1, 1]

    # --- r = 0.4 -> oracle_group ANSWER ---
    grp_lo = _build_group("Paris", 0.4, "q1", cA1_list, cA2_list, action="RETRIEVE")
    res_lo = S.compute_score(grp_lo, oracle_mode="group", dump_path=None)
    # every rollout sees the SAME group oracle = ANSWER; emitted action RETRIEVE -> oracle_match 0
    # re-derive expected group oracle for assertion clarity (p_A1/p_A2 are in the dump diag).
    p_A1 = sum(cA1_list) / len(cA1_list)   # 0.5
    p_A2 = sum(cA2_list) / len(cA2_list)   # 1.0
    oracle_lo = "ANSWER" if p_A1 >= p_A2 * 0.4 else "RETRIEVE"
    assert oracle_lo == "ANSWER", oracle_lo
    assert all(r["oracle_match"] == 0.0 for r in res_lo)  # emitted RETRIEVE != ANSWER

    # If rollouts had emitted ANSWER instead, they would MATCH at r=0.4
    grp_lo_ans = _build_group("Paris", 0.4, "q1", cA1_list, cA2_list, action="ANSWER")
    res_lo_ans = S.compute_score(grp_lo_ans, oracle_mode="group", dump_path=None)
    assert all(r["oracle_match"] == 1.0 for r in res_lo_ans)

    # --- r = 0.65 -> oracle_group RETRIEVE (THE FLIP) ---
    grp_hi = _build_group("Paris", 0.65, "q1", cA1_list, cA2_list, action="RETRIEVE")
    res_hi = S.compute_score(grp_hi, oracle_mode="group", dump_path=None)
    oracle_hi = "ANSWER" if p_A1 >= p_A2 * 0.65 else "RETRIEVE"
    assert oracle_hi == "RETRIEVE", oracle_hi
    assert all(r["oracle_match"] == 1.0 for r in res_hi)  # emitted RETRIEVE == RETRIEVE

    # Same group emitting ANSWER at r=0.65 -> mismatch
    grp_hi_ans = _build_group("Paris", 0.65, "q1", cA1_list, cA2_list, action="ANSWER")
    res_hi_ans = S.compute_score(grp_hi_ans, oracle_mode="group", dump_path=None)
    assert all(r["oracle_match"] == 0.0 for r in res_hi_ans)
    print("PASS test_group_oracle_r_sensitivity (oracle FLIPS ANSWER@r=0.4 -> RETRIEVE@r=0.65)")


def test_group_oracle_diag_dump():
    """The dump must carry p_A1, p_A2, oracle_group, group_size diagnostics."""
    import tempfile, json, os as _os
    cA1_list = [1, 1, 0, 0]; cA2_list = [1, 1, 1, 1]
    grp = _build_group("Paris", 0.65, "qd", cA1_list, cA2_list, action="RETRIEVE")
    fd, path = tempfile.mkstemp(suffix=".jsonl"); _os.close(fd)
    try:
        S.compute_score(grp, oracle_mode="group", dump_path=path)
        recs = [json.loads(l) for l in open(path) if l.strip()]
        assert len(recs) == 4
        for rec in recs:
            assert _approx(rec["p_A1"], 0.5), rec["p_A1"]
            assert _approx(rec["p_A2"], 1.0), rec["p_A2"]
            assert rec["oracle_group"] == "RETRIEVE", rec["oracle_group"]
            assert rec["group_size"] == 4, rec["group_size"]
            assert rec["oracle_mode"] == "group"
    finally:
        _os.remove(path)
    print("PASS test_group_oracle_diag_dump")


def test_group_size_one_fallback():
    """A group of size 1 -> p == that rollout's binary correctness (per-rollout-like)."""
    # single rollout: cA1=0, cA2=1, r=0.5. p_A1=0, p_A2=1 -> 0 >= 0.5 False -> RETRIEVE
    grp = _build_group("Paris", 0.5, "qs", [0], [1], action="RETRIEVE")
    res = S.compute_score(grp, oracle_mode="group", dump_path=None)
    assert res[0]["oracle_match"] == 1.0  # oracle RETRIEVE, emitted RETRIEVE
    # rollout-mode on the same single sample gives the identical oracle here
    res_roll = S.compute_score(grp, oracle_mode="rollout", dump_path=None)
    assert res_roll[0]["oracle_match"] == 1.0
    print("PASS test_group_size_one_fallback")


def test_rollout_mode_unchanged():
    """oracle_mode default ('rollout') must reproduce the legacy per-rollout behavior exactly,
    independent of any other samples in the batch (no cross-sample leakage)."""
    # legacy case: cA1=0, cA2=1, r=0.5 -> oracle RETRIEVE; emit RETRIEVE -> match 1
    rl = [_T1(ans="Berlin"), _T2(action="RETRIEVE"), _T3(ans="Paris")]
    ri = _ri_tid(rl, ["Paris"], 0.5, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]], "qx")
    # default (no kwarg) and explicit "rollout" both equal
    assert _score(ri)["oracle_match"] == 1.0
    assert S.compute_score([ri], oracle_mode="rollout")[0]["oracle_match"] == 1.0

    # Batch this WITH other samples of the same task_id: rollout-mode must IGNORE the group.
    # Add a sibling with cA1=1,cA2=1 (which in group-mode would change p_A1) — rollout-mode unaffected.
    sib = _ri_tid([_T1(ans="Paris"), _T2(action="RETRIEVE"), _T3(ans="Paris")], ["Paris"], 0.5,
                  [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]], "qx")
    batch = [ri, sib]
    res_roll = S.compute_score(batch, oracle_mode="rollout")
    # sample0: cA1=0,cA2=1 -> oracle RETRIEVE -> match 1 ; sample1: cA1=1,cA2=1,r=0.5 -> ANSWER -> emit RETRIEVE -> 0
    assert res_roll[0]["oracle_match"] == 1.0
    assert res_roll[1]["oracle_match"] == 0.0
    # group-mode: p_A1=0.5,p_A2=1.0,r=0.5 -> 0.5>=0.5 -> ANSWER for BOTH -> both emit RETRIEVE -> 0
    res_grp = S.compute_score(batch, oracle_mode="group")
    assert res_grp[0]["oracle_match"] == 0.0 and res_grp[1]["oracle_match"] == 0.0
    print("PASS test_rollout_mode_unchanged (no cross-sample leakage; group differs from rollout)")


# ---------------------------------------------------------------- verbal oracle: r-sensitivity
def test_verbal_oracle_r_sensitivity():
    """Verbal oracle = ANSWER iff c1 >= ec*r. With c1=0.5, ec=0.8:
       ANSWER iff 0.5 >= 0.8*r iff r <= 0.625.
       r=0.5 -> 0.5 >= 0.40 -> ANSWER ; r=0.7 -> 0.5 >= 0.56 (False) -> RETRIEVE."""
    # action ANSWER emitted; matches oracle at r=0.5, mismatches at r=0.7
    rl = [_T1(ans="Paris", conf="0.5"), _T2(action="ANSWER", ec="0.8"), _T3(ans="Paris")]
    ri_lo = _ri(rl, ["Paris"], 0.5, [["CONTINUE", None], ["ANSWER", None], ["ANSWER", "Paris"]])
    d_lo = S.compute_score([ri_lo], oracle_mode="verbal", dump_path=None)[0]
    assert d_lo["oracle_match"] == 1.0, ("r=0.5 should -> ANSWER oracle", d_lo)

    ri_hi = _ri(rl, ["Paris"], 0.7, [["CONTINUE", None], ["ANSWER", None], ["ANSWER", "Paris"]])
    d_hi = S.compute_score([ri_hi], oracle_mode="verbal", dump_path=None)[0]
    assert d_hi["oracle_match"] == 0.0, ("r=0.7 should -> RETRIEVE oracle, emitted ANSWER", d_hi)

    # Symmetric: emit RETRIEVE -> mismatch at r=0.5, match at r=0.7
    rl_ret = [_T1(ans="Paris", conf="0.5"), _T2(action="RETRIEVE", ec="0.8"), _T3(ans="Paris")]
    ri_lo_r = _ri(rl_ret, ["Paris"], 0.5, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]])
    assert S.compute_score([ri_lo_r], oracle_mode="verbal")[0]["oracle_match"] == 0.0
    ri_hi_r = _ri(rl_ret, ["Paris"], 0.7, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]])
    assert S.compute_score([ri_hi_r], oracle_mode="verbal")[0]["oracle_match"] == 1.0
    print("PASS test_verbal_oracle_r_sensitivity (ANSWER@r=0.5 -> RETRIEVE@r=0.7 for c1=0.5,ec=0.8)")


def test_verbal_oracle_missing_conf():
    """If c1 or ec is None (format fail), verbal oracle is None -> oracle_match=0."""
    # missing ec in T2 -> ec None -> oracle None -> match 0 (also format gate zeroes overall)
    bad_t2 = "<think>x</think><analysis>c</analysis><action>ANSWER</action>"
    rl = [_T1(ans="Paris", conf="0.5"), bad_t2, _T3(ans="Paris")]
    ri = _ri(rl, ["Paris"], 0.5, [["CONTINUE", None], ["ANSWER", None], ["ANSWER", "Paris"]])
    d = S.compute_score([ri], oracle_mode="verbal")[0]
    assert d["oracle_match"] == 0.0, d
    assert d["overall"] == 0.0  # format gate
    print("PASS test_verbal_oracle_missing_conf")


def test_verbal_oracle_diag_dump():
    """Dump carries oracle_verbal, c1, ec, r diagnostics for the verbal mode."""
    import tempfile, json, os as _os
    rl = [_T1(ans="Paris", conf="0.5"), _T2(action="ANSWER", ec="0.8"), _T3(ans="Paris")]
    ri = _ri(rl, ["Paris"], 0.7, [["CONTINUE", None], ["ANSWER", None], ["ANSWER", "Paris"]])
    fd, path = tempfile.mkstemp(suffix=".jsonl"); _os.close(fd)
    try:
        S.compute_score([ri], oracle_mode="verbal", dump_path=path)
        rec = [json.loads(l) for l in open(path) if l.strip()][0]
        assert rec["oracle_verbal"] == "RETRIEVE", rec["oracle_verbal"]  # 0.5 < 0.8*0.7=0.56
        assert _approx(rec["c1"], 0.5) and _approx(rec["ec"], 0.8)
        assert _approx(rec["r"], 0.7)
        assert rec["oracle_mode"] == "verbal"
        assert rec["oracle_mc"] is None  # not computed in verbal mode
    finally:
        _os.remove(path)
    print("PASS test_verbal_oracle_diag_dump")


# ---------------------------------------------------------------- combined oracle: 0.5/0.5 blend
def test_combined_oracle_blend():
    """combined: oracle_match = 0.5*1[action==oracle_mc] + 0.5*1[action==oracle_verbal].
    Construct a group where MC and verbal oracles DISAGREE and verify the blend per rollout.

    Group of 4 (task_id qc), r=0.5:
      cA1 = [1,1,0,0] -> p_A1=0.5 ; cA2 = [1,1,1,1] -> p_A2=1.0
      oracle_mc = ANSWER iff 0.5 >= 1.0*0.5 = 0.5 -> ANSWER (tie -> ANSWER).
    Give each rollout c1=0.3, ec=0.9 -> verbal: ANSWER iff 0.3 >= 0.9*0.5=0.45 (False) -> RETRIEVE.
    So oracle_mc=ANSWER, oracle_verbal=RETRIEVE (DISAGREE).
      emit ANSWER  -> 0.5*1 + 0.5*0 = 0.5
      emit RETRIEVE-> 0.5*0 + 0.5*1 = 0.5
    """
    gold = "Paris"; r = 0.5
    cA1_list = [1, 1, 0, 0]; cA2_list = [1, 1, 1, 1]

    def _grp(action):
        g = []
        for cA1, cA2 in zip(cA1_list, cA2_list):
            a1 = gold if cA1 else "Berlin"
            a2 = gold if cA2 else "Berlin"
            rl = [_T1(ans=a1, conf="0.3"), _T2(action=action, ec="0.9"), _T3(ans=a2)]
            aseq = [["CONTINUE", None], [action, None], ["ANSWER", a2]]
            g.append(_ri_tid(rl, [gold], r, aseq, "qc"))
        return g

    res_ans = S.compute_score(_grp("ANSWER"), oracle_mode="combined", dump_path=None)
    assert all(_approx(x["oracle_match"], 0.5) for x in res_ans), [x["oracle_match"] for x in res_ans]

    res_ret = S.compute_score(_grp("RETRIEVE"), oracle_mode="combined", dump_path=None)
    assert all(_approx(x["oracle_match"], 0.5) for x in res_ret), [x["oracle_match"] for x in res_ret]

    # Sanity: when they AGREE the blend is 0 or 1. Make verbal also ANSWER (c1=0.9, ec=0.5 -> 0.9>=0.25).
    def _grp_agree(action):
        g = []
        for cA1, cA2 in zip(cA1_list, cA2_list):
            a1 = gold if cA1 else "Berlin"
            a2 = gold if cA2 else "Berlin"
            rl = [_T1(ans=a1, conf="0.9"), _T2(action=action, ec="0.5"), _T3(ans=a2)]
            aseq = [["CONTINUE", None], [action, None], ["ANSWER", a2]]
            g.append(_ri_tid(rl, [gold], r, aseq, "qca"))
        return g
    # oracle_mc=ANSWER, oracle_verbal: 0.9 >= 0.5*0.5=0.25 -> ANSWER. emit ANSWER -> both match -> 1.0
    res_agree = S.compute_score(_grp_agree("ANSWER"), oracle_mode="combined")
    assert all(_approx(x["oracle_match"], 1.0) for x in res_agree), [x["oracle_match"] for x in res_agree]
    # emit RETRIEVE -> both mismatch -> 0.0
    res_agree_r = S.compute_score(_grp_agree("RETRIEVE"), oracle_mode="combined")
    assert all(_approx(x["oracle_match"], 0.0) for x in res_agree_r)
    print("PASS test_combined_oracle_blend (0.5 blend on disagreement; 0/1 on agreement)")


def test_combined_oracle_diag_dump():
    """combined dump carries BOTH oracle_mc and oracle_verbal."""
    import tempfile, json, os as _os
    gold = "Paris"; r = 0.5
    cA1_list = [1, 1, 0, 0]; cA2_list = [1, 1, 1, 1]
    g = []
    for cA1, cA2 in zip(cA1_list, cA2_list):
        a1 = gold if cA1 else "Berlin"
        a2 = gold if cA2 else "Berlin"
        rl = [_T1(ans=a1, conf="0.3"), _T2(action="ANSWER", ec="0.9"), _T3(ans=a2)]
        aseq = [["CONTINUE", None], ["ANSWER", None], ["ANSWER", a2]]
        g.append(_ri_tid(rl, [gold], r, aseq, "qcd"))
    fd, path = tempfile.mkstemp(suffix=".jsonl"); _os.close(fd)
    try:
        S.compute_score(g, oracle_mode="combined", dump_path=path)
        recs = [json.loads(l) for l in open(path) if l.strip()]
        assert len(recs) == 4
        for rec in recs:
            assert rec["oracle_mc"] == "ANSWER", rec["oracle_mc"]      # p_A1=0.5 >= 0.5
            assert rec["oracle_verbal"] == "RETRIEVE", rec["oracle_verbal"]  # 0.3 < 0.45
            assert rec["oracle_mode"] == "combined"
            assert rec["group_size"] == 4
    finally:
        _os.remove(path)
    print("PASS test_combined_oracle_diag_dump")


# ---------------------------------------------------------------- precomp oracle: r-sensitivity
def _ri_precomp(response_list, answers, r, action_seqs, p_A1, p_A2, task_id="qp"):
    d = _ri_tid(response_list, answers, r, action_seqs, task_id)
    d["p_A1"] = p_A1
    d["p_A2"] = p_A2
    return d


def test_precomp_oracle_r_sensitivity():
    """Precomp oracle = ANSWER iff p_A1 >= p_A2*r, from BAKED columns (NOT realized cA1/cA2).
    p_A1=0.4, p_A2=0.8 -> ANSWER iff 0.4 >= 0.8*r iff r <= 0.5.
      r=0.4 -> 0.4 >= 0.32 -> ANSWER
      r=0.6 -> 0.4 <  0.48 -> RETRIEVE
    Note: realized A1/A2 in the rollout are made WRONG/RIGHT to prove the columns (not cA1/cA2) drive it."""
    # Realized: A1 wrong, A2 right (cA1=0, cA2=1) -> the binary 'rollout' oracle would be RETRIEVE
    # regardless of r; precomp must instead use p_A1=0.4, p_A2=0.8 and flip with r.
    rl = [_T1(ans="Berlin"), _T2(action="ANSWER", ec="0.5"), _T3(ans="Paris")]

    # r=0.4 -> oracle ANSWER ; emit ANSWER -> match 1
    ri_lo = _ri_precomp(rl, ["Paris"], 0.4,
                        [["CONTINUE", None], ["ANSWER", None], ["ANSWER", "Paris"]], 0.4, 0.8)
    d_lo = S.compute_score([ri_lo], oracle_mode="precomp", dump_path=None)[0]
    assert d_lo["oracle_match"] == 1.0, ("r=0.4 should -> ANSWER oracle (p_A1>=p_A2*r)", d_lo)

    # r=0.6 -> oracle RETRIEVE ; emit ANSWER -> mismatch 0
    ri_hi = _ri_precomp(rl, ["Paris"], 0.6,
                        [["CONTINUE", None], ["ANSWER", None], ["ANSWER", "Paris"]], 0.4, 0.8)
    d_hi = S.compute_score([ri_hi], oracle_mode="precomp", dump_path=None)[0]
    assert d_hi["oracle_match"] == 0.0, ("r=0.6 should -> RETRIEVE oracle, emitted ANSWER", d_hi)

    # Symmetric: emit RETRIEVE -> mismatch at r=0.4, match at r=0.6
    rl_ret = [_T1(ans="Berlin"), _T2(action="RETRIEVE", ec="0.5"), _T3(ans="Paris")]
    ri_lo_r = _ri_precomp(rl_ret, ["Paris"], 0.4,
                          [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]], 0.4, 0.8)
    assert S.compute_score([ri_lo_r], oracle_mode="precomp")[0]["oracle_match"] == 0.0
    ri_hi_r = _ri_precomp(rl_ret, ["Paris"], 0.6,
                          [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]], 0.4, 0.8)
    assert S.compute_score([ri_hi_r], oracle_mode="precomp")[0]["oracle_match"] == 1.0
    print("PASS test_precomp_oracle_r_sensitivity (ANSWER@r=0.4 -> RETRIEVE@r=0.6 for p_A1=0.4,p_A2=0.8)")


def test_precomp_oracle_uses_columns_not_realized():
    """Precomp must use the baked p_A1/p_A2, IGNORING this rollout's realized cA1/cA2.
    Construct a rollout that is fully CORRECT (cA1=1,cA2=1) but columns say p_A1=0.0,p_A2=1.0:
      oracle = ANSWER iff 0.0 >= 1.0*r -> RETRIEVE for any r in (0,1]. emit RETRIEVE -> match."""
    rl = [_T1(ans="Paris"), _T2(action="RETRIEVE", ec="0.5"), _T3(ans="Paris")]  # cA1=cA2=1
    ri = _ri_precomp(rl, ["Paris"], 0.5,
                     [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]], 0.0, 1.0)
    d = S.compute_score([ri], oracle_mode="precomp")[0]
    assert d["oracle_match"] == 1.0, ("columns p_A1=0 -> RETRIEVE oracle despite cA1=1", d)
    # rollout-mode on the SAME sample: cA1=1>=cA2*0.5 -> ANSWER -> emit RETRIEVE -> mismatch 0 (proves divergence)
    d_roll = S.compute_score([ri], oracle_mode="rollout")[0]
    assert d_roll["oracle_match"] == 0.0
    print("PASS test_precomp_oracle_uses_columns_not_realized")


def test_precomp_oracle_missing_columns_fallback():
    """If p_A1/p_A2 columns are absent, precomp falls back to the rollout's binary correctness."""
    # cA1=0, cA2=1, r=0.5, NO columns -> falls back to 0 >= 1*0.5 -> RETRIEVE. emit RETRIEVE -> match.
    rl = [_T1(ans="Berlin"), _T2(action="RETRIEVE", ec="0.5"), _T3(ans="Paris")]
    ri = _ri_tid(rl, ["Paris"], 0.5,
                 [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]], "qpf")
    d = S.compute_score([ri], oracle_mode="precomp")[0]
    assert d["oracle_match"] == 1.0, d
    print("PASS test_precomp_oracle_missing_columns_fallback")


def test_precomp_oracle_negative_sentinel_fallback():
    """A negative sentinel (-1.0, baked into eval rows lacking a precomp estimate) is treated as
    missing -> falls back to the rollout's binary correctness (not a degenerate p=0 oracle)."""
    # cA1=0, cA2=1, r=0.5, columns=-1 -> fallback -> 0 >= 1*0.5 -> RETRIEVE. emit RETRIEVE -> 1.
    rl = [_T1(ans="Berlin"), _T2(action="RETRIEVE", ec="0.5"), _T3(ans="Paris")]
    ri = _ri_precomp(rl, ["Paris"], 0.5,
                     [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]], -1.0, -1.0)
    d = S.compute_score([ri], oracle_mode="precomp")[0]
    assert d["oracle_match"] == 1.0, d
    print("PASS test_precomp_oracle_negative_sentinel_fallback")


def test_precomp_oracle_diag_dump():
    """The dump carries p_A1, p_A2 (the BAKED values), oracle, oracle_mode for precomp."""
    import tempfile, json, os as _os
    rl = [_T1(ans="Berlin"), _T2(action="ANSWER", ec="0.5"), _T3(ans="Paris")]
    ri = _ri_precomp(rl, ["Paris"], 0.6,
                     [["CONTINUE", None], ["ANSWER", None], ["ANSWER", "Paris"]], 0.4, 0.8)
    fd, path = tempfile.mkstemp(suffix=".jsonl"); _os.close(fd)
    try:
        S.compute_score([ri], oracle_mode="precomp", dump_path=path)
        rec = [json.loads(l) for l in open(path) if l.strip()][0]
        assert _approx(rec["p_A1"], 0.4), rec["p_A1"]
        assert _approx(rec["p_A2"], 0.8), rec["p_A2"]
        assert rec["oracle"] == "RETRIEVE", rec["oracle"]   # 0.4 < 0.8*0.6=0.48
        assert rec["oracle_mode"] == "precomp"
        assert rec["oracle_mc"] is None  # not a group mode
    finally:
        _os.remove(path)
    print("PASS test_precomp_oracle_diag_dump")


def test_precomp_does_not_change_legacy_modes():
    """Regression: presence of p_A1/p_A2 columns must NOT alter rollout/group/verbal/combined.
    They only read p_A1/p_A2 in precomp mode."""
    rl = [_T1(ans="Berlin", conf="0.4"), _T2(action="RETRIEVE", ec="0.8"), _T3(ans="Paris")]
    # add bogus columns that WOULD flip a precomp oracle but must be ignored by other modes
    ri = _ri_precomp(rl, ["Paris"], 0.5,
                     [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]], 1.0, 0.0, "qpr")
    # rollout: cA1=0,cA2=1,r=0.5 -> RETRIEVE -> emit RETRIEVE -> 1 (unchanged by columns)
    assert S.compute_score([ri], oracle_mode="rollout")[0]["oracle_match"] == 1.0
    assert _score(ri)["oracle_match"] == 1.0  # default==rollout, columns ignored
    # verbal: c1=0.4,ec=0.8,r=0.5 -> ANSWER iff 0.4>=0.4 -> ANSWER; emit RETRIEVE -> 0 (unchanged)
    assert S.compute_score([ri], oracle_mode="verbal")[0]["oracle_match"] == 0.0
    print("PASS test_precomp_does_not_change_legacy_modes")


def test_new_modes_do_not_change_rollout_group():
    """Regression: adding verbal/combined must not alter rollout or group results."""
    # rollout: cA1=0,cA2=1,r=0.5 -> oracle RETRIEVE; emit RETRIEVE -> 1
    rl = [_T1(ans="Berlin", conf="0.4"), _T2(action="RETRIEVE", ec="0.8"), _T3(ans="Paris")]
    ri = _ri_tid(rl, ["Paris"], 0.5, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]], "qz")
    assert S.compute_score([ri], oracle_mode="rollout")[0]["oracle_match"] == 1.0
    assert _score(ri)["oracle_match"] == 1.0  # default == rollout
    # group on a real group (the canonical wave-2 case) still flips by r — unchanged.
    cA1_list = [1, 1, 0, 0]; cA2_list = [1, 1, 1, 1]
    grp = _build_group("Paris", 0.65, "qzg", cA1_list, cA2_list, action="RETRIEVE")
    assert all(x["oracle_match"] == 1.0 for x in S.compute_score(grp, oracle_mode="group"))
    print("PASS test_new_modes_do_not_change_rollout_group")


def _load_rollout_parser():
    """Extract `parse_turn_action_turn` from the rollout module WITHOUT importing vllm/torch.

    The function is pure-python (only needs the `re` module aliased as `_re_popqa`). We read the
    rollout source, slice out the function, and exec it in an isolated namespace. This lets the
    parser be unit-tested on a CPU-only login node / CI without a GPU.
    """
    import os, re as _re, types
    here = os.path.dirname(os.path.abspath(__file__))
    # rollout lives at EasyR1/verl/workers/rollout/multi_turn_rollout_popqa.py;
    # this test lives at EasyR1/training/reward_function/. Resolve relatively, with a /tmp fallback.
    candidates = [
        os.path.normpath(os.path.join(here, "..", "..", "verl", "workers", "rollout", "multi_turn_rollout_popqa.py")),
        "/tmp/rollout_new.py",
    ]
    src_path = next((c for c in candidates if os.path.exists(c)), None)
    assert src_path is not None, f"rollout source not found in {candidates}"
    src = open(src_path).read()
    start = src.index("def parse_turn_action_turn(")
    # find the next top-level def after it
    nxt = src.index("\ndef ", start + 1)
    func_src = src[start:nxt]
    ns = {"_re_popqa": _re}
    exec(func_src, ns)
    mod = types.SimpleNamespace(parse_turn_action_turn=ns["parse_turn_action_turn"])
    return mod


# ---------------------------------------------------------------- turn-aware parser logic
def test_turn_aware_parser():
    # Load the rollout module's turn-aware parser without importing vllm/torch.
    R = _load_rollout_parser()
    # T1: answer present, NO action -> CONTINUE (do not stop, do not inject ctx)
    t1_txt = _T1(ans="Paris")
    assert R.parse_turn_action_turn(t1_txt, 0, force_retrieve=True) == ("CONTINUE", "Paris")
    assert R.parse_turn_action_turn(t1_txt, 0, force_retrieve=False) == ("CONTINUE", "Paris")

    # T2 force_retrieve=True: always RETRIEVE regardless of emitted action; record emitted action
    t2_ans = _T2(action="ANSWER")
    out = R.parse_turn_action_turn(t2_ans, 1, force_retrieve=True)
    assert out[0] == "RETRIEVE", out  # forced
    t2_ret = _T2(action="RETRIEVE")
    assert R.parse_turn_action_turn(t2_ret, 1, force_retrieve=True)[0] == "RETRIEVE"

    # T2 force_retrieve=False (eval): follow emitted action
    assert R.parse_turn_action_turn(t2_ans, 1, force_retrieve=False)[0] == "ANSWER"
    assert R.parse_turn_action_turn(t2_ret, 1, force_retrieve=False)[0] == "RETRIEVE"

    # T3: answer present -> stop (ANSWER)
    t3_txt = _T3(ans="Paris")
    assert R.parse_turn_action_turn(t3_txt, 2, force_retrieve=True) == ("ANSWER", "Paris")

    # Truncated think at any turn -> TRUNCATED
    trunc = "<think>x<answer>Paris</answer>"
    assert R.parse_turn_action_turn(trunc, 0, force_retrieve=True)[0] == "TRUNCATED"
    assert R.parse_turn_action_turn(trunc, 1, force_retrieve=True)[0] == "TRUNCATED"

    # The recorded emitted action token (for the reward) should be ANSWER for the forced case
    out2 = R.parse_turn_action_turn(t2_ans, 1, force_retrieve=True)
    assert "ANSWER" in [str(x).upper() for x in out2], out2
    print("PASS test_turn_aware_parser")


# ------------------------------------------------- ec-BEFORE-action (ecfirst variant) parsing
def _T2_ecfirst(action="RETRIEVE", ec="0.8", analysis="decide"):
    """T2 with <estimated_confidence> emitted BEFORE <action> (the ecfirst variant)."""
    return (f"<think>x</think><analysis>{analysis}</analysis>"
            f"<estimated_confidence>{ec}</estimated_confidence><action>{action}</action>")


def test_ecfirst_rollout_parser_order_independent():
    """The rollout's <action> regex searches the whole body, so ec-before-action parses fine.
    Eval (force_retrieve=False) must still follow the emitted action; train forces RETRIEVE."""
    R = _load_rollout_parser()
    t2_ans = _T2_ecfirst(action="ANSWER", ec="0.8")
    t2_ret = _T2_ecfirst(action="RETRIEVE", ec="0.3")
    # eval: follow emitted action regardless of ec position
    assert R.parse_turn_action_turn(t2_ans, 1, force_retrieve=False)[0] == "ANSWER"
    assert R.parse_turn_action_turn(t2_ret, 1, force_retrieve=False)[0] == "RETRIEVE"
    # train: always RETRIEVE, record emitted action token
    out = R.parse_turn_action_turn(t2_ans, 1, force_retrieve=True)
    assert out[0] == "RETRIEVE" and "ANSWER" in [str(x).upper() for x in out], out
    print("PASS test_ecfirst_rollout_parser_order_independent")


def test_ecfirst_reward_parse_and_verbal():
    """Reward-side: ec-before-action T2 still parses action+ec; verbal oracle then works.
    c1=0.5, ec=0.8, r=0.5 -> ANSWER oracle; emit ANSWER -> match 1; format gate passes."""
    rl = [_T1(ans="Paris", conf="0.5"), _T2_ecfirst(action="ANSWER", ec="0.8"), _T3(ans="Paris")]
    ri = _ri(rl, ["Paris"], 0.5, [["CONTINUE", None], ["ANSWER", None], ["ANSWER", "Paris"]])
    d = S.compute_score([ri], oracle_mode="verbal")[0]
    assert d["format_reward"] == 1.0, d          # ec & action both parsed despite order
    assert d["oracle_match"] == 1.0, d           # verbal oracle ANSWER, emitted ANSWER
    print("PASS test_ecfirst_reward_parse_and_verbal")


# ---------------------------------------------------------------- w_ec calibration up-weight
def test_w_ec_default_is_identical():
    """KEY SAFETY CHECK: at w_ec=1.0 the weighted cal mean reduces EXACTLY to the plain mean,
    so cal_reward and overall are byte-identical to the pre-w_ec behavior for a worked example.

    Reuse the test_end_to_end_overall worked example (c1=0.9,ec=0.8,c2=0.95, cA1=cA2=1, r=0.5):
      cal_plain = (0.99 + 0.96 + 0.9975)/3 = 0.9825  (the pre-change value)
      core      = (1 + 0 + 1 + 0.9825)/4
      overall   = 0.1 + 0.9*core
    Assert that the explicit w_ec=1.0 result equals BOTH the hand value AND the default (no-kwarg) call.
    """
    rl = [_T1(ans="Paris", conf="0.9"), _T2(action="RETRIEVE", ec="0.8"), _T3(ans="Paris", conf="0.95")]
    ri = _ri(rl, ["Paris"], 0.5, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]])

    cal_plain = (0.99 + 0.96 + 0.9975) / 3.0          # pre-change plain mean
    core = (1.0 + 0.0 + 1.0 + cal_plain) / 4.0
    overall = 0.1 + 0.9 * core

    d_default = _score(ri)                              # default (no w_ec) == legacy
    d_w1 = _score(ri, w_ec=1.0)                         # explicit w_ec=1.0
    # 1) explicit w_ec=1.0 matches the hand-computed PRE-CHANGE cal/overall
    assert _approx(d_w1["cal_reward"], round(cal_plain, 6)), (d_w1["cal_reward"], cal_plain)
    assert _approx(d_w1["overall"], round(overall, 6)), (d_w1["overall"], overall)
    # 2) explicit w_ec=1.0 is byte-identical to the default call (no behavior change at default)
    assert d_w1["cal_reward"] == d_default["cal_reward"], (d_w1, d_default)
    assert d_w1["overall"] == d_default["overall"], (d_w1, d_default)
    print("PASS test_w_ec_default_is_identical (w_ec=1.0 == legacy plain mean, regression-safe)")


def test_w_ec_upweights_ec_term():
    """At w_ec=3.0 the ec Brier term dominates the weighted cal mean. Hand-computed.

    Same worked example: c1=0.9/cA1=1 -> 0.99 ; ec=0.8/y=1 -> 0.96 ; c2=0.95/y=1 -> 0.9975.
    Weighted mean with weights (1.0, 3.0, 1.0):
      cal = (1*0.99 + 3*0.96 + 1*0.9975) / (1 + 3 + 1)
          = (0.99 + 2.88 + 0.9975) / 5.0
          = 4.8675 / 5.0 = 0.9735
    This is LOWER than the plain mean 0.9825 because the (worse-calibrated) ec term is up-weighted.
      core    = (1 + 0 + 1 + 0.9735)/4
      overall = 0.1 + 0.9*core
    """
    rl = [_T1(ans="Paris", conf="0.9"), _T2(action="RETRIEVE", ec="0.8"), _T3(ans="Paris", conf="0.95")]
    ri = _ri(rl, ["Paris"], 0.5, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]])

    b_c1, b_ec, b_c2 = 0.99, 0.96, 0.9975
    cal = (1.0 * b_c1 + 3.0 * b_ec + 1.0 * b_c2) / (1.0 + 3.0 + 1.0)
    assert _approx(cal, 0.9735), cal
    core = (1.0 + 0.0 + 1.0 + cal) / 4.0
    overall = 0.1 + 0.9 * core

    d = _score(ri, w_ec=3.0)
    assert _approx(d["cal_reward"], round(cal, 6)), (d["cal_reward"], cal)
    assert _approx(d["overall"], round(overall, 6)), (d["overall"], overall)
    # the up-weighted ec term pulled cal BELOW the plain mean (0.9825) -> ec dominates as intended
    cal_plain = (b_c1 + b_ec + b_c2) / 3.0
    assert d["cal_reward"] < round(cal_plain, 6), (d["cal_reward"], cal_plain)
    print("PASS test_w_ec_upweights_ec_term (w_ec=3.0 -> ec Brier dominates cal: 0.9735 < 0.9825)")


def test_w_ec_absent_ec_unaffected():
    """When ec is absent (eval ANSWER path, no T3 retrieval) w_ec must not affect cal at all:
    only the c1 term is present, so cal == that term regardless of w_ec."""
    # Eval ANSWER: T1 + T2(ANSWER), no T3 -> ec present in T2 though. Use a no-T2-ec case instead:
    # build a 2-element rollout where T2 has ec, but to isolate "ec absent" we drop T2/T3 parsing of ec.
    # Simplest: T1 only + T2 ANSWER (ec present) is NOT ec-absent. Construct genuine ec-absent via
    # a single-turn-ish rollout where T2 is absent is a format fail; instead test that with ONLY c1
    # contributing (no ec, no c2) cal is invariant. We force that by a T2 lacking ec is a format
    # fail (overall 0) but cal still computed from present terms. Use rl with T1 + T2(no ec) + T3.
    bad_t2 = "<think>x</think><analysis>c</analysis><action>RETRIEVE</action>"  # no estimated_confidence
    rl = [_T1(ans="Paris", conf="0.9"), bad_t2, _T3(ans="Paris", conf="0.95")]
    ri = _ri(rl, ["Paris"], 0.5, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]])
    # cal here uses c1 (0.99) and c2 (0.9975) only; ec absent -> w_ec irrelevant.
    cal_expected = (0.99 + 0.9975) / 2.0
    d1 = _score(ri, w_ec=1.0)
    d3 = _score(ri, w_ec=3.0)
    assert _approx(d1["cal_reward"], round(cal_expected, 6)), d1["cal_reward"]
    assert d1["cal_reward"] == d3["cal_reward"], (d1["cal_reward"], d3["cal_reward"])
    print("PASS test_w_ec_absent_ec_unaffected (ec absent -> w_ec has no effect on cal)")


# ---------------------------------------------------------------- ec_target: distill p_A2 into ec Brier
def test_ec_target_default_cA2_identical():
    """REGRESSION: ec_target default ("cA2") must be byte-identical to the legacy behavior on a
    worked example, AND identical to an explicit ec_target="cA2" call. Reuses the
    test_end_to_end_overall example (c1=0.9,ec=0.8,c2=0.95, cA1=cA2=1, r=0.5):
      cal = (0.99 + 0.96 + 0.9975)/3 = 0.9825 ; core=(1+0+1+0.9825)/4 ; overall=0.1+0.9*core.
    Also: even WITH a p_A2 column present, ec_target=cA2 must IGNORE it (use cA2)."""
    rl = [_T1(ans="Paris", conf="0.9"), _T2(action="RETRIEVE", ec="0.8"), _T3(ans="Paris", conf="0.95")]
    ri = _ri(rl, ["Paris"], 0.5, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]])
    cal = (0.99 + 0.96 + 0.9975) / 3.0
    core = (1.0 + 0.0 + 1.0 + cal) / 4.0
    overall = 0.1 + 0.9 * core

    d_default = _score(ri)                       # default (no ec_target) == legacy
    d_explicit = _score(ri, ec_target="cA2")     # explicit cA2
    assert _approx(d_default["cal_reward"], round(cal, 6)), d_default["cal_reward"]
    assert _approx(d_default["overall"], round(overall, 6)), d_default["overall"]
    assert d_default["cal_reward"] == d_explicit["cal_reward"], (d_default, d_explicit)
    assert d_default["overall"] == d_explicit["overall"], (d_default, d_explicit)

    # p_A2 column present but ec_target=cA2 -> still uses cA2 (column ignored).
    ri_col = dict(ri); ri_col["p_A2"] = 0.2
    d_col = _score(ri_col, ec_target="cA2")
    assert d_col["cal_reward"] == d_default["cal_reward"], (d_col, d_default)
    print("PASS test_ec_target_default_cA2_identical (default==legacy; column ignored at cA2)")


def test_ec_target_p_a2_uses_precomputed_label():
    """ec_target="p_a2": ec Brier term scored vs the precomputed p_A2 column (continuous label),
    NOT this rollout's cA2. With p_A2=0.7, ec=0.6:
      ec Brier term = 1-(0.6-0.7)^2 = 1-0.01 = 0.99  (hand-check)
    Same example otherwise: c1=0.9/cA1=1 -> 0.99 ; c2=0.95/y(=cA2=1) -> 0.9975 (UNCHANGED targets).
      cal_p_a2 = (0.99 + 0.99 + 0.9975)/3
    The cA2-target value uses ec vs cA2=1 -> 1-(0.6-1)^2 = 0.84, so the two MUST differ."""
    rl = [_T1(ans="Paris", conf="0.9"), _T2(action="RETRIEVE", ec="0.6"), _T3(ans="Paris", conf="0.95")]
    ri = _ri(rl, ["Paris"], 0.5, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]])
    ri_p = dict(ri); ri_p["p_A2"] = 0.7

    b_ec_p = 1.0 - (0.6 - 0.7) ** 2   # 0.99
    assert _approx(b_ec_p, 0.99), b_ec_p
    cal_p = (0.99 + b_ec_p + 0.9975) / 3.0
    d_p = _score(ri_p, ec_target="p_a2")
    assert _approx(d_p["cal_reward"], round(cal_p, 6)), (d_p["cal_reward"], cal_p)

    # cA2-target value differs (ec vs cA2=1 -> 0.84).
    b_ec_cA2 = 1.0 - (0.6 - 1.0) ** 2   # 0.84
    cal_cA2 = (0.99 + b_ec_cA2 + 0.9975) / 3.0
    d_cA2 = _score(ri_p, ec_target="cA2")
    assert _approx(d_cA2["cal_reward"], round(cal_cA2, 6)), (d_cA2["cal_reward"], cal_cA2)
    assert d_p["cal_reward"] != d_cA2["cal_reward"], (d_p["cal_reward"], d_cA2["cal_reward"])

    # c1 and c2 Brier terms unchanged: verify via dump diag (brier_c1, brier_c2) equal across targets.
    import tempfile, json, os as _os
    def _diag(ri_in, tgt):
        fd, path = tempfile.mkstemp(suffix=".jsonl"); _os.close(fd)
        try:
            S.compute_score([ri_in], ec_target=tgt, dump_path=path)
            return [json.loads(l) for l in open(path) if l.strip()][0]
        finally:
            _os.remove(path)
    dg_p = _diag(ri_p, "p_a2"); dg_c = _diag(ri_p, "cA2")
    assert dg_p["brier_c1"] == dg_c["brier_c1"], (dg_p["brier_c1"], dg_c["brier_c1"])
    assert dg_p["brier_c2"] == dg_c["brier_c2"], (dg_p["brier_c2"], dg_c["brier_c2"])
    assert _approx(dg_p["ec_y"], 0.7), dg_p["ec_y"]          # ec target became p_A2
    assert _approx(dg_p["brier_ec"], 0.01), dg_p["brier_ec"] # (0.6-0.7)^2
    assert dg_p["ec_target"] == "p_a2"
    print("PASS test_ec_target_p_a2_uses_precomputed_label (ec vs p_A2=0.7; c1/c2 unchanged)")


def test_ec_target_p_a2_missing_falls_back_to_cA2():
    """ec_target="p_a2" with NO p_A2 column (or negative sentinel) -> falls back to cA2 target,
    so the result equals the cA2-target result (eval splits without precomp columns still work)."""
    rl = [_T1(ans="Paris", conf="0.9"), _T2(action="RETRIEVE", ec="0.6"), _T3(ans="Paris", conf="0.95")]
    # no p_A2 column at all
    ri = _ri(rl, ["Paris"], 0.5, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]])
    d_p = _score(ri, ec_target="p_a2")
    d_c = _score(ri, ec_target="cA2")
    assert d_p["cal_reward"] == d_c["cal_reward"], (d_p, d_c)
    # negative sentinel -> also fallback
    ri_neg = dict(ri); ri_neg["p_A2"] = -1.0
    d_neg = _score(ri_neg, ec_target="p_a2")
    assert d_neg["cal_reward"] == d_c["cal_reward"], (d_neg, d_c)
    print("PASS test_ec_target_p_a2_missing_falls_back_to_cA2")


def test_ec_target_independent_of_oracle_mode():
    """ec_target works under any oracle_mode (e.g. verbal) and only touches the ec term.
    Under oracle_mode=verbal with p_A2=0.7, ec=0.6: ec Brier vs 0.7 (0.99), c1 unchanged."""
    rl = [_T1(ans="Paris", conf="0.9"), _T2(action="ANSWER", ec="0.6"), _T3(ans="Paris", conf="0.95")]
    ri = _ri(rl, ["Paris"], 0.5, [["CONTINUE", None], ["ANSWER", None], ["ANSWER", "Paris"]])
    ri["p_A2"] = 0.7
    import tempfile, json, os as _os
    fd, path = tempfile.mkstemp(suffix=".jsonl"); _os.close(fd)
    try:
        S.compute_score([ri], oracle_mode="verbal", ec_target="p_a2", dump_path=path)
        rec = [json.loads(l) for l in open(path) if l.strip()][0]
        assert rec["oracle_mode"] == "verbal"
        assert _approx(rec["ec_y"], 0.7), rec["ec_y"]
        assert _approx(rec["brier_ec"], 0.01), rec["brier_ec"]
        assert rec["ec_target"] == "p_a2"
    finally:
        _os.remove(path)
    print("PASS test_ec_target_independent_of_oracle_mode")



# ---------------------------------------------------------------- per-confidence cal weights
def test_cal_weights_default_is_identical():
    """REGRESSION: at (w_c1cal, w_ec, w_c2cal) == (1,1,1) -- the defaults -- cal_reward and
    overall must be BYTE-IDENTICAL to the legacy (no-kwarg) call AND to the hand value.

    Reuse the worked example (c1=0.9,ec=0.8,c2=0.95, cA1=cA2=1, r=0.5):
      cal_plain = (0.99 + 0.96 + 0.9975)/3 = 0.9825
      core      = (1 + 0 + 1 + 0.9825)/4 ; overall = 0.1 + 0.9*core
    """
    rl = [_T1(ans="Paris", conf="0.9"), _T2(action="RETRIEVE", ec="0.8"), _T3(ans="Paris", conf="0.95")]
    ri = _ri(rl, ["Paris"], 0.5, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]])

    cal_plain = (0.99 + 0.96 + 0.9975) / 3.0
    core = (1.0 + 0.0 + 1.0 + cal_plain) / 4.0
    overall = 0.1 + 0.9 * core

    d_default = _score(ri)                                      # legacy (no kwargs)
    d_explicit = _score(ri, w_c1cal=1.0, w_ec=1.0, w_c2cal=1.0)  # explicit defaults
    # 1) explicit (1,1,1) matches the hand-computed PRE-CHANGE cal/overall
    assert _approx(d_explicit["cal_reward"], round(cal_plain, 6)), (d_explicit["cal_reward"], cal_plain)
    assert _approx(d_explicit["overall"], round(overall, 6)), (d_explicit["overall"], overall)
    # 2) explicit (1,1,1) is byte-identical to the default call (exact equality)
    assert d_explicit["cal_reward"] == d_default["cal_reward"], (d_explicit, d_default)
    assert d_explicit["overall"] == d_default["overall"], (d_explicit, d_default)
    print("PASS test_cal_weights_default_is_identical (w_c1cal=w_ec=w_c2cal=1 == legacy, exact)")


def test_cal_weights_ec_only():
    """ec-ONLY calibration: (w_c1cal=0, w_c2cal=0, w_ec=1) -> cal == the ec Brier term ALONE
    (no calibration on c1 or c2). Hand-check:
      ec=0.8, y=cA2=1 -> ec Brier term = 1-(0.8-1)^2 = 1-0.04 = 0.96
    cal must equal 0.96 exactly (c1=0.9 and c2=0.95 terms carry weight 0 -> dropped from num & den).
    Contrast: the default (1,1,1) cal is the 3-term mean 0.9825, so the two MUST differ.
    """
    rl = [_T1(ans="Paris", conf="0.9"), _T2(action="RETRIEVE", ec="0.8"), _T3(ans="Paris", conf="0.95")]
    ri = _ri(rl, ["Paris"], 0.5, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]])

    brier_ec = 1.0 - (0.8 - 1.0) ** 2   # 0.96 (term scored vs cA2=1)
    assert _approx(brier_ec, 0.96), brier_ec
    d = _score(ri, w_c1cal=0.0, w_c2cal=0.0, w_ec=1.0)
    # cal is EXACTLY the ec Brier term
    assert _approx(d["cal_reward"], round(brier_ec, 6)), (d["cal_reward"], brier_ec)
    # core/overall use ONLY the ec Brier as cal
    core = (1.0 + 0.0 + 1.0 + brier_ec) / 4.0
    overall = 0.1 + 0.9 * core
    assert _approx(d["overall"], round(overall, 6)), (d["overall"], overall)
    # and it differs from the default 3-term mean
    d_default = _score(ri)
    assert d["cal_reward"] != d_default["cal_reward"], (d["cal_reward"], d_default["cal_reward"])
    print("PASS test_cal_weights_ec_only (w_c1cal=0,w_c2cal=0,w_ec=1 -> cal == brier_ec == 0.96)")


def test_cal_weights_all_zero_guard():
    """Div-by-zero guard: if every present term's weight is 0, cal == 0.0 (no NaN/crash)."""
    rl = [_T1(ans="Paris", conf="0.9"), _T2(action="RETRIEVE", ec="0.8"), _T3(ans="Paris", conf="0.95")]
    ri = _ri(rl, ["Paris"], 0.5, [["CONTINUE", None], ["RETRIEVE", None], ["ANSWER", "Paris"]])
    d = _score(ri, w_c1cal=0.0, w_c2cal=0.0, w_ec=0.0)
    assert d["cal_reward"] == 0.0, d["cal_reward"]
    print("PASS test_cal_weights_all_zero_guard (all weights 0 -> cal=0.0)")


if __name__ == "__main__":
    test_oracle_cases()
    test_brier_values()
    test_w_ec_default_is_identical()
    test_w_ec_upweights_ec_term()
    test_w_ec_absent_ec_unaffected()
    test_cal_weights_default_is_identical()
    test_cal_weights_ec_only()
    test_cal_weights_all_zero_guard()
    test_ec_target_default_cA2_identical()
    test_ec_target_p_a2_uses_precomputed_label()
    test_ec_target_p_a2_missing_falls_back_to_cA2()
    test_ec_target_independent_of_oracle_mode()
    test_action_oracle_reward()
    test_format_gate()
    test_end_to_end_overall()
    test_eval_action_following()
    test_group_oracle_r_sensitivity()
    test_group_oracle_diag_dump()
    test_group_size_one_fallback()
    test_rollout_mode_unchanged()
    test_verbal_oracle_r_sensitivity()
    test_verbal_oracle_missing_conf()
    test_verbal_oracle_diag_dump()
    test_combined_oracle_blend()
    test_combined_oracle_diag_dump()
    test_precomp_oracle_r_sensitivity()
    test_precomp_oracle_uses_columns_not_realized()
    test_precomp_oracle_missing_columns_fallback()
    test_precomp_oracle_negative_sentinel_fallback()
    test_precomp_oracle_diag_dump()
    test_precomp_does_not_change_legacy_modes()
    test_new_modes_do_not_change_rollout_group()
    test_turn_aware_parser()
    test_ecfirst_rollout_parser_order_independent()
    test_ecfirst_reward_parse_and_verbal()
    print("\nALL TESTS PASSED")
