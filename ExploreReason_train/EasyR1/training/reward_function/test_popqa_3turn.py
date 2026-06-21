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


if __name__ == "__main__":
    test_oracle_cases()
    test_brier_values()
    test_action_oracle_reward()
    test_format_gate()
    test_end_to_end_overall()
    test_eval_action_following()
    test_group_oracle_r_sensitivity()
    test_group_oracle_diag_dump()
    test_group_size_one_fallback()
    test_rollout_mode_unchanged()
    test_turn_aware_parser()
    print("\nALL TESTS PASSED")
