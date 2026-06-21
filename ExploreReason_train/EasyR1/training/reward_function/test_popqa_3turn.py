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
    test_turn_aware_parser()
    print("\nALL TESTS PASSED")
