"""QA CTA-RL reward (RL#2). Mirror of explore_code_test_score.py:compute_score for PopQA.
Reward = correctness(popqa alias-match) * r^(num_retrieves), gated by per-turn RETRIEVE/ANSWER format.
"""
import re, string

def normalize_answer(s):
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())

def popqa_correct(pred, answers):
    if pred is None:
        return 0.0
    if isinstance(answers, str):
        answers = [answers]
    p = str(pred).strip(); pl = p.lower()
    for a in answers:
        a = str(a)
        if a in p or a.lower() in pl or a.capitalize() in p:
            return 1.0
    return 0.0

def parse_qa_action(text):
    if "<think>" in text and "</think>" not in text:
        return "TRUNCATED", None
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = text.replace("<|im_end|>", "").strip()
    m = re.search(r"ANSWER:\s*(.+)$", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return "ANSWER", m.group(1).strip()
    if re.match(r"^\s*RETRIEVE\b", text, flags=re.IGNORECASE):
        return "RETRIEVE", None
    return None, None

def compute_format_reward(response):
    a, _ = parse_qa_action(response)
    return 1.0 if a in ("RETRIEVE", "ANSWER") else 0.0

def count_retrieves(action_seqs):
    import numpy as _np
    n = 0
    for a in action_seqs:
        at = a
        while isinstance(at, (list, tuple, _np.ndarray)):
            if len(at) == 0:
                at = None
                break
            at = at[0]
        if str(at).strip() == "RETRIEVE":
            n += 1
    return n

def compute_score(reward_inputs, format_weight=0.1, log_dir=None, max_turns=4, **kw):
    results = []
    for ri in reward_inputs:
        pred = ri.get("pred_answers", "")
        answers = ri.get("possible_answers", ri.get("gold_answers", []))
        action_seqs = ri.get("action_seqs", [])
        r = float(ri.get("discount_factor", 1.0))
        response_list = ri.get("response_list", [])
        p_no_context = ri.get("p_no_context", None)
        p_ret = ri.get("p_ret", None)

        correctness = popqa_correct(pred, answers)
        num_retrieves = count_retrieves(action_seqs)
        discounted = round(correctness * (r ** num_retrieves), 6)
        if len(response_list) > 0:
            format_reward = 1.0 if all(compute_format_reward(str(x)) == 1.0 for x in response_list) else 0.0
        else:
            format_reward = 1.0
        trace_too_long = len(action_seqs) > max_turns
        overall = 0.0 if (format_reward == 0.0 or trace_too_long) else (format_weight * format_reward + (1.0 - format_weight) * discounted)
        oracle_match = 0.0
        if p_no_context is not None and p_ret is not None:
            oracle_retrieve = 1 if float(p_no_context) <= float(p_ret) * r else 0
            oracle_match = 1.0 if (int(num_retrieves > 0) == oracle_retrieve) else 0.0
        results.append({
            "overall": overall,
            "discounted_reward": discounted,
            "correctness": correctness,
            "num_retrieves": num_retrieves,
            "format_reward": format_reward,
            "oracle_match": oracle_match,
        })
    return results

if __name__ == "__main__":
    ri = [
        {"pred_answers": "Libreville", "possible_answers": ["Libreville"], "action_seqs": [["ANSWER", "Libreville"]], "discount_factor": 0.9, "response_list": ["ANSWER: Libreville"], "p_no_context": 0.9, "p_ret": 0.57},
        {"pred_answers": "Paris", "possible_answers": ["Libreville"], "action_seqs": [["RETRIEVE", None], ["ANSWER", "Paris"]], "discount_factor": 0.9, "response_list": ["RETRIEVE", "ANSWER: Paris"], "p_no_context": 0.2, "p_ret": 0.57},
    ]
    for r in compute_score(ri):
        print(r)
