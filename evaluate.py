import argparse
import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DEFAULT_EXPERIMENTS = (
    "full",
    "no_context",
    "rag_single",
    "ablation_no_rerank",
    "ablation_top1",
)

# Experiments where non-abstaining predictions may carry empty citations.
ALLOW_EMPTY_CITATIONS = frozenset({"no_context"})

# Experiments for which Q (composite metric) is not reported.
Q_EXCLUDED = frozenset({"no_context"})

DEFAULT_BASELINE = "rag_single"

RATIONALE_MAX = 1200
CITATION_MAX = 4

CSV_FIELDS = [
    "name", "answer_accuracy", "Q",
    "mean_input_tokens", "mean_output_tokens",
    "mean_llm_calls", "mean_cost", "median_latency_ms",
]

EXPECTED_CATEGORIES = frozenset(
    {"conditions", "synthesis", "application", "insufficient"}
)


# IO

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_jsonl_by_id(path):
    return {str(row["id"]): row for row in read_jsonl(path)}


def _normalize_bid(bid):
    """b_0001 -> b0001; defensive against naming drift."""
    return str(bid).replace("_", "").lower()


def _fmt(val, prec=4):
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.{prec}f}"
    return str(val)


# Gold helpers

def gold_evidence_sets(gold):
    for key in ("evidence_sets", "citations"):
        if key not in gold:
            continue
        raw = gold[key]
        if not isinstance(raw, list) or not raw:
            return []
        if isinstance(raw[0], list):
            return [list(s) for s in raw]
        return [list(raw)]
    return []


def citation_f1(pred_citations, gold_sets):
    """Best F1 between pred citations and any gold citation set."""
    gold_sets = [s for s in gold_sets if s]
    if not gold_sets:
        return 1.0

    pred = {_normalize_bid(b) for b in pred_citations}
    if not pred:
        return 0.0

    best = 0.0
    for g in gold_sets:
        g_norm = {_normalize_bid(b) for b in g}
        if not g_norm:
            continue
        tp = len(pred & g_norm)
        if tp == 0:
            continue
        p = tp / len(pred)
        r = tp / len(g_norm)
        f1 = 2 * p * r / (p + r)
        if f1 > best:
            best = f1
    return best


# Cost helpers. scoring model from the spec

@dataclass
class QuestionScore:
    qid: str
    category: str
    correctness: float = 0.0
    s: Optional[float] = None
    A: Optional[float] = None
    F: Optional[float] = None
    valid: bool = True
    error: Optional[str] = None
    missing: bool = False

def score_one(gold, pred, *, allow_empty_citations, compute_s):
    qid = str(gold["id"])
    cat = str(gold.get("category", "unknown"))
    g_abstain = bool(gold.get("abstain", False))
    s_zero_or_none = 0.0 if compute_s else None

    if pred is None:
        return QuestionScore(
            qid=qid, category=cat,
            correctness=0.0, s=s_zero_or_none,
            valid=False, error="missing prediction", missing=True,
        )

    p_abstain = pred["abstain"]

    if g_abstain:
        correctness = 1.0 if p_abstain else 0.0
        return QuestionScore(
            qid=qid, category=cat,
            correctness=correctness,
            s=correctness if compute_s else None,
        )

    if p_abstain:
        return QuestionScore(
            qid=qid, category=cat,
            correctness=0.0, s=s_zero_or_none,
            A=0.0, F=0.0 if compute_s else None,
        )

    g_ids = {str(x) for x in gold.get("answer_ids", [])}
    p_ids = {str(x) for x in pred["answer_ids"]}
    A = 1.0 if p_ids == g_ids else 0.0

    if compute_s:
        F = citation_f1(pred["citations"], gold_evidence_sets(gold))
        s = A * (0.8 + 0.2 * F)
    else:
        F = None
        s = None

    return QuestionScore(
        qid=qid, category=cat,
        correctness=A, s=s, A=A, F=F,
    )

def compute_cost(trace):
    return (
        trace.get("input_tokens", 0)
        + 4 * trace.get("output_tokens", 0)
        + 256 * trace.get("llm_calls", 0)
    )


def trace_has_violation(trace):
    budget = trace.get("budget")
    return bool(isinstance(budget, dict) and budget.get("violated"))


# Aggregation

@dataclass
class ExperimentResult:
    name: str
    n_questions: int
    n_valid: int
    n_invalid: int
    n_missing: int
    answer_accuracy: Optional[float] = None
    Q: Optional[float] = None
    category_means: dict = field(default_factory=dict)
    mean_input_tokens: Optional[float] = None
    mean_output_tokens: Optional[float] = None
    mean_llm_calls: Optional[float] = None
    mean_cost: Optional[float] = None
    median_latency_ms: Optional[float] = None
    budget_violations: int = 0
    missing_traces: int = 0
    E: Optional[float] = None
    final_grade: Optional[float] = None
    scores: list = field(default_factory=list)


def aggregate(name, scores, traces, golds, c_ref, verbose=True):
    compute_q = name not in Q_EXCLUDED

    correctness = [sc.correctness for sc in scores]
    answer_accuracy = statistics.mean(correctness) if correctness else None

    if compute_q:
        cat_buckets = {}
        for sc in scores:
            if sc.s is not None:
                cat_buckets.setdefault(sc.category, []).append(sc.s)
        cat_means = {c: statistics.mean(v) for c, v in cat_buckets.items()}

        if len(cat_means) == 4:
            Q = statistics.mean(cat_means.values())
        else:
            Q = None
            if verbose:
                print(
                    f"  WARN {name}: expected 4 categories, got "
                    f"{len(cat_means)}: {sorted(cat_means)}. Q not computed."
                )
        if verbose and set(cat_means) - EXPECTED_CATEGORIES:
            print(
                f"  WARN {name}: unexpected category slugs: "
                f"{sorted(set(cat_means) - EXPECTED_CATEGORIES)}"
            )
    else:
        cat_means, Q = {}, None

    costs, in_toks, out_toks, calls, latencies = [], [], [], [], []
    violations = 0
    missing_traces = 0
    for qid in golds:
        t = traces.get(qid)
        if t is None:
            missing_traces += 1
            continue
        if trace_has_violation(t):
            violations += 1
        costs.append(compute_cost(t))
        in_toks.append(t.get("input_tokens", 0))
        out_toks.append(t.get("output_tokens", 0))
        calls.append(t.get("llm_calls", 0))
        latencies.append(t.get("elapsed_ms", 0))

    mean_cost = statistics.mean(costs) if costs else None

    # Efficiency per spec: budget violation and missing trace mean 0
    if c_ref is None or mean_cost is None:
        E = None
    elif violations > 0 or missing_traces > 0:
        E = 0.0
    else:
        E = min(1.0, c_ref / max(mean_cost, 1.0))

    final_grade = (
        75 * Q + 25 * Q * E
        if Q is not None and E is not None
        else None
    )

    return ExperimentResult(
        name=name,
        n_questions=len(scores),
        n_valid=sum(1 for sc in scores if sc.valid),
        n_invalid=sum(1 for sc in scores if not sc.valid and not sc.missing),
        n_missing=sum(1 for sc in scores if sc.missing),
        answer_accuracy=answer_accuracy,
        Q=Q,
        category_means=cat_means,
        mean_input_tokens=statistics.mean(in_toks) if in_toks else None,
        mean_output_tokens=statistics.mean(out_toks) if out_toks else None,
        mean_llm_calls=statistics.mean(calls) if calls else None,
        mean_cost=mean_cost,
        median_latency_ms=statistics.median(latencies) if latencies else None,
        budget_violations=violations,
        missing_traces=missing_traces,
        E=E,
        final_grade=final_grade,
        scores=scores,
    )

# Output

def _print_result(r):
    print(f"\n{r.name}")
    print(f"  questions       : {r.n_questions} "
          f"(valid {r.n_valid}, invalid {r.n_invalid}, missing {r.n_missing})")
    print(f"  answer_accuracy : {_fmt(r.answer_accuracy)}")
    if r.name in Q_EXCLUDED:
        print(f"  Q               : (not applicable)")
    else:
        print(f"  Q               : {_fmt(r.Q)}")
        for cat in sorted(r.category_means):
            print(f"    {cat:<16} {_fmt(r.category_means[cat])}")
    print(f"  mean tokens in  : {_fmt(r.mean_input_tokens, 1)}")
    print(f"  mean tokens out : {_fmt(r.mean_output_tokens, 1)}")
    print(f"  mean llm_calls  : {_fmt(r.mean_llm_calls, 2)}")
    print(f"  mean cost       : {_fmt(r.mean_cost, 1)}")
    print(f"  median latency  : {_fmt(r.median_latency_ms, 0)} ms")
    if r.budget_violations:
        print(f"  budget violations: {r.budget_violations} (E forced to 0)")
    if r.missing_traces:
        print(f"  missing traces   : {r.missing_traces} (E forced to 0)")
    print(f"  E               : {_fmt(r.E)}")
    print(f"  Final score     : {_fmt(r.final_grade, 2)}")


def _write_csv(path, results):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({
                "name": r.name,
                "answer_accuracy": _fmt(r.answer_accuracy),
                "Q": _fmt(r.Q),
                "mean_input_tokens": _fmt(r.mean_input_tokens),
                "mean_output_tokens": _fmt(r.mean_output_tokens),
                "mean_llm_calls": _fmt(r.mean_llm_calls),
                "mean_cost": _fmt(r.mean_cost),
                "median_latency_ms": _fmt(r.median_latency_ms),
            })


def _write_details(path, results):
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            for sc in r.scores:
                row = {
                    "experiment": r.name,
                    "id": sc.qid,
                    "category": sc.category,
                    "correctness": round(sc.correctness, 4),
                    "valid": sc.valid,
                    "missing": sc.missing,
                }
                if sc.s is not None:
                    row["s"] = round(sc.s, 4)
                if sc.A is not None:
                    row["A"] = round(sc.A, 4)
                if sc.F is not None:
                    row["F"] = round(sc.F, 4)
                if sc.error:
                    row["error"] = sc.error
                f.write(json.dumps(row, ensure_ascii=False) + "\n")



def evaluate(
    gold_path,
    answers_dir=".",
    traces_dir=".",
    experiments=None,
    baseline=DEFAULT_BASELINE,
    csv_out="experiments.csv",
    details_out=None,
    verbose=True,
):
    '''Entry point'''
    answers_dir = Path(answers_dir)
    traces_dir = Path(traces_dir)

    golds = read_jsonl_by_id(gold_path)
    if verbose:
        print(f"Gold: {gold_path} ({len(golds)} questions)")

    exp_list = list(experiments) if experiments else list(DEFAULT_EXPERIMENTS)

    c_ref = None
    baseline_trace = traces_dir / f"dev_trace_{baseline}.jsonl"
    if baseline_trace.exists():
        baseline_traces = read_jsonl(baseline_trace)
        if baseline_traces:
            c_ref = statistics.mean(compute_cost(t) for t in baseline_traces)
            if verbose:
                print(f"C_ref (from {baseline}): {c_ref:.0f}")
    elif verbose:
        print(f"WARN: baseline trace not found: {baseline_trace}")

    results = []
    for name in exp_list:
        answers_path = answers_dir / f"dev_answers_{name}.jsonl"
        trace_path = traces_dir / f"dev_trace_{name}.jsonl"

        if not answers_path.exists():
            if verbose:
                print(f"Skipping {name}: {answers_path.name} not found")
            continue

        preds = read_jsonl_by_id(answers_path)
        traces = read_jsonl_by_id(trace_path) if trace_path.exists() else {}

        allow_empty = name in ALLOW_EMPTY_CITATIONS
        compute_s = name not in Q_EXCLUDED

        scores = [
            score_one(
                gold, preds.get(qid),
                allow_empty_citations=allow_empty,
                compute_s=compute_s,
            )
            for qid, gold in golds.items()
        ]

        result = aggregate(name, scores, traces, golds, c_ref, verbose=verbose)
        results.append(result)
        if verbose:
            _print_result(result)

    _write_csv(csv_out, results)
    if details_out:
        _write_details(details_out, results)

    if verbose:
        print(f"\nCSV     : {Path(csv_out).resolve()}")
        if details_out:
            print(f"Details : {Path(details_out).resolve()}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--answers-dir", default=".")
    parser.add_argument("--traces-dir", default=".")
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Experiment names to score. Default: " + ", ".join(DEFAULT_EXPERIMENTS),
    )
    # It has not been provided with the task, so not sure what to measure
    # against. so... keeping it as simplest correct RAG setup.
    parser.add_argument(
        "--baseline", default=DEFAULT_BASELINE,
        help=f"Trace used for C_ref. Default: {DEFAULT_BASELINE}",
    )
    parser.add_argument("--csv", default="experiments.csv")
    # Optional per-question json for debugging
    parser.add_argument(
        "--details", default=None,
        help="Optional per-question JSONL for debugging.",
    )
    args = parser.parse_args()

    evaluate(
        gold_path=args.gold,
        answers_dir=args.answers_dir,
        traces_dir=args.traces_dir,
        experiments=args.experiments,
        baseline=args.baseline,
        csv_out=args.csv,
        details_out=args.details,
        verbose=True,
    )


if __name__ == "__main__":
    main()
