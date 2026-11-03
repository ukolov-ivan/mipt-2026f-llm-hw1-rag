import argparse
import json
import os
import pickle
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from scripts.tokenizer import TOKENIZER_VERSION, tokenize

_PREVIEW = 300
_RATIONALE_MAX = 1200


# Experiments - data + policy objects

@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    retrieve: bool
    top_k: int | None = None
    retrieval_policy: str = "bm25_ranked"
    include_options_in_query: bool = True  # feed option texts into BM25 query
    retry_on_citation_error: bool = True
    allow_empty_citations: bool = False    # only no_context today

    # Reserved hooks. If set, main() raises (not yet implemented).
    two_stage: bool = False
    query_rewrite: bool = False
    reranker: str | None = None


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "no_context": ExperimentSpec(
        name="no_context",
        retrieve=False,
        retry_on_citation_error=False,
        allow_empty_citations=True,
    ),
    "rag_single": ExperimentSpec(
        name="rag_single",
        retrieve=True,
        retry_on_citation_error=False,
    ),
    "full": ExperimentSpec(
        name="full",
        retrieve=True,
    ),
    "ablation_no_rerank": ExperimentSpec(
        name="ablation_no_rerank",
        retrieve=True,
        # To-the-spec interpretation: BM25 selects top_k, blocks are then
        # presented in corpus order. Tests "does the LLM care about ranking
        # within the selected set, or only about which set was selected".
        retrieval_policy="bm25_then_reorder",
    ),
    "ablation_top1": ExperimentSpec(
        name="ablation_top1",
        retrieve=True,
        top_k=1,
    ),
}

VALID_EXPERIMENTS = frozenset(EXPERIMENTS)


class BM25Ranked:
    def retrieve(self, tokens, corpus, bm25, top_k):
        scores = bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))[:top_k]
        return [corpus[i] for i in order]


class BM25ThenReorder:
    """BM25-select top_k, then reorder by corpus position."""
    def retrieve(self, tokens, corpus, bm25, top_k):
        scores = bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))[:top_k]
        order.sort()
        return [corpus[i] for i in order]


class CorpusFirstK:
    def retrieve(self, tokens, corpus, bm25, top_k):
        return corpus[:top_k]


RETRIEVAL_POLICIES = {
    "bm25_ranked": BM25Ranked(),
    "bm25_then_reorder": BM25ThenReorder(),
    "corpus_first_k": CorpusFirstK(),
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_output_path(base_path: str, experiment_name: str) -> str:
    if not experiment_name or experiment_name in base_path:
        return base_path
    p = Path(base_path)
    return str(p.parent / f"{p.stem}_{experiment_name}{p.suffix}")


def load_bm25(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        for key in ("bm25", "index", "model"):
            if key in obj and hasattr(obj[key], "get_scores"):
                return obj[key]
    if hasattr(obj, "get_scores"):
        return obj
    raise TypeError(
        "index/bm25.pkl is not a BM25 object with get_scores(); "
        "adapt load_bm25() to build_index.py's format."
    )


# Input validation

def validate_question(q):
    qid = q.get("id")
    if not isinstance(qid, str) or not qid.strip():
        raise ValueError(f"question missing 'id': {q!r}")

    if not isinstance(q.get("question"), str) or not q["question"].strip():
        raise ValueError(f"{qid}: 'question' must be a non-empty string")

    options = q.get("options")
    if not isinstance(options, list):
        raise ValueError(f"{qid}: 'options' must be a list")
    if not 4 <= len(options) <= 6:
        raise ValueError(
            f"{qid}: expected 4-6 options, got {len(options)}"
        )

    seen = set()
    for o in options:
        oid = o.get("id")
        if not isinstance(oid, str) or not oid.strip():
            raise ValueError(f"{qid}: option missing 'id': {o!r}")
        if oid in seen:
            raise ValueError(f"{qid}: duplicate option id {oid!r}")
        seen.add(oid)
        if not isinstance(o.get("text"), str):
            raise ValueError(f"{qid}: option {oid!r} text must be a string")

    return qid


def validate_questions(questions):
    seen = set()
    for q in questions:
        qid = validate_question(q)
        if qid in seen:
            raise ValueError(f"duplicate question id: {qid!r}")
        seen.add(qid)


def validate_answer(ans, valid_options, allow_empty_citations):
    aid = ans.get("id")
    if not isinstance(aid, str) or not aid:
        raise ValueError(f"answer missing 'id': {ans!r}")
    if not isinstance(ans.get("abstain"), bool):
        raise ValueError(f"{aid}: 'abstain' must be bool")

    ids = ans.get("answer_ids", [])
    citations = ans.get("citations", [])
    rationale = ans.get("rationale", "")

    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise ValueError(f"{aid}: answer_ids must be list[str]")
    if not isinstance(citations, list) or not all(isinstance(x, str) for x in citations):
        raise ValueError(f"{aid}: citations must be list[str]")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError(f"{aid}: rationale must be a non-empty string")
    if len(rationale) > _RATIONALE_MAX:
        raise ValueError(f"{aid}: rationale exceeds {_RATIONALE_MAX} chars ({len(rationale)})")

    if len(set(ids)) != len(ids):
        raise ValueError(f"{aid}: duplicate answer_ids")
    if len(set(citations)) != len(citations):
        raise ValueError(f"{aid}: duplicate citations")
    if len(citations) > 4:
        raise ValueError(f"{aid}: more than 4 citations ({len(citations)})")

    if ans["abstain"]:
        if ids or citations:
            raise ValueError(f"{aid}: abstain=true but lists are not empty")
    else:
        if not ids:
            raise ValueError(f"{aid}: abstain=false but answer_ids is empty")
        if not citations and not allow_empty_citations:
            raise ValueError(f"{aid}: abstain=false but citations is empty")

    bad = set(ids) - valid_options
    if bad:
        raise ValueError(f"{aid}: unknown option ids {sorted(bad)}")

    return ans

# LLM call

def _preview(obj, limit=_PREVIEW):
    return (json.dumps(obj, ensure_ascii=False) if not isinstance(obj, str) else obj)[:limit]


def call_llm(messages, qid, timeout, max_tokens, temperature):
    base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    model = os.environ.get("LLM_MODEL", "")
    key = os.environ.get("LLM_API_KEY", "")
    submission_id = os.environ.get("SUBMISSION_ID", "")
    if not base or not model:
        raise RuntimeError(
            "Set LLM_BASE_URL and LLM_MODEL to an OpenAI-compatible endpoint."
        )
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Question-ID": qid,
        # WARN is printed once in main() if empty
        "X-Submission-ID": submission_id, 
    }
    if key:
        headers["Authorization"] = "Bearer " + key
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            body = json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"LLM HTTP {e.code}: {e.read().decode(errors='replace')[:_PREVIEW]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach LLM endpoint {url}: {e}") from e

    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"Malformed API response: {e}; body={_preview(body)}"
        ) from e

    if isinstance(text, list):
        text = "".join(x.get("text", "") for x in text if isinstance(x, dict))

    usage = body.get("usage", {})
    if "prompt_tokens" not in usage or "completion_tokens" not in usage:
        raise RuntimeError(
            "API omitted usage.prompt_tokens/completion_tokens; "
            "cannot create accurate token trace."
        )
    try:
        in_tok = int(usage["prompt_tokens"])
        out_tok = int(usage["completion_tokens"])
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"usage tokens not numeric: {usage!r}") from e

    return text, in_tok, out_tok

# TODO: Remove this, just log
@dataclass
class BudgetLimits:
    max_calls: int = 4
    max_input_tokens: int = 16384
    max_output_tokens: int = 4096
    max_per_call: int = 8192

    @classmethod
    def from_runtime(cls, runtime):
        b = runtime.get("budget", {})
        return cls(
            max_calls=int(b.get("max_llm_calls_per_question", 4)),
            max_input_tokens=int(b.get("max_input_tokens_total", 16384)),
            max_output_tokens=int(b.get("max_output_tokens_total", 4096)),
            max_per_call=int(b.get("max_input_plus_output_per_call", 8192)),
        )


@dataclass
class BudgetState:
    limits: BudgetLimits
    calls: int = 0
    in_tok: int = 0
    out_tok: int = 0
    violations: list[str] = field(default_factory=list)

    def check_pre_call(self, estimated_in: int, max_out: int) -> None:
        """Record (do not raise) pre-call budget concerns."""
        if self.calls + 1 > self.limits.max_calls:
            self.violations.append(
                f"would exceed max_calls={self.limits.max_calls}"
            )
        if estimated_in + max_out > self.limits.max_per_call:
            self.violations.append(
                f"in_est={estimated_in} + max_out={max_out} > "
                f"per_call={self.limits.max_per_call}"
            )

    def record(self, in_tok: int, out_tok: int) -> None:
        self.calls += 1
        self.in_tok += in_tok
        self.out_tok += out_tok
        if self.in_tok > self.limits.max_input_tokens:
            self.violations.append(
                f"in_tok={self.in_tok} > max={self.limits.max_input_tokens}"
            )
        if self.out_tok > self.limits.max_output_tokens:
            self.violations.append(
                f"out_tok={self.out_tok} > max={self.limits.max_output_tokens}"
            )

    def to_trace(self):
        return {
            "limits": {
                "max_calls": self.limits.max_calls,
                "max_input_tokens": self.limits.max_input_tokens,
                "max_output_tokens": self.limits.max_output_tokens,
                "max_per_call": self.limits.max_per_call,
            },
            "calls": self.calls,
            "input_tokens": self.in_tok,
            "output_tokens": self.out_tok,
            "violated": bool(self.violations),
            "violations": list(self.violations),
        }


def estimate_prompt_tokens(messages):
    """Conservative pre-call estimate. Russian + math ≈ 3 chars/token."""
    chars = sum(len(m["content"]) for m in messages)
    return chars // 3 + 4 * len(messages)


# Prompt builders

def format_evidence(passages):
    return "\n\n".join(
        f"[{d['block_id']}] с.{d.get('page','?')} §{d.get('section','?')}\n{d['text']}"
        for d in passages
    )


def _json_contract_line():
    return (
        "- Верни ТОЛЬКО JSON без Markdown, пробелов до/после:\n"
        '{"answer_ids":["A","C"],"abstain":false,"rationale":"Обоснование",'
        '"citations":["b_0001","b_0002"]}'
    )


def build_no_context_prompt(question_text, options_text):
    """No-retrieval prompt. This experiment intentionally violates the spec's
    output requirement that non-abstaining answers carry non-empty citations:
    there is nothing to cite. Output is still one-line-per-id, and F (citation
    F1) will be 0 by construction - that is the point of the ablation.
    """
    system = (
        "Ты эксперт по ТФКП. Отвечай на основе своих знаний. "
        "Не добавляй Markdown. Верни только JSON."
    )
    user = f"""Реши вопрос по ТФКП.

ПРАВИЛА ОТВЕТА:
- Выбери все верные варианты из списка.
- Если знаний недостаточно для уверенного ответа - установи abstain=true, оба списка пустыми.
- Если отвечаешь (abstain=false): answer_ids НЕПУСТОЙ, citations ПУСТОЙ (нет фрагментов).
- rationale: непустое обоснование, не более 300 символов (кратко).
{_json_contract_line()}

ВОПРОС: {question_text}

ВАРИАНТЫ:
{options_text}

JSON:"""
    return system, user


def build_main_prompt(passages, question_text, options_text):
    system = (
        "Ты эксперт по ТФКП. Отвечай строго по фрагментам. "
        "Не выдумывай факты. Не добавляй Markdown. Верни только JSON."
    )
    user = f"""Реши вопрос по ТФКП, опираясь только на фрагменты ниже.

ПРАВИЛА ОТВЕТА:
- Выбери все верные варианты из списка.
- Если фрагментов недостаточно для уверенного ответа - установи abstain=true, оба списка пустыми.
- Если в вопросе отсутствует существенное условие (например, ориентация контура, конкретное значение параметра),
  которое нужно для однозначного ответа - это признак недостаточности данных: abstain=true.
- Если отвечаешь (abstain=false): answer_ids и citations оба НЕПУСТЫЕ.
- citations: от 1 до 4 block_id только из фрагментов ниже. НЕ выдумывай block_id.
- rationale: непустое обоснование, не более 300 символов (кратко).
{_json_contract_line()}

ФРАГМЕНТЫ:
{format_evidence(passages)}

ВОПРОС: {question_text}

ВАРИАНТЫ:
{options_text}

JSON:"""
    return system, user


def build_prompt(q, passages, spec):
    options_text = "\n".join(f"{o['id']}: {o['text']}" for o in q["options"])
    if not spec.retrieve:
        system, user = build_no_context_prompt(q["question"], options_text)
    else:
        system, user = build_main_prompt(passages, q["question"], options_text)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_retry_prompt(q, passages, spec, prev_response, reason):
    """Retry prompt. If prev_response is None, omit the assistant turn."""
    options_text = "\n".join(f"{o['id']}: {o['text']}" for o in q["options"])
    system, user = build_main_prompt(passages, q["question"], options_text)
    available_ids = ", ".join(d["block_id"] for d in passages)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if prev_response is not None:
        messages.append({"role": "assistant", "content": prev_response})
    messages.append({
        "role": "user",
        "content": (
            f"ОШИБКА в предыдущем ответе: {reason}\n"
            f"Доступные block_id: {available_ids}\n"
            "Исправь ответ. Верни ТОЛЬКО корректный JSON:"
        ),
    })
    return messages


def parse_response(text):
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.I)

    def _loads(candidate):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    obj = _loads(text)
    if obj is None:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj = _loads(text[start:end + 1])
    if obj is None:
        raise ValueError(
            f"Model did not return a valid JSON object: {text[:_PREVIEW]}"
        )
    if not isinstance(obj, dict):
        raise ValueError(
            f"Model returned {type(obj).__name__}, expected JSON object: "
            f"{_preview(obj)}"
        )
    return obj


def validate_llm_output(obj, valid_options, block_ids, allow_empty_citations):
    """Returns (abstain, ids, citations, rationale) or raises ValueError."""
    abstain = obj.get("abstain", False)
    ids = obj.get("answer_ids", [])
    citations = obj.get("citations", [])
    rationale = obj.get("rationale", "")

    if not isinstance(abstain, bool):
        raise ValueError("'abstain' must be a boolean")
    if not isinstance(ids, list):
        raise ValueError("'answer_ids' must be a list")
    if not isinstance(citations, list):
        raise ValueError("'citations' must be a list")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("'rationale' must be a non-empty string")

    ids = list(dict.fromkeys(str(x) for x in ids))
    citations = list(dict.fromkeys(str(x) for x in citations))[:4]

    bad_opts = [x for x in ids if x not in valid_options]
    if bad_opts:
        raise ValueError(f"Unknown option IDs: {bad_opts}")

    if not allow_empty_citations:
        bad_cites = [x for x in citations if x not in block_ids]
        if bad_cites:
            raise ValueError(f"Citations not in retrieved blocks: {bad_cites}")

    if abstain:
        ids, citations = [], []
    else:
        if not ids:
            raise ValueError("Non-abstaining answer needs answer_ids")
        if not allow_empty_citations and not citations:
            raise ValueError("Non-abstaining answer needs citations")

    return abstain, ids, citations, rationale.strip()


# Orchestration

def plan_retrieval(q, cfg, spec, runtime, docs, bm25):
    """Return (passages, retrieved_ids). Empty for no-retrieval experiments."""
    if not spec.retrieve:
        return [], []

    top_k = (
        spec.top_k
        if spec.top_k is not None
        else int(runtime.get("retrieval", {}).get("top_k", 5))
    )

    query_parts = [q["question"]]
    if spec.include_options_in_query:
        query_parts.extend(o["text"] for o in q["options"])
    tokens = tokenize(" ".join(query_parts), cfg["tokenization"])

    policy = RETRIEVAL_POLICIES[spec.retrieval_policy]
    passages = policy.retrieve(tokens, docs, bm25, top_k)
    retrieved_ids = [d["block_id"] for d in passages]

    print(
        f"  retrieval: policy={spec.retrieval_policy} top_k={top_k} "
        f"ids={retrieved_ids}",
        file=sys.stderr,
    )
    return passages, retrieved_ids


def answer_question(q, docs, bm25, cfg, spec, runtime, budget_limits):
    qid = q["id"]
    start = time.perf_counter()
    budget = BudgetState(limits=budget_limits)
    valid_options = {str(o["id"]) for o in q["options"]}

    passages, retrieved_ids = plan_retrieval(q, cfg, spec, runtime, docs, bm25)
    block_ids = set(retrieved_ids)

    timeout = int(runtime.get("timeout_seconds", 180))
    gen = runtime.get("generation", {})
    max_tokens = int(gen.get("max_tokens", 1024))
    temperature = float(gen.get("temperature", 0.0))

    abstain: bool | None = None
    ids: list[str] = []
    citations: list[str] = []
    rationale: str = ""
    last_error: str | None = None
    raw: str | None = None

    # Initial call
    messages = build_prompt(q, passages, spec)
    budget.check_pre_call(estimate_prompt_tokens(messages), max_tokens)
    try:
        raw, in_tok, out_tok = call_llm(messages, qid, timeout, max_tokens, temperature)
        budget.record(in_tok, out_tok)
        parsed = parse_response(raw)
        abstain, ids, citations, rationale = validate_llm_output(
            parsed, valid_options, block_ids, spec.allow_empty_citations
        )
    except (RuntimeError, ValueError) as e:
        last_error = str(e)
        raw = None

    # Retry (citation error only)
    if (
        last_error is not None
        and spec.retry_on_citation_error
        and "citations" in last_error.lower()
        and budget.calls < budget.limits.max_calls
    ):
        print(f"  {qid}: retry ({last_error})", file=sys.stderr)
        retry_messages = build_retry_prompt(q, passages, spec, raw, last_error)
        budget.check_pre_call(estimate_prompt_tokens(retry_messages), max_tokens)
        try:
            raw2, in2, out2 = call_llm(
                retry_messages, qid, timeout, max_tokens, temperature
            )
            budget.record(in2, out2)
            parsed2 = parse_response(raw2)
            abstain, ids, citations, rationale = validate_llm_output(
                parsed2, valid_options, block_ids, spec.allow_empty_citations
            )
            last_error = None
        except (RuntimeError, ValueError) as e:
            last_error = f"Retry failed: {e}"

    # Emit answer (one line per question, always)
    if last_error is not None or abstain is None:
        answer = {
            "id": qid,
            "answer_ids": [],
            "abstain": True,
            "rationale": ("Ошибка: " + (last_error or "неизвестно"))[:_RATIONALE_MAX],
            "citations": [],
        }
    else:
        if len(rationale) > _RATIONALE_MAX:
            print(
                f"  {qid}: rationale truncated to {_RATIONALE_MAX} chars",
                file=sys.stderr,
            )
        answer = {
            "id": qid,
            "answer_ids": ids,
            "abstain": abstain,
            "rationale": rationale[:_RATIONALE_MAX],
            "citations": citations,
        }

    if budget.violations:
        print(f"  {qid}: BUDGET WARNING {budget.violations}", file=sys.stderr)

    elapsed_ms = round((time.perf_counter() - start) * 1000)
    trace = {
        "id": qid,
        "llm_calls": budget.calls,
        "input_tokens": budget.in_tok,
        "output_tokens": budget.out_tok,
        "elapsed_ms": elapsed_ms,
        "retrieved_block_ids": retrieved_ids,
        "budget": budget.to_trace(),
    }
    return answer, trace


def check_index_consistency(index_dir, cfg):
    """Cross-check tokenizer version + settings against index/manifest.json."""
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        print(
            "WARN: index/manifest.json not found; cannot verify tokenizer consistency",
            file=sys.stderr,
        )
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    m_tok = manifest.get("tokenization", {})
    c_tok = cfg.get("tokenization", {})

    # TODO: bump version and add match check 
    m_ver = m_tok.get("tokenizer_version")

    for key in ("lowercase", "include_math_symbols", "min_token_length"):
        if key in m_tok and key in c_tok and m_tok[key] != c_tok[key]:
            print(
                f"ERROR: tokenization.{key} mismatch: index={m_tok[key]} "
                f"cfg={c_tok[key]}. Rebuild or fix config.",
                file=sys.stderr,
            )
            sys.exit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime", default="runtime.json")
    parser.add_argument(
        "--experiment-name",
        default="full",
        help=(
            "Optional override. One of: "
            + ", ".join(sorted(VALID_EXPERIMENTS))
            + ". Falls back to runtime.default_experiment."
        ),
    )
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    runtime = json.loads(Path(args.runtime).read_text(encoding="utf-8"))    

    experiment_name = args.experiment_name
    if experiment_name not in VALID_EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment {experiment_name!r}. "
            f"Choose one of: {', '.join(sorted(VALID_EXPERIMENTS))}"
        )
    spec = EXPERIMENTS[experiment_name]

    # Reserved hooks asserted unimplemented.
    for hook in ("two_stage", "query_rewrite", "reranker"):
        if getattr(spec, hook):
            raise NotImplementedError(
                f"Experiment {experiment_name!r} sets {hook}=True, "
                f"which is not implemented yet."
            )

    output_path = resolve_output_path(args.output, experiment_name)
    trace_path = resolve_output_path(args.trace, experiment_name)

    print(f"Running experiment: {experiment_name}")
    print(f"  answers are written to {output_path}")
    print(f"  trace is being written to {trace_path}")

    # Warnings that only need to appear once.
    if not os.environ.get("SUBMISSION_ID"):
        print(
            "WARN: SUBMISSION_ID unset; X-Submission-ID header will be empty",
            file=sys.stderr,
        )
    # TODO: Add check for it
    # check_model_pinning(runtime)
    check_index_consistency(Path(args.index), cfg)

    docs = read_jsonl(Path(args.index) / "documents.jsonl")
    bm25 = load_bm25(Path(args.index) / "bm25.pkl")
    questions = read_jsonl(args.questions)
    validate_questions(questions)

    budget_limits = BudgetLimits.from_runtime(runtime)

    answers, traces = [], []
    for i, question in enumerate(questions, 1):
        answer, trace = answer_question(
            question, docs, bm25, cfg, spec, runtime, budget_limits
        )
        answers.append(answer)
        traces.append(trace)

        label = answer["answer_ids"] if answer["answer_ids"] else "Отказ"
        retry_note = (
            f" [retry, calls={trace['llm_calls']}]"
            if trace["llm_calls"] > 1
            else ""
        )
        print(f"[{i}/{len(questions)}] {question['id']}: {label}{retry_note}")

    # Post-hoc schema validation on the batch.
    valid_options = {q["id"]: {str(o["id"]) for o in q["options"]} for q in questions}
    for ans in answers:
        validate_answer(
            ans,
            valid_options[ans["id"]],
            allow_empty_citations=spec.allow_empty_citations,
        )

    if {a["id"] for a in answers} != {q["id"] for q in questions}:
        raise ValueError("answer IDs do not match question IDs 1:1")

    write_jsonl(output_path, answers)
    write_jsonl(trace_path, traces)

    violations = sum(1 for t in traces if t["budget"]["violated"])
    if violations:
        print(
            f"WARNING: {violations}/{len(traces)} questions had budget violations",
            file=sys.stderr,
        )
    print("Done!")
    print(f"  answers have been written to {output_path}")
    print(f"  trace has been written to {trace_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
