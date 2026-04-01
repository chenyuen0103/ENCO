import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Allow free-form reasoning before the final answer block.
# We only require that the completion ends with a well-formed <answer>...</answer>.
FORMAT_RE = re.compile(r"(?s)^.*<answer>.*?</answer>\s*$")
ANSWER_RE = re.compile(r"(?s)<answer>\s*(.*?)\s*</answer>")


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        if not completion:
            return ""
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(first)
    return str(completion)


def extract_answer_text(text: str) -> str:
    m = ANSWER_RE.search(text or "")
    return m.group(1) if m else (text or "")


def format_ok(text: str) -> int:
    return int(bool(FORMAT_RE.match(text or "")))


def cd_format_reward(completions, **kwargs):
    texts = [completion_to_text(c) for c in completions]
    return [1.0 if format_ok(t) else 0.0 for t in texts]


def build_cd_format_reward(scale: float = 0.2):
    if scale < 0:
        raise ValueError("scale must be >= 0")

    def _cd_format_reward(completions, **kwargs):
        texts = [completion_to_text(c) for c in completions]
        s = float(scale)
        return [s if format_ok(t) else 0.0 for t in texts]

    _cd_format_reward.__name__ = "cd_format_reward"
    return _cd_format_reward


def build_cd_partial_format_reward(scale: float = 0.25):
    """
    Dense shaping reward for causal-discovery formatting progress.
    Gives partial credit for structural progress even when strict format fails.
    """
    if scale < 0:
        raise ValueError("scale must be >= 0")

    target_cache: Dict[str, Optional[List[List[int]]]] = {}

    def _cached_target(raw: Any) -> Optional[List[List[int]]]:
        key = str(raw)
        if key in target_cache:
            return target_cache[key]
        mat = target_matrix_from_answer(raw)
        target_cache[key] = mat
        return mat

    def cd_partial_format_reward(completions, **kwargs):
        answers = kwargs.get("answer")
        answer_paths = kwargs.get("answer_path")
        rewards: List[float] = []
        s = float(scale)

        for i, completion in enumerate(completions):
            text = completion_to_text(completion)
            t = text or ""

            # infer expected matrix size from target answer when possible
            target_raw = None
            if answers is not None and i < len(answers):
                target_raw = answers[i]
            elif answer_paths is not None and i < len(answer_paths):
                target_raw = answer_paths[i]
            target_adj = _cached_target(target_raw)
            expected_n = len(target_adj) if target_adj is not None else None

            base = 0.0
            if "<answer>" in t:
                base += 0.2
            if "</answer>" in t:
                base += 0.2
            if "adjacency_matrix" in t:
                base += 0.1

            # parse in answer block first, then full text fallback
            ans_text = extract_answer_text(t)
            parsed = extract_adjacency_matrix(ans_text, expected_n=expected_n)
            if parsed is None:
                parsed = extract_adjacency_matrix(t, expected_n=expected_n)

            if parsed is not None:
                base += 0.4

            # keep shaping in [0, scale]
            base = max(0.0, min(1.0, float(base)))
            rewards.append(float(s * base))

        return rewards

    cd_partial_format_reward.__name__ = "cd_partial_format_reward"
    return cd_partial_format_reward


def _normalize_matrix(mat: Any, expected_n: Optional[int] = None) -> Optional[List[List[int]]]:
    if not isinstance(mat, list) or not mat:
        return None
    try:
        rows = [[int(x) for x in row] for row in mat]
    except Exception:
        return None
    n = len(rows)
    if expected_n is not None and n != expected_n:
        return None
    if any(len(r) != n for r in rows):
        return None
    for i in range(n):
        for j in range(n):
            if rows[i][j] not in (0, 1):
                return None
    return rows


def _matrix_from_obj(obj: Any, expected_n: Optional[int] = None) -> Optional[List[List[int]]]:
    if isinstance(obj, dict):
        if "adjacency_matrix" in obj:
            mat = _normalize_matrix(obj["adjacency_matrix"], expected_n=expected_n)
            if mat is not None:
                return mat
        ans = obj.get("answer")
        if isinstance(ans, dict) and "adjacency_matrix" in ans:
            mat = _normalize_matrix(ans["adjacency_matrix"], expected_n=expected_n)
            if mat is not None:
                return mat
    return _normalize_matrix(obj, expected_n=expected_n)


def _balanced_spans(text: str, open_ch: str, close_ch: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append((start, i + 1))
    return spans


def extract_adjacency_matrix(text: str, expected_n: Optional[int] = None) -> Optional[List[List[int]]]:
    if not text:
        return None

    candidates = [text, extract_answer_text(text)]
    for cand in candidates:
        cand = (cand or "").strip()
        if not cand:
            continue

        # 1) Full JSON
        try:
            obj = json.loads(cand)
            mat = _matrix_from_obj(obj, expected_n=expected_n)
            if mat is not None:
                return mat
        except Exception:
            pass

        # 2) Any JSON object spans
        for s, e in _balanced_spans(cand, "{", "}"):
            frag = cand[s:e]
            try:
                obj = json.loads(frag)
            except Exception:
                continue
            mat = _matrix_from_obj(obj, expected_n=expected_n)
            if mat is not None:
                return mat

        # 3) Any list-of-lists spans
        for s, e in _balanced_spans(cand, "[", "]"):
            frag = cand[s:e]
            if "[[" not in frag:
                continue
            try:
                obj = json.loads(frag)
            except Exception:
                continue
            mat = _normalize_matrix(obj, expected_n=expected_n)
            if mat is not None:
                return mat

    return None


def _load_json_from_raw(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # Prefer inline JSON first; long JSON strings can raise OSError if treated
    # as filesystem paths.
    try:
        return json.loads(s)
    except Exception:
        pass

    try:
        p = Path(s)
        if p.exists() and p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
    except OSError:
        return None
    return None


def target_matrix_from_answer(raw: Any) -> Optional[List[List[int]]]:
    obj = _load_json_from_raw(raw)
    if obj is None:
        return None
    return _matrix_from_obj(obj)


def _normalize_descendant_payload(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None

    payload = obj.get("answer") if isinstance(obj.get("answer"), dict) else obj
    if not isinstance(payload, dict):
        return None

    target = payload.get("target", payload.get("intervention_target"))
    descendants = payload.get("descendants", payload.get("affected_variables"))
    if target is None or not isinstance(descendants, list):
        return None

    target_str = str(target).strip()
    if not target_str:
        return None

    descendants_out: List[str] = []
    seen = set()
    for item in descendants:
        name = str(item).strip()
        if not name or name == target_str or name in seen:
            continue
        seen.add(name)
        descendants_out.append(name)

    return {
        "target": target_str,
        "descendants": descendants_out,
    }


def extract_descendant_payload(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    candidates = [text, extract_answer_text(text)]
    for cand in candidates:
        cand = (cand or "").strip()
        if not cand:
            continue

        try:
            obj = json.loads(cand)
            payload = _normalize_descendant_payload(obj)
            if payload is not None:
                return payload
        except Exception:
            pass

        for s, e in _balanced_spans(cand, "{", "}"):
            frag = cand[s:e]
            try:
                obj = json.loads(frag)
            except Exception:
                continue
            payload = _normalize_descendant_payload(obj)
            if payload is not None:
                return payload

    return None


def target_descendants_from_answer(raw: Any) -> Optional[Dict[str, Any]]:
    obj = _load_json_from_raw(raw)
    if obj is None:
        return None
    return _normalize_descendant_payload(obj)


def _set_f1(pred_items: List[str], target_items: List[str]) -> float:
    pred = set(str(x).strip() for x in pred_items if str(x).strip())
    gold = set(str(x).strip() for x in target_items if str(x).strip())
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def is_acyclic(adj: List[List[int]]) -> bool:
    n = len(adj)
    indeg = [0] * n
    out = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i][j] == 1:
                indeg[j] += 1
                out[i].append(j)

    q = [i for i in range(n) if indeg[i] == 0]
    seen = 0
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        seen += 1
        for v in out[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return seen == n


def edge_f1(pred: List[List[int]], target: List[List[int]]) -> float:
    n = len(target)
    tp = fp = fn = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = pred[i][j]
            t = target[i][j]
            if p == 1 and t == 1:
                tp += 1
            elif p == 1 and t == 0:
                fp += 1
            elif p == 0 and t == 1:
                fn += 1
    denom_p = tp + fp
    denom_r = tp + fn
    if denom_p == 0 and denom_r == 0:
        return 1.0
    precision = (tp / denom_p) if denom_p > 0 else 0.0
    recall = (tp / denom_r) if denom_r > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def normalized_shd(pred: List[List[int]], target: List[List[int]]) -> float:
    n = len(target)
    denom = float(max(n * (n - 1), 1))
    diff = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pred[i][j] != target[i][j]:
                diff += 1
    return float(diff) / denom


def _score_cd_prediction(
    pred_adj: List[List[int]],
    target_adj: List[List[int]],
    *,
    require_dag: bool,
    dag_penalty: float,
    shd_weight: float,
) -> Dict[str, float]:
    f1 = edge_f1(pred_adj, target_adj)
    shd_n = normalized_shd(pred_adj, target_adj)
    dag_ok = int(is_acyclic(pred_adj))

    reward = f1 - float(shd_weight) * shd_n
    if require_dag and not dag_ok:
        reward -= float(dag_penalty)
    reward = max(min(float(reward), 1.0), -1.0)

    return {
        "reward": reward,
        "edge_f1": float(f1),
        "shd_norm": float(shd_n),
        "dag_ok": float(dag_ok),
    }


def score_cd_completion(
    completion_text: str,
    target_answer: Any,
    require_dag: bool = True,
    dag_penalty: float = 0.1,
    shd_weight: float = 0.0,
) -> Dict[str, Any]:
    text = completion_to_text(completion_text)
    fmt = format_ok(text)
    target_adj = target_matrix_from_answer(target_answer)
    if target_adj is None:
        return {"reward": 0.0, "format_ok": fmt, "parse_ok": 0, "dag_ok": 0, "edge_f1": 0.0, "shd_norm": 1.0}

    pred_adj = extract_adjacency_matrix(text, expected_n=len(target_adj))
    if pred_adj is None:
        return {"reward": 0.0, "format_ok": fmt, "parse_ok": 0, "dag_ok": 0, "edge_f1": 0.0, "shd_norm": 1.0}

    scores = _score_cd_prediction(
        pred_adj,
        target_adj,
        require_dag=require_dag,
        dag_penalty=dag_penalty,
        shd_weight=shd_weight,
    )

    return {
        "reward": scores["reward"],
        "format_ok": fmt,
        "parse_ok": 1,
        "dag_ok": int(scores["dag_ok"]),
        "edge_f1": scores["edge_f1"],
        "shd_norm": scores["shd_norm"],
    }


def score_cd_descendants_completion(
    completion_text: str,
    target_answer: Any,
) -> Dict[str, Any]:
    text = completion_to_text(completion_text)
    fmt = format_ok(text)
    target_payload = target_descendants_from_answer(target_answer)
    if target_payload is None:
        return {"reward": 0.0, "format_ok": fmt, "parse_ok": 0, "target_ok": 0, "descendant_f1": 0.0}

    pred_payload = extract_descendant_payload(text)
    if pred_payload is None:
        return {"reward": 0.0, "format_ok": fmt, "parse_ok": 0, "target_ok": 0, "descendant_f1": 0.0}

    target_ok = int(pred_payload["target"] == target_payload["target"])
    f1 = _set_f1(pred_payload["descendants"], target_payload["descendants"]) if target_ok else 0.0
    return {
        "reward": float(f1),
        "format_ok": fmt,
        "parse_ok": 1,
        "target_ok": target_ok,
        "descendant_f1": float(f1),
    }


def build_cd_graph_reward(
    require_dag: bool = True,
    dag_penalty: float = 0.1,
    shd_weight: float = 0.0,
    scale: float = 1.0,
):
    if scale < 0:
        raise ValueError("scale must be >= 0")

    target_cache: Dict[str, Optional[List[List[int]]]] = {}

    def _cached_target(raw: Any) -> Optional[List[List[int]]]:
        key = str(raw)
        if key in target_cache:
            return target_cache[key]
        mat = target_matrix_from_answer(raw)
        target_cache[key] = mat
        return mat

    def cd_graph_reward(completions, **kwargs):
        answers = kwargs.get("answer")
        answer_paths = kwargs.get("answer_path")
        rewards: List[float] = []
        for i, completion in enumerate(completions):
            target_raw = None
            if answers is not None and i < len(answers):
                target_raw = answers[i]
            elif answer_paths is not None and i < len(answer_paths):
                target_raw = answer_paths[i]

            target_adj = _cached_target(target_raw)
            if target_adj is None:
                rewards.append(0.0)
                continue

            text = completion_to_text(completion)
            pred_adj = extract_adjacency_matrix(text, expected_n=len(target_adj))
            if pred_adj is None:
                rewards.append(0.0)
                continue

            f1 = edge_f1(pred_adj, target_adj)
            shd_n = normalized_shd(pred_adj, target_adj)
            reward = f1 - float(shd_weight) * shd_n
            if require_dag and not is_acyclic(pred_adj):
                reward -= float(dag_penalty)
            reward = float(scale) * float(reward)
            rewards.append(max(min(float(reward), 1.0), -1.0))
        return rewards

    cd_graph_reward.__name__ = "cd_graph_reward"
    return cd_graph_reward


def build_cd_descendant_partial_format_reward(scale: float = 0.25):
    if scale < 0:
        raise ValueError("scale must be >= 0")

    def cd_descendant_partial_format_reward(completions, **kwargs):
        rewards: List[float] = []
        s = float(scale)
        for completion in completions:
            t = completion_to_text(completion) or ""
            base = 0.0
            if "<answer>" in t:
                base += 0.2
            if "</answer>" in t:
                base += 0.2
            if '"target"' in t or "target" in t:
                base += 0.15
            if '"descendants"' in t or "descendants" in t:
                base += 0.15
            if extract_descendant_payload(t) is not None:
                base += 0.3
            rewards.append(float(s * max(0.0, min(1.0, base))))
        return rewards

    cd_descendant_partial_format_reward.__name__ = "cd_descendant_partial_format_reward"
    return cd_descendant_partial_format_reward


def build_cd_descendant_f1_reward(scale: float = 1.0):
    if scale < 0:
        raise ValueError("scale must be >= 0")

    target_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def _cached_target(raw: Any) -> Optional[Dict[str, Any]]:
        key = str(raw)
        if key in target_cache:
            return target_cache[key]
        payload = target_descendants_from_answer(raw)
        target_cache[key] = payload
        return payload

    def cd_descendant_f1_reward(completions, **kwargs):
        answers = kwargs.get("answer")
        answer_paths = kwargs.get("answer_path")
        rewards: List[float] = []
        s = float(scale)
        for i, completion in enumerate(completions):
            target_raw = None
            if answers is not None and i < len(answers):
                target_raw = answers[i]
            elif answer_paths is not None and i < len(answer_paths):
                target_raw = answer_paths[i]

            target_payload = _cached_target(target_raw)
            if target_payload is None:
                rewards.append(0.0)
                continue

            pred_payload = extract_descendant_payload(completion_to_text(completion))
            if pred_payload is None or pred_payload["target"] != target_payload["target"]:
                rewards.append(0.0)
                continue

            value = _set_f1(pred_payload["descendants"], target_payload["descendants"])
            rewards.append(float(s) * float(max(min(value, 1.0), 0.0)))
        return rewards

    cd_descendant_f1_reward.__name__ = "cd_descendant_f1_reward"
    return cd_descendant_f1_reward


def _build_cd_scalar_reward(
    *,
    name: str,
    scale: float,
    scorer,
):
    if scale < 0:
        raise ValueError("scale must be >= 0")

    target_cache: Dict[str, Optional[List[List[int]]]] = {}

    def _cached_target(raw: Any) -> Optional[List[List[int]]]:
        key = str(raw)
        if key in target_cache:
            return target_cache[key]
        mat = target_matrix_from_answer(raw)
        target_cache[key] = mat
        return mat

    def _reward_fn(completions, **kwargs):
        answers = kwargs.get("answer")
        answer_paths = kwargs.get("answer_path")
        rewards: List[float] = []
        s = float(scale)
        for i, completion in enumerate(completions):
            target_raw = None
            if answers is not None and i < len(answers):
                target_raw = answers[i]
            elif answer_paths is not None and i < len(answer_paths):
                target_raw = answer_paths[i]

            target_adj = _cached_target(target_raw)
            if target_adj is None:
                rewards.append(0.0)
                continue

            text = completion_to_text(completion)
            pred_adj = extract_adjacency_matrix(text, expected_n=len(target_adj))
            if pred_adj is None:
                rewards.append(0.0)
                continue

            value = float(scorer(pred_adj, target_adj))
            value = float(s) * value
            rewards.append(max(min(value, 1.0), -1.0))
        return rewards

    _reward_fn.__name__ = name
    return _reward_fn


def build_cd_edge_f1_reward(scale: float = 1.0):
    return _build_cd_scalar_reward(
        name="cd_edge_f1_reward",
        scale=scale,
        scorer=lambda pred_adj, target_adj: edge_f1(pred_adj, target_adj),
    )


def build_cd_low_shd_reward(scale: float = 1.0):
    return _build_cd_scalar_reward(
        name="cd_low_shd_reward",
        scale=scale,
        scorer=lambda pred_adj, target_adj: 1.0 - normalized_shd(pred_adj, target_adj),
    )


def build_cd_acyclic_reward(scale: float = 0.1):
    return _build_cd_scalar_reward(
        name="cd_acyclic_reward",
        scale=scale,
        scorer=lambda pred_adj, target_adj: float(is_acyclic(pred_adj)),
    )


def build_length_penalty_reward(
    tokenizer: Any,
    coef: float = 0.0,
    target_tokens: int = 0,
    max_abs: float = 1.0,
):
    if coef < 0:
        raise ValueError("coef must be >= 0")
    if target_tokens < 0:
        raise ValueError("target_tokens must be >= 0")

    def length_penalty_reward(completions, **kwargs):
        texts = [completion_to_text(c) for c in completions]
        tokenized = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        rewards: List[float] = []
        for ids in tokenized:
            # Length regularization (no free threshold): penalize every generated token.
            # `target_tokens` is kept only for backward CLI compatibility.
            penalty = float(coef) * float(len(ids))
            reward = -penalty
            if max_abs > 0:
                reward = max(reward, -float(max_abs))
            rewards.append(float(reward))
        return rewards

    length_penalty_reward.__name__ = "length_penalty_reward"
    return length_penalty_reward
