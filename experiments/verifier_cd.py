import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Accept either:
# 1) full standalone output: <think>...</think><answer>...</answer>
# 2) completion-only tail after prompt prefill: ...</think><answer>...</answer>
FORMAT_RE_FULL = re.compile(r"(?s)^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$")
FORMAT_RE_COMPLETION = re.compile(r"(?s)^\s*.+?</think>\s*<answer>.*?</answer>\s*$")
FORMAT_RE_ANSWER_TAIL = re.compile(r"(?s)^\s*(\{.*\}|\[.*\])\s*</answer>\s*$")
ANSWER_RE = re.compile(r"(?s)<answer>\s*(.*?)\s*</answer>")
PROMPT_COPY_MARKERS = (
    "OBSERVATIONAL SUMMARY",
    "INTERVENTION OF INTEREST",
    "tv_change_vs_obs",
    "obs_marginals=",
    "do_marginals=",
    "The intervention do(",
)
VARIABLE_ORDER_RE = re.compile(r"\b\d+\s*:\s*[^:\n]{1,80}states=\{")


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
    if m:
        return m.group(1)
    s = text or ""
    if "</answer>" in s:
        return s.rsplit("</answer>", 1)[0].strip()
    return s


def format_ok(text: str) -> int:
    s = text or ""
    return int(bool(FORMAT_RE_FULL.match(s) or FORMAT_RE_COMPLETION.match(s) or FORMAT_RE_ANSWER_TAIL.match(s)))


def cd_format_reward(completions, **kwargs):
    texts = [completion_to_text(c) for c in completions]
    return [1.0 if format_ok(t) else 0.0 for t in texts]


def _strict_cd_payload_ok(text: str) -> bool:
    """
    Strict formatting for reward purposes:
    require the outer format markers *and* a parseable causal-discovery payload
    inside the answer region (or, for answer-tail mode, in the completion tail).
    """
    if not format_ok(text):
        return False

    ans_text = extract_answer_text(text)
    if extract_descendant_payload(ans_text) is not None:
        return True
    if extract_descendant_payload(text) is not None:
        return True
    if extract_adjacency_matrix(ans_text) is not None:
        return True
    if extract_adjacency_matrix(text) is not None:
        return True
    return False


def _looks_like_prompt_copy(text: str) -> bool:
    s = str(text or "")
    if any(marker in s for marker in PROMPT_COPY_MARKERS):
        return True
    return bool(VARIABLE_ORDER_RE.search(s))


def _looks_like_null_answer(text: str) -> bool:
    s = str(text or "").strip().strip("`").strip().lower()
    return s in {"", "none", "null", "n/a", "na", "unknown", "[]", "{}"}


def build_cd_format_reward(scale: float = 0.2):
    if scale < 0:
        raise ValueError("scale must be >= 0")

    def _cd_format_reward(completions, **kwargs):
        texts = [completion_to_text(c) for c in completions]
        s = float(scale)
        return [s if _strict_cd_payload_ok(t) else 0.0 for t in texts]

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
            if "<think>" in t:
                base += 0.15
            if "</think>" in t:
                base += 0.15
            if "<answer>" in t:
                base += 0.15
            if "</answer>" in t:
                base += 0.15
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

    # If the string is a full completion with <answer>...</answer> tags,
    # extract the JSON from inside the tags (e.g. JSONL answer fields that
    # store the full staged completion rather than just the adjacency matrix).
    m = ANSWER_RE.search(s)
    if m:
        try:
            return json.loads(m.group(1).strip())
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


def _named_skeleton_edges_from_adj(adj: List[List[int]], variables: List[str]) -> set:
    """Return frozenset name-pairs for the undirected skeleton."""
    return {
        frozenset([variables[a], variables[b]])
        for a in range(len(adj))
        for b in range(a + 1, len(adj))
        if adj[a][b] == 1 or adj[b][a] == 1
    }


def _named_vstructs_from_adj(adj: List[List[int]], variables: List[str]) -> set:
    """Return normalized (parent1, collider, parent2) name-tuples for v-structures."""
    n = len(adj)
    vstructs = set()
    for k in range(n):
        parents = [i for i in range(n) if adj[i][k] == 1]
        for pi in range(len(parents)):
            for pj in range(pi + 1, len(parents)):
                i, j = parents[pi], parents[pj]
                if adj[i][j] == 0 and adj[j][i] == 0:
                    p1, p2 = sorted([variables[i], variables[j]])
                    vstructs.add((p1, variables[k], p2))
    return vstructs


def _named_directed_edges_from_adj(adj: List[List[int]], variables: List[str]) -> set:
    """Return (src_name, dst_name) pairs for all directed edges."""
    return {
        (variables[i], variables[j])
        for i in range(len(adj))
        for j in range(len(adj))
        if adj[i][j] == 1
    }


def _named_descendants_from_adj(adj: List[List[int]], variables: List[str], target_name: str) -> List[str]:
    """Return descendants of target_name in graph order, excluding the target itself."""
    if target_name not in variables:
        return []
    idx = variables.index(target_name)
    n = len(adj)
    seen = set()
    stack = [idx]
    while stack:
        u = stack.pop()
        for v in range(n):
            if adj[u][v] != 1 or v == idx or v in seen:
                continue
            seen.add(v)
            stack.append(v)
    return [variables[i] for i in range(n) if i in seen]


def build_cd_stage_targets(prompt: str, answer: Any) -> Optional[Dict[str, List[List[str]]]]:
    """
    Precompute ground-truth Stage 1/2/3 structures from the prompt VARIABLES block
    plus the target adjacency matrix.
    """
    variables = _variables_from_prompt(prompt or "")
    adj = target_matrix_from_answer(answer)
    if variables is None or adj is None or len(variables) != len(adj):
        return None

    skeleton = sorted(
        [sorted(list(edge)) for edge in _named_skeleton_edges_from_adj(adj, variables)],
        key=lambda pair: (pair[0], pair[1]),
    )
    vstructs = sorted(list(_named_vstructs_from_adj(adj, variables)))
    directed = sorted([list(edge) for edge in _named_directed_edges_from_adj(adj, variables)])
    return {
        "target_stage1_skeleton_edges": skeleton,
        "target_stage2_vstructures": vstructs,
        "target_stage3_directed_edges": directed,
    }


def _target_skeleton_edges_from_kwargs(kwargs: dict, index: int) -> Optional[set]:
    values = kwargs.get("target_stage1_skeleton_edges")
    if values is None or index >= len(values):
        return None
    raw = values[index]
    if not isinstance(raw, list):
        return None
    out = set()
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        a, b = str(item[0]), str(item[1])
        out.add(frozenset([a, b]))
    return out


def _target_vstructs_from_kwargs(kwargs: dict, index: int) -> Optional[set]:
    values = kwargs.get("target_stage2_vstructures")
    if values is None or index >= len(values):
        return None
    raw = values[index]
    if not isinstance(raw, list):
        return None
    out = set()
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return None
        out.add((str(item[0]), str(item[1]), str(item[2])))
    return out


def _target_directed_edges_from_kwargs(kwargs: dict, index: int) -> Optional[set]:
    values = kwargs.get("target_stage3_directed_edges")
    if values is None or index >= len(values):
        return None
    raw = values[index]
    if not isinstance(raw, list):
        return None
    out = set()
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        out.add((str(item[0]), str(item[1])))
    return out


_DO_TARGET_RE = re.compile(r"do\(\s*([A-Za-z0-9_]+)\s*=\s*[^)]*\)")


def _intervention_targets_from_prompt(prompt: str) -> List[str]:
    text = str(prompt or "")
    seen = set()
    out: List[str] = []
    for name in _DO_TARGET_RE.findall(text):
        name = str(name).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


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


_DESC_VAR_BLOCK_RE = re.compile(r"---\s*VARIABLE ORDER\s*---\s*\n(.*?)(?=\n---|\Z)", re.DOTALL)
_DESC_VAR_LINE_RE = re.compile(r"^\s*\d+\s*:\s*(\S+)", re.MULTILINE)
_DESC_TV_RE = re.compile(r"tv_change_vs_obs=(\[[^\n]*\])")
_DESC_DO_N_RE = re.compile(r"\bdo_n=(\d+)")
_DESC_STAGE1_HINT_RE = re.compile(r"\b(shift detection|tv change|total-variation|rank(?:ing)?|marginal shift)\b", re.IGNORECASE)
_DESC_STAGE2_HINT_RE = re.compile(r"\b(threshold analysis|noise threshold|sampling noise|stable|shifted)\b", re.IGNORECASE)
_DESC_STAGE3_HINT_RE = re.compile(r"\b(conclusion|descendants?\s+of|has no descendants|final descendants)\b", re.IGNORECASE)


def _value_at(values: Any, index: int) -> Any:
    if isinstance(values, (list, tuple)):
        return values[index] if index < len(values) else None
    return values


def _descendant_noise_threshold(do_n: int) -> float:
    if do_n <= 0:
        return 0.20
    if do_n <= 5:
        return 0.15
    if do_n <= 20:
        return 0.10
    if do_n <= 50:
        return 0.07
    return 0.05


def _descendant_variables_from_prompt(prompt: str) -> List[str]:
    m = _DESC_VAR_BLOCK_RE.search(prompt or "")
    if not m:
        return []
    return [str(name) for name in _DESC_VAR_LINE_RE.findall(m.group(1))]


def _extract_descendant_prompt_targets(prompt: str, answer: Any) -> Optional[Dict[str, Any]]:
    payload = target_descendants_from_answer(answer)
    if payload is None:
        return None

    prompt_text = str(prompt or "")
    tv_match = _DESC_TV_RE.search(prompt_text)
    tv_changes: List[Tuple[str, float]] = []
    if tv_match:
        try:
            raw = json.loads(tv_match.group(1))
            for item in raw:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                tv_changes.append((str(item[0]).strip(), float(item[1])))
        except Exception:
            tv_changes = []

    do_n = 0
    do_match = _DESC_DO_N_RE.search(prompt_text)
    if do_match:
        try:
            do_n = int(do_match.group(1))
        except Exception:
            do_n = 0

    target = str(payload["target"])
    descendants = [str(x) for x in payload["descendants"]]
    desc_set = set(descendants)
    threshold = _descendant_noise_threshold(do_n)

    variables = _descendant_variables_from_prompt(prompt_text)
    if variables:
        non_target_vars = [v for v in variables if v != target]
    else:
        non_target_vars = [v for v, _ in tv_changes if v != target]

    tv_map = {v: float(tv) for v, tv in tv_changes if v and v != target}
    ranking = [v for v, _ in tv_changes if v and v != target]
    if not ranking:
        ranking = [v for v in non_target_vars if v != target]

    shift_labels = {
        v: ("shifted" if float(tv_map.get(v, 0.0)) > threshold else "stable")
        for v in non_target_vars
    }
    descendant_labels = {v: ("descendant" if v in desc_set else "not descendant") for v in non_target_vars}
    return {
        "target": target,
        "descendants": descendants,
        "ranking": ranking,
        "shift_labels": shift_labels,
        "descendant_labels": descendant_labels,
        "variables": non_target_vars,
        "threshold": float(threshold),
        "do_n": int(do_n),
    }


def _descendant_targets_from_kwargs(kwargs: dict, index: int, cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    prompt = _value_at(kwargs.get("prompt_raw"), index)
    if prompt is None:
        prompt = _value_at(kwargs.get("prompt"), index)
    answer = _value_at(kwargs.get("answer"), index)
    if answer is None:
        answer = _value_at(kwargs.get("answer_path"), index)
    key = (str(prompt or ""), str(answer or ""))
    if key in cache:
        return cache[key]
    cache[key] = _extract_descendant_prompt_targets(str(prompt or ""), answer)
    return cache[key]


def _vars_in_text_order(text: str, variables: List[str]) -> List[str]:
    if not text or not variables:
        return []
    pattern = re.compile(r"\b(" + "|".join(re.escape(v) for v in sorted(set(variables), key=len, reverse=True)) + r")\b")
    seen = set()
    out: List[str] = []
    for match in pattern.finditer(text):
        name = match.group(1)
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _pairwise_order_score(pred_order: List[str], target_order: List[str]) -> float:
    pred_pos = {name: idx for idx, name in enumerate(pred_order)}
    total = 0
    correct = 0
    for i in range(len(target_order)):
        for j in range(i + 1, len(target_order)):
            a, b = target_order[i], target_order[j]
            if a not in pred_pos or b not in pred_pos:
                continue
            total += 1
            if pred_pos[a] < pred_pos[b]:
                correct += 1
    if total == 0:
        return 0.0
    return float(correct) / float(total)


def _descendant_stage2_lines(stage2_text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for raw_line in (stage2_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        out.append((line, line.lower()))
    return out


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
            ans_text = extract_answer_text(t)
            payload = extract_descendant_payload(t)
            base = 0.0

            if "</think>" in t:
                base += 0.10
            if "<answer>" in t:
                base += 0.05
            if "</answer>" in t:
                base += 0.05
            if '"target"' in ans_text or "target" in ans_text:
                base += 0.20
            if '"descendants"' in ans_text or "descendants" in ans_text:
                base += 0.20
            if payload is not None:
                base += 0.40

            # Do not reward degenerate answers like <answer>None</answer>.
            if _looks_like_null_answer(ans_text):
                base = min(base, 0.02)

            # Strongly downweight prompt-copy behavior unless it still parses.
            if _looks_like_prompt_copy(t):
                base *= 0.15 if payload is None else 0.5

            rewards.append(float(s * max(0.0, min(1.0, base))))
        return rewards

    cd_descendant_partial_format_reward.__name__ = "cd_descendant_partial_format_reward"
    return cd_descendant_partial_format_reward


def build_cd_descendant_answer_tail_partial_format_reward(scale: float = 0.25):
    if scale < 0:
        raise ValueError("scale must be >= 0")

    def cd_descendant_answer_tail_partial_format_reward(completions, **kwargs):
        rewards: List[float] = []
        s = float(scale)
        for completion in completions:
            t = completion_to_text(completion) or ""
            ans_text = extract_answer_text(t)
            payload = extract_descendant_payload(t)
            base = 0.0

            if ans_text.lstrip().startswith("{"):
                base += 0.20
            if ans_text.lstrip().startswith("["):
                base += 0.05
            if '"target"' in ans_text or "target" in ans_text:
                base += 0.25
            if '"descendants"' in ans_text or "descendants" in ans_text:
                base += 0.25
            if "</answer>" in t:
                base += 0.10

            if payload is not None:
                base += 0.35

            if _looks_like_null_answer(ans_text):
                base = min(base, 0.02)

            if _looks_like_prompt_copy(t):
                base *= 0.10 if payload is None else 0.5

            rewards.append(float(s * max(0.0, min(1.0, base))))
        return rewards

    cd_descendant_answer_tail_partial_format_reward.__name__ = (
        "cd_descendant_answer_tail_partial_format_reward"
    )
    return cd_descendant_answer_tail_partial_format_reward


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


def build_cd_descendant_cot_structure_reward(scale: float = 0.1):
    if scale < 0:
        raise ValueError("scale must be >= 0")

    def _reward_fn(completions, **kwargs):
        rewards: List[float] = []
        s = float(scale)
        for completion in completions:
            text = completion_to_text(completion)
            think = _extract_think_block(text)
            if not think:
                rewards.append(0.0)
                continue
            score = 0.0
            if _STAGE1_RE.search(think) or _DESC_STAGE1_HINT_RE.search(think):
                score += 1.0 / 3.0
            if _STAGE2_RE.search(think) or _DESC_STAGE2_HINT_RE.search(think):
                score += 1.0 / 3.0
            if _STAGE3_RE.search(think) or _DESC_STAGE3_HINT_RE.search(think):
                score += 1.0 / 3.0
            rewards.append(float(s * score))
        return rewards

    _reward_fn.__name__ = "cd_descendant_cot_structure_reward"
    return _reward_fn


def build_cd_descendant_shift_ranking_reward(scale: float = 0.2):
    if scale < 0:
        raise ValueError("scale must be >= 0")

    target_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}

    def _reward_fn(completions, **kwargs):
        rewards: List[float] = []
        s = float(scale)
        for i, completion in enumerate(completions):
            targets = _descendant_targets_from_kwargs(kwargs, i, target_cache)
            if not targets:
                rewards.append(0.0)
                continue

            target_order = [v for v in targets["ranking"] if isinstance(v, str)]
            if len(target_order) < 2:
                rewards.append(0.0)
                continue

            think = _extract_think_block(completion_to_text(completion))
            stage1 = _extract_stage(think, _STAGE1_RE, _STAGE2_RE) if think else ""
            if not stage1:
                rewards.append(0.0)
                continue

            pred_order = _vars_in_text_order(stage1, target_order)
            rewards.append(float(s * _pairwise_order_score(pred_order, target_order)))
        return rewards

    _reward_fn.__name__ = "cd_descendant_shift_ranking_reward"
    return _reward_fn


def build_cd_descendant_variable_classification_reward(scale: float = 0.2):
    if scale < 0:
        raise ValueError("scale must be >= 0")

    target_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}

    def _reward_fn(completions, **kwargs):
        rewards: List[float] = []
        s = float(scale)
        for i, completion in enumerate(completions):
            targets = _descendant_targets_from_kwargs(kwargs, i, target_cache)
            if not targets:
                rewards.append(0.0)
                continue

            shift_targets = dict(targets["shift_labels"])
            desc_targets = dict(targets["descendant_labels"])
            if not shift_targets and not desc_targets:
                rewards.append(0.0)
                continue

            think = _extract_think_block(completion_to_text(completion))
            stage2 = _extract_stage(think, _STAGE2_RE, _STAGE3_RE) if think else ""
            if not stage2:
                rewards.append(0.0)
                continue

            total = 0
            correct = 0
            for line, line_lower in _descendant_stage2_lines(stage2):
                vars_in_line = _vars_in_text_order(line, targets["variables"])
                if not vars_in_line:
                    continue
                var = vars_in_line[0]

                pred_shift: Optional[str] = None
                if "stable" in line_lower:
                    pred_shift = "stable"
                elif "shifted" in line_lower or "shift" in line_lower:
                    pred_shift = "shifted"

                pred_desc: Optional[str] = None
                if "not descendant" in line_lower:
                    pred_desc = "not descendant"
                elif "descendant" in line_lower:
                    pred_desc = "descendant"

                if pred_shift is not None and var in shift_targets:
                    total += 1
                    if pred_shift == shift_targets[var]:
                        correct += 1
                if pred_desc is not None and var in desc_targets:
                    total += 1
                    if pred_desc == desc_targets[var]:
                        correct += 1

            rewards.append(float(s * (float(correct) / float(total) if total > 0 else 0.0)))
        return rewards

    _reward_fn.__name__ = "cd_descendant_variable_classification_reward"
    return _reward_fn


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


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Stage header splitters
_STAGE1_RE = re.compile(r"Stage\s+1\s*\([^)]*\)[^:]*:", re.IGNORECASE)
_STAGE2_RE = re.compile(r"Stage\s+2\s*\([^)]*\)[^:]*:", re.IGNORECASE)
_STAGE3_RE = re.compile(r"Stage\s+3\s*\([^)]*\)[^:]*:", re.IGNORECASE)

# Stage 1 keywords: skeleton identification
_SKELETON_RE = re.compile(
    r"\b(skeleton|adjacent|undirected|connected pair|no direct|not adjacent"
    r"|are connected|are not connected|edge between|no edge)\b",
    re.IGNORECASE,
)
# Stage 2 keywords: v-structure / collider identification
_VSTRUCT_RE = re.compile(
    r"\b(collider|v.structure|v structure|unshielded|immorality|triple|blocked|d.separat)\b",
    re.IGNORECASE,
)
# Stage 3 keywords: orientation / Meek rules
_ORIENT_RE = re.compile(
    r"\b(meek|orient|direction|acyclic|propagat|rule [123]|R[123]:)\b",
    re.IGNORECASE,
)

# Edge patterns used for parsing think-block stages
_EDGE_DIRECTED_RE = re.compile(r"\b(\w+)\s*->\s*(\w+)\b")
_EDGE_UNDIRECTED_RE = re.compile(r"\b(\w+)\s*--\s*(\w+)\b")
# Collider triple: (parent1, collider, parent2)
_COLLIDER_TRIPLE_RE = re.compile(r"\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)")

# Variable block in prompt: "0: X1", "1: Pollution", etc.
# Matches both "--- VARIABLES ---" and "--- VARIABLE ORDER (...) ---"
_VAR_BLOCK_RE = re.compile(r"---\s*VARIABLES\s*---\s*\n(.*?)(?=\n---|\Z)", re.DOTALL)
_VAR_ORDER_BLOCK_RE = re.compile(r"---\s*VARIABLE ORDER[^-]*---\s*\n(.*?)(?=\n---|\Z)", re.DOTALL)
_VAR_LINE_RE = re.compile(r"^\s*\d+\s*:\s*(\S+)", re.MULTILINE)


def _extract_think_block(text: str) -> str:
    """
    Return the contents of the think block.

    Supports both:
    - full outputs: <think>...</think><answer>...</answer>
    - completion-only tails after prompt prefill: ...</think><answer>...</answer>
    """
    s = text or ""
    m = _THINK_RE.search(s)
    if m:
        return m.group(1)

    # GRPO commonly prefills the prompt through the opening <think> tag, so the
    # reward function may receive only the generated tail:
    #   "Stage 1 ... </think><answer>..."
    end_idx = s.find("</think>")
    if end_idx != -1:
        return s[:end_idx]
    return ""


def _extract_stage(think: str, start_re: re.Pattern, end_re: re.Pattern) -> str:
    """Return the text of one stage section, bounded by its header and the next stage header."""
    m = start_re.search(think)
    if not m:
        return ""
    start = m.end()
    m2 = end_re.search(think, start)
    end = m2.start() if m2 else len(think)
    return think[start:end].strip()


def _variables_from_prompt(prompt: str) -> Optional[List[str]]:
    """Extract ordered variable names from the VARIABLES block in a prompt.

    Handles both summary_joint format (--- VARIABLES ---) and matrix format
    (--- VARIABLE ORDER (ORDER MATTERS) ---).
    """
    m = _VAR_BLOCK_RE.search(prompt or "") or _VAR_ORDER_BLOCK_RE.search(prompt or "")
    if not m:
        return None
    names = _VAR_LINE_RE.findall(m.group(1))
    return names if names else None


def _prompt_text_from_kwargs(kwargs: dict, index: int) -> str:
    prompt = _value_at(kwargs.get("prompt_raw"), index)
    if prompt is None:
        prompt = _value_at(kwargs.get("prompt"), index)
    return str(prompt or "")


def _skeleton_from_adj(adj: List[List[int]]) -> set:
    """Return index-pair set representing the undirected skeleton."""
    n = len(adj)
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 1 or adj[j][i] == 1:
                edges.add((i, j))
    return edges


def _skeleton_from_stage1_text(stage1_text: str) -> set:
    """Parse 'X -- Y' lines from Stage 1 text into a set of frozenset name-pairs."""
    edges = set()
    for m in _EDGE_UNDIRECTED_RE.finditer(stage1_text):
        a, b = m.group(1), m.group(2)
        edges.add(frozenset([a, b]))
    return edges


def build_cd_cot_structure_reward(scale: float = 0.1):
    """
    Soft shaping reward for staged reasoning structure inside <think>.
    Awards credit proportionally for including all three stages.

    Prefer explicit stage headers when present, since the training/eval format
    requires:
      - Stage 1 (Skeleton)
      - Stage 2 (V-structures)
      - Stage 3 (Orientation)

    Fall back to the older keyword heuristics for partially formatted outputs.
    - Stage 1 (Skeleton): 1/3 of scale
    - Stage 2 (V-structures): 1/3 of scale
    - Stage 3 (Orientation): 1/3 of scale
    """
    if scale < 0:
        raise ValueError("scale must be >= 0")

    def _cd_cot_structure_reward(completions, **kwargs):
        rewards: List[float] = []
        s = float(scale)
        for completion in completions:
            text = completion_to_text(completion)
            think = _extract_think_block(text)
            if not think:
                rewards.append(0.0)
                continue
            score = 0.0
            if _STAGE1_RE.search(think) or _SKELETON_RE.search(think):
                score += 1.0 / 3.0
            if _STAGE2_RE.search(think) or _VSTRUCT_RE.search(think):
                score += 1.0 / 3.0
            if _STAGE3_RE.search(think) or _ORIENT_RE.search(think):
                score += 1.0 / 3.0
            rewards.append(float(s * score))
        return rewards

    _cd_cot_structure_reward.__name__ = "cd_cot_structure_reward"
    return _cd_cot_structure_reward


def build_cd_skeleton_f1_reward(scale: float = 0.2):
    """
    Hard shaping reward that parses Stage 1 (Skeleton) text from <think> and computes
    skeleton F1 against the ground truth.

    Parses 'X -- Y' lines from the Stage 1 section by name, then maps names to indices
    using the VARIABLES block in the prompt. Falls back to 0 if Stage 1 is missing or
    variable names cannot be resolved.
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

    def _named_skeleton_f1(pred_edges: set, target_edges: set) -> float:
        """F1 over sets of frozenset name-pairs."""
        if not pred_edges and not target_edges:
            return 1.0
        if not pred_edges or not target_edges:
            return 0.0
        tp = len(pred_edges & target_edges)
        precision = tp / len(pred_edges)
        recall = tp / len(target_edges)
        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def _cd_skeleton_f1_reward(completions, **kwargs):
        answers = kwargs.get("answer")
        answer_paths = kwargs.get("answer_path")
        rewards: List[float] = []
        s = float(scale)

        for i, completion in enumerate(completions):
            target_edges = _target_skeleton_edges_from_kwargs(kwargs, i)
            prompt_text = _prompt_text_from_kwargs(kwargs, i)
            variables = _variables_from_prompt(prompt_text) if prompt_text else None
            if target_edges is None:
                target_raw = None
                if answers is not None and i < len(answers):
                    target_raw = answers[i]
                elif answer_paths is not None and i < len(answer_paths):
                    target_raw = answer_paths[i]

                target_adj = _cached_target(target_raw)
                if target_adj is None:
                    rewards.append(0.0)
                    continue

                if variables is None or len(variables) != len(target_adj):
                    rewards.append(0.0)
                    continue

                target_edges = _named_skeleton_edges_from_adj(target_adj, variables)

            text = completion_to_text(completion)
            think = _extract_think_block(text)
            think_score = 0.0
            if think:
                stage1_text = _extract_stage(think, _STAGE1_RE, _STAGE2_RE)
                pred_edges = _skeleton_from_stage1_text(stage1_text)
                think_score = _named_skeleton_f1(pred_edges, target_edges)

            answer_score = 0.0
            if variables is not None:
                pred_adj = extract_adjacency_matrix(text, expected_n=len(variables))
                if pred_adj is not None:
                    answer_edges = _named_skeleton_edges_from_adj(pred_adj, variables)
                    answer_score = _named_skeleton_f1(answer_edges, target_edges)

            value = max(float(think_score), float(answer_score))
            rewards.append(float(s) * float(max(min(value, 1.0), 0.0)))
        return rewards

    _cd_skeleton_f1_reward.__name__ = "cd_skeleton_f1_reward"
    return _cd_skeleton_f1_reward


def build_cd_vstruct_f1_reward(scale: float = 0.15):
    """
    Shaping reward that parses Stage 2 (V-structures) text from <think> and computes
    F1 over unshielded collider triples against ground truth.

    Ground truth v-structures are derived from the adjacency matrix: a triple (i, k, j)
    is a v-structure iff i->k, j->k, and i-k-j is NOT in the skeleton (unshielded).
    Predicted triples are parsed from '(parent1, collider, parent2)' lines.
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

    def _vstruct_set_from_stage2_text(stage2_text: str) -> set:
        """Parse '(parent1, collider, parent2)' triples; normalize by sorting parents."""
        vstructs = set()
        for m in _COLLIDER_TRIPLE_RE.finditer(stage2_text):
            p1, col, p2 = m.group(1), m.group(2), m.group(3)
            sp1, sp2 = sorted([p1, p2])
            vstructs.add((sp1, col, sp2))
        return vstructs

    def _f1(pred: set, target: set) -> float:
        if not pred and not target:
            return 1.0
        if not pred or not target:
            return 0.0
        tp = len(pred & target)
        prec = tp / len(pred)
        rec = tp / len(target)
        if prec + rec == 0.0:
            return 0.0
        return 2.0 * prec * rec / (prec + rec)

    def _cd_vstruct_f1_reward(completions, **kwargs):
        answers = kwargs.get("answer")
        answer_paths = kwargs.get("answer_path")
        rewards: List[float] = []
        s = float(scale)

        for i, completion in enumerate(completions):
            target_vstructs = _target_vstructs_from_kwargs(kwargs, i)
            prompt_text = _prompt_text_from_kwargs(kwargs, i)
            variables = _variables_from_prompt(prompt_text) if prompt_text else None
            if target_vstructs is None:
                target_raw = None
                if answers is not None and i < len(answers):
                    target_raw = answers[i]
                elif answer_paths is not None and i < len(answer_paths):
                    target_raw = answer_paths[i]

                target_adj = _cached_target(target_raw)
                if target_adj is None:
                    rewards.append(0.0)
                    continue

                if variables is None or len(variables) != len(target_adj):
                    rewards.append(0.0)
                    continue

                target_vstructs = _named_vstructs_from_adj(target_adj, variables)

            text = completion_to_text(completion)
            think = _extract_think_block(text)
            think_score = 0.0
            if think:
                stage2_text = _extract_stage(think, _STAGE2_RE, _STAGE3_RE)
                pred_vstructs = _vstruct_set_from_stage2_text(stage2_text)
                think_score = _f1(pred_vstructs, target_vstructs)

            answer_score = 0.0
            if variables is not None:
                pred_adj = extract_adjacency_matrix(text, expected_n=len(variables))
                if pred_adj is not None:
                    answer_vstructs = _named_vstructs_from_adj(pred_adj, variables)
                    answer_score = _f1(answer_vstructs, target_vstructs)

            value = max(float(think_score), float(answer_score))
            rewards.append(float(s) * float(max(min(value, 1.0), 0.0)))
        return rewards

    _cd_vstruct_f1_reward.__name__ = "cd_vstruct_f1_reward"
    return _cd_vstruct_f1_reward


def build_cd_orientation_f1_reward(scale: float = 0.15):
    """
    Shaping reward that parses Stage 3 (Orientation) text from <think> and computes
    F1 over directed edges against ground truth.

    Predicted edges are parsed from 'X -> Y' lines in the Stage 3 section.
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

    def _directed_set_from_stage3_text(stage3_text: str) -> set:
        """Parse 'X -> Y' lines from Stage 3 text."""
        return {
            (m.group(1), m.group(2))
            for m in _EDGE_DIRECTED_RE.finditer(stage3_text)
        }

    def _f1(pred: set, target: set) -> float:
        if not pred and not target:
            return 1.0
        if not pred or not target:
            return 0.0
        tp = len(pred & target)
        prec = tp / len(pred)
        rec = tp / len(target)
        if prec + rec == 0.0:
            return 0.0
        return 2.0 * prec * rec / (prec + rec)

    def _cd_orientation_f1_reward(completions, **kwargs):
        answers = kwargs.get("answer")
        answer_paths = kwargs.get("answer_path")
        rewards: List[float] = []
        s = float(scale)

        for i, completion in enumerate(completions):
            target_directed = _target_directed_edges_from_kwargs(kwargs, i)
            prompt_text = _prompt_text_from_kwargs(kwargs, i)
            variables = _variables_from_prompt(prompt_text) if prompt_text else None
            if target_directed is None:
                target_raw = None
                if answers is not None and i < len(answers):
                    target_raw = answers[i]
                elif answer_paths is not None and i < len(answer_paths):
                    target_raw = answer_paths[i]

                target_adj = _cached_target(target_raw)
                if target_adj is None:
                    rewards.append(0.0)
                    continue

                if variables is None or len(variables) != len(target_adj):
                    rewards.append(0.0)
                    continue

                target_directed = _named_directed_edges_from_adj(target_adj, variables)

            text = completion_to_text(completion)
            think = _extract_think_block(text)
            think_score = 0.0
            if think:
                # Stage 3 runs to end of think block (no following stage header)
                m3 = _STAGE3_RE.search(think)
                stage3_text = think[m3.end():].strip() if m3 else ""
                pred_directed = _directed_set_from_stage3_text(stage3_text)
                think_score = _f1(pred_directed, target_directed)

            answer_score = 0.0
            if variables is not None:
                pred_adj = extract_adjacency_matrix(text, expected_n=len(variables))
                if pred_adj is not None:
                    answer_directed = _named_directed_edges_from_adj(pred_adj, variables)
                    answer_score = _f1(answer_directed, target_directed)

            value = max(float(think_score), float(answer_score))
            rewards.append(float(s) * float(max(min(value, 1.0), 0.0)))
        return rewards

    _cd_orientation_f1_reward.__name__ = "cd_orientation_f1_reward"
    return _cd_orientation_f1_reward


def build_cd_descendant_consistency_reward(scale: float = 0.1):
    """
    Reward consistency of the final predicted graph with intervention-local descendant sets.

    For each intervention target mentioned in the prompt, compare the descendants implied by
    the predicted adjacency matrix against the true descendants from the target graph and
    average descendant-set F1 across targets.
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
            prompt_text = _prompt_text_from_kwargs(kwargs, i)
            variables = _variables_from_prompt(prompt_text) if prompt_text else None
            if target_adj is None or variables is None or len(variables) != len(target_adj):
                rewards.append(0.0)
                continue

            intervention_targets = _intervention_targets_from_prompt(prompt_text)
            intervention_targets = [name for name in intervention_targets if name in variables]
            if not intervention_targets:
                rewards.append(0.0)
                continue

            text = completion_to_text(completion)
            pred_adj = extract_adjacency_matrix(text, expected_n=len(target_adj))
            if pred_adj is None:
                rewards.append(0.0)
                continue

            per_target_scores: List[float] = []
            for target_name in intervention_targets:
                gold_desc = _named_descendants_from_adj(target_adj, variables, target_name)
                pred_desc = _named_descendants_from_adj(pred_adj, variables, target_name)
                per_target_scores.append(_set_f1(pred_desc, gold_desc))

            value = sum(per_target_scores) / len(per_target_scores) if per_target_scores else 0.0
            rewards.append(float(s) * float(max(min(value, 1.0), 0.0)))
        return rewards

    _reward_fn.__name__ = "cd_descendant_consistency_reward"
    return _reward_fn


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
