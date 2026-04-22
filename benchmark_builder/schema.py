from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "benchmark_spec/v1"


class BenchmarkSpecError(ValueError):
    """Raised when a benchmark manifest is invalid."""


@dataclass
class DatasetSpec:
    name: str
    graph_source: str
    graph_path: str = ""
    graph_params: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    benchmark_card: str | None = None
    role: str = "main"
    notes: str = ""


@dataclass
class PromptCellSpec:
    style: str
    obs_per_prompt: int
    int_per_combo: int
    anonymize: bool = False
    row_order: str = "random"
    col_order: str = "original"
    causal_rules: bool = False
    give_steps: bool = False
    def_int: bool = False
    intervene_vars: str = "all"
    enabled: bool = True
    label: str | None = None

    @property
    def naming_regime(self) -> str:
        return "anonymized" if self.anonymize else "real"


@dataclass
class NamesOnlySpec:
    enabled: bool = False
    col_order: str = "original"


@dataclass
class ModelSpec:
    name: str
    provider: str = "auto"
    temperature: float = 0.0
    enabled: bool = True
    backend_version: str = "unknown"
    tags: list[str] = field(default_factory=list)


@dataclass
class BaselineSpec:
    name: str
    enabled: bool = True
    scope: str = "all"
    sample_size_obs: int | None = None
    sample_size_inters: int | None = None
    max_inters: int = -1
    seed: int | None = None
    checkpoint_dir: str | None = None
    library: str | None = None
    pc_variant: str = "stable"
    pc_ci_test: str = "chi_square"
    pc_significance_level: float = 0.01
    pc_max_cond_vars: int = 5
    ges_scoring_method: str = "bic-d"
    ges_min_improvement: float = 1e-6
    model: str | None = None
    provider: str = "auto"
    temperature: float = 0.0
    max_new_tokens: int | None = None
    num_samples: int = 5
    edge_threshold: float = 0.5
    takayama_pattern: int = 2
    takayama_bootstrap_samples: int = 100
    notes: str = ""


@dataclass
class EvaluatorSpec:
    tau: float = 0.7
    answer_col: str | None = None
    pred_col: str = "prediction"
    parser_version: str = "query_api.extract_adjacency"
    evaluator_version: str = "evaluate.py"


@dataclass
class ProvenanceSpec:
    schema_version: str = SCHEMA_VERSION
    parser_version: str = "query_api.extract_adjacency"
    evaluator_version: str = "evaluate.py"
    tags: list[str] = field(default_factory=list)


@dataclass
class ExecutionSpec:
    prompt_storage: str = "disk"
    prompt_retention: str = "full"
    example_prompt_dir: str | None = None


@dataclass
class BenchmarkSpec:
    name: str
    role: str
    description: str
    task_family: str
    datasets: list[DatasetSpec]
    prompt_cells: list[PromptCellSpec]
    names_only: NamesOnlySpec
    models: list[ModelSpec]
    baselines: list[BaselineSpec]
    evaluator: EvaluatorSpec
    provenance: ProvenanceSpec
    execution: ExecutionSpec
    num_prompts: int
    shuffles_per_graph: int
    seed: int
    output_root: str = "benchmark_runs"
    notes: str = ""
    benchmark_card: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkSpecError(message)


def _normalize_dataset_items(raw: dict[str, Any]) -> list[dict[str, Any]]:
    if "datasets" in raw:
        items = raw["datasets"]
        _expect(isinstance(items, list) and items, "`datasets` must be a non-empty list.")
        return items

    dataset_name = raw.get("dataset")
    bif_file = raw.get("bif_file")
    graph_file = raw.get("graph_file") or bif_file
    _expect(bool(dataset_name) and bool(graph_file), "Legacy manifests require `dataset` and `bif_file`.")
    return [
        {
            "name": dataset_name,
            "graph_source": raw.get("graph_source", "bif"),
            "graph_path": graph_file,
            "benchmark_card": raw.get("benchmark_card"),
            "role": raw.get("role", "main"),
            "notes": raw.get("notes", ""),
        }
    ]


def _normalize_models(raw: dict[str, Any]) -> list[dict[str, Any]]:
    if "models" in raw:
        return raw["models"]
    if "model_roster" in raw:
        return raw["model_roster"]
    if "model" in raw:
        return [
            {
                "name": raw["model"],
                "temperature": raw.get("temperature", 0.0),
                "provider": "auto",
                "enabled": True,
            }
        ]
    return []


def _normalize_baselines(raw: dict[str, Any]) -> list[dict[str, Any]]:
    if "baselines" in raw:
        return raw["baselines"]
    if "baseline_roster" in raw:
        return raw["baseline_roster"]

    out: list[dict[str, Any]] = []
    legacy_enco = raw.get("enco")
    if isinstance(legacy_enco, dict) and legacy_enco.get("enabled"):
        out.append(
            {
                "name": "ENCO",
                "enabled": True,
                "scope": "all",
                "sample_size_obs": legacy_enco.get("sample_size_obs"),
                "sample_size_inters": legacy_enco.get("sample_size_inters"),
                "max_inters": legacy_enco.get("max_inters", -1),
                "seed": legacy_enco.get("seed"),
                "checkpoint_dir": legacy_enco.get("checkpoint_dir"),
                "library": "internal",
            }
        )
    return out


def _normalize_prompt_cells(raw: dict[str, Any]) -> list[dict[str, Any]]:
    items = raw.get("prompt_cells", [])
    _expect(isinstance(items, list), "`prompt_cells` must be a list.")
    return items


def _normalize_names_only(raw: dict[str, Any]) -> dict[str, Any]:
    names_only = raw.get("names_only", {})
    if not names_only:
        return {"enabled": False}
    return names_only


def _build_dataset(item: dict[str, Any]) -> DatasetSpec:
    graph_path = item.get("graph_path") or item.get("bif_file") or ""
    _expect(bool(item.get("name")), "Each dataset entry requires `name`.")
    _expect(bool(item.get("graph_source")), f"Dataset {item.get('name')} requires `graph_source`.")
    if item.get("graph_source") != "synthetic":
        _expect(bool(graph_path), f"Dataset {item.get('name')} requires `graph_path`.")
    return DatasetSpec(
        name=item["name"],
        graph_source=item["graph_source"],
        graph_path=graph_path,
        graph_params=dict(item.get("graph_params", {})),
        tags=list(item.get("tags", [])),
        benchmark_card=item.get("benchmark_card"),
        role=item.get("role", "main"),
        notes=item.get("notes", ""),
    )


def _build_prompt_cell(item: dict[str, Any]) -> PromptCellSpec:
    _expect(item.get("style") in {"summary_joint", "matrix"}, "Supported prompt styles are `summary_joint` and `matrix`.")
    return PromptCellSpec(
        style=item["style"],
        obs_per_prompt=int(item["obs_per_prompt"]),
        int_per_combo=int(item["int_per_combo"]),
        anonymize=bool(item.get("anonymize", False)),
        row_order=item.get("row_order", "random"),
        col_order=item.get("col_order", "original"),
        causal_rules=bool(item.get("causal_rules", False)),
        give_steps=bool(item.get("give_steps", False)),
        def_int=bool(item.get("def_int", False)),
        intervene_vars=item.get("intervene_vars", "all"),
        enabled=bool(item.get("enabled", True)),
        label=item.get("label"),
    )


def _build_model(item: dict[str, Any]) -> ModelSpec:
    _expect(bool(item.get("name")), "Each model entry requires `name`.")
    return ModelSpec(
        name=item["name"],
        provider=item.get("provider", "auto"),
        temperature=float(item.get("temperature", 0.0)),
        enabled=bool(item.get("enabled", True)),
        backend_version=item.get("backend_version", "unknown"),
        tags=list(item.get("tags", [])),
    )


def _build_baseline(item: dict[str, Any]) -> BaselineSpec:
    _expect(bool(item.get("name")), "Each baseline entry requires `name`.")
    return BaselineSpec(
        name=item["name"],
        enabled=bool(item.get("enabled", True)),
        scope=item.get("scope", "all"),
        sample_size_obs=item.get("sample_size_obs"),
        sample_size_inters=item.get("sample_size_inters"),
        max_inters=int(item.get("max_inters", -1)),
        seed=item.get("seed"),
        checkpoint_dir=item.get("checkpoint_dir"),
        library=item.get("library"),
        pc_variant=item.get("pc_variant", "stable"),
        pc_ci_test=item.get("pc_ci_test", "chi_square"),
        pc_significance_level=float(item.get("pc_significance_level", 0.01)),
        pc_max_cond_vars=int(item.get("pc_max_cond_vars", 5)),
        ges_scoring_method=item.get("ges_scoring_method", "bic-d"),
        ges_min_improvement=float(item.get("ges_min_improvement", 1e-6)),
        model=item.get("model"),
        provider=item.get("provider", "auto"),
        temperature=float(item.get("temperature", 0.0)),
        max_new_tokens=item.get("max_new_tokens"),
        num_samples=int(item.get("num_samples", 5)),
        edge_threshold=float(item.get("edge_threshold", 0.5)),
        takayama_pattern=int(item.get("takayama_pattern", 2)),
        takayama_bootstrap_samples=int(item.get("takayama_bootstrap_samples", 100)),
        notes=item.get("notes", ""),
    )


def load_benchmark_spec(path: str | Path) -> BenchmarkSpec:
    manifest_path = Path(path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = [_build_dataset(item) for item in _normalize_dataset_items(raw)]
    prompt_cells = [_build_prompt_cell(item) for item in _normalize_prompt_cells(raw)]
    models = [_build_model(item) for item in _normalize_models(raw)]
    baselines = [_build_baseline(item) for item in _normalize_baselines(raw)]
    names_only = NamesOnlySpec(**_normalize_names_only(raw))
    evaluator_raw = raw.get("evaluator", {})
    provenance_raw = raw.get("provenance", {})

    _expect(bool(raw.get("name")), "Manifest requires `name`.")
    _expect(bool(raw.get("role")), "Manifest requires `role`.")
    _expect(bool(datasets), "Manifest must define at least one dataset.")
    _expect(bool(prompt_cells) or names_only.enabled, "Manifest must define prompt cells or enable names-only.")
    _expect(bool(models), "Manifest must define at least one model.")

    spec = BenchmarkSpec(
        name=raw["name"],
        role=raw["role"],
        description=raw.get("description") or raw.get("notes") or "",
        task_family=raw.get("task_family", "llm_causal_discovery"),
        datasets=datasets,
        prompt_cells=prompt_cells,
        names_only=names_only,
        models=models,
        baselines=baselines,
        evaluator=EvaluatorSpec(
            tau=float(evaluator_raw.get("tau", 0.7)),
            answer_col=evaluator_raw.get("answer_col"),
            pred_col=evaluator_raw.get("pred_col", "prediction"),
            parser_version=evaluator_raw.get("parser_version", "query_api.extract_adjacency"),
            evaluator_version=evaluator_raw.get("evaluator_version", "evaluate.py"),
        ),
        provenance=ProvenanceSpec(
            schema_version=provenance_raw.get("schema_version", SCHEMA_VERSION),
            parser_version=provenance_raw.get("parser_version", "query_api.extract_adjacency"),
            evaluator_version=provenance_raw.get("evaluator_version", "evaluate.py"),
            tags=list(provenance_raw.get("tags", [])),
        ),
        execution=ExecutionSpec(
            prompt_storage=raw.get("execution", {}).get("prompt_storage", "disk"),
            prompt_retention=raw.get("execution", {}).get("prompt_retention", "full"),
            example_prompt_dir=raw.get("execution", {}).get("example_prompt_dir"),
        ),
        num_prompts=int(raw.get("num_prompts", 5)),
        shuffles_per_graph=int(raw.get("shuffles_per_graph", 1)),
        seed=int(raw.get("seed", 42)),
        output_root=raw.get("output_root", "benchmark_runs"),
        notes=raw.get("notes", ""),
        benchmark_card=raw.get("benchmark_card"),
    )
    _expect(spec.execution.prompt_storage in {"disk", "in_memory"}, "`execution.prompt_storage` must be `disk` or `in_memory`.")
    _expect(
        spec.execution.prompt_retention in {"full", "example", "none"},
        "`execution.prompt_retention` must be `full`, `example`, or `none`.",
    )
    if spec.execution.prompt_storage == "in_memory" and spec.execution.prompt_retention == "full":
        raise BenchmarkSpecError("In-memory prompting cannot use `prompt_retention=full`; use `example` or `none`.")
    return spec
