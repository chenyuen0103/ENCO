from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .schema import BaselineSpec, BenchmarkSpec, DatasetSpec, EvaluatorSpec, ModelSpec, PromptCellSpec


@dataclass(frozen=True)
class CommandPlan:
    label: str
    cmd: list[str]
    cwd: Path


class DatasetAdapter(ABC):
    @abstractmethod
    def resolve_graph_path(self, dataset: DatasetSpec, spec: BenchmarkSpec, *, dry_run: bool) -> Path:
        raise NotImplementedError


class ModelAdapter(ABC):
    @abstractmethod
    def build_query_command(
        self,
        *,
        prompt_csv: Path,
        model: ModelSpec,
        spec: BenchmarkSpec,
    ) -> CommandPlan:
        raise NotImplementedError


class BaselineAdapter(ABC):
    @abstractmethod
    def applies_to(self, baseline: BaselineSpec, cell: PromptCellSpec) -> bool:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        *,
        baseline: BaselineSpec,
        dataset: DatasetSpec,
        graph_path: Path,
        cell: PromptCellSpec,
        spec: BenchmarkSpec,
        dry_run: bool,
    ) -> Path:
        raise NotImplementedError


class Evaluator(ABC):
    @abstractmethod
    def build_eval_command(self, *, response_csv: Path, evaluator: EvaluatorSpec) -> CommandPlan:
        raise NotImplementedError
