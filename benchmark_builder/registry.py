from __future__ import annotations

from pathlib import Path

from .schema import BenchmarkSpec, load_benchmark_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIRS = (
    REPO_ROOT / "benchmark_specs",
    REPO_ROOT / "paper_slices",
)


class BenchmarkRegistry:
    def __init__(self, roots: tuple[Path, ...] = DEFAULT_CONFIG_DIRS) -> None:
        self.roots = roots

    def list_configs(self) -> list[Path]:
        out: list[Path] = []
        for root in self.roots:
            if root.exists():
                out.extend(sorted(root.rglob("*.json")))
        return out

    def resolve(self, identifier: str) -> Path:
        candidate = Path(identifier)
        if candidate.exists():
            return candidate.resolve()
        if candidate.is_absolute():
            raise FileNotFoundError(f"Benchmark config not found: {identifier}")

        matches: list[Path] = []
        for path in self.list_configs():
            stem = path.stem
            rel = path.relative_to(REPO_ROOT)
            if identifier in {stem, str(rel), rel.as_posix()}:
                matches.append(path)
        if not matches:
            raise FileNotFoundError(f"Benchmark config not found: {identifier}")
        if len(matches) > 1:
            options = ", ".join(str(p.relative_to(REPO_ROOT)) for p in matches)
            raise FileNotFoundError(f"Benchmark identifier is ambiguous: {identifier} -> {options}")
        return matches[0].resolve()

    def load(self, identifier: str) -> BenchmarkSpec:
        return load_benchmark_spec(self.resolve(identifier))
