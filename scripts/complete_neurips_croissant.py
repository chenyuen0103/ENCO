#!/usr/bin/env python3
"""Complete Hugging Face Croissant metadata for the NeurIPS data submission.

Hugging Face can generate Croissant metadata for the small/synthetic configs,
but the large CSV files are too large and have a slightly mixed schema. This
script preserves the generated metadata and appends direct CSV metadata for the
large files using the columns common to all large CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASET_URL = "https://huggingface.co/datasets/mixcausalbench/anonymous-data"
DATASET_GIT_URL = f"{DATASET_URL}.git"
REPO_SHA256_PLACEHOLDER = "https://github.com/mlcommons/croissant/issues/80"

TYPE_BY_COLUMN = {
    "anonymize": "cr:Int64",
    "append_format_hint": "cr:Int64",
    "config_index": "cr:Int64",
    "data_idx": "cr:Int64",
    "hist_mass_keep_frac": "cr:Float64",
    "int_per_combo": "cr:Int64",
    "obs_per_prompt": "cr:Int64",
    "prompt_tokens_ref": "cr:Int64",
    "shuffle_idx": "cr:Int64",
    "shuffles_per_graph": "cr:Int64",
}


def csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def common_large_columns(large_dir: Path) -> list[str]:
    csv_paths = sorted(large_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {large_dir}")

    headers = [csv_header(path) for path in csv_paths]
    common = set(headers[0])
    for header in headers[1:]:
        common &= set(header)

    return [column for column in headers[0] if column in common]


def upsert_by_id(items: list[dict], new_item: dict) -> None:
    new_id = new_item.get("@id")
    for index, item in enumerate(items):
        if item.get("@id") == new_id:
            items[index] = new_item
            return
    items.append(new_item)


def add_context(metadata: dict) -> None:
    context = metadata.setdefault("@context", {})
    context.setdefault("rai", "http://mlcommons.org/croissant/RAI/")
    context.setdefault("prov", "http://www.w3.org/ns/prov#")
    context.setdefault("value", "cr:value")


def add_rai_metadata(metadata: dict) -> None:
    metadata.setdefault("citeAs", "Anonymous Data. Hugging Face dataset repository.")
    metadata.setdefault("datePublished", "2026-05-03")
    metadata["license"] = "https://opensource.org/license/bsd-2-clause"
    metadata.setdefault("version", "1.0.0")
    metadata["rai:dataCollection"] = (
        "The dataset was generated programmatically from Bayesian-network graph "
        "assets from the bnlearn Bayesian Network Repository, synthetic graph "
        "specifications, prompt templates, fixed random seeds, and "
        "observational/interventional data-generation utilities adapted from the "
        "ENCO paper/codebase. No new human-subject data collection was performed. "
        "The collection method is simulation/programmatic benchmark generation "
        "from existing graph assets and synthetic graph seeds. The hosted records "
        "contain generated benchmark prompts, task metadata, reference answers, "
        "and hashes used to identify prompt and answer content."
    )
    metadata["rai:dataCollectionType"] = "Programmatic benchmark generation."
    metadata["rai:dataBiases"] = (
        "Coverage is determined by the selected causal graph families, graph "
        "sizes, benchmark configurations, prompt styles, and generation seeds. "
        "The benchmark intentionally over-represents controlled causal graph "
        "settings with known reference answers and under-represents messy, "
        "partially observed, cyclic, confounded, non-tabular, multilingual, "
        "domain-specific, and culturally situated real-world causal-discovery "
        "settings. There are no demographic population groups represented as "
        "human subjects, but model behavior on this dataset may reflect "
        "adaptation to the included graph families, English-language prompt "
        "templates, and formatting conventions rather than general causal "
        "reasoning ability."
    )
    metadata["rai:dataLimitations"] = (
        "The dataset is intended for evaluation-only benchmarking of language "
        "models on causal-discovery prompts. It should not be interpreted as a "
        "representative sample of real-world decision-making data, and scores on "
        "this benchmark should be complemented by other analyses. It is not "
        "recommended for model training or fine-tuning, validating models for "
        "medical, financial, legal, policy, or other high-stakes decisions, "
        "estimating causal effects in real populations, or making claims about "
        "causal relationships outside the represented benchmark graph families "
        "and prompt formats. Validity is limited to the hosted English-language "
        "prompt format, graph families, generation settings, and reference-answer "
        "metrics described in the accompanying anonymized submission."
    )
    data_use_cases = (
        "The dataset is intended to measure language-model performance on "
        "causal-discovery benchmark tasks: given text prompts containing graph, "
        "observational, and/or interventional information, a model should infer "
        "the target causal graph structure and return the requested edge-level "
        "answer. Validated use cases are benchmark evaluation and comparison of "
        "models or prompt configurations under the hosted small, large, and "
        "synthetic test splits, using the reference answers and the evaluation "
        "metrics reported in the accompanying anonymized submission. The dataset "
        "has not been validated for training production models, deployment in "
        "medical, policy, financial, or other high-stakes decision workflows, "
        "estimating causal effects in real populations, or assessing general "
        "scientific causal reasoning outside the represented benchmark graph "
        "families and prompt formats. The synthetic subset is programmatically "
        "generated to test controlled graph structures; its utility is for "
        "benchmark evaluation rather than as evidence that the synthetic data "
        "matches any particular real-world population distribution."
    )
    metadata["rai:dataUseCases"] = data_use_cases
    metadata["rai:useCases"] = data_use_cases
    metadata["rai:personalSensitiveInformation"] = (
        "The dataset does not intentionally contain personal or sensitive "
        "personal information. None of the listed sensitive categories are "
        "intentionally present: gender, socio-economic status, geography, "
        "language as a personal attribute, age, culture, experience or seniority, "
        "health or medical data about individuals, or political or religious "
        "beliefs. The dataset contains generated benchmark prompts, task "
        "metadata, reference answers, and graph-derived data; no names, contact "
        "details, national identifiers, human-subject records, or direct personal "
        "identifiers are intentionally included."
    )
    metadata["rai:dataSocialImpact"] = (
        "The dataset may support more reproducible and transparent evaluation of "
        "causal-discovery behavior in language models, benefiting researchers who "
        "need controlled benchmark prompts, reference answers, and public hosted "
        "data. Potential negative impacts include over-reliance on benchmark "
        "scores as a proxy for real-world causal reasoning, optimizing models "
        "only for this benchmark, or using benchmark performance to justify "
        "deployment in high-stakes settings without external validation. Because "
        "the dataset is open, no access-control mitigation is applied; mitigations "
        "instead include documenting the intended evaluation-only use, explicitly "
        "listing out-of-scope uses, releasing reference answers and metadata for "
        "reproducibility, and flagging that the synthetic subset is not a "
        "validated simulation of any particular real-world population."
    )
    metadata["rai:hasSyntheticData"] = True
    metadata["prov:wasGeneratedBy"] = [
        {
            "@type": "prov:Activity",
            "prov:type": {"@id": "https://www.wikidata.org/wiki/Q4929239"},
            "prov:label": "Collection",
            "sc:description": (
                "No new human-subject data collection was performed. The dataset "
                "was constructed programmatically from Bayesian-network graph "
                "assets from the bnlearn Bayesian Network Repository, synthetic "
                "graph specifications, prompt templates, fixed benchmark "
                "configurations, and observational/interventional data-generation "
                "utilities adapted from the ENCO paper/codebase. The hosted "
                "records contain generated benchmark prompts, task metadata, "
                "reference answers, and content hashes. The collection method was "
                "simulation/programmatic generation rather than surveys, "
                "interviews, observations, web scraping, or crowdsourcing."
            ),
            "prov:wasAttributedTo": [
                {
                    "@type": "prov:SoftwareAgent",
                    "@id": "benchmark_generation_pipeline",
                    "prov:label": "Automated benchmark-generation pipeline",
                    "prov:description": (
                        "Repository scripts used to instantiate benchmark prompts "
                        "and reference answers from graph assets, synthetic graph "
                        "specifications, prompt templates, and fixed configuration "
                        "files."
                    ),
                },
                {
                    "@type": "prov:Agent",
                    "@id": "anonymous_research_team",
                    "prov:label": "Anonymous research team",
                    "prov:description": (
                        "The anonymized submission authors configured and ran the "
                        "automated benchmark-generation pipeline. No external "
                        "crowdworkers or human-subject participants were used."
                    ),
                },
            ],
        },
        {
            "@type": "prov:Activity",
            "prov:type": {"@id": "https://www.wikidata.org/wiki/Q1172378"},
            "prov:label": "Preprocessing and packaging",
            "sc:description": (
                "Benchmark prompts and answers were generated into CSV files, "
                "grouped into small, large, and synthetic test configurations, "
                "and uploaded to a public Hugging Face dataset repository. The "
                "small and synthetic configurations were converted by Hugging "
                "Face into viewer-backed Parquet metadata. The large "
                "configuration is provided as direct CSV files because the files "
                "are too large for automatic Hugging Face viewer conversion, and "
                "is therefore described manually in the Croissant metadata."
            ),
            "prov:wasAttributedTo": [
                {
                    "@type": "prov:SoftwareAgent",
                    "@id": "huggingface_hub",
                    "prov:label": "Hugging Face Hub and dataset viewer",
                    "prov:description": (
                        "Hosting platform used for public dataset distribution and "
                        "automatic viewer-backed Parquet/Croissant conversion for "
                        "the small and synthetic configs."
                    ),
                },
                {
                    "@type": "prov:SoftwareAgent",
                    "@id": "croissant_completion_script",
                    "prov:label": "Croissant completion script",
                    "prov:description": (
                        "Local metadata script used to add direct-download large "
                        "CSV coverage and Responsible AI fields to the generated "
                        "Croissant metadata."
                    ),
                },
            ],
        },
        {
            "@type": "prov:Activity",
            "prov:type": {"@id": "https://www.wikidata.org/wiki/Q109719325"},
            "prov:label": "Annotation",
            "sc:description": (
                "No crowdsourcing, paid human annotation, or external annotation "
                "platform was used for this hosted dataset. Reference answers "
                "were generated from the known benchmark graph structures and "
                "generation pipeline rather than collected from human annotators "
                "or LLM annotators. No inter-annotator agreement score is "
                "applicable because labels are programmatic reference answers."
            ),
            "prov:wasAttributedTo": [
                {
                    "@type": "prov:SoftwareAgent",
                    "@id": "reference_answer_generator",
                    "prov:label": "Programmatic reference-answer generator",
                    "prov:description": (
                        "Automated code path that derives reference answers from "
                        "known benchmark graph structures and task specifications."
                    ),
                }
            ],
        },
        {
            "@type": "prov:Activity",
            "prov:type": {"@id": "https://www.wikidata.org/wiki/Q3306762"},
            "prov:label": "Quality review",
            "sc:description": (
                "The hosted files were checked for public accessibility, split "
                "naming, file layout, and Croissant metadata coverage. The "
                "Croissant metadata was manually completed to include the large "
                "direct-download CSV files and the required Responsible AI fields "
                "for the NeurIPS submission."
            ),
            "prov:wasAttributedTo": [
                {
                    "@type": "prov:Agent",
                    "@id": "anonymous_research_team",
                    "prov:label": "Anonymous research team",
                    "prov:description": (
                        "The anonymized submission authors reviewed public access, "
                        "split naming, file layout, and metadata completeness."
                    ),
                }
            ],
        },
    ]
    metadata["prov:wasDerivedFrom"] = [
        {
            "@type": "prov:Entity",
            "@id": "https://www.bnlearn.com/bnrepository/",
            "prov:label": "bnlearn Bayesian Network Repository",
            "description": (
                "Bayesian-network benchmark graph structures used as source "
                "graph assets for benchmark prompt generation."
            ),
            "sc:license": "https://creativecommons.org/licenses/by-sa/4.0/",
            "prov:wasAttributedTo": {
                "@type": "prov:Agent",
                "prov:label": "bnlearn / Marco Scutari",
                "prov:description": (
                    "Source repository for benchmark Bayesian-network graph "
                    "assets."
                ),
            },
        },
        {
            "@type": "prov:Entity",
            "@id": "https://openreview.net/forum?id=eYciPrLuUhG",
            "prov:label": "ENCO paper/codebase data-generation utilities",
            "description": (
                "Observational and interventional data-generation and graph "
                "utility code adapted from ENCO: Efficient Neural Causal "
                "Discovery without Acyclicity Constraints."
            ),
            "sc:license": "https://opensource.org/license/bsd-2-clause",
        },
        {
            "@type": "prov:Entity",
            "@id": f"{DATASET_URL}/tree/main/graphs/small",
            "prov:label": "Small causal graph benchmark source files",
            "description": (
                "Hosted small benchmark graph CSV and manifest files used to "
                "instantiate small causal-discovery evaluation prompts."
            ),
            "sc:license": "https://opensource.org/license/bsd-2-clause",
        },
        {
            "@type": "prov:Entity",
            "@id": f"{DATASET_URL}/tree/main/graphs/large",
            "prov:label": "Large causal graph benchmark source files",
            "description": (
                "Hosted large benchmark graph CSV and manifest files used to "
                "instantiate large causal-discovery evaluation prompts."
            ),
            "sc:license": "https://opensource.org/license/bsd-2-clause",
        },
        {
            "@type": "prov:Entity",
            "@id": f"{DATASET_URL}/tree/main/graphs/synthetic",
            "prov:label": "Synthetic graph specifications and generated source files",
            "description": (
                "Programmatically generated synthetic graph families and fixed "
                "seeds used for the synthetic evaluation config."
            ),
            "sc:license": "https://opensource.org/license/bsd-2-clause",
        },
    ]


def add_large_distribution(metadata: dict) -> None:
    distribution = metadata.setdefault("distribution", [])
    upsert_by_id(
        distribution,
        {
            "@type": "cr:FileObject",
            "@id": "hf-main-repo",
            "name": "hf-main-repo",
            "description": "The public Hugging Face dataset repository main branch.",
            "contentUrl": DATASET_GIT_URL,
            "encodingFormat": "git+https",
            "sha256": REPO_SHA256_PLACEHOLDER,
        },
    )
    upsert_by_id(
        distribution,
        {
            "@type": "cr:FileSet",
            "@id": "csv-files-for-config-large",
            "name": "large CSV files",
            "description": (
                "Direct CSV files for the large evaluation config. These files "
                "are described manually because they are too large for the "
                "Hugging Face dataset viewer conversion."
            ),
            "containedIn": {"@id": "hf-main-repo"},
            "encodingFormat": "text/csv",
            "includes": "graphs/large/*.csv",
        },
    )


def add_large_recordsets(metadata: dict, columns: list[str]) -> None:
    record_sets = metadata.setdefault("recordSet", [])
    upsert_by_id(
        record_sets,
        {
            "@type": "cr:RecordSet",
            "dataType": "cr:Split",
            "key": {"@id": "large_splits/split_name"},
            "@id": "large_splits",
            "name": "large_splits",
            "description": "Splits for the large config.",
            "field": [
                {
                    "@type": "cr:Field",
                    "@id": "large_splits/split_name",
                    "dataType": "sc:Text",
                }
            ],
            "data": [{"large_splits/split_name": "test"}],
        },
    )

    fields = [
        {
            "@type": "cr:Field",
            "@id": "large/split",
            "dataType": "sc:Text",
            "value": "test",
            "references": {"field": {"@id": "large_splits/split_name"}},
        }
    ]
    for column in columns:
        fields.append(
            {
                "@type": "cr:Field",
                "@id": f"large/{column}",
                "dataType": TYPE_BY_COLUMN.get(column, "sc:Text"),
                "source": {
                    "fileSet": {"@id": "csv-files-for-config-large"},
                    "extract": {"column": column},
                },
            }
        )

    upsert_by_id(
        record_sets,
        {
            "@type": "cr:RecordSet",
            "@id": "large",
            "description": (
                "mixcausalbench/anonymous-data - 'large' subset, described as "
                "direct CSV files because the files are too large for automatic "
                "Hugging Face viewer conversion."
            ),
            "field": fields,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="croissant.json", type=Path)
    parser.add_argument("--output", default="croissant_neurips.json", type=Path)
    parser.add_argument(
        "--large-dir", default=Path("benchmark_data/graphs/large"), type=Path
    )
    args = parser.parse_args()

    metadata = json.loads(args.input.read_text(encoding="utf-8"))
    columns = common_large_columns(args.large_dir)

    add_context(metadata)
    add_rai_metadata(metadata)
    add_large_distribution(metadata)
    add_large_recordsets(metadata, columns)

    args.output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} with {len(columns)} common large columns.")


if __name__ == "__main__":
    main()
