#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="HF-only wrapper around query_gemini.py (provider=hf)."
    )
    ap.add_argument("--csv", required=True, help="Input prompt CSV path.")
    ap.add_argument("--model", required=True, help="HF model id/path.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--hf-device-map", default="auto")
    ap.add_argument("--hf-dtype", default="auto")
    ap.add_argument("--hf-batch-size", type=int, default=1)
    ap.add_argument(
        "--hf-trust-remote-code",
        dest="hf_trust_remote_code",
        action="store_true",
        help="Allow loading custom model code (recommended for Qwen).",
    )
    ap.add_argument(
        "--no-hf-trust-remote-code",
        dest="hf_trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code.",
    )
    ap.set_defaults(hf_trust_remote_code=True)

    ap.add_argument("--prompt-col", default="prompt_path")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-csv", default=None)

    args = ap.parse_args()

    query_script = (Path(__file__).parent / "query_gemini.py").resolve()
    if not query_script.exists():
        raise SystemExit(f"query script not found: {query_script}")

    cmd = [
        sys.executable,
        str(query_script),
        "--csv",
        str(args.csv),
        "--model",
        args.model,
        "--provider",
        "hf",
        "--temperature",
        str(args.temperature),
        "--prompt-col",
        args.prompt_col,
        "--hf-device-map",
        args.hf_device_map,
        "--hf-dtype",
        args.hf_dtype,
        "--hf-batch-size",
        str(args.hf_batch_size),
    ]

    if args.max_new_tokens is not None:
        cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if args.hf_trust_remote_code:
        cmd.append("--hf-trust-remote-code")
    else:
        cmd.append("--no-hf-trust-remote-code")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.max_rows is not None:
        cmd.extend(["--max-rows", str(args.max_rows)])
    if args.dry_run:
        cmd.append("--dry-run")
    if args.out_csv is not None:
        cmd.extend(["--out-csv", args.out_csv])

    print("[running]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
