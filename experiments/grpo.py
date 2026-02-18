import os
import re
import argparse
import json
import time
import socket
import gc
import sys
import subprocess
import csv
import io
from contextlib import redirect_stdout, redirect_stderr
import torch
from pathlib import Path
from urllib import request, error
from urllib.parse import urlparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_callback import PrinterCallback
from peft import LoraConfig, get_peft_model
from math_verify import LatexExtractionConfig, parse, verify
from trl import GRPOConfig, GRPOTrainer

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> and </think> tags, and <answer> and </answer> tags, respectively."
)

FORMAT_RE = re.compile(r"(?s)^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$")
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


class CompactMetricsCallback(TrainerCallback):
    """Print a compact metric line instead of full raw log dictionaries."""

    _KEYS = [
        ("epoch", "epoch"),
        ("loss", "loss"),
        ("reward", "reward"),
        ("rewards/accuracy_reward/mean", "acc"),
        ("rewards/format_reward/mean", "fmt"),
        ("entropy", "ent"),
        ("step_time", "step_s"),
    ]

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero or not logs:
            return

        parts = []
        for key, name in self._KEYS:
            if key not in logs:
                continue
            value = logs[key]
            if isinstance(value, float):
                parts.append(f"{name}={value:.4f}")
            else:
                parts.append(f"{name}={value}")

        if parts:
            print("[train] " + " | ".join(parts))


def build_argparser():
    p = argparse.ArgumentParser(description="GRPO training with TRL + vLLM server mode")
    p.add_argument("--mode", type=str, default="train", choices=["train", "ab_eval"])
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    p.add_argument("--dataset_id", type=str, default="AI-MO/NuminaMath-TIR")
    p.add_argument("--train_split", type=str, default="train[:5%]")
    p.add_argument("--test_split", type=str, default="test[:5%]")
    p.add_argument("--output_dir", type=str, default="Qwen2-0.5B-GRPO-vLLM-server")

    # Prompt / generation
    p.add_argument("--max_prompt_tokens", type=int, default=128)
    p.add_argument("--max_completion_length", type=int, default=64)
    p.add_argument("--num_generations", type=int, default=4)

    # Train
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)

    # vLLM server
    p.add_argument(
        "--vllm_server_base_url",
        type=str,
        default=os.environ.get("VLLM_SERVER_BASE_URL", "http://127.0.0.1:8000"),
        help="vLLM server URL. Can also be set via VLLM_SERVER_BASE_URL.",
    )
    p.add_argument("--vllm_server_timeout", type=float, default=240.0)
    p.add_argument("--vllm_group_port", type=int, default=51216)
    p.add_argument("--vllm_preflight_timeout", type=float, default=15.0)
    p.add_argument("--use-vllm", dest="use_vllm", action="store_true")
    p.add_argument("--no-use-vllm", dest="use_vllm", action="store_false")
    p.set_defaults(use_vllm=True)
    p.add_argument(
        "--enable_vllm_preflight",
        action="store_true",
        help="Run an early communicator init/close probe before trainer startup.",
    )
    p.add_argument(
        "--auto-launch-vllm-server",
        action="store_true",
        help="Launch TRL vLLM server automatically on first visible GPU and train on the rest.",
    )
    p.add_argument(
        "--vllm_server_model_id",
        type=str,
        default=None,
        help="Model id/path for auto-launched vLLM server. Defaults to --model_id.",
    )
    p.add_argument("--vllm_server_startup_timeout", type=float, default=120.0)
    p.add_argument("--vllm_server_gpu_memory_utilization", type=float, default=0.2)
    p.add_argument("--vllm_server_max_model_len", type=int, default=2048)
    p.add_argument("--vllm_server_log_file", type=str, default="vllm_server.log")

    # Logging
    p.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "none"])
    p.add_argument("--run_name", type=str, default="qwen2-grpo-vllm-server")
    p.add_argument("--wandb_project", type=str, default="enco-grpo")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--dataloader_num_workers", type=int, default=4)

    # A/B eval (base vs GRPO on frozen set)
    p.add_argument("--ab_base_model", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    p.add_argument("--ab_grpo_model", type=str, default=None)
    p.add_argument("--ab_eval_split", type=str, default="test")
    p.add_argument("--ab_eval_n", type=int, default=200)
    p.add_argument("--ab_eval_seed", type=int, default=1234)
    p.add_argument("--ab_pass_k", type=int, default=1)
    p.add_argument("--ab_max_new_tokens", type=int, default=128)
    p.add_argument("--ab_temperature", type=float, default=0.0)
    p.add_argument("--ab_top_p", type=float, default=0.95)
    p.add_argument("--ab_do_sample", action="store_true")
    p.add_argument("--ab_output_json", type=str, default=None)
    p.add_argument("--ab_debug_csv", type=str, default=None)

    return p


def build_prompt(problem: str) -> str:
    return f"system\n{SYSTEM_PROMPT}\nuser\n{problem}\nassistant\n"


def _completion_to_text(completion):
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


def _extract_answer_text(text: str) -> str:
    m = ANSWER_RE.search(text)
    return m.group(1) if m else text


def _is_correct(answer_text: str, solution: str) -> float:
    gold = parse(
        solution,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    pred = parse(
        answer_text,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )

    if len(gold) == 0:
        return 1.0
    try:
        return float(verify(pred, gold))
    except Exception:
        return 0.0


def _score_answer_with_meta(answer_text: str, solution: str):
    gold = parse(
        solution,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    pred = parse(
        answer_text,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )

    if len(gold) == 0:
        return 1.0, False, False

    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            score = float(verify(pred, gold))
        log_text = buf.getvalue().lower()
        timed_out = "timeout" in log_text
        return score, timed_out, False
    except Exception as exc:
        msg = f"{exc}".lower()
        log_text = buf.getvalue().lower()
        timed_out = ("timeout" in msg) or ("timeout" in log_text)
        return 0.0, timed_out, True


def _load_eval_model(model_id_or_path: str):
    is_adapter = (Path(model_id_or_path) / "adapter_config.json").exists()
    if is_adapter:
        from peft import AutoPeftModelForCausalLM

        return AutoPeftModelForCausalLM.from_pretrained(
            model_id_or_path,
            dtype="auto",
            device_map="auto",
        ).eval()

    return AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        dtype="auto",
        device_map="auto",
    ).eval()


def _eval_on_dataset(
    model_id_or_path: str,
    eval_dataset,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    pass_k: int,
    debug_model_label: str = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_eval_model(model_id_or_path)
    input_device = next(model.parameters()).device

    fmt_ok = 0
    acc_sum = 0.0
    pass_k_ok = 0.0
    verify_timeouts = 0
    verify_errors = 0
    debug_rows = []

    for idx, example in enumerate(eval_dataset):
        prompt = build_prompt(example["problem"])
        inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
        prompt_len = inputs["input_ids"].shape[1]

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": pass_k,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.no_grad():
            out = model.generate(
                **inputs,
                **generation_kwargs,
            )

        completions = [
            tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            for seq in out
        ]
        first_completion = completions[0]
        fmt_ok += int(bool(FORMAT_RE.match(first_completion)))

        sample_scores = []
        for sample_idx, completion_text in enumerate(completions):
            answer_text = _extract_answer_text(completion_text)
            score, timed_out, had_error = _score_answer_with_meta(answer_text, example["solution"])
            sample_scores.append(score)
            verify_timeouts += int(timed_out)
            verify_errors += int(had_error)

            if debug_model_label is not None:
                debug_rows.append(
                    {
                        "model": debug_model_label,
                        "example_idx": idx,
                        "sample_idx": sample_idx,
                        "problem": example["problem"],
                        "solution": example["solution"],
                        "completion": completion_text,
                        "answer_text": answer_text,
                        "format_ok": int(bool(FORMAT_RE.match(completion_text))),
                        "is_correct": score,
                        "verify_timed_out": int(timed_out),
                        "verify_error": int(had_error),
                    }
                )

        acc_sum += sample_scores[0]
        pass_k_ok += float(max(sample_scores))

    n = len(eval_dataset)
    metrics = {
        "model": model_id_or_path,
        "n": n,
        "format_rate": fmt_ok / n,
        "accuracy": acc_sum / n,
        f"pass@{pass_k}": pass_k_ok / n,
        "verify_timeouts": verify_timeouts,
        "verify_errors": verify_errors,
    }
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, debug_rows


def run_ab_eval(args):
    if not args.ab_grpo_model:
        raise ValueError("--ab_grpo_model is required when --mode ab_eval")
    if args.ab_eval_n <= 0:
        raise ValueError("--ab_eval_n must be > 0")
    if args.ab_pass_k <= 0:
        raise ValueError("--ab_pass_k must be > 0")

    full_eval = load_dataset(args.dataset_id, split=args.ab_eval_split)
    n = min(args.ab_eval_n, len(full_eval))
    frozen_eval = full_eval.shuffle(seed=args.ab_eval_seed).select(range(n))

    do_sample = args.ab_do_sample or args.ab_temperature > 0 or args.ab_pass_k > 1
    if args.ab_pass_k > 1 and not do_sample:
        raise ValueError("pass@k with k>1 requires sampling; set --ab_do_sample or --ab_temperature > 0")

    base_metrics, base_debug_rows = _eval_on_dataset(
        model_id_or_path=args.ab_base_model,
        eval_dataset=frozen_eval,
        max_new_tokens=args.ab_max_new_tokens,
        temperature=args.ab_temperature,
        top_p=args.ab_top_p,
        do_sample=do_sample,
        pass_k=args.ab_pass_k,
        debug_model_label="base" if args.ab_debug_csv else None,
    )
    grpo_metrics, grpo_debug_rows = _eval_on_dataset(
        model_id_or_path=args.ab_grpo_model,
        eval_dataset=frozen_eval,
        max_new_tokens=args.ab_max_new_tokens,
        temperature=args.ab_temperature,
        top_p=args.ab_top_p,
        do_sample=do_sample,
        pass_k=args.ab_pass_k,
        debug_model_label="grpo" if args.ab_debug_csv else None,
    )

    pass_k_key = f"pass@{args.ab_pass_k}"
    result = {
        "dataset_id": args.dataset_id,
        "split": args.ab_eval_split,
        "seed": args.ab_eval_seed,
        "n": n,
        "generation": {
            "max_new_tokens": args.ab_max_new_tokens,
            "do_sample": do_sample,
            "temperature": args.ab_temperature if do_sample else 0.0,
            "top_p": args.ab_top_p if do_sample else 1.0,
            "pass_k": args.ab_pass_k,
        },
        "base": base_metrics,
        "grpo": grpo_metrics,
        "delta_grpo_minus_base": {
            "format_rate": grpo_metrics["format_rate"] - base_metrics["format_rate"],
            "accuracy": grpo_metrics["accuracy"] - base_metrics["accuracy"],
            pass_k_key: grpo_metrics[pass_k_key] - base_metrics[pass_k_key],
        },
    }

    print(json.dumps(result, indent=2))
    if args.ab_output_json:
        with open(args.ab_output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print("Saved eval JSON to:", args.ab_output_json)
    if args.ab_debug_csv:
        all_rows = base_debug_rows + grpo_debug_rows
        fieldnames = [
            "model",
            "example_idx",
            "sample_idx",
            "problem",
            "solution",
            "completion",
            "answer_text",
            "format_ok",
            "is_correct",
            "verify_timed_out",
            "verify_error",
        ]
        with open(args.ab_debug_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print("Saved debug CSV to:", args.ab_debug_csv)


def _parse_visible_cuda_devices():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip():
        return [d.strip() for d in cvd.split(",") if d.strip()]
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []


def _wait_for_health(url: str, timeout_s: float):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with request.urlopen(url, timeout=2.0) as resp:
                if resp.status < 400:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _build_trl_vllm_serve_cmd(args, host: str, port: int):
    model_id = args.vllm_server_model_id or args.model_id
    return [
        "trl",
        "vllm-serve",
        "--model",
        model_id,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(args.vllm_server_gpu_memory_utilization),
        "--max-model-len",
        str(args.vllm_server_max_model_len),
    ]


def _maybe_launch_local_vllm_and_reexec_train(args):
    if args.mode != "train" or not args.use_vllm or not args.auto_launch_vllm_server:
        return
    if os.environ.get("GRPO_TRAIN_CHILD") == "1":
        return
    if os.environ.get("LOCAL_RANK") is not None:
        raise RuntimeError(
            "--auto-launch-vllm-server should be run with plain python (not inside accelerate launch). "
            "The script will launch accelerate for the training child process automatically."
        )

    visible_gpus = _parse_visible_cuda_devices()
    if len(visible_gpus) < 2:
        raise RuntimeError(
            "Auto vLLM launch needs at least 2 visible GPUs: one for server and one or more for training. "
            f"Visible GPUs: {visible_gpus}"
        )

    server_gpu = visible_gpus[0]
    train_gpus = visible_gpus[1:]
    parsed = urlparse(args.vllm_server_base_url.rstrip("/"))
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    health_url = f"http://{host}:{port}/health"

    server_env = os.environ.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = server_gpu
    log_file = open(args.vllm_server_log_file, "w", encoding="utf-8")

    server_cmd = _build_trl_vllm_serve_cmd(args, host=host, port=port)
    try:
        server_proc = subprocess.Popen(
            server_cmd,
            env=server_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        log_file.close()
        raise RuntimeError(
            "Failed to start auto vLLM server: `trl` command not found in this environment. "
            "Install TRL CLI or start the server manually and omit --auto-launch-vllm-server."
        ) from exc

    try:
        ready = _wait_for_health(health_url, args.vllm_server_startup_timeout)
        if not ready:
            raise RuntimeError(
                "Auto-launched vLLM server did not become healthy in time. "
                f"Check logs: {args.vllm_server_log_file}"
            )

        child_env = os.environ.copy()
        child_env["GRPO_TRAIN_CHILD"] = "1"
        child_env["CUDA_VISIBLE_DEVICES"] = ",".join(train_gpus)
        if len(train_gpus) > 1:
            child_cmd = [
                "accelerate",
                "launch",
                "--num_processes",
                str(len(train_gpus)),
                str(Path(__file__).resolve()),
                *sys.argv[1:],
            ]
        else:
            child_cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]

        print(
            f"[orchestrator] vLLM server on GPU {server_gpu}; training child on GPUs {train_gpus}; "
            f"server url={args.vllm_server_base_url}"
        )
        try:
            rc = subprocess.call(child_cmd, env=child_env)
        except FileNotFoundError as exc:
            if child_cmd and child_cmd[0] == "accelerate":
                raise RuntimeError(
                    "Failed to start training child: `accelerate` command not found. "
                    "Install accelerate or run with only one training GPU."
                ) from exc
            raise
        raise SystemExit(rc)
    finally:
        try:
            server_proc.terminate()
            server_proc.wait(timeout=10)
        except Exception:
            try:
                server_proc.kill()
            except Exception:
                pass
        log_file.close()


def run_train(args):
    args.vllm_server_base_url = args.vllm_server_base_url.rstrip("/")

    # Throughput-oriented defaults for Ampere/Hopper GPUs.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if args.use_vllm:
        # Fail fast if the configured vLLM endpoint is unreachable.
        health_url = f"{args.vllm_server_base_url}/health"
        try:
            with request.urlopen(health_url, timeout=10) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status} from {health_url}")
        except (error.URLError, RuntimeError) as e:
            raise RuntimeError(
                f"Could not reach vLLM server at {health_url}. "
                "Start the server first or set --vllm_server_base_url / VLLM_SERVER_BASE_URL."
            ) from e

    def _json_get(url, timeout=5.0):
        with request.urlopen(url, timeout=timeout) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status} from {url}")
            return json.loads(resp.read().decode("utf-8"))

    def _json_post(url, payload=None, timeout=5.0):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        with request.urlopen(req, timeout=timeout) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status} from {url}")
            body = resp.read()
            return json.loads(body.decode("utf-8")) if body else {}

    def _wait_tcp_ready(host, port, total_timeout):
        deadline = time.time() + total_timeout
        while time.time() < deadline:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return True
            except OSError:
                time.sleep(0.2)
        return False

    def _preflight_vllm_communicator():
        # Only the local rank 0 process probes the communicator to avoid races.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank != 0:
            return

        parsed = urlparse(args.vllm_server_base_url)
        host = parsed.hostname or "127.0.0.1"

        try:
            world = _json_get(f"{args.vllm_server_base_url}/get_world_size/", timeout=5.0)
            vllm_world_size = int(world["world_size"])
        except Exception as e:
            raise RuntimeError(
                "vLLM server does not expose /get_world_size/. "
                "Make sure the server was started with TRL's `trl vllm-serve`."
            ) from e

        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            client_device_uuid = str(torch.cuda.get_device_properties(device_idx).uuid)
        else:
            client_device_uuid = "cpu"

        init_url = f"{args.vllm_server_base_url}/init_communicator/"
        close_url = f"{args.vllm_server_base_url}/close_communicator/"
        try:
            _json_post(
                init_url,
                payload={
                    "host": "0.0.0.0",
                    "port": args.vllm_group_port,
                    "world_size": vllm_world_size + 1,
                    "client_device_uuid": client_device_uuid,
                },
                timeout=5.0,
            )
            ready = _wait_tcp_ready(host, args.vllm_group_port, args.vllm_preflight_timeout)
            if not ready:
                raise RuntimeError(
                    f"Timed out waiting for communicator TCPStore at {host}:{args.vllm_group_port}. "
                    "The vLLM server did not bring up the weight-sync group."
                )
        finally:
            try:
                _json_post(close_url, payload=None, timeout=3.0)
            except Exception:
                pass

    if args.use_vllm and args.enable_vllm_preflight:
        _preflight_vllm_communicator()

    # ---- W&B import + TRL profiling workaround (prevents NameError you saw)
    if args.report_to == "wandb":
        # Configure W&B from CLI flags for reproducible logging behavior.
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        os.environ["WANDB_MODE"] = args.wandb_mode
        import wandb  # noqa: F401
        import trl.extras.profiling as trl_profiling
        trl_profiling.wandb = __import__("wandb")

    # ---- Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Dataset
    train_dataset, test_dataset = load_dataset(
        args.dataset_id,
        split=[args.train_split, args.test_split],
    )

    def add_prompt(example, max_prompt_tokens=args.max_prompt_tokens):
        text = build_prompt(example["problem"])
        ids = tokenizer(text, truncation=True, max_length=max_prompt_tokens)["input_ids"]
        return {"prompt": tokenizer.decode(ids, skip_special_tokens=True)}

    train_dataset = train_dataset.map(add_prompt)
    test_dataset = test_dataset.map(add_prompt)

    # Keep only what GRPO + reward needs
    keep_cols = ["prompt", "solution"]
    train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in keep_cols])
    test_dataset = test_dataset.remove_columns([c for c in test_dataset.column_names if c not in keep_cols])

    # ---- Model + LoRA
    distributed_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    model_load_kwargs = {
        "dtype": "auto",
    }
    # `device_map="auto"` is incompatible with distributed Accelerate launches.
    if distributed_world_size == 1:
        model_load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        **model_load_kwargs,
    )

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)

    # ---- Reward functions
    def format_reward(completions, **kwargs):
        texts = [_completion_to_text(c) for c in completions]
        return [1.0 if FORMAT_RE.match(t) else 0.0 for t in texts]


    def accuracy_reward(completions, **kwargs):
        solutions = kwargs["solution"]
        rewards = []
        for completion, solution in zip(completions, solutions):
            text = _completion_to_text(completion)
            answer_text = _extract_answer_text(text)
            rewards.append(_is_correct(answer_text, solution))
        return rewards

    # ---- GRPO config (TRL 0.28.0)
    report_to = ["wandb"] if args.report_to == "wandb" else "none"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    generation_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    )
    if generation_batch_size % args.num_generations != 0:
        valid_num_generations = [
            str(n)
            for n in range(1, generation_batch_size + 1)
            if generation_batch_size % n == 0
        ]
        raise ValueError(
            "Invalid GRPO settings: "
            f"generation_batch_size={generation_batch_size} "
            "(per_device_train_batch_size * gradient_accumulation_steps * WORLD_SIZE) "
            f"must be divisible by num_generations={args.num_generations}. "
            f"For this launch, choose --num_generations from {{{', '.join(valid_num_generations)}}} "
            "or adjust batch/accum/world size."
        )

    grpo_kwargs = {}
    if args.use_vllm:
        grpo_kwargs.update(
            {
                "vllm_mode": "server",
                "vllm_server_base_url": args.vllm_server_base_url,
                "vllm_server_timeout": args.vllm_server_timeout,
                "vllm_group_port": args.vllm_group_port,
            }
        )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        remove_unused_columns=False,
        num_train_epochs=args.num_train_epochs,

        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        bf16=torch.cuda.is_available(),

        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,

        report_to=report_to,
        run_name=args.run_name,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,

        save_strategy="steps",
        save_steps=args.save_steps,
        push_to_hub=False,

        # vLLM integration (server mode)
        use_vllm=args.use_vllm,
        **grpo_kwargs,
    )

    try:
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[format_reward, accuracy_reward],
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[CompactMetricsCallback()],
        )
        trainer.remove_callback(PrinterCallback)
    except RuntimeError as e:
        msg = str(e)
        if "NCCL error: remote process exited or there was a network error" in msg:
            raise RuntimeError(
                "Failed to initialize TRL<->vLLM NCCL communicator. "
                "Most common cause is overlapping GPUs between trainer and vLLM server. "
                "Use disjoint devices (for example, server on GPU 0 and trainer on GPUs 1,2,3), "
                "and ensure --vllm_group_port is free and consistent for this run."
            ) from e
        raise

    trainer.train()
    trainer.save_model(args.output_dir)
    print("Saved to:", args.output_dir)


def main():
    args = build_argparser().parse_args()
    _maybe_launch_local_vllm_and_reexec_train(args)
    if args.mode == "ab_eval":
        run_ab_eval(args)
        return
    run_train(args)


if __name__ == "__main__":
    main()
