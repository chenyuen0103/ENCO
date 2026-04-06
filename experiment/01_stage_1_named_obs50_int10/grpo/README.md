---
base_model: Qwen/Qwen3-4B-Thinking-2507
library_name: peft
model_name: grpo
tags:
- base_model:adapter:Qwen/Qwen3-4B-Thinking-2507
- grpo
- lora
- transformers
- trl
licence: license
pipeline_tag: text-generation
---

# Model Card for grpo

This model is a fine-tuned version of [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with GRPO, a method introduced in [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

### Framework versions

- PEFT 0.18.1
- TRL: 0.24.0
- Transformers: 4.57.1
- Pytorch: 2.8.0
- Datasets: 4.3.0
- Tokenizers: 0.22.2

## Citations

Cite GRPO as:

```bibtex
@article{shao2024deepseekmath,
    title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
    author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
    year         = 2024,
    eprint       = {arXiv:2402.03300},
}

```

Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```

## Training Config

Saved automatically from the GRPO training launch.

### Command

```bash
/u/chenyuen0103/anaconda3/envs/enco/bin/python --mode train --task cd_descendants --model_id Qwen/Qwen3-4B-Thinking-2507 --cd-train-csv /u/chenyuen0103/ENCO/experiment/01_stage_1_named_obs50_int10/train_mixed.csv --output_dir /u/chenyuen0103/ENCO/experiment/01_stage_1_named_obs50_int10/grpo --cd-test-csv /u/chenyuen0103/ENCO/experiments/prompts/cd_descendants/sachs/splits/stage_1_named_obs50_int10_eval.csv --no-use-vllm --max_completion_length 128 --num_generations 2 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --learning_rate 5e-6 --cd-format-reward-scale 0.2 --cd-partial-format-reward-scale 0.15 --cd-graph-reward-scale 1.0 --length_penalty_coef 0.0001 --length_penalty_max_abs 0 --save_steps 20 --logging_steps 1 --report_to none
```

### Parsed Arguments

```json
{
  "auto_launch_vllm_server": false,
  "cd_acyclic_reward_scale": 0.0,
  "cd_append_format_hint": true,
  "cd_bif_file": null,
  "cd_config_causal_rules": false,
  "cd_config_def_int": false,
  "cd_config_file": null,
  "cd_config_give_steps": false,
  "cd_config_intervene_vars": "all",
  "cd_config_num_prompts": 5,
  "cd_config_seed": 0,
  "cd_config_thinking_tags": true,
  "cd_edge_f1_reward_scale": 0.0,
  "cd_format_hint_text": "Use exactly this structure: <think>...</think><answer>...</answer>. Inside <answer>, output only the final graph answer.",
  "cd_format_reward_scale": 0.2,
  "cd_graph_reward_scale": 1.0,
  "cd_low_shd_reward_scale": 0.0,
  "cd_max_test_samples": 0,
  "cd_max_train_samples": 0,
  "cd_partial_format_reward_scale": 0.15,
  "cd_reward_dag_penalty": 0.1,
  "cd_reward_require_dag": true,
  "cd_reward_shd_weight": 0.0,
  "cd_test_csv": [
    "/u/chenyuen0103/ENCO/experiments/prompts/cd_descendants/sachs/splits/stage_1_named_obs50_int10_eval.csv"
  ],
  "cd_test_fraction": 0.1,
  "cd_train_csv": [
    "/u/chenyuen0103/ENCO/experiment/01_stage_1_named_obs50_int10/train_mixed.csv"
  ],
  "cd_wrap_system_prompt": true,
  "dataloader_num_workers": 4,
  "dataset_id": "AI-MO/NuminaMath-TIR",
  "enable_thinking": false,
  "enable_vllm_preflight": false,
  "eval_batch_size": 1,
  "eval_debug_csv": null,
  "eval_do_sample": false,
  "eval_max_new_tokens": 128,
  "eval_model": null,
  "eval_n": 200,
  "eval_output_json": null,
  "eval_pass_k": 1,
  "eval_responses_max_chars": 0,
  "eval_seed": 1234,
  "eval_split": "test",
  "eval_temperature": 0.0,
  "eval_top_p": 0.95,
  "export_csv": null,
  "export_limit": 0,
  "gradient_accumulation_steps": 8,
  "learning_rate": 5e-06,
  "length_penalty_coef": 0.0001,
  "length_penalty_max_abs": 0.0,
  "length_penalty_target_tokens": 0,
  "logging_steps": 1,
  "max_completion_length": 128,
  "max_prompt_tokens": 256000,
  "mode": "train",
  "model_id": "Qwen/Qwen3-4B-Thinking-2507",
  "num_generations": 2,
  "num_train_epochs": 1.0,
  "output_dir": "/u/chenyuen0103/ENCO/experiment/01_stage_1_named_obs50_int10/grpo",
  "per_device_train_batch_size": 2,
  "report_to": "none",
  "run_name": "qwen2-grpo-vllm-server",
  "sample_completions_every": 0,
  "sample_completions_k": 2,
  "sample_completions_max_chars": 320,
  "save_eval_responses": false,
  "save_steps": 20,
  "save_total_limit": 1,
  "stop_sequence": [
    "</answer>"
  ],
  "task": "cd_descendants",
  "test_split": "test[:5%]",
  "train_log_jsonl": null,
  "train_split": "train[:5%]",
  "train_temperature": 1.0,
  "train_top_p": 1.0,
  "use_vllm": false,
  "vllm_group_port": 51216,
  "vllm_preflight_timeout": 15.0,
  "vllm_server_base_url": "http://127.0.0.1:8000",
  "vllm_server_gpu_memory_utilization": 0.2,
  "vllm_server_log_file": "vllm_server.log",
  "vllm_server_max_model_len": null,
  "vllm_server_model_id": null,
  "vllm_server_startup_timeout": 120.0,
  "vllm_server_timeout": 240.0,
  "wandb_entity": null,
  "wandb_mode": "online",
  "wandb_project": "enco-grpo"
}
```
