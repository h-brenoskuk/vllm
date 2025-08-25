# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project

OpenAI-compatible multi-experiment benchmark runner.

This mirrors the multi-run/manager flow of async_llm.py but drives
the client-side benchmark defined in serve.py against an OpenAI-
compatible endpoint.

YAML schema example:

api:
  endpoint-type: openai-chat       # or "openai"
  backend: openai                  # must be an OPENAI_COMPATIBLE_BACKEND
  base-url: https://api.openai.com # or your gateway
  endpoint: /v1/chat/completions   # or /v1/completions
  model: gpt-4o-mini               # model id in request body
  served-model-name: gpt-4o-mini   # optional; display/name override
  tokenizer: gpt2                  # optional, for client-side metrics only
  tokenizer-mode: auto             # optional
  trust-remote-code: false         # optional
  logprobs: null                   # optional, e.g. 1

datasets:
  text_random_300x40:
    dataset-name: random
    seed: 42
    num-prompts: 100
    random-prefix-len: 25
    random-input-len: 300
    random-output-len: 40
    random-range-ratio: 0.2

  mm_two_images_two_sizes:
    dataset-name: random-mm
    seed: 42
    num-prompts: 100
    random-prefix-len: 25
    random-input-len: 300
    random-output-len: 40
    random-range-ratio: 0.2
    random-mm-base-items-per-request: 2
    random-mm-num-mm-items-range-ratio: 0
    random-mm-limit-mm-per-prompt: '{"image":3,"video":0}'
    random-mm-bucket-config: '{(256, 256, 1): 0.5, (720, 1280, 1): 0.5}'

experiments:
  - name: openai-text-10c
    dataset: text_random_300x40
    args:
      max-concurrency: 10
      request-rate: 50
      ignore-eos: true
      temperature: 0.0
      top-p: 0.95

  - name: openai-mm-10c
    dataset: mm_two_images_two_sizes
    args:
      max-concurrency: 10
      request-rate: 30
      ignore-eos: true
      temperature: 0.0
      top-p: 0.95
"""

import argparse
import asyncio
import json
from contextlib import suppress
from copy import deepcopy
from typing import Any, cast

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency for YAML configs
    yaml = None  # type: ignore

from transformers import PreTrainedTokenizerBase

from vllm.benchmarks.datasets import get_samples  # type: ignore
from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser
from vllm.benchmarks.serve import benchmark as serve_benchmark
from vllm.benchmarks.serve import check_goodput_args
from vllm.transformers_utils.tokenizer import get_tokenizer


def _parse_kv_dict(
    arg_val: str | None,
    value_parser=None,
) -> dict[str, Any] | None:
    """Parse simple comma-separated k=v pairs into a dict.

    Example: "image=3,video=0" -> {"image": 3, "video": 0}
    """
    if arg_val is None:
        return None
    result: dict[str, Any] = {}
    for pair in arg_val.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"Invalid k=v pair: {pair}")
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()
        if value_parser is not None:
            result[k] = value_parser(v)
        else:
            # Try int -> float -> str
            try:
                result[k] = int(v)
            except ValueError:
                try:
                    result[k] = float(v)
                except ValueError:
                    # Support true/false
                    if v.lower() in ("true", "false"):
                        result[k] = v.lower() == "true"
                    else:
                        result[k] = v
    return result


def _normalize_kv_input(val: Any) -> dict[str, Any]:
    """Normalize CLI/YAML kv inputs into dictionaries.

    Accepts either a dict (returned as-is), a JSON/k=v string, or None.
    """
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        s = val.strip()
        # Try JSON first
        if s.startswith("{") and s.endswith("}"):
            try:
                return cast(dict[str, Any], json.loads(s))
            except Exception:
                pass
        # Fallback to simple k=v parsing
        try:
            return _parse_kv_dict(s) or {}
        except Exception:
            return {}
    return {}


def _parse_mm_bucket_config_value(v: Any) -> dict[tuple[int, int, int], float]:
    """Parse random-mm bucket config from YAML overrides.

    Supports dict with tuple or string keys, or a string that can be
    ast.literal_eval'ed into a dict.
    """
    import ast as _ast  # local import to avoid global namespace clutter

    def _normalize(d: dict) -> dict[tuple[int, int, int], float]:
        out: dict[tuple[int, int, int], float] = {}
        for k, val in d.items():
            key: Any = k
            if isinstance(key, str):
                with suppress(Exception):
                    key = _ast.literal_eval(key)
            if not (
                isinstance(key, tuple) and len(key) == 3 and
                all(isinstance(x, int) for x in key)
            ):
                raise ValueError(
                    f"Invalid bucket key {k!r}. Expected tuple (H, W, T)."
                )
            out[(int(key[0]), int(key[1]), int(key[2]))] = float(val)
        return out

    if isinstance(v, dict):
        return _normalize(v)
    if isinstance(v, str):
        with suppress(Exception):
            parsed = _ast.literal_eval(v)
            if isinstance(parsed, dict):
                return _normalize(parsed)
        raise ValueError(
            "Unsupported value for random_mm_bucket_config override."
        )
    raise ValueError(
        "random_mm_bucket_config override must be dict or string."
    )


def _coerce_override_value(key: str, value: Any) -> Any:
    """Coerce YAML override values to expected CLI types.

    Mirrors async_llm.py so dataset overrides behave identically.
    """
    float_keys = {
        "request_rate",
        "top_p",
        "min_p",
        "temperature",
        "gpu_memory_utilization",
        "burstiness",
    }
    int_keys = {
        "max_concurrency",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "max_model_len",
        "top_k",
        "num_prompts",
        "logprobs",
        "ready_check_timeout_sec",
    }
    bool_keys = {"ignore_eos", "enable_prefix_caching", "disable_tqdm",
                 "trust_remote_code"}

    # Dataset-related numeric fields
    float_keys.update({
        "random_mm_num_mm_items_range_ratio",
        "random_range_ratio",
    })
    int_keys.update({
        "random_mm_base_items_per_request",
        "random_prefix_len",
        "random_input_len",
        "random_output_len",
        "num_prompts",
    })

    if key in float_keys:
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"inf", "+inf", "infinity", "+infinity"}:
                return float("inf")
            if v in {"-inf", "-infinity"}:
                return float("-inf")
            try:
                return float(value)
            except Exception:
                return value
        try:
            return float(value)
        except Exception:
            return value

    if key in int_keys:
        try:
            return int(value)
        except Exception:
            return value

    if key in bool_keys:
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    # Dict-like overrides that may arrive as JSON or k=v strings
    if key in {"mm_processor_kwargs", "limit_mm_per_prompt"}:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            return _normalize_kv_input(value)
        return value

    # Multimodal dataset dict overrides
    if key == "random_mm_limit_mm_per_prompt":
        if isinstance(value, dict):
            return {str(k): int(v) for k, v in value.items()}
        if isinstance(value, str):
            try:
                d = cast(dict[str, Any], json.loads(value))
            except Exception:
                d = _normalize_kv_input(value)
            return {str(k): int(v) for k, v in d.items()}
        return value

    if key == "random_mm_bucket_config":
        try:
            return _parse_mm_bucket_config_value(value)
        except Exception:
            return value

    return value


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    # Dataset-related arguments (shared with serve.py)
    add_dataset_parser(parser)  # type: ignore[arg-type]

    # API/client args
    parser.add_argument(
        "--endpoint-type",
        type=str,
        default="openai",
        help="Endpoint type used to serve model (e.g., openai, openai-chat).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        help="Backend identifier; must be openai-compatible in serve.py.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help=(
            "API base url (e.g., https://api.openai.com). "
            "If using --config-yaml, specify this under api.base-url."
        ),
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/chat/completions",
        help="API endpoint path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Model id sent to the API (e.g., gpt-4o-mini).",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="Optional served model display/name override.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name/path for client-side token counting.",
    )
    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        choices=["auto", "slow", "mistral", "custom"],
        help="Tokenizer mode.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--logprobs", type=int, default=None)

    # Traffic/control args
    parser.add_argument("--request-rate",
                        type=float,
                        default=float("inf"),
                        help="Requests per second. 'inf' sends all at t=0.")
    parser.add_argument("--burstiness",
                        type=float,
                        default=1.0,
                        help="Burstiness factor for request generation.")
    parser.add_argument("--max-concurrency",
                        type=int,
                        default=32,
                        help="Max concurrent in-flight requests.")
    parser.add_argument("--disable-tqdm",
                        action="store_true",
                        help="Disable progress bar.")
    parser.add_argument("--ignore-eos",
                        action="store_true",
                        help="Ignore EOS during generation.")
    parser.add_argument(
        "--request-id-prefix",
        type=str,
        required=False,
        default="openai-bench",
        help="Specify the prefix of request id.",
    )
    parser.add_argument(
        "--config-yaml",
        type=str,
        default=None,
        help=(
            "Path to a YAML file describing multiple experiments to run "
            "sequentially. If provided, overrides CLI single-run mode."
        ),
    )

    # Sampling params (OpenAI-compatible)
    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument("--top-p", type=float, default=None)
    sampling_group.add_argument("--top-k", type=int, default=None)
    sampling_group.add_argument("--min-p", type=float, default=None)
    sampling_group.add_argument("--temperature", type=float, default=0.0)

    # Percentiles/goodput (optional parity with serve.py)
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help=(
            "Comma-separated list of metric names for percentile reporting. "
            "Allowed: ttft,tpot,itl,e2el"
        ),
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help=(
            "Comma-separated list of percentiles to report (e.g., 25,50,75)."
        ),
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help=(
            "Service level objectives for goodput as KEY:VALUE in ms. "
            "Allowed keys: ttft,tpot,e2el"
        ),
    )
    parser.add_argument(
        "--ready-check-timeout-sec",
        type=int,
        default=600,
        help="Timeout in seconds waiting for endpoint readiness.",
    )


class OpenAIBenchmarkManager:
    """Run multiple experiments sequentially with API reuse.

    YAML schema keys:
      - api: OpenAI/OpenAI-compatible connection and model fields
      - datasets: dataset configurations (same as serve.py/add_dataset_parser)
      - experiments: list of experiments that select a dataset and override
        traffic/sampling knobs
    """

    _ALLOWED_EXPERIMENT_ARG_KEYS: set[str] = {
        "max_concurrency",
        "request_rate",
        "burstiness",
        "ignore_eos",
        "disable_tqdm",
        # sampling
        "top_p",
        "top_k",
        "min_p",
        "temperature",
    }

    def _apply_experiment_args(self, base_args: argparse.Namespace,
                               exp_args: dict[str, Any]) -> argparse.Namespace:
        new_args = deepcopy(base_args)
        for k, v in (exp_args or {}).items():
            attr = k.replace("-", "_")
            if attr in self._ALLOWED_EXPERIMENT_ARG_KEYS:
                setattr(new_args, attr, _coerce_override_value(attr, v))
        return new_args

    def _apply_overrides(self, base_args: argparse.Namespace,
                         overrides: dict[str, Any]) -> argparse.Namespace:
        new_args = deepcopy(base_args)
        for k, v in overrides.items():
            attr = k.replace("-", "_")
            setattr(new_args, attr, _coerce_override_value(attr, v))
        return new_args

    async def run_from_yaml(self, base_args: argparse.Namespace) -> None:
        assert base_args.config_yaml is not None
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required for --config-yaml but not installed.")

        with open(base_args.config_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("YAML must be a mapping with 'experiments' list")

        api_overrides = cfg.get("api", {}) or {}
        experiments = cfg.get("experiments", [])
        datasets_map = cfg.get("datasets", {}) or {}

        if not isinstance(experiments, list):
            raise ValueError("'experiments' must be a list")
        if not isinstance(datasets_map, dict):
            raise ValueError("'datasets' must be a mapping of name -> config")

        # Apply top-level API overrides once (single tokenizer across runs)
        if api_overrides:
            normalized_overrides = {
                k.replace("-", "_"): v for k, v in api_overrides.items()
            }
            base_args = self._apply_overrides(base_args, normalized_overrides)

        # Resolve tokenizer id
        tokenizer_id = (base_args.tokenizer if base_args.tokenizer is not None
                        else base_args.model)
        if tokenizer_id is None:
            raise ValueError(
                "Model must be specified via 'api.model' or CLI --model")

        tokenizer = get_tokenizer(tokenizer_id,
                                  tokenizer_mode=base_args.tokenizer_mode,
                                  trust_remote_code=bool(
                                      base_args.trust_remote_code))

        # Pre-group experiments by dataset
        experiments_by_dataset: dict[str, list[dict[str, Any]]] = {}
        for idx, exp in enumerate(experiments):
            if not isinstance(exp, dict):
                raise ValueError("Each experiment entry must be a mapping")
            dataset_name = exp.get("dataset")
            if not dataset_name:
                name = exp.get("name", f"exp-{idx}")
                raise ValueError(
                    f"Experiment '{name}' missing required 'dataset' key")
            if dataset_name not in datasets_map:
                name = exp.get("name", f"exp-{idx}")
                raise ValueError(
                    f"Experiment '{name}' references unknown dataset "
                    f"'{dataset_name}'")
            experiments_by_dataset.setdefault(dataset_name, []).append(exp)

        # Iterate datasets in YAML-defined order
        for dataset_name, dataset_cfg in datasets_map.items():
            if dataset_name not in experiments_by_dataset:
                continue

            dataset_args = self._apply_overrides(base_args, {
                k.replace("-", "_"): v for k, v in (dataset_cfg or {}).items()
            })

            if not getattr(dataset_args, "model", None):
                raise ValueError(
                    "Model must be specified via 'api.model' or CLI --model"
                )

            # Precompute and cache sample requests for this dataset once
            precomputed_requests: list[SampleRequest] = get_samples(
                dataset_args, cast(PreTrainedTokenizerBase, tokenizer))

            # Run all experiments for this dataset
            for idx, exp in enumerate(experiments_by_dataset[dataset_name]):
                name = exp.get("name", f"exp-{idx}")
                print("=" * 50)
                print(f"Starting experiment: {name} (dataset: {dataset_name})")

                args_for_run = self._apply_experiment_args(
                    dataset_args, cast(dict[str, Any], exp.get("args", {})))

                # Collect sampling parameters
                sampling_params = {
                    k: v
                    for k, v in {
                        "top_p": args_for_run.top_p,
                        "top_k": args_for_run.top_k,
                        "min_p": args_for_run.min_p,
                        "temperature": args_for_run.temperature,
                    }.items() if v is not None
                }
                if "temperature" not in sampling_params:
                    sampling_params["temperature"] = 0.0

                # Goodput config (optional)
                goodput_config_dict = check_goodput_args(args_for_run)

                api_url = f"{args_for_run.base_url}{args_for_run.endpoint}"
                base_url = f"{args_for_run.base_url}"

                await serve_benchmark(
                    endpoint_type=args_for_run.endpoint_type,
                    api_url=api_url,
                    base_url=base_url,
                    model_id=args_for_run.model,
                    model_name=args_for_run.served_model_name,
                    tokenizer=cast(PreTrainedTokenizerBase, tokenizer),
                    input_requests=precomputed_requests,
                    logprobs=args_for_run.logprobs,
                    request_rate=args_for_run.request_rate,
                    burstiness=args_for_run.burstiness,
                    disable_tqdm=args_for_run.disable_tqdm,
                    profile=False,
                    selected_percentile_metrics=args_for_run.
                    percentile_metrics.split(","),
                    selected_percentiles=[
                        float(p)
                        for p in args_for_run.metric_percentiles.split(",")
                    ],
                    ignore_eos=args_for_run.ignore_eos,
                    goodput_config_dict=goodput_config_dict,
                    max_concurrency=args_for_run.max_concurrency,
                    lora_modules=None,
                    extra_body=sampling_params,
                    ramp_up_strategy=None,
                    ramp_up_start_rps=None,
                    ramp_up_end_rps=None,
                    ready_check_timeout_sec=args_for_run.
                    ready_check_timeout_sec,
                )

        print("All experiments completed.")


async def main_async(args: argparse.Namespace) -> None:
    # If a YAML config is provided, use the manager path
    if getattr(args, "config_yaml", None):
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required for --config-yaml but not installed. "
                "Install pyyaml or remove --config-yaml.")
        manager = OpenAIBenchmarkManager()
        await manager.run_from_yaml(args)
        return

    # Single-run path: run one benchmark using CLI args
    if args.model is None:
        raise SystemExit(
            "Missing --model. Provide --model for single-run CLI or use "
            "--config-yaml with an 'api.model' override.")

    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = get_tokenizer(tokenizer_id,
                              tokenizer_mode=args.tokenizer_mode,
                              trust_remote_code=args.trust_remote_code)

    input_requests = get_samples(args, tokenizer)

    # Sampling parameters (OpenAI-compatible)
    sampling_params = {
        k: v
        for k, v in {
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "temperature": args.temperature,
        }.items() if v is not None
    }
    if "temperature" not in sampling_params:
        sampling_params["temperature"] = 0.0

    goodput_config_dict = check_goodput_args(args)

    api_url = f"{args.base_url}{args.endpoint}"
    base_url = f"{args.base_url}"

    await serve_benchmark(
        endpoint_type=args.endpoint_type,
        api_url=api_url,
        base_url=base_url,
        model_id=args.model,
        model_name=args.served_model_name,
        tokenizer=cast(PreTrainedTokenizerBase, tokenizer),
        input_requests=input_requests,
        logprobs=args.logprobs,
        request_rate=args.request_rate,
        burstiness=args.burstiness,
        disable_tqdm=args.disable_tqdm,
        profile=False,
        selected_percentile_metrics=args.percentile_metrics.split(","),
        selected_percentiles=[float(p)
                              for p in args.metric_percentiles.split(",")],
        ignore_eos=args.ignore_eos,
        goodput_config_dict=goodput_config_dict,
        max_concurrency=args.max_concurrency,
        lora_modules=None,
        extra_body=sampling_params,
        ramp_up_strategy=None,
        ramp_up_start_rps=None,
        ramp_up_end_rps=None,
        ready_check_timeout_sec=args.ready_check_timeout_sec,
    )

"""
python3 benchmarks/openai_bench.py \
  --base-url http://127.0.0.1:8000 \
  --endpoint "/v1/chat/completions" \
  --endpoint-type openai-chat \
  --backend openai \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset-name random \
  --num-prompts 100 \
  --max-concurrency 10 \
  --random-prefix-len 25 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-range-ratio 0.2 \
  --request-rate inf \
  --ignore-eos \
  --seed 42
"""

"""
python3 benchmarks/openai_bench.py --config-yaml \
  benchmarks/config_openai.yaml
"""



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    cli_args = parser.parse_args()
    asyncio.run(main_async(cli_args))


