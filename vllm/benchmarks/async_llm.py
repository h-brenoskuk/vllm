# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator, Iterable
from contextlib import suppress
from copy import deepcopy

import numpy as np
from tqdm.asyncio import tqdm

from vllm import SamplingParams
from vllm.benchmarks.datasets import get_samples  # type: ignore
from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser
from vllm.utils import random_uuid

# Set environment variable for V1 engine
os.environ["VLLM_USE_V1"] = "1"
from typing import Any, Optional, cast

from transformers import PreTrainedTokenizerBase

from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         parse_chat_messages)
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import (FinishedRequestStats, IterationStats,
                                   SchedulerStats)

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency for YAML configs
    yaml = None  # type: ignore


def convert_openai_to_vllm_format(
    request: SampleRequest,
    model_config,
    tokenizer,
) -> dict[str, object]:
    """
    Convert OpenAI API format from SampleRequest to vLLM internal format.

    Currently only support images.
    """

    if not request.multi_modal_data:
        return {"prompt": request.prompt}

    # Create OpenAI chat message format
    content: list[Any] = [
        {"type": "text", "text": request.prompt}
    ]

    # Handle both single image and multiple images
    if isinstance(request.multi_modal_data, list):
        # Raise error if the list contains anything other than images.
        for item in request.multi_modal_data:
            if item.get("type") != "image_url":
                raise ValueError("Only images are supported in "
                "the multimodal data.")
        content.extend(cast(list[Any], request.multi_modal_data))
    else:
        # Raise error if the item isn't image_url.
        if request.multi_modal_data.get("type") != "image_url":
            raise ValueError("Only images are supported in "
                "the multimodal data.")
        content.append(cast(Any, request.multi_modal_data))

    messages: list[Any] = [
        {"role": "user", "content": content}
    ]

    # Use vLLM's built-in conversion
    conversation, mm_data = parse_chat_messages(
        messages=cast(list[ChatCompletionMessageParam], messages),
        model_config=model_config,
        tokenizer=tokenizer,
        content_format="openai",
    )

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_str = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_str = conversation[0]["content"]

    return {"prompt": prompt_str, "multi_modal_data": mm_data}

def apply_openai_to_vllm_format(
    requests: list[SampleRequest],
    model_config,
    tokenizer,
) -> list[SampleRequest]:
    """
    Apply OpenAI to vLLM format to the requests.
    """
    # Pre-convert multimodal requests so conversion cost is outside timed path
    # For images, we desserialize the image data and convert it to a string.
    # This is done here so that the conversion cost is outside the timed path.
    for req in requests:
        try:
            if getattr(req, "multi_modal_data", None):
                converted = convert_openai_to_vllm_format(
                    req, model_config, tokenizer
                )
                # Store converted dict as prompt and clear multimodal field
                req.prompt = converted
                req.multi_modal_data = None
        except Exception as _:
            pass
    return requests

class BenchmarkMetricsCollector(StatLoggerBase):
    """Collects benchmark metrics from the v1 engine's internal stats."""

    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        self.engine_index = engine_index
        self.vllm_config = vllm_config

        # Timing collections
        self.ttft_values: list[float] = []
        self.tpot_values: list[float] = []
        self.itl_values: list[float] = []

        # Request tracking
        self.finished_requests: list[FinishedRequestStats] = []
        self.total_prompt_tokens = 0
        self.total_generation_tokens = 0
        self.total_requests_completed = 0

        # Benchmark timing
        self.benchmark_start_time: float | None = None
        self.benchmark_end_time: float | None = None

    def start_benchmark(self):
        """Mark the start of the benchmark."""
        self.benchmark_start_time = time.perf_counter()
        self.reset_stats()

    def end_benchmark(self):
        """Mark the end of the benchmark."""
        self.benchmark_end_time = time.perf_counter()

    def reset_stats(self):
        """Reset all collected stats."""
        self.ttft_values.clear()
        self.tpot_values.clear()
        self.itl_values.clear()
        self.finished_requests.clear()
        self.total_prompt_tokens = 0
        self.total_generation_tokens = 0
        self.total_requests_completed = 0

    def record(
        self,
        scheduler_stats: Optional[SchedulerStats],
        iteration_stats: Optional[IterationStats],
        engine_idx: int = 0,
    ) -> None:
        """Collect metrics from engine iterations."""
        if iteration_stats is None:
            return

        # Collect timing data
        # This is mirroring the  e2e benchmark at serve.py.
        # TPOT is calculated at serve.py as (latency - TTFT) / (N - 1)
        # ITL is calculated at lib/endpoint_request_func.py as
        # the interval between consecutive chunks.
        self.ttft_values.extend(iteration_stats.time_to_first_tokens_iter)
        self.itl_values.extend(iteration_stats.time_per_output_tokens_iter)

        # Collect token counts
        self.total_prompt_tokens += iteration_stats.num_prompt_tokens
        self.total_generation_tokens += iteration_stats.num_generation_tokens

        # Collect finished requests
        self.finished_requests.extend(iteration_stats.finished_requests)
        self.total_requests_completed += len(iteration_stats.finished_requests)

    def log_engine_initialized(self):
        """Required by StatLoggerBase interface."""
        pass

    def get_duration(self) -> float:
        """Get benchmark duration in seconds."""
        if self.benchmark_start_time and self.benchmark_end_time:
            return self.benchmark_end_time - self.benchmark_start_time
        return 0.0

    def calculate_percentiles(
        self,
        values: list[float],
        percentiles: list[float],
    ) -> list[tuple[float, float]]:
        """Calculate percentiles for given values."""
        if not values:
            return [(p, 0.0) for p in percentiles]
        return [(p, float(np.percentile(values, p))) for p in percentiles]

    def print_metrics(self):
        """Print metrics in the same format as serve.py."""
        duration = self.get_duration()

        if duration == 0:
            print("Benchmark duration not available")
            return

        # Convert timing values from seconds to milliseconds
        ttft_ms = [t * 1000 for t in self.ttft_values]

        # Compute per-request TPOT from finished requests at end of run.
        # TPOT (excluding first token) is decode_time / (gen_tokens - 1).
        per_request_tpot_s = [
            fr.decode_time / (fr.num_generation_tokens - 1)
            for fr in self.finished_requests
            if fr.num_generation_tokens > 1 and fr.decode_time > 0
        ]
        tpot_ms = [t * 1000 for t in per_request_tpot_s]
        # ITL distribution comes from inter-token (inter-yield) latencies
        itl_ms = [t * 1000 for t in self.itl_values]

        # Calculate throughput metrics
        request_throughput = self.total_requests_completed / duration
        output_throughput = self.total_generation_tokens / duration
        total_throughput = (
            self.total_prompt_tokens + self.total_generation_tokens
        ) / duration

        print(
            "{s:{c}^{n}}".format(
                s=" Serving Benchmark Result ", n=50, c="="
            )
        )
        print(
            "{:<40} {:<10}".format(
                "Successful requests:", self.total_requests_completed
            )
        )
        print(
            "{:<40} {:<10.2f}".format("Benchmark duration (s):", duration)
        )
        print(
            "{:<40} {:<10}".format(
                "Total input tokens:", self.total_prompt_tokens
            )
        )
        print(
            "{:<40} {:<10}".format(
                "Total generated tokens:", self.total_generation_tokens
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Request throughput (req/s):", request_throughput
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Output token throughput (tok/s):", output_throughput
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Total Token throughput (tok/s):", total_throughput
            )
        )

        # TTFT metrics
        if ttft_ms:
            print(
                "{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-")
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Mean TTFT (ms):", float(np.mean(ttft_ms))
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Median TTFT (ms):", float(np.median(ttft_ms))
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "P99 TTFT (ms):", float(np.percentile(ttft_ms, 99))
                )
            )

        # TPOT metrics
        if tpot_ms:
            print(
                "{s:{c}^{n}}".format(
                    s="Time per Output Token (excl. 1st token)", n=50, c="-"
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Mean TPOT (ms):", float(np.mean(tpot_ms))
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Median TPOT (ms):", float(np.median(tpot_ms))
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "P99 TPOT (ms):", float(np.percentile(tpot_ms, 99))
                )
            )

        # ITL metrics
        if itl_ms:
            print(
                "{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-")
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Mean ITL (ms):", float(np.mean(itl_ms))
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Median ITL (ms):", float(np.median(itl_ms))
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "P99 ITL (ms):", float(np.percentile(itl_ms, 99))
                )
            )

        print("=" * 50)


async def create_async_llm_engine_with_metrics(
    async_vllm_args: dict[str, Any],
    # returns a reference to the metrics collector
) -> tuple[AsyncLLM, BenchmarkMetricsCollector]:
    """Create an AsyncLLM engine with metrics collection."""
    async_engine_args = AsyncEngineArgs(**async_vllm_args)

    # Store reference to the metrics collector so we can access it later
    metrics_collector_ref: dict[str, BenchmarkMetricsCollector | None] = {
        "instance": None
    }

    # Create custom stat logger factory
    def metrics_factory(
        vllm_config: VllmConfig,
        engine_index: int,
    ) -> BenchmarkMetricsCollector:
        collector = BenchmarkMetricsCollector(vllm_config, engine_index)
        metrics_collector_ref["instance"] = collector  # Store reference
        return collector

    engine = AsyncLLM.from_engine_args(
        engine_args=async_engine_args,
        stat_loggers=[metrics_factory],
    )

    return engine, cast(
        BenchmarkMetricsCollector,
        metrics_collector_ref["instance"],
    )


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

    Handles common numeric and boolean fields so downstream code
    receives the correct types (e.g., request_rate as float).
    """
    float_keys = {
        "request_rate",
        "top_p",
        "min_p",
        "temperature",
        "gpu_memory_utilization",
    }
    int_keys = {
        "max_concurrency",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "max_model_len",
        "top_k",
        "num_prompts",
    }
    bool_keys = {"ignore_eos", "enable_prefix_caching", "disable_tqdm"}

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


def add_cli_args(parser) -> None:
    # Dataset-related arguments (shared with serve.py)
    add_dataset_parser(parser)  # type: ignore[arg-type]
    parser.add_argument(
        "--endpoint-type",
        type=str,
        default="openai-chat",
        choices=["openai", "openai-chat", "openai-audio"],
        help=("Endpoint type used only for dataset compatibility checks."),
    )

    # Traffic/control args
    parser.add_argument("--request-rate",
                        type=float,
                        default=float("inf"),
                        help="Requests per second. 'inf' sends all at t=0.")
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
    parser.add_argument("--request-id-prefix",
                        type=str,
                        default="async-llm",
                        help="Prefix used to assign request IDs.")
    parser.add_argument(
        "--config-yaml",
        type=str,
        default=None,
        help=(
            "Path to a YAML file describing multiple experiments to run "
            "sequentially. If provided, overrides CLI single-run mode."
        ),
    )

    # Sampling params (mirrors serve.py subset)
    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument("--top-p", type=float, default=None)
    sampling_group.add_argument("--top-k", type=int, default=None)
    sampling_group.add_argument("--min-p", type=float, default=None)
    sampling_group.add_argument("--temperature", type=float, default=0.0)

    # Engine/model args
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Name or path of the model. Required unless --config-yaml is "
            "provided with an 'engine.model' key."
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument(
        "--limit-mm-per-prompt",
        type=str,
        default="image=3,video=0",
        help="Comma-separated k=v, e.g. 'image=3,video=0'",
    )
    parser.add_argument(
        "--mm-processor-kwargs",
        type=str,
        default="max_pixels=1003520",
        help="Comma-separated k=v, e.g. 'max_pixels=1003520'",
    )
    parser.add_argument(
        "--guided-decoding-backend",
        type=str,
        default="xgrammar",
        help="Guided decoding backend name.",
    )
    parser.add_argument("--enable-prefix-caching",
                        action="store_true",
                        help="Enable prefix caching.")


async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
) -> AsyncGenerator[SampleRequest, None]:
    """
    Asynchronously generates requests at a specified rate.
    """
    requests_iterator: Iterable[SampleRequest] = iter(input_requests)

    for request in requests_iterator:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Calculate interval between requests
        interval = 1.0 / request_rate
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)





async def run_single_experiment(
    args: argparse.Namespace,
    engine: AsyncLLM | None = None,
    metrics_collector: BenchmarkMetricsCollector | None = None,
    precomputed_requests: list[SampleRequest] | None = None,
) -> tuple[AsyncLLM, BenchmarkMetricsCollector]:
    """Run one experiment, optionally reusing an existing engine.

    Returns engine and metrics collector (for reuse).
    """
    # Build engine args from CLI/YAML (normalize both string and dict)
    limit_mm_dict = _normalize_kv_input(args.limit_mm_per_prompt)
    mm_kwargs_dict = _normalize_kv_input(args.mm_processor_kwargs)

    vllm_args: dict[str, Any] = {
        "model": args.model,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "mm_processor_kwargs": mm_kwargs_dict,
        "guided_decoding_backend": args.guided_decoding_backend,
        "limit_mm_per_prompt": limit_mm_dict,
        "max_model_len": args.max_model_len,
        "enable_prefix_caching": bool(args.enable_prefix_caching),
    }

    # Create or reuse engine with metrics collection
    if engine is None or metrics_collector is None:
        async_vllm_args = {**vllm_args, "disable_log_requests": True}
        engine, metrics_collector = await create_async_llm_engine_with_metrics(
            async_vllm_args
        )

    assert engine is not None and metrics_collector is not None

    # Tokenizer and (optional) model config for multimodal conversion
    tokenizer = await engine.get_tokenizer()
    model_config = await engine.get_model_config()

    # Build or reuse input requests; convert to vLLM format if needed
    if precomputed_requests is not None:
        input_requests: list[SampleRequest] = precomputed_requests
    else:
        input_requests = apply_openai_to_vllm_format(
            get_samples(args, cast(PreTrainedTokenizerBase, tokenizer)),
            model_config,
            tokenizer,
        )

    total_requests = len(input_requests)
    print(f"Traffic request rate: {args.request_rate}")
    print(f"Maximum request concurrency: {args.max_concurrency}")

    # Concurrency limiter
    semaphore = (asyncio.Semaphore(args.max_concurrency)
                 if args.max_concurrency else None)

    # request calls the async engine generate method
    async def request_func(
        request_func_input: SampleRequest,
        pbar: tqdm | None,
    ):
        try:
            # Create sampling parameters with the expected output length
            sp_kwargs: dict[str, Any] = {
                "max_tokens": request_func_input.expected_output_len,
                "temperature": args.temperature,
                "ignore_eos": args.ignore_eos,
            }
            if args.top_p is not None:
                sp_kwargs["top_p"] = args.top_p
            if args.top_k is not None:
                sp_kwargs["top_k"] = args.top_k
            if args.min_p is not None:
                sp_kwargs["min_p"] = args.min_p
            sampling_params = SamplingParams(**sp_kwargs)

            # Prepare prompt; convert multimodal OpenAI-like content
            prompt_to_generate: Any
            if request_func_input.multi_modal_data:
                converted = convert_openai_to_vllm_format(
                    request_func_input, model_config, tokenizer
                )
                prompt_to_generate = converted
            else:
                prompt_to_generate = request_func_input.prompt

            # Generate using the async engine
            outputs = []
            async for output in engine.generate(
                prompt=cast(Any, prompt_to_generate),
                sampling_params=sampling_params,
                request_id=random_uuid(),
            ):
                outputs.append(output)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)

            # Return the final output (complete generation)
            return outputs[-1] if outputs else None

        except Exception as e:
            if pbar is not None:
                pbar.update(1)
            print(f"Exception occurred: {type(e).__name__}: {str(e)}")
            import traceback

            traceback.print_exc()
            return f"Error: {str(e)}"

    # Readiness/warmup run using the first request (mirrors serve.py behavior)
    if total_requests > 0:
        # This mirrors the behavior of serve.py
        print("Starting initial single prompt test run...")
        first_req = input_requests[0]
        await request_func(request_func_input=first_req, pbar=None)
        print("Initial test run completed. Starting main benchmark run...")

    # Create progress bar for measured run over full dataset (including first)
    pbar = None if args.disable_tqdm else tqdm(total=total_requests)
    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(
                request_func_input=request_func_input, pbar=pbar
            )
        async with semaphore:
            return await request_func(
                request_func_input=request_func_input, pbar=pbar
            )

    # Start metrics collection
    metrics_collector.start_benchmark()

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []
    async for request in get_request(input_requests, args.request_rate):
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request, pbar=pbar)
            )
        )

    await asyncio.gather(*tasks)
    elapsed_time = time.perf_counter() - benchmark_start_time

    # End metrics collection
    metrics_collector.end_benchmark()

    print(f"Benchmark completed in {elapsed_time:.2f} seconds")

    if pbar is not None:
        pbar.close()

    # Print detailed metrics
    print()  # Add some space
    metrics_collector.print_metrics()

    return engine, metrics_collector


class BenchmarkManager:
    """Run multiple experiments sequentially with engine reuse.

    YAML schema (example):
      engine:
        model: Qwen/Qwen2.5-VL-3B-Instruct
        tensor-parallel-size: 1
        pipeline-parallel-size: 1
        dtype: bfloat16
        gpu-memory-utilization: 0.9
        max-model-len: 16384
        limit-mm-per-prompt: "image=3,video=0"
        mm-processor-kwargs: "max_pixels=1003520"
        guided-decoding-backend: xgrammar
        enable-prefix-caching: false

      datasets:
        text_random_300x40:
          dataset-name: random
          seed: 42
          num-prompts: 100
          random-prefix-len: 25
          random-input-len: 300
          random-output-len: 40
          random-range-ratio: 0.2

        mm_zero_images:
          dataset-name: random-mm
          seed: 42
          num-prompts: 100
          random-prefix-len: 25
          random-input-len: 300
          random-output-len: 40
          random-range-ratio: 0.2
          random-mm-base-items-per-request: 0
          random-mm-num-mm-items-range-ratio: 0
          random-mm-limit-mm-per-prompt: '{"image":3,"video":0}'
          random-mm-bucket-config: '{(256, 256, 1): 0.25, (720, 1280, 1): 0.75}'

      experiments:
        - name: text-10c
          dataset: text_random_300x40
          args:
            max-concurrency: 10
            request-rate: inf
            ignore-eos: true

        - name: mm-zero-10c
          dataset: mm_zero_images
          args:
            max-concurrency: 10
            request-rate: inf
            ignore-eos: true
    """

    def __init__(self) -> None:
        # Engine is created in run_from_yaml (async context)
        pass

    # Only allow experiments to change traffic/sampling knobs.
    # Engine and dataset parameters are immutable per config.
    _ALLOWED_EXPERIMENT_ARG_KEYS: set[str] = {
        "max_concurrency",
        "request_rate",
        "ignore_eos",
        "disable_tqdm",
        # sampling
        "top_p",
        "top_k",
        "min_p",
        "temperature",
        # optional utility
        "request_id_prefix",
    }

    def _apply_experiment_args(self, base_args: argparse.Namespace,
                               exp_args: dict[str, Any]) -> argparse.Namespace:
        """Apply only allowed experiment-level args.

        Ignores any keys outside _ALLOWED_EXPERIMENT_ARG_KEYS to avoid
        overriding engine or dataset configuration.
        """
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
            # support kebab-case in YAML by converting to underscore
            attr = k.replace("-", "_")
            # quick solution to convert string to int, float, bool, etc.
            # TODO: improve this (use BaseModel)
            coerced = _coerce_override_value(attr, v)
            setattr(new_args, attr, coerced)
        return new_args


    async def run_from_yaml(self, base_args: argparse.Namespace) -> None:
        assert base_args.config_yaml is not None
        with open(base_args.config_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("YAML must be a mapping with 'experiments' list")

        engine_overrides = cfg.get("engine", {}) or {}
        experiments = cfg.get("experiments", [])
        datasets_map = cfg.get("datasets", {}) or {}

        if not isinstance(experiments, list):
            raise ValueError("'experiments' must be a list")
        if not isinstance(datasets_map, dict):
            raise ValueError("'datasets' must be a mapping of name -> config")

        # Apply top-level engine overrides once (single engine across runs)
        if engine_overrides:
            normalized_engine_overrides = {
                k.replace("-", "_"): v
                for k, v in engine_overrides.items()
            }
            base_args = self._apply_overrides(
                base_args, normalized_engine_overrides)

        # Create a single engine for all experiments
        engine_args_dict = {
            "model": base_args.model,
            "pipeline_parallel_size": base_args.pipeline_parallel_size,
            "tensor_parallel_size": base_args.tensor_parallel_size,
            "dtype": base_args.dtype,
            "gpu_memory_utilization": base_args.gpu_memory_utilization,
            "mm_processor_kwargs": _normalize_kv_input(
                base_args.mm_processor_kwargs),
            "guided_decoding_backend": base_args.guided_decoding_backend,
            "limit_mm_per_prompt": _normalize_kv_input(
                base_args.limit_mm_per_prompt),
            "max_model_len": base_args.max_model_len,
            "disable_log_requests": True,
            "enable_prefix_caching": bool(base_args.enable_prefix_caching),
        }
        engine, metrics_collector = await create_async_llm_engine_with_metrics(
            engine_args_dict
        )

        # Group experiments by dataset to avoid re-sampling and ensure
        # dataset config is the single source of truth.
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
                continue  # no experiments for this dataset

            # Build base args for this dataset (no experiment overrides)
            dataset_args = self._apply_overrides(base_args, {
                k.replace("-", "_"): v for k, v in (dataset_cfg or {}).items()
            })

            # Ensure model is defined via engine overrides or CLI
            if not getattr(dataset_args, "model", None):
                raise ValueError(
                    "Model must be specified via 'engine.model' or CLI --model"
                )

            # Precompute and cache sample requests for this dataset once
            tokenizer = await engine.get_tokenizer()
            model_config = await engine.get_model_config()
            precomputed_requests = apply_openai_to_vllm_format(
                get_samples(
                    dataset_args,
                    cast(PreTrainedTokenizerBase, tokenizer),
                ),
                model_config,
                tokenizer,
            )

            # Run all experiments that use this dataset, reusing requests
            for idx, exp in enumerate(experiments_by_dataset[dataset_name]):
                name = exp.get("name", f"exp-{idx}")
                print("=" * 50)
                print(f"Starting experiment: {name} (dataset: {dataset_name})")

                # Apply only allowed experiment-level args
                args_for_run = self._apply_experiment_args(
                    dataset_args, cast(dict[str, Any], exp.get("args", {}))
                )

                await run_single_experiment(
                    args_for_run,
                    engine,
                    metrics_collector,
                    precomputed_requests,
                )

        # Cleanup engine after all experiments finish
        with suppress(Exception):
            engine.shutdown()

        print("All experiments completed.")


"""
python3 benchmarks/async_llm.py \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384 \
    --limit-mm-per-prompt "image=3,video=0" \
    --mm-processor-kwargs "max_pixels=1003520" \
    --guided-decoding-backend xgrammar \
    --dataset-name random \
    --num-prompts 100 \
    --max-concurrency 10 \
    --random-prefix-len 25 \
    --random-input-len 300 \
    --random-output-len 40 \
    --random-range-ratio 0.2 \
    --request-rate inf \
    --ignore-eos \
    --endpoint-type openai-chat \
    --seed 42
"""

"""
python3 benchmarks/async_llm.py \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384 \
    --limit-mm-per-prompt "image=3,video=0" \
    --mm-processor-kwargs "max_pixels=1003520" \
    --guided-decoding-backend xgrammar \
    --dataset-name random-mm \
    --num-prompts 100 \
    --max-concurrency 10 \
    --random-prefix-len 25 \
    --random-input-len 300 \
    --random-output-len 40 \
    --random-range-ratio 0.2 \
    --random-mm-base-items-per-request 2 \
    --random-mm-num-mm-items-range-ratio 0 \
    --random-mm-limit-mm-per-prompt '{"image":3,"video":0}' \
    --random-mm-bucket-config '{(256, 256, 1): 0.25, (720, 1280, 1): 0.75}' \
    --request-rate inf \
    --ignore-eos \
    --endpoint-type openai-chat \
    --seed 42 
"""

"""
python3 benchmarks/async_llm.py --config-yaml \
    benchmarks/experiments/random_mm_test.yaml
"""

async def main_async(args: argparse.Namespace) -> None:
    # Enforce that a model is provided in single-run mode.
    if (not getattr(args, "config_yaml", None)
            and not getattr(args, "model", None)):
        raise SystemExit(
            "Missing --model. Provide --model for single-run CLI or use "
            "--config-yaml with an 'engine.model' or per-experiment 'model' "
            "override."
        )
    # If a YAML config is provided, use the BenchmarkManager path
    if getattr(args, "config_yaml", None):
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required for --config-yaml but not installed. "
                "Install pyyaml or remove --config-yaml."
            )
        manager = BenchmarkManager()
        await manager.run_from_yaml(args)
        return
    # Single-run path: delegate to unified runner and shutdown engine
    engine, _metrics = await run_single_experiment(args)
    with suppress(Exception):
        engine.shutdown()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args()
    asyncio.run(main_async(args))