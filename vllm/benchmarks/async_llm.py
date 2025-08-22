# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio
import os
import time
from collections.abc import AsyncGenerator, Iterable

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


def convert_openai_to_vllm_format(
    request: SampleRequest,
    model_config,
    tokenizer,
) -> dict[str, object]:
    """
    Convert OpenAI API format from SampleRequest to vLLM internal format.
    """

    if not request.multi_modal_data:
        return {"prompt": request.prompt}

    # Create OpenAI chat message format
    content: list[Any] = [
        {"type": "text", "text": request.prompt}
    ]

    # Handle both single image and multiple images
    if isinstance(request.multi_modal_data, list):
        content.extend(cast(list[Any], request.multi_modal_data))
    else:
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
        self.ttft_values.extend(iteration_stats.time_to_first_tokens_iter)
        self.tpot_values.extend(iteration_stats.time_per_output_tokens_iter)

        # Inter-token latency is roughly the same as TPOT for our purposes
        # In the real serve.py, ITL is calculated differently
        # but this gives similar insights
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
        tpot_ms = [t * 1000 for t in self.tpot_values]
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

    # Sampling params (mirrors serve.py subset)
    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument("--top-p", type=float, default=None)
    sampling_group.add_argument("--top-k", type=int, default=None)
    sampling_group.add_argument("--min-p", type=float, default=None)
    sampling_group.add_argument("--temperature", type=float, default=0.0)

    # Engine/model args
    parser.add_argument("--model", type=str, required=True)
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


async def main_async(args: argparse.Namespace) -> None:
    # Build engine args from CLI
    limit_mm_dict = _parse_kv_dict(args.limit_mm_per_prompt) or {}
    mm_kwargs_dict = _parse_kv_dict(args.mm_processor_kwargs) or {}

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

    async_vllm_args = {**vllm_args, "disable_log_requests": True}

    # Create engine with metrics collection
    engine, metrics_collector = await create_async_llm_engine_with_metrics(
        async_vllm_args
    )

    # Tokenizer and (optional) model config for multimodal conversion
    tokenizer = await engine.get_tokenizer()
    model_config = await engine.get_model_config()

    # Build input requests via the shared dataset loader
    input_requests = get_samples(args, cast(PreTrainedTokenizerBase, tokenizer))

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

    _ = await asyncio.gather(*tasks)
    elapsed_time = time.perf_counter() - benchmark_start_time

    # End metrics collection
    metrics_collector.end_benchmark()

    print(f"Benchmark completed in {elapsed_time:.2f} seconds")

    if pbar is not None:
        pbar.close()

    # Print detailed metrics
    print()  # Add some space
    metrics_collector.print_metrics()


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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args()
    asyncio.run(main_async(args))