# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import time
from collections.abc import AsyncGenerator, Iterable

import numpy as np
from tqdm.asyncio import tqdm

from vllm import SamplingParams
from vllm.benchmarks.datasets import (RandomDataset, RandomMultiModalDataset,
                                      SampleRequest)
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


async def benchmark(
    num_requests: int = 100,
    prefix_len: int = 0,
    input_len: int = 100,
    output_len: int = 10,
    request_rate: float = float("inf"),
    max_concurrency: int = 32,
    disable_tqdm: bool = False,
    use_multimodal: bool = False,
    image_width: int = 224,
    image_height: int = 224,
    ignore_eos: bool = True,
):
    vllm_args = {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.9,
        "mm_processor_kwargs": {"max_pixels": 1003520},
        "guided_decoding_backend": "xgrammar",
        "limit_mm_per_prompt": {"image": 3, "video": 0},
        "max_model_len": 16384,
        "enable_prefix_caching": False,
    }

    async_vllm_args = {**vllm_args, "disable_log_requests": True}

    # Create engine with metrics collection
    engine, metrics_collector = await create_async_llm_engine_with_metrics(
        async_vllm_args
    )

    tokenizer = await engine.get_tokenizer()

    if use_multimodal:
        mm_dataset_generator = RandomMultiModalDataset(random_seed=42)
        random_dataset = mm_dataset_generator.sample(
            tokenizer=cast(PreTrainedTokenizerBase, tokenizer),
            num_requests=num_requests,
            prefix_len=prefix_len,
            input_len=input_len,
            output_len=output_len,
            width=image_width,
            height=image_height,
        )
    else:
        text_dataset_generator = RandomDataset(random_seed=42)
        random_dataset = text_dataset_generator.sample(
            tokenizer=cast(PreTrainedTokenizerBase, tokenizer),
            num_requests=num_requests,
            prefix_len=prefix_len,
            input_len=input_len,
            output_len=output_len,
        )

    print(f"Traffic request rate: {request_rate}")
    print(f"Maximum request concurrency: {max_concurrency}")

    # Create progress bar
    pbar = None if disable_tqdm else tqdm(total=num_requests)

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    # Start metrics collection
    metrics_collector.start_benchmark()

    # request calls the async engine generate method
    async def request_func(
        request_func_input: SampleRequest,
        pbar: tqdm | None,
    ):
        try:
            # Create sampling parameters with the expected output length
            sampling_params = SamplingParams(
                max_tokens=request_func_input.expected_output_len,
                temperature=0.0,  # Greedy decoding for consistency
                ignore_eos=ignore_eos,
            )

            # Use vLLM's native conversion pipeline from OpenAI format to
            # internal format
            # Prepare prompt
            prompt_to_generate: Any = None
            if use_multimodal and request_func_input.multi_modal_data:
                # Get model config from the engine for proper conversion
                model_config = await engine.get_model_config()
                tokenizer = await engine.get_tokenizer()

                # Convert using vLLM's built-in functions
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

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(
                request_func_input=request_func_input, pbar=pbar
            )
        async with semaphore:
            return await request_func(
                request_func_input=request_func_input, pbar=pbar
            )

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []
    async for request in get_request(random_dataset, request_rate):
        request_func_input = request
        tasks.append(
            asyncio.create_task(
                limited_request_func(
                    request_func_input=request_func_input, pbar=pbar
                )
            )
        )

    outputs: list[list[str]] = await asyncio.gather(*tasks)
    elapsed_time = time.perf_counter() - benchmark_start_time

    # End metrics collection
    metrics_collector.end_benchmark()

    print(f"Benchmark completed in {elapsed_time:.2f} seconds")

    if pbar is not None:
        pbar.close()

    # Print detailed metrics
    print()  # Add some space
    metrics_collector.print_metrics()
    return outputs


if __name__ == "__main__":
    # Simple test with default parameters
    # print("Running text-only benchmark...")
    # asyncio.run(
    #    benchmark(
    #        num_requests=5,  # Small number for quick testing
    #        input_len=100,
    #        output_len=20,
    #        disable_tqdm=False,
    #        use_multimodal=False,
    #    )
    # )

    # print("\n" + "="*50 + "\n")

    # Test multimodal benchmark
    print("Running multimodal benchmark...")
    asyncio.run(
        benchmark(
            num_requests=100,
            max_concurrency=32,
            input_len=512,
            output_len=32,
            disable_tqdm=False,
            use_multimodal=True,
            image_width=224,
            image_height=224,
        )
    )