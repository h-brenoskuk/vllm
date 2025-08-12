# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This module defines a framework for sampling benchmark requests from various
datasets. Each dataset subclass of BenchmarkDataset must implement sample
generation. Supported dataset types include:
  - ShareGPT
  - Random (synthetic)
  - Sonnet
  - BurstGPT
  - HuggingFace
  - VisionArena
"""
import base64
import io
import json
import logging
import random
import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from io import BytesIO
from typing import Any, Callable, Optional, Union, cast

import numpy as np
from PIL import Image
from transformers import PreTrainedTokenizerBase

from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.multimodal import MultiModalDataDict
from vllm.multimodal.image import convert_image_mode
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_lora_tokenizer
from vllm.utils import PlaceholderModule

try:
    from datasets import load_dataset
except ImportError:
    datasets = PlaceholderModule("datasets")
    load_dataset = datasets.placeholder_attr("load_dataset")

try:
    import pandas as pd
except ImportError:
    pd = PlaceholderModule("pandas")

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class SampleRequest:
    """
    Represents a single inference request for benchmarking.
    """

    prompt: Union[str, Any]
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[
        Union[MultiModalDataDict, dict, list[dict]]
    ] = None
    lora_request: Optional[LoRARequest] = None


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


class BenchmarkDataset(ABC):
    DEFAULT_SEED = 0
    IS_MULTIMODAL = False

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        random_seed: int = DEFAULT_SEED,
    ) -> None:
        """
        Initialize the BenchmarkDataset with an optional dataset path and random
        seed.  
        
        Args:
            dataset_path (Optional[str]): Path to the dataset. If None, it
            indicates that a default or random dataset might be used.
            random_seed (int): Seed value for reproducible shuffling or
            sampling. Defaults to DEFAULT_SEED.
        """
        self.dataset_path = dataset_path
        # Set the random seed, ensuring that a None value is replaced with the
        # default seed.
        self.random_seed = (random_seed
                            if random_seed is not None else self.DEFAULT_SEED)
        self.data = None

    def apply_multimodal_chat_transformation(
            self,
            prompt: str,
            mm_content: Optional[
                Union[MultiModalDataDict, dict, list[dict]]
            ] = None) -> list[dict]:
        """
        Transform a prompt and optional multimodal content into a chat format.
        This method is used for chat models that expect a specific conversation
        format.
        """
        content: list[dict[str, Any]] = [{"text": prompt, "type": "text"}]
        if mm_content is not None:
            if isinstance(mm_content, list):
                content.extend(cast(list[dict[str, Any]], mm_content))
            elif isinstance(mm_content, dict):
                content.append(mm_content)
            else:
                raise TypeError(
                    "Could not process multimodal content"
                )
        return [{"role": "user", "content": content}]

    def load_data(self) -> None:
        """
        Load data from the dataset path into self.data.

        This method must be overridden by subclasses since the method to load
        data will vary depending on the dataset format and source.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        # TODO (jenniferzhao): add support for downloading data
        raise NotImplementedError(
            "load_data must be implemented in subclasses.")

    def get_random_lora_request(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_loras: Optional[int] = None,
        lora_path: Optional[str] = None,
    ) -> tuple[Optional[LoRARequest], AnyTokenizer]:
        """
        Optionally select a random LoRA request and return its associated
        tokenizer.

        This method is used when LoRA parameters are provided.  It randomly
        selects a LoRA based on max_loras and retrieves a cached tokenizer for
        that LoRA if available. Otherwise, it returns the base tokenizer.

        Args:
            tokenizer (PreTrainedTokenizerBase): The base tokenizer to use if no
                LoRA is selected.
            max_loras (Optional[int]): The maximum number of LoRAs available.
                If `None`, LoRA is not used.
            lora_path (Optional[str]): Path to the LoRA parameters on disk.
                If `None`, LoRA is not used.

        Returns:
            A tuple with the following elements:
                - A new [LoRARequest][] (or `None` if not applicable).
                - The tokenizer associated with the LoRA request
                  (or the base tokenizer).
        """
        if max_loras is None or lora_path is None:
            return None, tokenizer

        # Generate a random LoRA ID in the range [1, max_loras].
        lora_id = random.randint(1, max_loras)
        lora_request = LoRARequest(
            lora_name=str(lora_id),
            lora_int_id=lora_id,
            lora_path=lora_path_on_disk(lora_path),
        )
        if lora_id not in lora_tokenizer_cache:
            lora_tokenizer_cache[lora_id] = get_lora_tokenizer(lora_request)
        # Return lora_request and the cached tokenizer if available; otherwise,
        # return the base tokenizer
        return lora_request, lora_tokenizer_cache[lora_id] or tokenizer

    @abstractmethod
    def sample(self, tokenizer: PreTrainedTokenizerBase,
               num_requests: int) -> list[SampleRequest]:
        """
        Abstract method to generate sample requests from the dataset.

        Subclasses must override this method to implement dataset-specific logic
        for generating a list of SampleRequest objects.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used
                for processing the dataset's text.
            num_requests (int): The number of sample requests to generate.

        Returns:
            list[SampleRequest]: A list of sample requests generated from the
            dataset.
        """
        raise NotImplementedError("sample must be implemented in subclasses.")

    def maybe_oversample_requests(self, requests: list[SampleRequest],
                                  num_requests: int) -> None:
        """
        Oversamples the list of requests if its size is less than the desired
        number.

        Args:
            requests (List[SampleRequest]): The current list of sampled
                requests.
            num_requests (int): The target number of requests.
        """
        if len(requests) < num_requests:
            random.seed(self.random_seed)
            additional = random.choices(requests,
                                        k=num_requests - len(requests))
            requests.extend(additional)
            logger.info("Oversampled requests to reach %d total samples.",
                        num_requests)


# -----------------------------------------------------------------------------
# Utility Functions and Global Caches
# -----------------------------------------------------------------------------


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
    skip_min_output_len_check: bool = False,
) -> bool:
    """
    Validate a sequence based on prompt and output lengths.

    Default pruning criteria are copied from the original `sample_hf_requests`
    and `sample_sharegpt_requests` functions in benchmark_serving.py, as well as
    from `sample_requests` in benchmark_throughput.py.
    """
    # Check for invalid conditions
    prompt_too_short = prompt_len < min_len
    output_too_short = (not skip_min_output_len_check) and (output_len
                                                            < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len

    # Return True if none of the invalid conditions are met
    return not (prompt_too_short or output_too_short or prompt_too_long
                or combined_too_long)


@cache
def lora_path_on_disk(lora_path: str) -> str:
    return get_adapter_absolute_path(lora_path)


# Global cache for LoRA tokenizers.
lora_tokenizer_cache: dict[int, AnyTokenizer] = {}


def process_image(image: Any) -> Mapping[str, Any]:
    """
    Process a single image input and return a multimedia content dictionary.

    Supports three input types:

    1. Dictionary with raw image bytes: - Expects a dict with a 'bytes' key
       containing raw image data.  - Loads the bytes as a PIL.Image.Image.

    2. PIL.Image.Image input: - Converts the image to RGB.  - Saves the image as
       a JPEG in memory.  - Encodes the JPEG data as a base64 string.  - Returns
       a dictionary with the image as a base64 data URL.

    3. String input: - Treats the string as a URL or local file path.  -
       Prepends "file://" if the string doesn't start with "http://" or
       "file://".  - Returns a dictionary with the image URL.

    Raises:
        ValueError: If the input is not a supported type.
    """
    if isinstance(image, dict) and 'bytes' in image:
        image = Image.open(BytesIO(image['bytes']))
    if isinstance(image, Image.Image):
        image = convert_image_mode(image, "RGB")
        with io.BytesIO() as image_data:
            image.save(image_data, format="JPEG")
            image_base64 = base64.b64encode(
                image_data.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            },
        }

    if isinstance(image, str):
        image_url = (image if image.startswith(
            ("http://", "file://")) else f"file://{image}")
        return {"type": "image_url", "image_url": {"url": image_url}}

    raise ValueError(f"Invalid image input {image}. Must be a PIL.Image.Image"
                     " or str or dictionary with raw image bytes.")


# -----------------------------------------------------------------------------
# Random Dataset Implementation (Synthetic Data)
# -----------------------------------------------------------------------------



class RandomDataset(BenchmarkDataset):
    """
    Generate synthetic prompts with lengths sampled around targets.

    Behavior
    --------
    For each request, this dataset:
      1) Samples an **inner** input length `L_in` and an output length `L_out`
         independently from
         `[floor(X*(1 - range_ratio)), ceil(X*(1 + range_ratio))]`,
         where `X` is `input_len - tokenizer.num_special_tokens_to_add()` for
         inputs, and `X` is `output_len` for outputs.
         `L_out` is clamped to >= 1.
      2) Builds a single random prefix of length `prefix_len` once per call to
         `sample(...)` and prepends it to every request.
      3) Appends a structured “ramp” sequence of token ids:
         `((offset + index) + arange(L_in)) % vocab_size`.
      4) Performs buffered canonicalization: decode → re-encode without
         special tokens. If canonicalized tokens are shorter than the target
         total (`prefix_len + L_in`), it over-generates a small token-id
         buffer, decodes and re-encodes again up to `max_iters` retries,
         extending the ramp deterministically each time. Once the canonical
         re-encoding length is at least the target, it takes exactly the first
         `target_total` tokens and decodes.

    Notes
    -----
    - No special tokens are added to the prompt by this dataset. The subtraction
      of `tokenizer.num_special_tokens_to_add()` from `input_len` is a budgeting
      choice so downstream code can add specials without exceeding the intended
      total length. If you will not add specials later, set `input_len`
      accordingly.
    - Re-encoding always uses `add_special_tokens=False` and decoding uses
      `clean_up_tokenization_spaces=False` to reduce non-determinism from
      whitespace cleanup.
    - Canonicalization ensures the full sequence round-trips to the same IDs;
      prefixes may not.
    - There is no general guarantee that buffered canonicalization will exactly
      hit the target across every tokenizer configuration or normalization
      setting. It works for typical configurations and falls back to "at most K"
      behavior when disabled.

    TODO
    ----
    - Consider a graph-based canonical subset approach (safe transitions) as a
      more principled solution for exact-length construction across tokenizers.

    Reproducibility
    ---------------
      Uses a per-instance NumPy Generator
      (`np.random.default_rng(self.random_seed)`) for all sampling, providing
      deterministic results without mutating global RNG state.

    Args:
        tokenizer: Hugging Face tokenizer used for encode/decode. Re-encoding
            uses `add_special_tokens=False`.
        num_requests: Number of requests to generate.
        prefix_len: Number of random prefix tokens (shared by all requests in
            this batch).
        range_ratio: Relative half-width (b in [0,1)) of the sampling interval.
            Must be < 1.0.
        input_len: Target input length used to derive the inner ramp length by
            subtracting `tokenizer.num_special_tokens_to_add()`.
        output_len: Target mean output token length used to sample `L_out`.

        Buffered canonicalization kwargs:
        - buffer_ratio (float): Proportion of the target total to over-generate.
          Default 0.15. Set to 0 to disable buffering and revert to "at most K".
        - min_buffer (int): Minimum over-generation tokens. Default 8.
        - max_iters (int): Max extension attempts when still short. Default 2.
        - extra_margin_floor (int): Extra tokens to add per retry. Default 4.

    Returns:
        List[SampleRequest]: each with
            - prompt: Decoded prompt text
            - prompt_len: Actual tokenized length after canonicalization
            - expected_output_len: Sampled `L_out`
    """

    DEFAULT_PREFIX_LEN = 0
    DEFAULT_RANGE_RATIO = 0.0
    DEFAULT_INPUT_LEN = 1024
    DEFAULT_OUTPUT_LEN = 128
    # Buffered canonicalization defaults
    DEFAULT_BUFFER_RATIO = 0.15
    DEFAULT_MIN_BUFFER = 16
    DEFAULT_MAX_ITERS = 4
    DEFAULT_EXTRA_MARGIN_FLOOR = 8

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Use numpy's default_rng for deterministic sampling
        self._np_rng = np.random.default_rng(self.random_seed)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        range_ratio: float = DEFAULT_RANGE_RATIO,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        **kwargs,
    ) -> list[SampleRequest]:
        """
        Generate `num_requests` synthetic prompts and expected output lengths.

        Parameters
        - tokenizer: Hugging Face tokenizer used to convert token IDs to text.
          Special tokens are not added when re-encoding 
          (add_special_tokens=False).
        - num_requests: Number of requests to generate.
        - prefix_len: Fixed number of random prefix tokens prepended to every 
        prompt.
        - range_ratio: Relative half-width of the sampling interval for lengths.
          For a target X, actual lengths are sampled uniformly from the integer
          interval [floor(X*(1 - range_ratio)), ceil(X*(1 + range_ratio))].
          Must be < 1.0.
        - input_len: Target input token length (including `prefix_len`), before
          any special tokens. The final prompt is canonicalized using a buffered
          decode→encode strategy to reach the exact sampled target when
          possible.
        - output_len: Target output token length used to set
          `expected_output_len` per request; not used to generate text.
        - buffer_ratio, min_buffer, max_iters, extra_margin_floor: See class
          docstring for details.

        Returns
        - List of `SampleRequest` with:
          - prompt: Decoded prompt text
          - prompt_len: Final tokenized length of `prompt`
          - expected_output_len: Sampled output length

        Raises
        - AssertionError: if `range_ratio >= 1.0`.
        - ValueError: if computed sampling intervals are invalid (e.g., empty).

        Implementation details
        - To avoid drift caused by tokenizer merges/normalization, the method
          decodes a candidate sequence, re-encodes without special tokens, and
          if under length, over-generates by a small buffer and retries up to a
          bounded number of times. On success, the canonical re-encoding is
          sliced to the exact target token count.
        - Logs the input and output sampling intervals at INFO level.
        """
        input_lens, output_lens, offsets = self.get_text_sampling_params(
            num_requests, range_ratio, input_len, output_len, tokenizer
        )

        # Generate prefix once
        prefix_token_ids = self.get_prefix(tokenizer, prefix_len)
        vocab_size = tokenizer.vocab_size

        # Buffered canonicalization config
        buffer_ratio: float = kwargs.get("buffer_ratio",
                                         self.DEFAULT_BUFFER_RATIO)
        min_buffer: int = kwargs.get("min_buffer", self.DEFAULT_MIN_BUFFER)
        max_iters: int = kwargs.get("max_iters", self.DEFAULT_MAX_ITERS)
        extra_margin_floor: int = kwargs.get(
            "extra_margin_floor", self.DEFAULT_EXTRA_MARGIN_FLOOR)

        total_retry_requests = 0
        total_retries = 0

        requests = []
        for i in range(num_requests):
            (prompt, total_input_len, retries_taken) = \
                self.generate_token_sequence(
                tokenizer=tokenizer,
                prefix_token_ids=prefix_token_ids,
                prefix_len=prefix_len,
                vocab_size=vocab_size,
                input_len=int(input_lens[i]),
                offset=int(offsets[i]),
                index=i,
                buffer_ratio=buffer_ratio,
                min_buffer=min_buffer,
                max_iters=max_iters,
                extra_margin_floor=extra_margin_floor,
            )
            if retries_taken > 0:
                total_retry_requests += 1
                total_retries += retries_taken
            logger.debug(
                "Buffered canonicalization retries=%d final_prompt_len=%d",
                retries_taken,
                total_input_len,
            )
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=total_input_len,
                    expected_output_len=int(output_lens[i]),
                )
            )
        if total_retry_requests:
            share = total_retry_requests / max(1, num_requests)
            logger.debug(
                "Buffered canonicalization: requests_with_retry=%d (%.2f%%), "
                "total_retries=%d",
                total_retry_requests,
                100.0 * share,
                total_retries,
            )
        return requests

    def get_prefix(
        self, tokenizer: PreTrainedTokenizerBase, prefix_len: int
    ) -> list[int]:
        """
        Get the prefix for the dataset.
        """
        if prefix_len <= 0:
            return []
        vocab_size = tokenizer.vocab_size
        # Exclude special token IDs from random draws
        special_ids: set[int] = set(getattr(tokenizer, "all_special_ids", [])
                                    or [])
        allowed_ids = [i for i in range(vocab_size) if i not in special_ids]
        if not allowed_ids:
            raise ValueError(
                "No non-special token IDs available for prefix generation.")
        return self._np_rng.choice(allowed_ids,
                                   size=prefix_len,
                                   replace=True).tolist()

    def get_text_sampling_params(
        self,
        num_requests: int,
        range_ratio: float,
        input_len: int,
        output_len: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the sampling parameters for the dataset.
        """
        # Enforce range_ratio < 1
        assert range_ratio < 1.0, (
            "random_range_ratio must be < 1.0 to ensure a valid sampling range"
        )
        num_special_tokens = int(tokenizer.num_special_tokens_to_add())
        real_input_len = max(0, int(input_len) - num_special_tokens)
        # Bounds use floor for low and ceil for high
        input_low = math.floor(real_input_len * (1 - range_ratio))
        input_high = math.ceil(real_input_len * (1 + range_ratio))
        output_low = math.floor(output_len * (1 - range_ratio))
        output_high = math.ceil(output_len * (1 + range_ratio))
        output_low = max(output_low, 1)

        if input_low > input_high:
            raise ValueError(
                f"Invalid input sampling interval: low={input_low} > high={input_high}"
            )
        if output_low > output_high:
            raise ValueError(
                f"Invalid output sampling interval: low={output_low} > high={output_high}"
            )

        # Add logging for debugging
        logger.info(
            "Sampling input_len from [%s, %s] and output_len from [%s, %s]",
            input_low,
            input_high,
            output_low,
            output_high,
        )

        input_lens = self._np_rng.integers(input_low, input_high + 1,
                                           size=num_requests)
        output_lens = self._np_rng.integers(output_low, output_high + 1,
                                            size=num_requests)
        # Sample offsets excluding special token IDs
        special_ids: set[int] = set(getattr(tokenizer, "all_special_ids", [])
                                    or [])
        allowed_ids = [i for i in range(tokenizer.vocab_size)
                       if i not in special_ids]
        if not allowed_ids:
            raise ValueError(
                "No non-special token IDs available for offset generation.")
        offsets = self._np_rng.choice(allowed_ids, size=num_requests,
                                      replace=True)
        return input_lens, output_lens, offsets

    def generate_token_sequence(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        prefix_token_ids: list[int],
        prefix_len: int,
        vocab_size: int,
        input_len: int,
        offset: int,
        index: int,
        buffer_ratio: float,
        min_buffer: int,
        max_iters: int,
        extra_margin_floor: int,
    ) -> tuple[str, int, int]:
        """
        Construct a candidate sequence and run buffered canonicalization.

        Returns (prompt, total_input_len, retries_taken).
        """
        # Build the deterministic inner sequence by sampling sequentially
        # from the vocabulary. We will extend this sequence in-place if needed.
        base_inner_len = int(input_len)
        inner_seq = (
            (offset + index + np.arange(base_inner_len)) % vocab_size
        ).tolist()

        target_total = prefix_len + base_inner_len

        # Fast path: buffering disabled → current "at most K" behavior.
        if buffer_ratio <= 0.0:
            token_sequence = prefix_token_ids + inner_seq
            prompt = tokenizer.decode(
                token_sequence, clean_up_tokenization_spaces=False)
            re_encoded_sequence = tokenizer.encode(
                prompt,
                add_special_tokens=False,
            )[:target_total]
            prompt = tokenizer.decode(
                re_encoded_sequence, clean_up_tokenization_spaces=False)
            return prompt, len(re_encoded_sequence), 0

        # Initial over-generation buffer
        initial_buffer = max(
            min_buffer, int(round(buffer_ratio * target_total))
        )
        if initial_buffer > 0:
            extension = (
                (
                    offset
                    + index
                    + np.arange(
                        base_inner_len,
                        base_inner_len + initial_buffer,
                    )
                )
                % vocab_size
            ).tolist()
            inner_seq.extend(extension)

        retries_taken = 0
        token_sequence = prefix_token_ids + inner_seq
        prompt = tokenizer.decode(
            token_sequence, clean_up_tokenization_spaces=False
        )
        canonical_ids = tokenizer.encode(prompt, add_special_tokens=False)

        # If still short after initial buffer, extend with extra margins
        # up to max_iters.
        while (len(canonical_ids) < target_total
               and retries_taken < max_iters):
            retries_taken += 1
            extra = max(extra_margin_floor, initial_buffer // 2)
            # Extend deterministically continuing the ramp
            start = (
                base_inner_len + initial_buffer + (retries_taken - 1) * extra
            )
            extension = (
                (offset + index + np.arange(start, start + extra)) % vocab_size
            ).tolist()
            inner_seq.extend(extension)
            token_sequence = prefix_token_ids + inner_seq
            prompt = tokenizer.decode(
                token_sequence, clean_up_tokenization_spaces=False)
            canonical_ids = tokenizer.encode(prompt, add_special_tokens=False)

        # At this point, either canonical_ids is long enough,
        # or we give the best-effort slice. Log warning if we undershoot.
        if len(canonical_ids) < target_total:
            logger.warning(
                "Buffered canonicalization undershoot: got=%d target=%d retries=%d",
                len(canonical_ids),
                target_total,
                retries_taken,
            )
        canonical_ids = canonical_ids[:target_total]

        prompt = tokenizer.decode(canonical_ids,
                                  clean_up_tokenization_spaces=False)
        return prompt, len(canonical_ids), retries_taken


# -----------------------------------------------------------------------------
# MultiModalDataset Implementation
# -----------------------------------------------------------------------------
   

class RandomMultiModalDataset(RandomDataset):
    """
    Random multimodal dataset that generates synthetic images with random
    text.

    This class is used for generating requests with random dimensions and
    number of images. Images are in base64 format; `multi_modal_data` is a
    list of dicts. It shouldn't be used for generating requests on the fly,
    but rather for generating a dataset. Hence, it is not optimized for
    speed, but rather for ease of use.
    """

    IS_MULTIMODAL = True
    DEFAULT_HEIGHT = 224
    DEFAULT_WIDTH = 224
    DEFAULT_NUM_IMAGES = 1
    DEFAULT_NUM_IMAGES_RANGE_RATIO = 0.0
    DEFAULT_DIMENSION_RANGE_RATIO = 0.0
    DEFAULT_ENABLE_MULTIMODAL_CHAT = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


    def generate_synthetic_image(self, width: int, height: int) -> Image.Image:
        """Generate synthetic PIL image with random RGB values."""
        random_pixels = self._np_rng.integers(
            0,
            256,
            (height, width, 3),
            dtype=np.uint8,
        )
        return Image.fromarray(random_pixels)

    def get_image_sampling_params(
        self,
        num_images_range_ratio: float,
        dimension_range_ratio: float,
        width: int,
        height: int,
        num_images: int,
    ) -> tuple[int, int, int, int, int, int]:
        """
        Get the sampling parameters for the image dimensions.
        """
        # Enforce num_images_range_ratio < 1
        assert num_images_range_ratio < 1.0, (
            "num_images_range_ratio must be < 1.0 to ensure a valid sampling "
            "range"
        )
        max_num_images = int(num_images * (1 + num_images_range_ratio))
        # ensure min num images is zero
        min_num_images = max(int(num_images * (1 - num_images_range_ratio)), 0)
        # Enforce dimension_range_ratio < 1
        assert dimension_range_ratio < 1.0, (
            "dimension_range_ratio must be < 1.0 to ensure a valid sampling "
            "range"
        )
        min_width = int(width * (1 - dimension_range_ratio))
        max_width = int(width * (1 + dimension_range_ratio))
        min_height = int(height * (1 - dimension_range_ratio))
        max_height = int(height * (1 + dimension_range_ratio))
        return (
            min_num_images,
            max_num_images,
            min_width,
            max_width,
            min_height,
            max_height,
        )

    def get_image_dimensions_iterator(
        self,
        min_num_images: int,
        max_num_images: int,
        min_width: int,
        max_width: int,
        min_height: int,
        max_height: int,
    ) -> Iterator[tuple[int, int]]:
        """
        Iterator over the image dimensions for each request
        whose size is between min_num_images and max_num_images.
        """
        request_num_images = int(
            self._np_rng.integers(min_num_images, max_num_images + 1)
        )
        for _ in range(request_num_images):
            yield (
                int(self._np_rng.integers(min_width, max_width + 1)),
                int(self._np_rng.integers(min_height, max_height + 1)),
            )

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        prefix_len: int = RandomDataset.DEFAULT_PREFIX_LEN,
        range_ratio: float = RandomDataset.DEFAULT_RANGE_RATIO,
        input_len: int = RandomDataset.DEFAULT_INPUT_LEN,
        output_len: int = RandomDataset.DEFAULT_OUTPUT_LEN,
        num_images: int = DEFAULT_NUM_IMAGES,
        num_images_range_ratio: float = DEFAULT_NUM_IMAGES_RANGE_RATIO,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        dimension_range_ratio: float = DEFAULT_DIMENSION_RANGE_RATIO,
        enable_multimodal_chat: bool = DEFAULT_ENABLE_MULTIMODAL_CHAT,
        **kwargs,
    ) -> list[SampleRequest]:
        """
        Standard sample method compatible with serve.py and other datasets.
        Returns OpenAI API format for compatibility with serve.py.

        Args:
            tokenizer: The tokenizer to use for processing
            num_requests: Number of requests to generate
            width: Image width in pixels
            height: Image height in pixels
            num_images: Number of images per request
            num_images_range_ratio: Relative half-width of the sampling
                interval for number of images.
            dimension_range_ratio: Relative half-width of the sampling
                interval for image dimensions.
            enable_multimodal_chat: Whether to apply multimodal chat
                transformation
            **kwargs: Additional arguments passed to parent sample method

        Returns:
            List of SampleRequest objects with properly formatted OpenAI
            multimodal data.
        """
        input_lens, output_lens, offsets = self.get_text_sampling_params(
            num_requests, range_ratio, input_len, output_len, tokenizer
        )

        (
            min_num_images,
            max_num_images,
            min_width,
            max_width,
            min_height,
            max_height,
        ) = self.get_image_sampling_params(
            num_images_range_ratio,
            dimension_range_ratio,
            width,
            height,
            num_images,
        )

        # Generate prefix once
        prefix_token_ids = self.get_prefix(tokenizer, prefix_len)
        vocab_size = tokenizer.vocab_size
        # Add synthetic images to each request
        mm_requests = []
        for i in range(num_requests):
            prompt: str | list[dict[str, Any]] = ""
            # Reuse RandomDataset buffered canonicalization settings
            buffer_ratio: float = kwargs.get("buffer_ratio",
                                             self.DEFAULT_BUFFER_RATIO)
            min_buffer: int = kwargs.get("min_buffer",
                                         self.DEFAULT_MIN_BUFFER)
            max_iters: int = kwargs.get("max_iters",
                                        self.DEFAULT_MAX_ITERS)
            extra_margin_floor: int = kwargs.get(
                "extra_margin_floor", self.DEFAULT_EXTRA_MARGIN_FLOOR)

            prompt, total_input_len, _retries = self.generate_token_sequence(
                tokenizer=tokenizer,
                prefix_token_ids=prefix_token_ids,
                prefix_len=prefix_len,
                vocab_size=vocab_size,
                input_len=int(input_lens[i]),
                offset=int(offsets[i]),
                index=i,
                buffer_ratio=buffer_ratio,
                min_buffer=min_buffer,
                max_iters=max_iters,
                extra_margin_floor=extra_margin_floor,
            )
            # Get image dimension iterator for a given request
            image_dimensions_iterator = self.get_image_dimensions_iterator(
                min_num_images,
                max_num_images,
                min_width,
                max_width,
                min_height,
                max_height,
            )
            # Create synthetic images
            # The process_image returns
            # {"type": "image_input", "image_url": f"{base64_image}"}
            # This follows the OpenAI API chat completions
            # https://github.com/openai/openai-python
            mm_content = cast(list[dict[str, Any]], [
                process_image(
                    self.generate_synthetic_image(width, height)
                )
                for width, height in image_dimensions_iterator
            ])
            if enable_multimodal_chat:
                prompt = self.apply_multimodal_chat_transformation(
                    prompt, mm_content)
            mm_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=total_input_len,
                    expected_output_len=int(output_lens[i]),
                    multi_modal_data=mm_content,
                )
            )
        return mm_requests



# -----------------------------------------------------------------------------
# ShareGPT Dataset Implementation
# -----------------------------------------------------------------------------


class ShareGPTDataset(BenchmarkDataset):
    """
    Implements the ShareGPT dataset.  Loads data from a JSON file and generates
    sample requests based on conversation turns.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = json.load(f)
        # Filter entries with at least two conversation turns.
        self.data = [
            entry for entry in self.data
            if "conversations" in entry and len(entry["conversations"]) >= 2
        ]
        random.seed(self.random_seed)
        random.shuffle(self.data)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        lora_path: Optional[str] = None,
        max_loras: Optional[int] = None,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        samples: list = []
        for entry in self.data:
            if len(samples) >= num_requests:
                break
            prompt, completion = (
                entry["conversations"][0]["value"],
                entry["conversations"][1]["value"],
            )

            lora_request, tokenizer = self.get_random_lora_request(
                tokenizer=tokenizer, max_loras=max_loras, lora_path=lora_path)
            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            new_output_len = (len(completion_ids)
                              if output_len is None else output_len)
            if not is_valid_sequence(prompt_len,
                                     new_output_len,
                                     skip_min_output_len_check=output_len
                                     is not None):
                continue
            if enable_multimodal_chat:
                prompt = self.apply_multimodal_chat_transformation(
                    prompt, None)
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=new_output_len,
                    lora_request=lora_request,
                ))
        self.maybe_oversample_requests(samples, num_requests)
        return samples


def add_dataset_parser(parser: FlexibleArgumentParser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=[
            "sharegpt",
            "burstgpt",
            "sonnet",
            "random",
            "random-mm",
            "hf",
            "custom",
        ],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Do not load the dataset in streaming mode.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the sharegpt/sonnet dataset. "
        "Or the huggingface dataset ID if using HF dataset.",
    )

    # group for dataset specific arguments
    custom_group = parser.add_argument_group("custom dataset options")
    custom_group.add_argument(
        "--custom-output-len",
        type=int,
        default=256,
        help=
        "Number of output tokens per request, used only for custom dataset.",
    )
    custom_group.add_argument(
        "--custom-skip-chat-template",
        action="store_true",
        help=
        "Skip applying chat template to prompt, used only for custom dataset.",
    )

    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help=
        "Number of input tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help=
        "Number of output tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help=
        "Number of prefix tokens per request, used only for sonnet dataset.",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.",
    )

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help=
        "Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help=
        "Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for sampling input/output length, "
        "used only for random sampling. Must be in the range [0, 1) to define "
        "a symmetric sampling range"
        "[length * (1 - range_ratio), length * (1 + range_ratio)].",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help=("Number of fixed prefix tokens before the random context "
              "in a request. "
              "The total input length is the sum of `random-prefix-len` and "
              "a random "
              "context length sampled from [input_len * (1 - range_ratio), "
              "input_len * (1 + range_ratio)]."),
    )

    # random multimodal dataset options
    random_mm_group = parser.add_argument_group(
        "random multimodal dataset options")
    random_mm_group.add_argument(
        "--random-mm-input-len",
        type=int,
        default=RandomDataset.DEFAULT_INPUT_LEN,
        help=(
            "Number of input tokens per request, used only for random-mm "
            "sampling."
        ),
    )
    random_mm_group.add_argument(
        "--random-mm-output-len",
        type=int,
        default=RandomDataset.DEFAULT_OUTPUT_LEN,
        help=(
            "Number of output tokens per request, used only for random-mm "
            "sampling."
        ),
    )
    random_mm_group.add_argument(
        "--random-mm-range-ratio",
        type=float,
        default=RandomDataset.DEFAULT_RANGE_RATIO,
        help=(
            "Range ratio for sampling input/output length, used only for "
            "random-mm sampling. Must be in the range [0, 1) to define a "
            "symmetric sampling range [length * (1 - range_ratio), length * "
            "(1 + range_ratio)]."
        ),
    )
    random_mm_group.add_argument(
        "--random-mm-prefix-len",
        type=int,
        default=RandomDataset.DEFAULT_PREFIX_LEN,
        help=(
            "Number of fixed prefix tokens before the random context in a "
            "request for random-mm."
        ),
    )
    random_mm_group.add_argument(
        "--random-mm-width",
        type=int,
        default=RandomMultiModalDataset.DEFAULT_WIDTH,
        help="Image width in pixels per image for random-mm dataset.",
    )
    random_mm_group.add_argument(
        "--random-mm-height",
        type=int,
        default=RandomMultiModalDataset.DEFAULT_HEIGHT,
        help="Image height in pixels per image for random-mm dataset.",
    )
    random_mm_group.add_argument(
        "--random-mm-num-images",
        type=int,
        default=RandomMultiModalDataset.DEFAULT_NUM_IMAGES,
        help="Number of images per request for random-mm dataset.",
    )
    random_mm_group.add_argument(
        "--random-mm-num-images-range-ratio",
        type=float,
        default=RandomMultiModalDataset.DEFAULT_NUM_IMAGES_RANGE_RATIO,
        help=(
            "Relative half-width of the sampling interval for number of "
            "images in random-mm dataset. Must be in [0, 1)."
        ),
    )
    random_mm_group.add_argument(
        "--random-mm-dimension-range-ratio",
        type=float,
        default=RandomMultiModalDataset.DEFAULT_DIMENSION_RANGE_RATIO,
        help=(
            "Relative half-width of the sampling interval for image "
            "dimensions in random-mm dataset. Must be in [0, 1)."
        ),
    )


    hf_group = parser.add_argument_group("hf dataset options")
    hf_group.add_argument("--hf-subset",
                          type=str,
                          default=None,
                          help="Subset of the HF dataset.")
    hf_group.add_argument("--hf-split",
                          type=str,
                          default=None,
                          help="Split of the HF dataset.")
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths "
        "from the sampled HF dataset.",
    )


def get_samples(args, tokenizer) -> list[SampleRequest]:
    if args.dataset_name == "custom":
        dataset = CustomDataset(dataset_path=args.dataset_path)
        input_requests = dataset.sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.custom_output_len,
            skip_chat_template=args.custom_skip_chat_template,
        )

    elif args.dataset_name == "sonnet":
        dataset = SonnetDataset(dataset_path=args.dataset_path)
        # For the "sonnet" dataset, formatting depends on the backend.
        if args.endpoint_type == "openai-chat":
            input_requests = dataset.sample(
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=False,
            )
        else:
            assert tokenizer.chat_template or tokenizer.default_chat_template, (
                "Tokenizer/model must have chat template for sonnet dataset.")
            input_requests = dataset.sample(
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=True,
            )

    elif args.dataset_name == "hf":
        # all following datasets are implemented from the
        # HuggingFaceDataset base class
        if args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = VisionArenaDataset
            args.hf_split = "train"
            args.hf_subset = None
        elif args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = InstructCoderDataset
            args.hf_split = "train"
        elif args.dataset_path in MTBenchDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = MTBenchDataset
            args.hf_split = "train"
        elif args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = ConversationDataset
        elif args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS:
            dataset_class = AIMODataset
            args.hf_split = "train"
        elif args.dataset_path in NextEditPredictionDataset.SUPPORTED_DATASET_PATHS:  # noqa: E501
            dataset_class = NextEditPredictionDataset
            args.hf_split = "train"
        elif args.dataset_path in ASRDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = ASRDataset
            args.hf_split = "train"
        elif args.dataset_path in MLPerfDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = MLPerfDataset
            args.hf_split = "train"
        else:
            supported_datasets = set([
                dataset_name for cls in HuggingFaceDataset.__subclasses__()
                for dataset_name in cls.SUPPORTED_DATASET_PATHS
            ])
            raise ValueError(
                f"Unsupported dataset path: {args.dataset_path}. "
                "Huggingface dataset only supports dataset_path"
                f" from one of following: {supported_datasets}. "
                "Please consider contributing if you would "
                "like to add support for additional dataset formats.")

        if dataset_class.IS_MULTIMODAL and args.endpoint_type not in [
                "openai-chat",
                "openai-audio",
        ]:
            # multi-modal benchmark is only available on OpenAI Chat backend.
            raise ValueError(
                "Multi-modal content is only supported on 'openai-chat' and "
                "'openai-audio' backend.")
        input_requests = dataset_class(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
            random_seed=args.seed,
            no_stream=args.no_stream,
        ).sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.hf_output_len,
        )

    else:
        # For datasets that follow a similar structure, use a mapping.
        dataset_mapping = {
            "sharegpt":
            lambda: ShareGPTDataset(random_seed=args.seed,
                                    dataset_path=args.dataset_path).sample(
                                        tokenizer=tokenizer,
                                        num_requests=args.num_prompts,
                                        output_len=args.sharegpt_output_len,
                                    ),
            "burstgpt":
            lambda: BurstGPTDataset(random_seed=args.seed,
                                    dataset_path=args.dataset_path).
            sample(tokenizer=tokenizer, num_requests=args.num_prompts),
            "random":
            lambda: RandomDataset(random_seed=args.seed,
                                  dataset_path=args.dataset_path).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.random_prefix_len,
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                range_ratio=args.random_range_ratio,
            ),
            "random-mm":
            lambda: RandomMultiModalDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.random_mm_prefix_len,
                range_ratio=args.random_mm_range_ratio,
                input_len=args.random_mm_input_len,
                output_len=args.random_mm_output_len,
                width=args.random_mm_width,
                height=args.random_mm_height,
                num_images=args.random_mm_num_images,
                num_images_range_ratio=args.random_mm_num_images_range_ratio,
                dimension_range_ratio=args.random_mm_dimension_range_ratio,
            ),
        }

        try:
            # Enforce endpoint compatibility for multimodal datasets.
            if args.dataset_name == "random-mm" and args.endpoint_type not in [
                    "openai-chat"]:
                raise ValueError(
                    "Multi-modal content (images) is only supported on "
                    "'openai-chat' backend."
                )
            input_requests = dataset_mapping[args.dataset_name]()
        except KeyError as err:
            raise ValueError(f"Unknown dataset: {args.dataset_name}") from err

    return input_requests


# -----------------------------------------------------------------------------
# Custom Dataset Implementation
# -----------------------------------------------------------------------------


class CustomDataset(BenchmarkDataset):
    """
    Implements the Custom dataset.  Loads data from a JSONL file and generates
    sample requests based on conversation turns. E.g.,
    ```
    {"prompt": "What is the capital of India?"}
    {"prompt": "What is the capital of Iran?"}
    {"prompt": "What is the capital of China?"}
    ```
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        # self.data will be a list of dictionaries
        # e.g., [{"prompt": "What is the capital of India?"}, ...]
        # This will be the standardized format which load_data()
        # has to convert into depending on the filetype of dataset_path.
        # sample() will assume this standardized format of self.data
        self.data = []

        # Load the JSONL file
        if self.dataset_path.endswith(".jsonl"):
            jsonl_data = pd.read_json(path_or_buf=self.dataset_path,
                                      lines=True)

            # check if the JSONL file has a 'prompt' column
            if "prompt" not in jsonl_data.columns:
                raise ValueError("JSONL file must contain a 'prompt' column.")

            # Convert each row to a dictionary and append to self.data
            # This will convert the DataFrame to a list of dictionaries
            # where each dictionary corresponds to a row in the DataFrame.
            # This is the standardized format we want for self.data
            for _, row in jsonl_data.iterrows():
                self.data.append(row.to_dict())
        else:
            raise NotImplementedError(
                "Only JSONL format is supported for CustomDataset.")

        random.seed(self.random_seed)
        random.shuffle(self.data)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        lora_path: Optional[str] = None,
        max_loras: Optional[int] = None,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        skip_chat_template: bool = False,
        **kwargs,
    ) -> list:
        sampled_requests = []
        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            prompt = item["prompt"]

            # apply template
            if not skip_chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{
                        "role": "user",
                        "content": prompt
                    }],
                    add_generation_prompt=True,
                    tokenize=False,
                )

            prompt_len = len(tokenizer(prompt).input_ids)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                ))
        self.maybe_oversample_requests(sampled_requests, num_requests)

        return sampled_requests


# -----------------------------------------------------------------------------
# Sonnet Dataset Implementation
# -----------------------------------------------------------------------------


class SonnetDataset(BenchmarkDataset):
    """
    Simplified implementation of the Sonnet dataset.  Loads poem lines from a
    text file and generates sample requests.  Default values here copied from
    `benchmark_serving.py` for the sonnet dataset.
    """

    DEFAULT_PREFIX_LEN = 200
    DEFAULT_INPUT_LEN = 550
    DEFAULT_OUTPUT_LEN = 150

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if not self.dataset_path:
            raise ValueError("dataset_path must be provided.")
        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = f.readlines()

    def sample(
        self,
        tokenizer,
        num_requests: int,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        return_prompt_formatted: bool = False,
        **kwargs,
    ) -> list:
        # Calculate average token length for a poem line.
        tokenized_lines = [tokenizer(line).input_ids for line in self.data]
        avg_len = sum(len(tokens)
                      for tokens in tokenized_lines) / len(tokenized_lines)

        # Build the base prompt.
        base_prompt = "Pick as many lines as you can from these poem lines:\n"
        base_msg = [{"role": "user", "content": base_prompt}]
        base_fmt = tokenizer.apply_chat_template(base_msg,
                                                 add_generation_prompt=True,
                                                 tokenize=False)
        base_offset = len(tokenizer(base_fmt).input_ids)
        if input_len <= base_offset:
            raise ValueError(
                f"'input_len' must be higher than the base prompt length "
                f"({base_offset}).")

        # Determine how many poem lines to use.
        num_input_lines = round((input_len - base_offset) / avg_len)
        num_prefix_lines = max(round((prefix_len - base_offset) / avg_len), 0)
        prefix_lines = self.data[:num_prefix_lines]

        samples = []
        while len(samples) < num_requests:
            extra_lines = random.choices(self.data,
                                         k=num_input_lines - num_prefix_lines)
            prompt = f"{base_prompt}{''.join(prefix_lines + extra_lines)}"
            msg = [{"role": "user", "content": prompt}]
            prompt_formatted = tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False)
            prompt_len = len(tokenizer(prompt_formatted).input_ids)
            if prompt_len <= input_len:
                samples.append(
                    SampleRequest(
                        prompt=prompt_formatted
                        if return_prompt_formatted else prompt,
                        prompt_len=prompt_len,
                        expected_output_len=output_len,
                    ))
        return samples


# -----------------------------------------------------------------------------
# BurstGPT Dataset Implementation
# -----------------------------------------------------------------------------


class BurstGPTDataset(BenchmarkDataset):
    """
    Implements the BurstGPT dataset.  Loads data from a CSV file and generates
    sample requests based on synthetic prompt generation. Only rows with Model
    "GPT-4" and positive response tokens are used.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self, ):
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        df = pd.read_csv(self.dataset_path)
        # Filter to keep only GPT-4 rows.
        gpt4_df = df[df["Model"] == "GPT-4"]
        # Remove failed requests (where Response tokens is 0 or less).
        gpt4_df = gpt4_df[gpt4_df["Response tokens"] > 0]
        # Sample the desired number of rows.
        self.data = gpt4_df

    def _sample_loaded_data(self, num_requests: int) -> list:
        if num_requests <= len(self.data):
            data = self.data.sample(n=num_requests,
                                    random_state=self.random_seed)
        else:
            data = self.data.sample(
                n=num_requests,
                random_state=self.random_seed,
                replace=True,
            )
        # Convert the dataframe to a list of lists.
        return data.values.tolist()

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        max_loras: Optional[int] = None,
        lora_path: Optional[str] = None,
        **kwargs,
    ) -> list[SampleRequest]:
        samples = []
        data = self._sample_loaded_data(num_requests=num_requests)
        for i in range(num_requests):
            input_len = int(data[i][2])
            output_len = int(data[i][3])
            lora_req, tokenizer = self.get_random_lora_request(
                tokenizer=tokenizer, max_loras=max_loras, lora_path=lora_path)
            vocab_size = tokenizer.vocab_size
            # Generate a synthetic prompt: a list of token IDs computed as (i +
            # j) modulo vocab_size.
            token_ids = [(i + j) % vocab_size for j in range(input_len)]
            prompt = tokenizer.decode(token_ids)
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=input_len,
                    expected_output_len=output_len,
                    lora_request=lora_req,
                ))
        return samples


# -----------------------------------------------------------------------------
# HuggingFace Dataset Base Implementation
# -----------------------------------------------------------------------------
class HuggingFaceDataset(BenchmarkDataset):
    """Base class for datasets hosted on HuggingFace."""

    SUPPORTED_DATASET_PATHS: Union[set[str], dict[str, Callable]] = set()

    def __init__(
        self,
        dataset_path: str,
        dataset_split: str,
        no_stream: bool = False,
        dataset_subset: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(dataset_path=dataset_path, **kwargs)

        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset
        self.load_stream = not no_stream
        self.load_data()

    def load_data(self) -> None:
        """Load data from HuggingFace datasets."""
        self.data = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
            streaming=self.load_stream,
        )
        self.data = self.data.shuffle(seed=self.random_seed)


# -----------------------------------------------------------------------------
# Conversation Dataset Implementation
# -----------------------------------------------------------------------------


class ConversationDataset(HuggingFaceDataset):
    """Dataset for conversation data with multimodal support."""
    SUPPORTED_DATASET_PATHS = {
        'lmms-lab/LLaVA-OneVision-Data', 'Aeala/ShareGPT_Vicuna_unfiltered'
    }
    IS_MULTIMODAL = True

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               num_requests: int,
               output_len: Optional[int] = None,
               enable_multimodal_chat: bool = False,
               **kwargs) -> list:
        # Filter examples with at least 2 conversations
        filtered_data = self.data.filter(
            lambda x: len(x["conversations"]) >= 2)
        sampled_requests = []
        dynamic_output = output_len is None

        for item in filtered_data:
            if len(sampled_requests) >= num_requests:
                break
            conv = item["conversations"]
            prompt, completion = conv[0]["value"], conv[1]["value"]

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            completion_len = len(completion_ids)
            output_len = completion_len if dynamic_output else output_len
            assert isinstance(output_len, int) and output_len > 0
            if dynamic_output and not is_valid_sequence(
                    prompt_len, completion_len):
                continue
            mm_content = process_image(
                item["image"]) if "image" in item else None
            if enable_multimodal_chat:
                # Note: when chat is enabled the request prompt_len is no longer
                # accurate and we will be using request output to count the
                # actual prompt len and output len
                prompt = self.apply_multimodal_chat_transformation(
                    prompt, mm_content)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
                ))
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests


# -----------------------------------------------------------------------------
# Vision Arena Dataset Implementation
# -----------------------------------------------------------------------------


class VisionArenaDataset(HuggingFaceDataset):
    """
    Vision Arena Dataset.
    """

    DEFAULT_OUTPUT_LEN = 128
    SUPPORTED_DATASET_PATHS = {
        "lmarena-ai/VisionArena-Chat":
        lambda x: x["conversation"][0][0]["content"],
        "lmarena-ai/vision-arena-bench-v0.1":
        lambda x: x["turns"][0][0]["content"]
    }
    IS_MULTIMODAL = True

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        output_len = (output_len
                      if output_len is not None else self.DEFAULT_OUTPUT_LEN)
        sampled_requests = []
        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            parser_fn = self.SUPPORTED_DATASET_PATHS.get(self.dataset_path)
            if parser_fn is None:
                raise ValueError(
                    f"Unsupported dataset path: {self.dataset_path}")
            prompt = parser_fn(item)
            mm_content = process_image(item["images"][0])
            prompt_len = len(tokenizer(prompt).input_ids)
            if enable_multimodal_chat:
                # Note: when chat is enabled the request prompt_len is no longer
                # accurate and we will be using request output to count the
                # actual prompt len
                prompt = self.apply_multimodal_chat_transformation(
                    prompt, mm_content)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
                ))
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests


# -----------------------------------------------------------------------------
# Instruct Coder Dataset Implementation
# -----------------------------------------------------------------------------


class InstructCoderDataset(HuggingFaceDataset):
    """
    InstructCoder Dataset.
    https://huggingface.co/datasets/likaixin/InstructCoder

    InstructCoder is the dataset designed for general code editing.  It consists
    of 114,239 instruction-input-output triplets, and covers multiple distinct
    code editing scenario.
    """

    DEFAULT_OUTPUT_LEN = 200  # this is the average default output length
    SUPPORTED_DATASET_PATHS = {
        "likaixin/InstructCoder",
    }

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               num_requests: int,
               output_len: Optional[int] = None,
               enable_multimodal_chat: bool = False,
               **kwargs) -> list:
        output_len = (output_len
                      if output_len is not None else self.DEFAULT_OUTPUT_LEN)
        sampled_requests = []
        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            prompt = f"{item['input']}\n\n{item['instruction']} Just output \
            the code, do not include any explanation."

            # apply template
            prompt = tokenizer.apply_chat_template(
                [{
                    "role": "user",
                    "content": prompt
                }],
                add_generation_prompt=True,
                tokenize=False,
            )

            prompt_len = len(tokenizer(prompt).input_ids)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                ))
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests


# -----------------------------------------------------------------------------
# MT-Bench Dataset Implementation
# -----------------------------------------------------------------------------


class MTBenchDataset(HuggingFaceDataset):
    """
    MT-Bench Dataset.
    https://huggingface.co/datasets/philschmid/mt-bench

    We create a single turn dataset for MT-Bench.
    This is similar to Spec decoding benchmark setup in vLLM
    https://github.com/vllm-project/vllm/blob/9d98ab5ec/examples/offline_inference/eagle.py#L14-L18
    """  # noqa: E501

    DEFAULT_OUTPUT_LEN = 256  # avg len used in SD bench in vLLM
    SUPPORTED_DATASET_PATHS = {
        "philschmid/mt-bench",
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        output_len = (output_len
                      if output_len is not None else self.DEFAULT_OUTPUT_LEN)
        sampled_requests = []

        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            prompt = item["turns"][0]

            # apply template
            prompt = tokenizer.apply_chat_template(
                [{
                    "role": "user",
                    "content": prompt
                }],
                add_generation_prompt=True,
                tokenize=False,
            )

            prompt_len = len(tokenizer(prompt).input_ids)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                ))
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests


# -----------------------------------------------------------------------------
# AIMO Dataset Implementation
# -----------------------------------------------------------------------------


class AIMODataset(HuggingFaceDataset):
    """
    Dataset class for processing a AIMO dataset with reasoning questions.
    """
    SUPPORTED_DATASET_PATHS = {
        "AI-MO/aimo-validation-aime", "AI-MO/NuminaMath-1.5",
        "AI-MO/NuminaMath-CoT"
    }

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               num_requests: int,
               output_len: Optional[int] = None,
               **kwargs) -> list:
        sampled_requests = []
        dynamic_output = output_len is None

        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            prompt, completion = item['problem'], item["solution"]

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            completion_len = len(completion_ids)
            output_len = completion_len if dynamic_output else output_len
            assert isinstance(output_len, int) and output_len > 0
            if dynamic_output and not is_valid_sequence(prompt_len,
                                                        completion_len,
                                                        max_prompt_len=2048,
                                                        max_total_len=32000):
                continue
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=None,
                ))
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests


# -----------------------------------------------------------------------------
# Next Edit Prediction Dataset Implementation
# -----------------------------------------------------------------------------


zeta_prompt = """### Instruction:
You are a code completion assistant and your task is to analyze user edits and then rewrite an excerpt that the user provides, suggesting the appropriate edits within the excerpt, taking into account the cursor location.

### User Edits:

{}

### User Excerpt:

{}

### Response:

""" # noqa: E501


def _format_zeta_prompt(
        sample: dict,
        original_start_marker: str = "<|editable_region_start|>") -> dict:
    """Format the zeta prompt for the Next Edit Prediction (NEP) dataset.

    This function formats examples from the NEP dataset
    into prompts and expected outputs. It could be
    further extended to support more NEP datasets.

    Args:
        sample: The dataset sample containing events,
            inputs, and outputs.
        original_start_marker: The marker indicating the
            start of the editable region. Defaults to
            "<|editable_region_start|>".

    Returns:
        A dictionary with the formatted prompts and expected outputs.
    """
    events = sample["events"]
    input = sample["input"]
    output = sample["output"]
    prompt = zeta_prompt.format(events, input)

    # following the original implementation, extract the focused region
    # from the raw output
    output_start_index = output.find(original_start_marker)
    output_focused_region = output[output_start_index:]
    expected_output = output_focused_region

    return {"prompt": prompt, "expected_output": expected_output}


class NextEditPredictionDataset(HuggingFaceDataset):
    """
    Dataset class for processing a Next Edit Prediction dataset.
    """

    SUPPORTED_DATASET_PATHS = {
        "zed-industries/zeta",
    }
    MAPPING_PROMPT_FUNCS = {
        "zed-industries/zeta": _format_zeta_prompt,
    }

    def sample(self, tokenizer: PreTrainedTokenizerBase, num_requests: int,
               **kwargs):
        formatting_prompt_func = self.MAPPING_PROMPT_FUNCS.get(
            self.dataset_path)
        if formatting_prompt_func is None:
            raise ValueError(f"Unsupported dataset path: {self.dataset_path}")
        samples = []
        for sample in self.data:
            sample = formatting_prompt_func(sample)
            samples.append(
                SampleRequest(
                    prompt=sample["prompt"],
                    prompt_len=len(tokenizer(sample["prompt"]).input_ids),
                    expected_output_len=len(
                        tokenizer(sample["expected_output"]).input_ids),
                ))
            if len(samples) >= num_requests:
                break
        self.maybe_oversample_requests(samples, num_requests)
        return samples


# -----------------------------------------------------------------------------
# ASR Dataset Implementation
# -----------------------------------------------------------------------------


class ASRDataset(HuggingFaceDataset):
    """
    Dataset class for processing a ASR dataset for transcription.
    Tested on the following set:

    +----------------+----------------------------------------+--------------------------+-----------------------------+
    | Dataset        | Domain                                 | Speaking Style           | hf-subset                   |
    +----------------+----------------------------------------+--------------------------+-----------------------------+
    | TED-LIUM       | TED talks                              | Oratory                  | release1, release2, release3|
    |                |                                        |                          | release3-speaker-adaptation |
    | VoxPopuli      | European Parliament                    | Oratory                  | en, de, it, fr,  ...        |
    | LibriSpeech    | Audiobook                              | Narrated                 | "LIUM/tedlium"              |
    | GigaSpeech     | Audiobook, podcast, YouTube            | Narrated, spontaneous    | xs, s, m, l, xl, dev, test  |
    | SPGISpeech     | Financial meetings                     | Oratory, spontaneous     | S, M, L, dev, test          |
    | AMI            | Meetings                               | Spontaneous              | ihm, sdm                    |
    +----------------+----------------------------------------+--------------------------+-----------------------------+

    """  # noqa: E501

    SUPPORTED_DATASET_PATHS = {
        "openslr/librispeech_asr",
        "facebook/voxpopuli",
        "LIUM/tedlium",
        "edinburghcstr/ami",
        "speechcolab/gigaspeech",
        "kensho/spgispeech",
    }

    DEFAULT_OUTPUT_LEN = 128
    IS_MULTIMODAL = True

    # TODO Whisper-specific. Abstract interface when more models are supported.
    TRANSCRIPTION_PREAMBLE = (
        "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>")
    skip_long_audios: bool = True

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        **kwargs,
    ) -> list:
        output_len = (output_len
                      if output_len is not None else self.DEFAULT_OUTPUT_LEN)
        prompt = ASRDataset.TRANSCRIPTION_PREAMBLE
        prompt_len = len(tokenizer(prompt).input_ids)
        sampled_requests = []
        skipped = 0
        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            audio = item["audio"]
            y, sr = audio["array"], audio["sampling_rate"]
            duration_s = librosa.get_duration(y=y, sr=sr)
            # Whisper max supported duration
            if self.skip_long_audios and duration_s > 30:
                skipped += 1
                continue

            mm_content = {"audio": (y, sr)}
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
                ))
        if skipped:
            logger.warning(
                "%d samples discarded from dataset due to"
                " their length being greater than"
                " what Whisper supports.",
                skipped,
            )
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests


# -----------------------------------------------------------------------------
# MLPerf Dataset Implementation
# -----------------------------------------------------------------------------


class MLPerfDataset(HuggingFaceDataset):
    """
    MLPerf Inference Dataset.

    Dataset on HF:
    https://huggingface.co/datasets/mgoin/mlperf-inference-llama2-data
    https://huggingface.co/datasets/mgoin/mlperf-inference-llama3.1-data

    Each record contains:
      - "system_prompt": system role instruction.
      - "question": user question.
      - "output": reference answer.

    We combine the system prompt and question into a chat-formatted prompt
    (using the tokenizer's chat template) and set the expected output length to
    the tokenized length of the provided reference answer.
    """

    SUPPORTED_DATASET_PATHS = {
        "mgoin/mlperf-inference-llama2-data",
        "mgoin/mlperf-inference-llama3.1-data",
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        **kwargs,
    ) -> list[SampleRequest]:
        # Force dynamic output length based on reference completion.
        dynamic_output = output_len is None
        sampled_requests: list[SampleRequest] = []

        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break

            system_prompt = item["system_prompt"]
            question = item["question"]
            reference_answer = item["output"]

            # Build chat-style prompt using tokenizer template, if available.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            prompt_formatted = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompt_len = len(tokenizer(prompt_formatted).input_ids)

            # Determine output length from reference answer tokens.
            ref_out_len = len(
                tokenizer(reference_answer, add_special_tokens=False).input_ids
            )
            expected_output_len = ref_out_len if dynamic_output else output_len

            # Validate sequence lengths.
            if not is_valid_sequence(prompt_len, expected_output_len):
                continue

            sampled_requests.append(
                SampleRequest(
                    prompt=prompt_formatted,
                    prompt_len=prompt_len,
                    expected_output_len=expected_output_len,
                )
            )

        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests
