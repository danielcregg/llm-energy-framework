"""Layer 2: Inference Execution Engine — model loading and standardised workloads.

Loads any Hugging Face model with configurable precision (fp16, int8, int4),
runs the standard benchmark prompt set, and returns precise token counts
and timing information.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "benchmark_prompts.json"


@dataclass
class InferenceResult:
    """Results from a single inference run."""

    prompt_id: str
    task_type: str
    prompt_tokens: int
    output_tokens: int
    generation_time_seconds: float
    output_text: str
    batch_size: int


def load_prompts() -> list[dict]:
    """Load the standard benchmark prompt set."""
    with open(PROMPTS_PATH) as f:
        return json.load(f)


def load_model(model_name: str, precision: str = "fp16"):
    """Load a Hugging Face model and tokenizer with the specified precision.

    Args:
        model_name: Hugging Face model identifier (e.g. 'meta-llama/Llama-3.2-1B-Instruct').
        precision: One of 'fp16', 'int8', 'int4'.

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": "auto"}

    if precision == "fp16":
        kwargs["dtype"] = torch.float16
    elif precision == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "int4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    logger.info("Loading model: %s (precision=%s)", model_name, precision)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    logger.info("Model loaded successfully on %s", model.device if hasattr(model, 'device') else 'auto')
    return model, tokenizer


@torch.inference_mode()
def run_inference(model, tokenizer, prompt: str, prompt_id: str = "",
                  task_type: str = "", max_new_tokens: int = 200,
                  batch_size: int = 1) -> InferenceResult:
    """Run inference on a single prompt and return results with precise token counts.

    Args:
        model: Loaded HF model.
        tokenizer: Loaded HF tokenizer.
        prompt: Input text.
        prompt_id: Identifier for the prompt.
        task_type: Category of the prompt (summarisation, qa, code, etc.).
        max_new_tokens: Maximum tokens to generate.
        batch_size: Number of copies to process in parallel.

    Returns:
        InferenceResult with token counts and timing.

    Raises:
        torch.cuda.OutOfMemoryError: Re-raised after cleanup if OOM occurs.
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Replicate for batch_size > 1
    if batch_size > 1:
        input_ids = input_ids.repeat(batch_size, 1)
        attention_mask = attention_mask.repeat(batch_size, 1)

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    prompt_len = input_ids.shape[1]

    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # GREEDY ONLY
        )

        torch.cuda.synchronize()
        t1 = time.perf_counter()
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA OOM for prompt_id=%s, batch_size=%d", prompt_id, batch_size)
        torch.cuda.empty_cache()
        gc.collect()
        raise

    output_tokens_per_seq = outputs.shape[1] - prompt_len
    total_output_tokens = output_tokens_per_seq * batch_size
    output_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    return InferenceResult(
        prompt_id=prompt_id,
        task_type=task_type,
        prompt_tokens=prompt_len * batch_size,
        output_tokens=total_output_tokens,
        generation_time_seconds=t1 - t0,
        output_text=output_text,
        batch_size=batch_size,
    )
