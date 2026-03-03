"""Layer 2: Inference Execution Engine — model loading and standardised workloads.

Loads any Hugging Face model with configurable precision (fp16, int8, int4),
runs the standard benchmark prompt set, and returns precise token counts
and timing information.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

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
    # TODO: Implement model loading with precision support
    # - fp16: torch_dtype=torch.float16
    # - int8: load_in_8bit=True via bitsandbytes
    # - int4: load_in_4bit=True via bitsandbytes
    raise NotImplementedError("To be implemented on GPU machine")


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
    """
    # TODO: Implement inference with precise token counting
    # - Tokenize input, record input token count
    # - Generate with model.generate()
    # - Count output tokens (generated - input)
    # - Time the generation
    raise NotImplementedError("To be implemented on GPU machine")
