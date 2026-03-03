"""Tests for src/inference.py — token counting and prompt loading."""

from pathlib import Path

from src.inference import InferenceResult, load_prompts


class TestLoadPrompts:
    def test_loads_five_prompts(self):
        prompts = load_prompts()
        assert len(prompts) == 5

    def test_prompt_fields(self):
        prompts = load_prompts()
        for p in prompts:
            assert "id" in p
            assert "task" in p
            assert "prompt" in p
            assert isinstance(p["prompt"], str)
            assert len(p["prompt"]) > 10

    def test_expected_task_types(self):
        prompts = load_prompts()
        tasks = {p["task"] for p in prompts}
        expected = {"summarisation", "qa", "code", "longform", "reasoning"}
        assert tasks == expected

    def test_expected_ids(self):
        prompts = load_prompts()
        ids = {p["id"] for p in prompts}
        expected = {"sum_short", "qa_factual", "code_simple", "longform", "reasoning"}
        assert ids == expected


class TestInferenceResult:
    def test_dataclass_creation(self):
        result = InferenceResult(
            prompt_id="test",
            task_type="qa",
            prompt_tokens=50,
            output_tokens=200,
            generation_time_seconds=2.0,
            output_text="test output",
            batch_size=1,
        )
        assert result.prompt_tokens == 50
        assert result.output_tokens == 200
        assert result.batch_size == 1

    def test_batch_token_counting(self):
        """Output tokens should include all sequences in the batch."""
        result = InferenceResult(
            prompt_id="test",
            task_type="qa",
            prompt_tokens=50 * 4,  # 50 per seq * 4 batch
            output_tokens=200 * 4,  # 200 per seq * 4 batch
            generation_time_seconds=3.0,
            output_text="test output",
            batch_size=4,
        )
        assert result.output_tokens == 800
        assert result.prompt_tokens == 200
