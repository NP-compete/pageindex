"""Vertex AI Gemini client for LLM operations."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

if TYPE_CHECKING:
    from pageindex.config import PageIndexConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper for Vertex AI Gemini API calls."""

    def __init__(self, config: PageIndexConfig):
        self.config = config
        self._initialized = False
        self._model: GenerativeModel | None = None

    def _ensure_initialized(self) -> None:
        """Initialize Vertex AI if not already done."""
        if not self._initialized:
            vertexai.init(
                project=self.config.project_id,
                location=self.config.location,
            )
            self._model = GenerativeModel(self.config.model)
            self._initialized = True

    @property
    def model(self) -> GenerativeModel:
        """Get the initialized model."""
        self._ensure_initialized()
        assert self._model is not None
        return self._model

    def chat(
        self,
        prompt: str,
        max_retries: int = 10,
        chat_history: list[dict] | None = None,
    ) -> str:
        """Synchronous chat completion with retry logic."""
        self._ensure_initialized()

        generation_config = GenerationConfig(
            temperature=0,
            max_output_tokens=8192,
        )

        for attempt in range(max_retries):
            try:
                if chat_history:
                    contents = []
                    for msg in chat_history:
                        role = "user" if msg["role"] == "user" else "model"
                        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
                    contents.append({"role": "user", "parts": [{"text": prompt}]})
                else:
                    contents = prompt

                response = self.model.generate_content(
                    contents,
                    generation_config=generation_config,
                )
                return response.text

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    logger.error(f"Max retries reached for prompt: {prompt[:100]}...")
                    raise

        return "Error"

    def chat_with_finish_reason(
        self,
        prompt: str,
        max_retries: int = 10,
        chat_history: list[dict] | None = None,
    ) -> tuple[str, str]:
        """Chat completion that also returns finish reason."""
        self._ensure_initialized()

        generation_config = GenerationConfig(
            temperature=0,
            max_output_tokens=8192,
        )

        for attempt in range(max_retries):
            try:
                if chat_history:
                    contents = []
                    for msg in chat_history:
                        role = "user" if msg["role"] == "user" else "model"
                        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
                    contents.append({"role": "user", "parts": [{"text": prompt}]})
                else:
                    contents = prompt

                response = self.model.generate_content(
                    contents,
                    generation_config=generation_config,
                )

                finish_reason = "finished"
                if response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, "finish_reason"):
                        fr = str(candidate.finish_reason)
                        if "MAX_TOKENS" in fr or "LENGTH" in fr:
                            finish_reason = "max_output_reached"

                return response.text, finish_reason

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    logger.error(f"Max retries reached for prompt: {prompt[:100]}...")
                    raise

        return "Error", "error"

    async def chat_async(
        self,
        prompt: str,
        max_retries: int = 10,
    ) -> str:
        """Asynchronous chat completion with retry logic."""
        self._ensure_initialized()

        generation_config = GenerationConfig(
            temperature=0,
            max_output_tokens=8192,
        )

        for attempt in range(max_retries):
            try:
                response = await self.model.generate_content_async(
                    prompt,
                    generation_config=generation_config,
                )
                return response.text

            except Exception as e:
                logger.warning(f"Async attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Max retries reached for prompt: {prompt[:100]}...")
                    raise

        return "Error"

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using Vertex AI's token counter."""
        if not text:
            return 0
        self._ensure_initialized()
        try:
            response = self.model.count_tokens(text)
            return response.total_tokens
        except Exception:
            # Fallback: rough estimate (4 chars per token)
            return len(text) // 4


# Module-level client instance (lazy initialization)
_client: LLMClient | None = None


def get_client(config: PageIndexConfig) -> LLMClient:
    """Get or create the LLM client singleton."""
    global _client
    if _client is None:
        _client = LLMClient(config)
    return _client


def chat(config: PageIndexConfig, prompt: str, chat_history: list[dict] | None = None) -> str:
    """Convenience function for synchronous chat."""
    return get_client(config).chat(prompt, chat_history=chat_history)


def chat_with_finish_reason(
    config: PageIndexConfig,
    prompt: str,
    chat_history: list[dict] | None = None,
) -> tuple[str, str]:
    """Convenience function for chat with finish reason."""
    return get_client(config).chat_with_finish_reason(prompt, chat_history=chat_history)


async def chat_async(config: PageIndexConfig, prompt: str) -> str:
    """Convenience function for async chat."""
    return await get_client(config).chat_async(prompt)


def count_tokens(config: PageIndexConfig, text: str) -> int:
    """Convenience function for token counting."""
    return get_client(config).count_tokens(text)
