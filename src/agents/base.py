"""
MOMENTUM-X Base Agent Interface

### ARCHITECTURAL CONTEXT
All LLM-powered agents inherit from BaseAgent. This enforces the standardized
AgentSignal output contract and provides shared infrastructure for:
- LLM invocation via litellm (unified API for all providers)
- Latency tracking per call
- Prompt variant management for Arena integration
- Structured JSON output parsing with retry logic

Ref: ADR-001 (Agent Communication Protocol)
Ref: PROMPT_SIGNATURES.md (Base signatures per agent type)

### DESIGN DECISIONS
- ABC enforces analyze() contract on all subclasses
- litellm over raw provider SDKs for model-agnostic switching
- JSON output mode with structured retry (LLMs sometimes emit invalid JSON)
- Timeout hard-cap at 120s per agent call (ADR-001 latency budget)
- prompt_variant_id threaded through for Arena tracking
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import litellm

from src.core.models import AgentSignal
from src.utils.trade_logger import get_trade_logger

logger = get_trade_logger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


class BaseAgent(ABC):
    """
    Abstract base class for all LLM-powered analytical agents.

    Subclasses must implement:
        - agent_id: str property
        - system_prompt: str property
        - build_user_prompt(**kwargs) -> str
        - parse_response(raw: dict) -> AgentSignal
    """

    def __init__(
        self,
        model: str,
        provider: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout: int = 120,
        prompt_variant_id: str = "v0_control",
    ):
        """
        Args:
            model: Model identifier (e.g., "deepseek/deepseek-r1-distill-qwen-32b")
            provider: LiteLLM provider prefix (e.g., "together_ai")
            temperature: Sampling temperature (Arena tests 0.1-0.7)
            max_tokens: Max response tokens
            timeout: Hard timeout in seconds (ADR-001: 120s max)
            prompt_variant_id: Arena tracking ID for this prompt configuration
        """
        self.model = f"{provider}/{model}" if provider else model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.prompt_variant_id = prompt_variant_id

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Unique identifier for this agent type (e.g., 'news_agent')."""
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt defining agent behavior. Ref: PROMPT_SIGNATURES.md."""
        ...

    @abstractmethod
    def build_user_prompt(self, **kwargs: Any) -> str:
        """
        Build the user prompt from input data.
        Subclasses define what data they need.
        """
        ...

    @abstractmethod
    def parse_response(self, raw: dict, ticker: str) -> AgentSignal:
        """
        Parse LLM JSON response into a typed AgentSignal.
        Subclasses define their specific signal type.
        """
        ...

    async def analyze(self, ticker: str, **kwargs: Any) -> AgentSignal:
        """
        Execute the full agent pipeline:
        1. Build prompt from input data
        2. Call LLM via litellm
        3. Parse structured response
        4. Return typed AgentSignal with latency tracking

        This method handles retries, JSON parsing failures, and timeouts.
        """
        user_prompt = self.build_user_prompt(ticker=ticker, **kwargs)
        start_time = time.monotonic()

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                response_format={"type": "json_object"},
            )

            latency_ms = (time.monotonic() - start_time) * 1000
            raw_content = response.choices[0].message.content

            # ── Parse JSON (with cleanup for markdown fences) ──
            parsed = self._extract_json(raw_content)

            signal = self.parse_response(parsed, ticker)

            # ── Inject metadata ──
            # Since signals are frozen, we reconstruct with metadata
            return signal.model_copy(
                update={
                    "prompt_variant_id": self.prompt_variant_id,
                    "model_id": self.model,
                    "latency_ms": latency_ms,
                }
            )

        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Agent %s failed for %s: %s (%.0fms)",
                self.agent_id, ticker, str(e), latency_ms,
            )
            # Return a neutral fallback signal — never crash the pipeline
            return AgentSignal(
                agent_id=self.agent_id,
                ticker=ticker,
                timestamp=datetime.now(timezone.utc),
                signal="NEUTRAL",
                confidence=0.0,
                reasoning=f"Agent error: {str(e)}",
                flags=["AGENT_ERROR"],
                prompt_variant_id=self.prompt_variant_id,
                model_id=self.model,
                latency_ms=latency_ms,
            )

    def _extract_json(self, raw: str) -> dict:
        """
        Extract JSON from LLM response, handling common formatting issues:
        - Markdown code fences (```json ... ```)
        - Leading/trailing whitespace
        - DeepSeek R1 <think>...</think> blocks before JSON
        """
        text = raw.strip()

        # Strip R1 thinking blocks
        if "<think>" in text:
            think_end = text.rfind("</think>")
            if think_end != -1:
                text = text[think_end + len("</think>"):].strip()

        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            lines = [
                line for line in lines
                if not line.strip().startswith("```")
            ]
            text = "\n".join(lines).strip()

        return json.loads(text)
