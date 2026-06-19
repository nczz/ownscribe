"""Abstract base class for summarizers."""

from __future__ import annotations

import abc


class Summarizer(abc.ABC):
    """Base class for summarization backends."""

    @abc.abstractmethod
    def summarize(self, transcript_text: str) -> str:
        """Summarize a transcript and return the summary text."""

    @abc.abstractmethod
    def generate_title(self, summary_text: str) -> str:
        """Generate a short meeting title from a summary."""

    @abc.abstractmethod
    def chat(
        self, system_prompt: str, user_prompt: str,
        json_mode: bool = False, json_schema: dict | None = None,
    ) -> str:
        """Send a chat completion request and return the response text."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the summarization backend is reachable."""

    def close(self) -> None:  # noqa: B027 — intentional optional hook, not abstract
        """Release any native resources. No-op by default; must be idempotent."""

    def __enter__(self) -> Summarizer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
