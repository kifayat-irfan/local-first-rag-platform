"""
LLM client for answer generation. Ollama by default, matching the "works
offline, no cloud APIs" requirement — talks to a local Ollama server over
plain HTTP, no extra SDK.
"""

from __future__ import annotations

from typing import Optional

import aiohttp


class OllamaClient:
    """
    Args:
        base_url: Ollama server URL (default assumes a locally-running instance).
        model: Model tag, e.g. "llama3.1:8b". Must already be pulled (`ollama pull llama3.1:8b`).
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1:8b"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> str:
        """
        Args:
            prompt: User-turn content.
            system: System prompt.
            temperature: Sampling temperature; low by default since this is
                a factual QA context where creativity is undesirable.

        Returns:
            The model's response text.

        Raises:
            aiohttp.ClientError: on connection failure (e.g. Ollama not running).
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {"temperature": temperature},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("response", "")


class GroqLLMClient:
    """
    Public-mode LLM client — calls Groq's hosted chat completions API
    (OpenAI-compatible) instead of a local Ollama instance. Used only when
    the user explicitly switches to "Public mode" (e.g. a deployed demo
    where a local Ollama isn't reachable). Local mode (OllamaClient above)
    stays the default everywhere else in this platform.

    Args:
        api_key: Groq API key. Get a free one at https://console.groq.com/keys.
            NEVER hardcode a real key here or in any config/prompt — always
            load it from an environment variable (GROQ_API_KEY / RAG_GROQ_API_KEY).
        model: Groq-hosted model id, e.g. "llama-3.3-70b-versatile".
    """

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    async def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> str:
        if not self.api_key:
            return (
                "⚠️ No Groq API key configured. Set GROQ_API_KEY (or RAG_GROQ_API_KEY) "
                "to use Public mode — get a free key at https://console.groq.com/keys"
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history, status=resp.status, message=body[:300]
                    )
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
