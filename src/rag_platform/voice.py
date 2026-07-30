"""
Voice input via Groq's hosted Whisper API (free tier).

This is the one deliberate exception to "everything local" in this platform:
speech-to-text models are heavy enough that running Whisper locally on a
laptop CPU is slow, and Groq's free tier makes a hosted call the pragmatic
choice for a chat "microphone button" feature. Text queries and all
retrieval/generation remain fully local — voice transcription is the only
step that leaves the machine, and only when the user explicitly records
audio.
"""

from __future__ import annotations

import aiohttp


class GroqTranscriptionError(Exception):
    pass


class GroqWhisperClient:
    def __init__(self, api_key: str, model: str = "whisper-large-v3-turbo"):
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """
        Args:
            audio_bytes: Raw audio file bytes (wav/mp3/m4a/webm — anything
                Whisper accepts).
            filename: Used only to hint the file extension/mimetype to the API.

        Returns:
            The transcribed text.

        Raises:
            GroqTranscriptionError: if no API key is configured, or the API call fails.
        """
        if not self.api_key:
            raise GroqTranscriptionError(
                "No Groq API key configured. Set GROQ_API_KEY (or RAG_GROQ_API_KEY) "
                "in your environment — get a free key at https://console.groq.com/keys"
            )

        form = aiohttp.FormData()
        form.add_field("file", audio_bytes, filename=filename, content_type="application/octet-stream")
        form.add_field("model", self.model)

        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, data=form, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise GroqTranscriptionError(f"Groq API error {resp.status}: {body[:300]}")
                data = await resp.json()
                return data.get("text", "").strip()
