"""A small client for any OpenAI-compatible chat completions endpoint.

Written instead of reusing SunFounder's `picarx.llm`, which sends requests
without a timeout and eventually drops the system prompt from its history.
Only `requests` is needed, which the SunFounder stack already installs.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field

import requests

RETRY_DELAY_S = 1.0


class AIUnavailable(Exception):
    """The endpoint could not be reached or returned an error."""


@dataclass
class AIReply:
    say: str
    actions: list[str] = field(default_factory=list)


def parse_reply(content: str) -> AIReply:
    """Read the model's JSON reply; fall back to treating it as plain speech."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return AIReply(say=content.strip())
    if not isinstance(data, dict):
        return AIReply(say=content.strip())

    say = data.get("say") or data.get("risposta") or ""
    actions = data.get("actions") or data.get("azioni") or []
    if isinstance(actions, str):
        actions = [a.strip() for a in actions.split(",")]
    if not isinstance(actions, list):
        actions = []
    return AIReply(say=str(say).strip(), actions=[str(a) for a in actions])


class ChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        timeout_s: float = 20.0,
        max_history_turns: int = 8,
        temperature: float = 0.8,
        retries: int = 1,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.retries = retries
        # One turn is a user message plus the assistant's answer.
        self.history: deque[dict[str, str]] = deque(maxlen=max_history_turns * 2)
        self._session = requests.Session()

    def reset(self) -> None:
        self.history.clear()

    def ask(self, user_text: str) -> AIReply:
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.history,
            {"role": "user", "content": user_text},
        ]
        content = self._complete(messages)
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": content})
        return parse_reply(content)

    def _complete(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(RETRY_DELAY_S)
            try:
                response = self._session.post(
                    self.url, json=payload, headers=headers, timeout=self.timeout_s
                )
            except requests.RequestException as error:
                last_error = error
                continue
            if response.status_code >= 500 or response.status_code == 429:
                last_error = AIUnavailable(f"HTTP {response.status_code}: {response.text[:200]}")
                continue
            if response.status_code >= 400:
                raise AIUnavailable(f"HTTP {response.status_code}: {response.text[:200]}")
            try:
                return response.json()["choices"][0]["message"]["content"] or ""
            except (ValueError, KeyError, IndexError) as error:
                raise AIUnavailable(f"Unexpected response: {response.text[:200]}") from error
        raise AIUnavailable(str(last_error))
