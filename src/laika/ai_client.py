"""Provider-independent client for OpenAI-compatible chat completions endpoints.

Written instead of reusing SunFounder's `picarx.llm`, which sends requests
without a timeout and eventually drops the system prompt from its history.
Only `requests` is needed, which the SunFounder stack already installs; no
provider SDK is used. URLs, credentials and provider quirks live in `Endpoint`,
so the rest of Laika only sees `ChatClient`, `ToolSpec`, `ToolCall` and
`AssistantTurn`.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests

RETRY_DELAY_S = 1.0
MAX_RETRY_AFTER_S = 5.0
PROVIDERS = ("openai", "azure", "openai-compatible")
DEFAULT_AZURE_API_VERSION = "2024-10-21"
IMAGE_DETAILS = ("low", "high", "auto")


class AIUnavailable(Exception):
    """The endpoint could not be reached or returned an error.

    `rejected` is True when the endpoint answered but refused or garbled the request
    (HTTP 4xx, unexpected body): a configuration problem, not an outage.
    """

    def __init__(self, message: str, *, rejected: bool = False) -> None:
        super().__init__(message)
        self.rejected = rejected


class AIConfigError(ValueError):
    """The AI settings are unusable (unknown provider, insecure URL, missing field)."""


def _is_local_host(host: str) -> bool:
    host = host.lower().rstrip(".")
    if host == "localhost" or host.endswith((".local", ".lan", ".localhost")):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private


def check_base_url(base_url: str) -> None:
    """Cloud endpoints must use HTTPS; plain HTTP is only for local/private hosts."""
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise AIConfigError(f"[ai] base_url must be an http(s) URL, got {base_url!r}")
    if parsed.scheme == "http" and not _is_local_host(parsed.hostname):
        raise AIConfigError(
            f"[ai] base_url {base_url!r} uses plain HTTP; use HTTPS, or a local/LAN host "
            "(localhost, 127.0.0.1, *.local, *.lan or a private IP address)"
        )


@dataclass(frozen=True)
class Endpoint:
    """Everything provider-specific: where to POST, which headers, which model name."""

    url: str
    headers: dict[str, str]
    model: str

    @classmethod
    def build(
        cls,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        azure_deployment: str = "",
        azure_api_version: str = DEFAULT_AZURE_API_VERSION,
    ) -> Endpoint:
        if provider not in PROVIDERS:
            raise AIConfigError(f"[ai] provider must be one of: {', '.join(PROVIDERS)}")
        check_base_url(base_url)
        base = base_url.rstrip("/")
        if provider == "azure":
            deployment = azure_deployment or model
            if not deployment:
                raise AIConfigError('[ai] azure_deployment is required for provider "azure"')
            return cls(
                url=(
                    f"{base}/openai/deployments/{deployment}/chat/completions"
                    f"?api-version={azure_api_version}"
                ),
                headers={"api-key": api_key},
                model=deployment,
            )
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        return cls(url=f"{base}/chat/completions", headers=headers, model=model)


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call; `parameters` is a JSON schema object."""

    name: str
    description: str
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolCall:
    """A tool call proposed by the model. Never trusted: `error` is set when unusable."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None

    @classmethod
    def parse(cls, raw: Any, index: int = 0) -> ToolCall:
        function = raw.get("function") if isinstance(raw, dict) else None
        if not isinstance(function, dict):
            return cls(id=f"call_{index}", name="", error="Malformed tool call")
        call_id = str(raw.get("id") or f"call_{index}")
        name = str(function.get("name") or "")
        arguments = function.get("arguments")
        if isinstance(arguments, dict):  # some local servers return an object, not a string
            return cls(call_id, name, arguments, json.dumps(arguments))
        text = arguments if isinstance(arguments, str) else ""
        error = None if name else "Tool call has no name"
        parsed: dict[str, Any] = {}
        if text.strip():
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                error = error or f"Arguments are not valid JSON: {exc.msg}"
            else:
                if isinstance(value, dict):
                    parsed = value
                else:
                    error = error or "Arguments must be a JSON object"
        return cls(call_id, name, parsed, text, error)


@dataclass(frozen=True)
class AssistantTurn:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None

    def to_message(self) -> dict[str, Any]:
        """The assistant message to append to the conversation before tool results."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw_arguments or "{}"},
                }
                for call in self.tool_calls
            ]
        return message


def system_message(content: str) -> dict[str, Any]:
    return {"role": "system", "content": content}


def user_message(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


def image_message(
    text: str, image: bytes, *, detail: str = "low", mime: str = "image/jpeg"
) -> dict[str, Any]:
    """A user message with a prompt and one image, as an inline data URL.

    The same shape is accepted by OpenAI, Azure OpenAI and the OpenAI-compatible
    local servers (LM Studio, Ollama, vLLM) when the model can see.
    """
    if detail not in IMAGE_DETAILS:
        raise ValueError(f"image detail must be one of: {', '.join(IMAGE_DETAILS)}")
    url = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": url, "detail": detail}},
        ],
    }


def assistant_message(turn: AssistantTurn) -> dict[str, Any]:
    return turn.to_message()


def tool_message(tool_call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


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


class _BadRequest(Exception):
    """HTTP 400, kept apart so `complete` can retry without response_format."""


def _retry_after(response: Any) -> float:
    """Honour a numeric Retry-After header, capped so the robot never waits long."""
    headers = getattr(response, "headers", None) or {}
    try:
        return min(max(float(headers.get("Retry-After", RETRY_DELAY_S)), 0.0), MAX_RETRY_AFTER_S)
    except (TypeError, ValueError):
        return RETRY_DELAY_S


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
        provider: str = "openai",
        azure_deployment: str = "",
        azure_api_version: str = DEFAULT_AZURE_API_VERSION,
    ) -> None:
        self.endpoint = Endpoint.build(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
            azure_deployment=azure_deployment,
            azure_api_version=azure_api_version,
        )
        self.system_prompt = system_prompt
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.retries = retries
        # One turn is a user message plus the assistant's answer.
        self.history: deque[dict[str, Any]] = deque(maxlen=max_history_turns * 2)
        self._session = requests.Session()

    def reset(self) -> None:
        self.history.clear()

    def ask(self, user_text: str) -> AIReply:
        messages = [
            system_message(self.system_prompt),
            *self.history,
            user_message(user_text),
        ]
        content = self.complete(messages, json_mode=True).content
        self.history.append(user_message(user_text))
        self.history.append({"role": "assistant", "content": content})
        return parse_reply(content)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        tool_choice: str = "auto",
        json_mode: bool = False,
    ) -> AssistantTurn:
        """One stateless chat-completions request; the caller owns the messages."""
        payload: dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = [tool.to_openai() for tool in tools]
            payload["tool_choice"] = tool_choice
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            data = self._post(payload)
        except _BadRequest as error:
            # Some local servers do not know response_format: retry once without it.
            if not (json_mode and "response_format" in str(error)):
                raise AIUnavailable(str(error), rejected=True) from error
            retry_payload = {k: v for k, v in payload.items() if k != "response_format"}
            try:
                data = self._post(retry_payload)
            except _BadRequest as again:
                raise AIUnavailable(str(again), rejected=True) from again
        return self._parse_turn(data)

    @staticmethod
    def _parse_turn(data: Any) -> AssistantTurn:
        try:
            choice = data["choices"][0]
            message = choice["message"]
            raw_calls = message.get("tool_calls") or []
            content = message.get("content") or ""
            usage = data.get("usage")
            return AssistantTurn(
                content=content if isinstance(content, str) else "",
                tool_calls=tuple(ToolCall.parse(raw, i) for i, raw in enumerate(raw_calls)),
                finish_reason=choice.get("finish_reason"),
                usage=usage if isinstance(usage, dict) else None,
            )
        except (KeyError, IndexError, TypeError, AttributeError) as error:
            raise AIUnavailable(f"Unexpected response: {str(data)[:200]}", rejected=True) from error

    def _post(self, payload: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        delay = RETRY_DELAY_S
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(delay)
            delay = RETRY_DELAY_S
            try:
                response = self._session.post(
                    self.endpoint.url,
                    json=payload,
                    headers=self.endpoint.headers,
                    timeout=self.timeout_s,
                )
            except requests.RequestException as error:
                last_error = error
                continue
            status = response.status_code
            if status >= 500 or status == 429:
                last_error = AIUnavailable(f"HTTP {status}: {response.text[:200]}")
                delay = _retry_after(response)
                continue
            if status == 400:
                raise _BadRequest(f"HTTP 400: {response.text[:200]}")
            if status >= 400:
                raise AIUnavailable(f"HTTP {status}: {response.text[:200]}", rejected=True)
            try:
                return response.json()
            except ValueError as error:
                raise AIUnavailable(f"Unexpected response: {response.text[:200]}", rejected=True) from error
        raise AIUnavailable(str(last_error))
