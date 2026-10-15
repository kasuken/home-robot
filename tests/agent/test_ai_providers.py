"""Providers, tool calls, the HTTPS rule and an end-to-end run against a local server."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

from laika.ai_client import (
    AIConfigError,
    AIUnavailable,
    ChatClient,
    Endpoint,
    ToolCall,
    ToolSpec,
    assistant_message,
    tool_message,
    user_message,
)

WEATHER = ToolSpec(
    "get_weather",
    "Weather in a city",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)

OPENAI_TOOL_RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Roma"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 50, "completion_tokens": 12, "total_tokens": 62},
}

# Ollama /v1 and LM Studio often omit usage, may send an empty content string,
# and Ollama can return arguments as an object.
LOCAL_TOOL_RESPONSE = {
    "id": "chatcmpl-9",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_9",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "Milano"}},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
}


class FakeResponse:
    def __init__(self, status_code=200, body=None, headers=None):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.headers = headers or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json, headers, timeout):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def text_response(text):
    return FakeResponse(body={"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})


def make_client(session=None, **kwargs):
    settings = {
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-test",
        "model": "gpt-4o-mini",
        "system_prompt": "system",
    }
    settings.update(kwargs)
    client = ChatClient(**settings)
    if session is not None:
        client._session = session
    return client


class EndpointTests(unittest.TestCase):
    def test_openai_uses_bearer_and_chat_completions(self):
        client = make_client(FakeSession(text_response("ok")))
        client.complete([user_message("ciao")])
        call = client._session.calls[0]
        self.assertEqual(call["url"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(call["headers"], {"Authorization": "Bearer sk-test"})
        self.assertEqual(call["json"]["model"], "gpt-4o-mini")

    def test_azure_uses_deployment_url_and_api_key_header(self):
        client = make_client(
            FakeSession(text_response("ok")),
            provider="azure",
            base_url="https://res.openai.azure.com/",
            azure_deployment="laika-dep",
            azure_api_version="2024-10-21",
        )
        client.complete([user_message("ciao")])
        call = client._session.calls[0]
        self.assertEqual(
            call["url"],
            "https://res.openai.azure.com/openai/deployments/laika-dep/chat/completions"
            "?api-version=2024-10-21",
        )
        self.assertEqual(call["headers"], {"api-key": "sk-test"})
        self.assertEqual(call["json"]["model"], "laika-dep")

    def test_azure_falls_back_to_the_model_as_deployment(self):
        endpoint = Endpoint.build(
            provider="azure", base_url="https://r.openai.azure.com", api_key="k", model="dep"
        )
        self.assertIn("/deployments/dep/", endpoint.url)

    def test_local_server_needs_no_key(self):
        endpoint = Endpoint.build(
            provider="openai-compatible",
            base_url="http://192.168.1.20:1234/v1",
            api_key="",
            model="qwen",
        )
        self.assertEqual(endpoint.url, "http://192.168.1.20:1234/v1/chat/completions")
        self.assertEqual(endpoint.headers, {})

    def test_local_server_sends_bearer_when_a_key_is_set(self):
        endpoint = Endpoint.build(
            provider="openai-compatible", base_url="http://127.0.0.1:8000/v1", api_key="k", model="m"
        )
        self.assertEqual(endpoint.headers, {"Authorization": "Bearer k"})

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(AIConfigError):
            Endpoint.build(provider="bard", base_url="https://x.test", api_key="k", model="m")


class HttpsRuleTests(unittest.TestCase):
    def build(self, url):
        return Endpoint.build(provider="openai-compatible", base_url=url, api_key="", model="m")

    def test_plain_http_is_allowed_for_local_hosts(self):
        for url in (
            "http://localhost:11434/v1",
            "http://127.0.0.1:9/v1",
            "http://127.5.5.5/v1",
            "http://[::1]:8000/v1",
            "http://studio.local:1234/v1",
            "http://pc.lan:1234/v1",
            "http://192.168.1.20:1234/v1",
            "http://10.0.0.5/v1",
            "http://172.16.3.4/v1",
            "http://[fd00::5]:1234/v1",
        ):
            with self.subTest(url=url):
                self.build(url)

    def test_plain_http_is_refused_for_cloud_hosts(self):
        for url in (
            "http://api.openai.com/v1",
            "http://8.8.8.8/v1",
            "http://example.com/v1",
            "http://localhost.evil.com/v1",
        ):
            with self.subTest(url=url):
                with self.assertRaises(AIConfigError):
                    self.build(url)

    def test_https_is_always_allowed_and_other_schemes_are_not(self):
        self.build("https://api.openai.com/v1")
        for url in ("ftp://localhost/v1", "api.openai.com/v1", ""):
            with self.subTest(url=url):
                with self.assertRaises(AIConfigError):
                    self.build(url)


class ToolCallTests(unittest.TestCase):
    def test_parses_a_realistic_openai_tool_call(self):
        client = make_client(FakeSession(FakeResponse(body=OPENAI_TOOL_RESPONSE)))
        turn = client.complete([user_message("meteo?")], tools=[WEATHER])
        self.assertEqual(turn.finish_reason, "tool_calls")
        self.assertEqual(turn.content, "")
        self.assertEqual(turn.usage["total_tokens"], 62)
        (call,) = turn.tool_calls
        self.assertTrue(call.valid)
        self.assertEqual((call.id, call.name, call.arguments), ("call_abc123", "get_weather", {"city": "Roma"}))
        self.assertEqual(call.raw_arguments, '{"city": "Roma"}')

    def test_sends_tools_and_tool_choice(self):
        session = FakeSession(text_response("ok"))
        make_client(session).complete([user_message("x")], tools=[WEATHER], tool_choice="required")
        payload = session.calls[0]["json"]
        self.assertEqual(payload["tool_choice"], "required")
        self.assertEqual(
            payload["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Weather in a city",
                        "parameters": WEATHER.parameters,
                    },
                }
            ],
        )
        self.assertNotIn("response_format", payload)

    def test_no_tools_means_no_tool_fields(self):
        session = FakeSession(text_response("ok"))
        make_client(session).complete([user_message("x")])
        self.assertNotIn("tools", session.calls[0]["json"])
        self.assertNotIn("tool_choice", session.calls[0]["json"])

    def test_parses_a_local_server_response_with_object_arguments(self):
        client = make_client(
            FakeSession(FakeResponse(body=LOCAL_TOOL_RESPONSE)),
            provider="openai-compatible",
            base_url="http://127.0.0.1:11434/v1",
            api_key="",
        )
        turn = client.complete([user_message("meteo?")], tools=[WEATHER])
        (call,) = turn.tool_calls
        self.assertTrue(call.valid)
        self.assertEqual(call.arguments, {"city": "Milano"})
        self.assertIsNone(turn.usage)

    def test_invalid_json_arguments_do_not_crash(self):
        call = ToolCall.parse(
            {"id": "c1", "function": {"name": "drive", "arguments": '{"cm": 10'}}
        )
        self.assertFalse(call.valid)
        self.assertEqual(call.arguments, {})
        self.assertEqual(call.raw_arguments, '{"cm": 10')
        self.assertIn("not valid JSON", call.error)

    def test_non_object_and_malformed_calls_are_marked_invalid(self):
        self.assertFalse(ToolCall.parse({"id": "c", "function": {"name": "a", "arguments": "[1]"}}).valid)
        self.assertFalse(ToolCall.parse({"id": "c", "function": {"arguments": "{}"}}).valid)
        self.assertFalse(ToolCall.parse("nonsense").valid)

    def test_empty_arguments_are_an_empty_object_and_missing_id_is_generated(self):
        call = ToolCall.parse({"function": {"name": "stop", "arguments": ""}}, index=2)
        self.assertTrue(call.valid)
        self.assertEqual((call.id, call.arguments), ("call_2", {}))

    def test_tool_round_trip_messages_are_valid_openai(self):
        session = FakeSession(
            FakeResponse(body=OPENAI_TOOL_RESPONSE), text_response("A Roma c'è il sole.")
        )
        client = make_client(session)
        messages = [user_message("meteo a Roma?")]
        turn = client.complete(messages, tools=[WEATHER])
        messages.append(assistant_message(turn))
        messages.append(tool_message(turn.tool_calls[0].id, '{"sky": "sunny"}'))
        final = client.complete(messages, tools=[WEATHER])

        self.assertEqual(final.content, "A Roma c'è il sole.")
        sent = session.calls[1]["json"]["messages"]
        self.assertEqual(
            sent[1],
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Roma"}'},
                    }
                ],
            },
        )
        self.assertEqual(
            sent[2], {"role": "tool", "tool_call_id": "call_abc123", "content": '{"sky": "sunny"}'}
        )

    def test_complete_is_stateless(self):
        client = make_client(FakeSession(text_response("ok")))
        client.complete([user_message("x")])
        self.assertEqual(len(client.history), 0)

    def test_unexpected_response_shape_is_unavailable(self):
        client = make_client(FakeSession(FakeResponse(body={"choices": []})))
        with self.assertRaises(AIUnavailable):
            client.complete([user_message("x")])


class RetryTests(unittest.TestCase):
    def setUp(self):
        import laika.ai_client as module

        self.module = module
        self.sleeps = []
        self._sleep = module.time.sleep
        module.time.sleep = self.sleeps.append

    def tearDown(self):
        self.module.time.sleep = self._sleep

    def test_retries_on_429_and_5xx_then_succeeds(self):
        session = FakeSession(
            FakeResponse(429, headers={"Retry-After": "3"}),
            FakeResponse(503),
            text_response("ok"),
        )
        turn = make_client(session, retries=2).complete([user_message("x")])
        self.assertEqual(turn.content, "ok")
        self.assertEqual(self.sleeps, [3.0, 1.0])

    def test_retry_after_is_capped(self):
        session = FakeSession(FakeResponse(429, headers={"Retry-After": "600"}), text_response("ok"))
        make_client(session).complete([user_message("x")])
        self.assertEqual(self.sleeps, [self.module.MAX_RETRY_AFTER_S])

    def test_retries_are_bounded_and_never_on_other_4xx(self):
        session = FakeSession(requests.Timeout("a"), requests.Timeout("b"), text_response("never"))
        with self.assertRaises(AIUnavailable):
            make_client(session, retries=1).complete([user_message("x")])
        self.assertEqual(len(session.calls), 2)

        for status in (401, 403, 404, 422):
            session = FakeSession(FakeResponse(status), text_response("never"))
            with self.assertRaises(AIUnavailable):
                make_client(session).complete([user_message("x")])
            self.assertEqual(len(session.calls), 1)

    def test_every_request_has_a_finite_timeout(self):
        session = FakeSession(text_response("ok"))
        make_client(session, timeout_s=4).complete([user_message("x")])
        self.assertEqual(session.calls[0]["timeout"], 4)

    def test_json_mode_retries_once_without_response_format(self):
        session = FakeSession(
            FakeResponse(400, body={"error": "'response_format.type' must be 'json_schema'"}),
            text_response('{"say": "Ciao"}'),
        )
        reply = make_client(session).ask("ciao")
        self.assertEqual(reply.say, "Ciao")
        self.assertIn("response_format", session.calls[0]["json"])
        self.assertNotIn("response_format", session.calls[1]["json"])

    def test_other_400_errors_are_not_retried(self):
        session = FakeSession(FakeResponse(400, body={"error": "bad model"}), text_response("never"))
        with self.assertRaises(AIUnavailable):
            make_client(session).ask("ciao")
        self.assertEqual(len(session.calls), 1)


class _Handler(BaseHTTPRequestHandler):
    requests_seen: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests_seen.append(
            {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
        )
        if any(m["role"] == "tool" for m in body["messages"]):
            message = {"role": "assistant", "content": "A Roma c'è il sole."}
            finish = "stop"
        else:
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Roma"}'},
                    }
                ],
            }
            finish = "tool_calls"
        payload = json.dumps(
            {"choices": [{"index": 0, "message": message, "finish_reason": finish}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class LocalServerEndToEndTests(unittest.TestCase):
    def test_tool_call_round_trip_over_real_http(self):
        _Handler.requests_seen = []
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        client = ChatClient(
            provider="openai-compatible",
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            api_key="",
            model="local-model",
            system_prompt="system",
            timeout_s=5,
        )
        messages = [user_message("meteo a Roma?")]
        turn = client.complete(messages, tools=[WEATHER])
        self.assertEqual(turn.finish_reason, "tool_calls")
        (call,) = turn.tool_calls
        self.assertEqual((call.name, call.arguments), ("get_weather", {"city": "Roma"}))

        messages += [assistant_message(turn), tool_message(call.id, '{"sky": "sunny"}')]
        final = client.complete(messages, tools=[WEATHER])
        self.assertEqual(final.content, "A Roma c'è il sole.")

        first, second = _Handler.requests_seen
        self.assertEqual(first["path"], "/v1/chat/completions")
        self.assertIsNone(first["auth"])
        self.assertEqual(first["body"]["tools"][0]["function"]["name"], "get_weather")
        self.assertEqual(second["body"]["messages"][-1]["role"], "tool")


if __name__ == "__main__":
    unittest.main()
