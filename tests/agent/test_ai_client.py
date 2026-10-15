import json
import unittest

import requests

from laika.ai_client import AIUnavailable, ChatClient, parse_reply


class FakeResponse:
    def __init__(self, status_code=200, content="", body=None):
        self.status_code = status_code
        self._body = body if body is not None else {"choices": [{"message": {"content": content}}]}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json, headers, timeout):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_client(session, **kwargs):
    client = ChatClient(
        base_url="https://example.test/v1/",
        api_key="test",
        model="test-model",
        system_prompt="system",
        **kwargs,
    )
    client._session = session
    return client


class ParseReplyTests(unittest.TestCase):
    def test_reads_json(self):
        reply = parse_reply('{"say": "Ciao!", "actions": ["nod", "wave"]}')
        self.assertEqual(reply.say, "Ciao!")
        self.assertEqual(reply.actions, ["nod", "wave"])

    def test_reads_json_in_a_code_fence(self):
        reply = parse_reply('```json\n{"say": "Ciao!", "actions": []}\n```')
        self.assertEqual(reply.say, "Ciao!")

    def test_plain_text_becomes_speech(self):
        reply = parse_reply("Ciao, sono Laika.")
        self.assertEqual(reply.say, "Ciao, sono Laika.")
        self.assertEqual(reply.actions, [])

    def test_accepts_comma_separated_actions(self):
        self.assertEqual(parse_reply('{"say": "Sì", "actions": "nod, honk"}').actions, ["nod", "honk"])


class ChatClientTests(unittest.TestCase):
    def test_sends_system_prompt_and_history_with_a_timeout(self):
        session = FakeSession(
            FakeResponse(content='{"say": "Uno", "actions": []}'),
            FakeResponse(content='{"say": "Due", "actions": []}'),
        )
        client = make_client(session, timeout_s=7)
        client.ask("prima")
        client.ask("seconda")

        last = session.calls[-1]
        self.assertEqual(last["url"], "https://example.test/v1/chat/completions")
        self.assertEqual(last["timeout"], 7)
        roles = [m["role"] for m in last["json"]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_history_is_bounded_and_keeps_the_system_prompt(self):
        session = FakeSession(*[FakeResponse(content='{"say": "ok"}') for _ in range(5)])
        client = make_client(session, max_history_turns=2)
        for i in range(5):
            client.ask(f"domanda {i}")
        messages = session.calls[-1]["json"]["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "system"})
        self.assertEqual(len(messages), 1 + 4 + 1)

    def test_retries_once_on_connection_errors(self):
        session = FakeSession(requests.ConnectionError("down"), FakeResponse(content='{"say": "ok"}'))
        client = make_client(session)
        client_reply = client.ask("ciao")
        self.assertEqual(client_reply.say, "ok")

    def test_raises_when_the_endpoint_stays_down(self):
        session = FakeSession(requests.Timeout("slow"), requests.Timeout("slow"))
        client = make_client(session)
        with self.assertRaises(AIUnavailable):
            client.ask("ciao")
        self.assertEqual(len(client.history), 0)

    def test_does_not_retry_client_errors(self):
        session = FakeSession(FakeResponse(401, body={"error": {"message": "bad key"}}))
        with self.assertRaises(AIUnavailable):
            make_client(session).ask("ciao")


if __name__ == "__main__":
    unittest.main()
