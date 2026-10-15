import json
import unittest

import requests

from laika.web import (
    BRAVE_URL,
    MAX_RESULT_CHARS,
    MAX_SNIPPET_CHARS,
    USER_AGENT,
    BraveSearchClient,
    WebGuard,
    WikipediaClient,
    cap_results,
    check_https,
    clean_text,
)

SEARCH_IT = {
    "batchcomplete": "",
    "query": {
        "searchinfo": {"totalhits": 2},
        "search": [
            {
                "ns": 0,
                "title": "Laika",
                "pageid": 1,
                "snippet": 'La cagnolina <span class="searchmatch">Laika</span> &quot;imbarcata&quot; su Sputnik 2',
            },
            {"ns": 0, "title": "Laika (cane)", "pageid": 2, "snippet": "Altro &amp; altro"},
        ],
    },
}
SUMMARY_IT = {
    "type": "standard",
    "title": "Laika",
    "extract": "Laika è uno dei nomi con cui è nota la cagnolina imbarcata a bordo dello Sputnik 2.",
    "content_urls": {"desktop": {"page": "https://it.wikipedia.org/wiki/Laika"}},
}
EMPTY = {"batchcomplete": "", "query": {"searchinfo": {"totalhits": 0}, "search": []}}
BRAVE = {
    "type": "search",
    "query": {"original": "meteo roma", "more_results_available": True},
    "web": {
        "type": "search",
        "results": [
            {
                "title": "Meteo <strong>Roma</strong>",
                "url": "https://example.com/meteo",
                "description": "Domani <strong>sole</strong> &amp; 24 gradi",
                "age": "2 hours ago",
            },
            {"title": "Strano", "url": "javascript:alert(1)", "description": "x" * 1000},
        ],
    },
}


class Response:
    def __init__(self, status=200, data=None, text=""):
        self.status_code = status
        self._data = data
        self.text = text
        self.headers = {}

    def json(self):
        if self._data is None:
            raise ValueError("not json")
        return self._data


class FakeSession:
    """Answers GETs from a function (url, params) -> Response or an exception; records the calls."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        answer = self.handler(url, params)
        if isinstance(answer, Exception):
            raise answer
        return answer


def wiki_handler(search_by_lang):
    def handle(url, params):
        lang = url.split("//")[1].split(".")[0]
        if url.endswith("/w/api.php"):
            return Response(data=search_by_lang.get(lang, EMPTY))
        return Response(data=SUMMARY_IT)

    return handle


class TextTests(unittest.TestCase):
    def test_strips_tags_and_entities(self):
        self.assertEqual(clean_text('<b>Ciao</b> &amp;  &quot;mondo&quot;\n<script>x</script>'), 'Ciao & "mondo" x')

    def test_caps_length(self):
        self.assertLessEqual(len(clean_text("a" * 1000)), MAX_SNIPPET_CHARS)

    def test_non_strings_are_empty(self):
        self.assertEqual(clean_text(None), "")

    def test_https_only_except_local(self):
        check_https("https://api.search.brave.com/x")
        check_https("http://127.0.0.1:8000/x")
        for url in ("http://example.com/x", "ftp://example.com", "file:///etc/passwd"):
            with self.assertRaises(ValueError):
                check_https(url)

    def test_whole_result_cap(self):
        big = [{"title": "t", "snippet": "s" * 900, "url": "", "source": "x"} for _ in range(10)]
        kept = cap_results(big)
        self.assertLessEqual(sum(len(str(v)) for r in kept for v in r.values()), MAX_RESULT_CHARS)
        self.assertGreaterEqual(len(kept), 1)


class WikipediaTests(unittest.TestCase):
    def client(self, handler, **kwargs):
        self.session = FakeSession(handler)
        kwargs.setdefault("retry_delay_s", 0)
        return WikipediaClient(session=self.session, **kwargs)

    def test_request_shapes(self):
        client = self.client(wiki_handler({"it": SEARCH_IT}), max_results=3, timeout_s=4)
        result = client.search("Laika cagnolina")
        search, summary = self.session.calls
        self.assertEqual(search["url"], "https://it.wikipedia.org/w/api.php")
        self.assertEqual(
            search["params"],
            {"action": "query", "list": "search", "srsearch": "Laika cagnolina", "format": "json", "srlimit": 3, "utf8": 1},
        )
        self.assertEqual(search["headers"]["User-Agent"], USER_AGENT)
        self.assertIn("github.com/kasuken/home-robot", USER_AGENT)
        self.assertEqual(search["timeout"], 4)
        self.assertEqual(summary["url"], "https://it.wikipedia.org/api/rest_v1/page/summary/Laika")
        self.assertEqual(summary["headers"]["User-Agent"], USER_AGENT)
        self.assertTrue(result["ok"])
        self.assertEqual(result["source"], "Wikipedia (it)")
        first, second = result["results"]
        self.assertTrue(first["snippet"].startswith("Laika è uno dei nomi"))
        self.assertEqual(first["url"], "https://it.wikipedia.org/wiki/Laika")
        self.assertEqual(second["snippet"], "Altro & altro")
        self.assertEqual(second["url"], "https://it.wikipedia.org/wiki/Laika_%28cane%29")
        self.assertNotIn("<", json.dumps(result))

    def test_falls_back_to_the_other_language(self):
        client = self.client(wiki_handler({"en": SEARCH_IT}))
        result = client.search("Laika")
        hosts = [c["url"].split("//")[1].split("/")[0] for c in self.session.calls]
        self.assertEqual(hosts[:2], ["it.wikipedia.org", "en.wikipedia.org"])
        self.assertEqual(result["source"], "Wikipedia (en)")
        self.assertTrue(result["results"])

    def test_no_results_is_ok_and_empty(self):
        result = self.client(wiki_handler({})).search("zzzz")
        self.assertTrue(result["ok"])
        self.assertEqual(result["results"], [])

    def test_summary_failure_keeps_the_search_snippets(self):
        def handle(url, params):
            return Response(data=SEARCH_IT) if url.endswith("api.php") else Response(404, text="nope")

        result = self.client(handle).search("Laika")
        self.assertTrue(result["ok"])
        self.assertTrue(result["results"][0]["snippet"].startswith("La cagnolina Laika"))

    def test_cache_hit_sends_nothing(self):
        client = self.client(wiki_handler({"it": SEARCH_IT}))
        first = client.search("Laika")
        count = len(self.session.calls)
        self.assertEqual(client.search("  laika "), first)
        self.assertEqual(len(self.session.calls), count)

    def test_errors_never_raise(self):
        for answer in (requests.ConnectionError("down"), Response(500), Response(403), Response(200, None), RuntimeError("x")):
            client = self.client(lambda u, p, a=answer: a)
            result = client.search("Laika")
            self.assertFalse(result["ok"], answer)
            self.assertIn("Wikipedia", result["error"])

    def test_one_retry_on_connection_error_and_5xx(self):
        answers = [requests.ConnectionError("x"), Response(503), Response(data=SEARCH_IT)]
        client = self.client(lambda u, p: answers.pop(0))
        self.assertFalse(client.search("a b")["ok"])  # two attempts, both failed
        self.assertEqual(len(self.session.calls), 2)
        answers = [Response(429), Response(data=SEARCH_IT), Response(data=SUMMARY_IT)]
        client = self.client(lambda u, p: answers.pop(0))
        self.assertTrue(client.search("c d")["ok"])

    def test_hourly_limit(self):
        now = [0.0]
        guard = WebGuard(2, clock=lambda: now[0])
        client = self.client(wiki_handler({"it": SEARCH_IT}), guard=guard)
        self.assertTrue(client.search("uno")["ok"])
        self.assertTrue(client.search("due")["ok"])
        limited = client.search("tre")
        self.assertFalse(limited["ok"])
        self.assertIn("limit reached", limited["error"])
        now[0] = 3601.0
        self.assertTrue(client.search("tre")["ok"])

    def test_plain_http_is_refused_for_remote_hosts(self):
        client = self.client(wiki_handler({}), base_url="http://{lang}.wikipedia.org")
        result = client.search("Laika")
        self.assertFalse(result["ok"])
        self.assertEqual(self.session.calls, [])


class BraveTests(unittest.TestCase):
    KEY = "BSA-secret-key-123"

    def client(self, handler, **kwargs):
        self.session = FakeSession(handler)
        kwargs.setdefault("retry_delay_s", 0)
        return BraveSearchClient(self.KEY, session=self.session, **kwargs)

    def test_request_shape(self):
        result = self.client(lambda u, p: Response(data=BRAVE), max_results=4, timeout_s=5).search("meteo roma")
        (call,) = self.session.calls
        self.assertEqual(call["url"], f"{BRAVE_URL}/res/v1/web/search")
        self.assertEqual(call["headers"]["X-Subscription-Token"], self.KEY)
        self.assertEqual(
            call["params"],
            {"q": "meteo roma", "count": 4, "country": "IT", "search_lang": "it", "safesearch": "strict"},
        )
        self.assertEqual(call["timeout"], 5)
        first, second = result["results"]
        self.assertEqual(first["title"], "Meteo Roma")
        self.assertEqual(first["snippet"], "Domani sole & 24 gradi")
        self.assertEqual(first["published"], "2 hours ago")
        self.assertEqual(first["source"], "Brave Search")
        self.assertEqual(second["url"], "")
        self.assertLessEqual(len(second["snippet"]), MAX_SNIPPET_CHARS)

    def test_safesearch_values(self):
        self.client(lambda u, p: Response(data=BRAVE), safesearch="moderate").search("x y")
        self.assertEqual(self.session.calls[0]["params"]["safesearch"], "moderate")
        with self.assertRaises(ValueError):
            BraveSearchClient("k", safesearch="off")

    def test_cache_and_limit_shared_through_the_guard(self):
        guard = WebGuard(1)
        client = self.client(lambda u, p: Response(data=BRAVE), guard=guard)
        wiki = WikipediaClient(session=FakeSession(wiki_handler({"it": SEARCH_IT})), guard=guard)
        self.assertTrue(client.search("meteo roma")["ok"])
        self.assertTrue(client.search("Meteo Roma")["ok"])  # cache: free
        self.assertEqual(len(self.session.calls), 1)
        self.assertIn("limit reached", wiki.search("Laika")["error"])

    def test_errors_never_raise_and_never_leak_the_key(self):
        for answer in (
            requests.ConnectionError(f"boom {self.KEY}"),
            Response(401, text=self.KEY),
            Response(429),
            Response(200, None),
        ):
            result = self.client(lambda u, p, a=answer: a).search("meteo")
            self.assertFalse(result["ok"])
            self.assertNotIn(self.KEY, json.dumps(result))

    def test_key_is_not_in_a_successful_result(self):
        result = self.client(lambda u, p: Response(data=BRAVE)).search("meteo")
        self.assertNotIn(self.KEY, json.dumps(result))


if __name__ == "__main__":
    unittest.main()
