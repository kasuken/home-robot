"""Vision requests: request shape, validation, limits, outages and the full pipeline."""

import base64
import contextlib
import io
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

import requests

from laika.ai_client import image_message
from laika.camera import Camera
from laika.config import AIConfig, VisionConfig, load_config, validate_vision
from laika.events import AIEndpointMonitor, EventBus, EventType
from laika.house import Pose, default_house
from laika.observations import InvalidObservation, Observer
from laika.robot import ImageObservation
from laika.safety import SafetyController
from laika.simulator import SimulatedRobot
from laika.vision import VisionDescriber, build_vision_client, shrink_jpeg
from laika.world_model import WorldModel, WorldModelUpdater
from tests.agent.test_ai_providers import FakeResponse, FakeSession, text_response
from tests.fakes import FakeClock

try:
    from PIL import Image
except ImportError:
    Image = None

FAKE_JPEG = b"\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9"
REPLY = {
    "description": "A sofa and a lamp in a living room.",
    "objects": [
        {"name": "Sofa", "confidence": 0.95, "position": "left"},
        {"name": "lamp", "confidence": 0.7, "position": "right"},
    ],
    "people": 1,
    "confidence": 0.9,
}
EXAMPLE = Path(__file__).resolve().parents[2] / "config" / "laika.example.toml"


def frame(jpeg=FAKE_JPEG, **kwargs):
    return ImageObservation(
        jpeg=jpeg, width=640, height=480, camera_pan=15.0, camera_tilt=-5.0, **kwargs
    )


def describer(session, *, ai=None, vision=None, monitor=None, clock=None):
    ai = ai or AIConfig(api_key="sk-test")
    vision = vision or VisionConfig()
    client = build_vision_client(ai, vision)
    client._session = session
    client.retries = 0
    kwargs = {"clock": clock} if clock else {}
    return VisionDescriber(client, vision, monitor, **kwargs)


def data_url(jpeg):
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


class RequestShapeTests(unittest.TestCase):
    def test_image_message_is_a_text_plus_data_url(self):
        message = image_message("what is this?", b"abc", detail="high")
        text, image = message["content"]
        self.assertEqual(message["role"], "user")
        self.assertEqual(text, {"type": "text", "text": "what is this?"})
        self.assertEqual(image["type"], "image_url")
        self.assertEqual(image["image_url"]["detail"], "high")
        self.assertEqual(image["image_url"]["url"], data_url(b"abc"))
        with self.assertRaises(ValueError):
            image_message("x", b"abc", detail="ultra")

    def test_openai_request(self):
        session = FakeSession(text_response(json.dumps(REPLY)))
        describer(session, vision=VisionConfig(model="gpt-4o", detail="auto"))(frame(), "kitchen")
        (call,) = session.calls
        body = call["json"]
        self.assertEqual(call["url"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(call["headers"], {"Authorization": "Bearer sk-test"})
        self.assertEqual(call["timeout"], 30.0)
        self.assertEqual(body["model"], "gpt-4o")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertNotIn("tools", body)
        system, user = body["messages"]
        self.assertEqual(system["role"], "system")
        self.assertIn("STRICT JSON", system["content"])
        self.assertIn("kitchen", user["content"][0]["text"])
        image = user["content"][1]["image_url"]
        self.assertEqual(image["detail"], "auto")
        self.assertEqual(image["url"], data_url(FAKE_JPEG))

    def test_the_ai_model_is_used_when_the_vision_model_is_empty(self):
        session = FakeSession(text_response(json.dumps(REPLY)))
        describer(session, ai=AIConfig(api_key="k", model="gpt-4o-mini"))(frame(), None)
        self.assertEqual(session.calls[0]["json"]["model"], "gpt-4o-mini")
        self.assertNotIn("robot is in", session.calls[0]["json"]["messages"][1]["content"][0]["text"])

    def test_azure_request_uses_the_vision_deployment(self):
        ai = AIConfig(
            provider="azure",
            base_url="https://res.openai.azure.com",
            api_key="az-key",
            azure_deployment="chat-deployment",
        )
        session = FakeSession(text_response(json.dumps(REPLY)))
        describer(session, ai=ai, vision=VisionConfig(model="vision-deployment"))(frame(), None)
        call = session.calls[0]
        self.assertIn("/openai/deployments/vision-deployment/chat/completions", call["url"])
        self.assertEqual(call["headers"], {"api-key": "az-key"})
        self.assertEqual(call["json"]["messages"][1]["content"][1]["type"], "image_url")

    def test_azure_falls_back_to_the_chat_deployment(self):
        ai = AIConfig(provider="azure", base_url="https://res.openai.azure.com", azure_deployment="chat-dep")
        self.assertIn("/deployments/chat-dep/", build_vision_client(ai, VisionConfig()).endpoint.url)

    def test_the_vision_client_is_separate_and_has_the_vision_timeout(self):
        client = build_vision_client(AIConfig(api_key="k"), VisionConfig(timeout_s=45))
        self.assertEqual(client.timeout_s, 45)
        self.assertEqual(len(client.history), 0)


class ResultTests(unittest.TestCase):
    def test_a_good_reply_becomes_a_validated_observation(self):
        observation = describer(FakeSession(text_response(json.dumps(REPLY))))(frame(), "living_room")
        self.assertEqual(observation.object_names, ["sofa", "lamp"])
        self.assertEqual(observation.people, 1)
        self.assertEqual(observation.confidence, 0.9)
        self.assertEqual(observation.source, "vision")
        self.assertEqual(observation.location, "living_room")
        self.assertEqual((observation.camera_pan, observation.camera_tilt), (15.0, -5.0))
        self.assertEqual(observation.objects[0].position, "left")

    def test_a_fenced_reply_is_still_accepted(self):
        reply = "```json\n" + json.dumps(REPLY) + "\n```"
        self.assertEqual(describer(FakeSession(text_response(reply)))(frame(), None).people, 1)

    def test_a_malformed_reply_is_rejected_with_invalid_observation(self):
        for reply in ("I see a sofa.", "", json.dumps({**REPLY, "confidence": 7}), json.dumps([1])):
            with self.subTest(reply=reply), self.assertRaises(InvalidObservation):
                describer(FakeSession(text_response(reply)))(frame(), None)

    def test_the_observer_logs_a_malformed_reply_once_and_makes_no_observation(self):
        bus = EventBus()
        bus.start()
        self.addCleanup(bus.close)
        created = []
        bus.subscribe(created.append, EventType.OBSERVATION_CREATED)
        observer = Observer(bus, describe=describer(FakeSession(text_response("not json"))))
        self.addCleanup(observer.close)
        image = frame()
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertNoLogs("laika.vision", "WARNING"):
            observer.submit(image, "kitchen")
            self.assertIsNone(observer.wait_for(image, 5))
        bus.drain()
        self.assertEqual(out.getvalue().count("observation rejected"), 1)
        self.assertEqual(len(observer.rejections), 1)
        self.assertEqual(created, [])


class LimitTests(unittest.TestCase):
    def test_an_oversized_image_is_not_sent(self):
        session = FakeSession(text_response(json.dumps(REPLY)))
        vision = VisionConfig(max_image_kb=10)
        with mock.patch.dict(sys.modules, {"PIL": None}):  # no Pillow: nothing can shrink it
            with self.assertLogs("laika.vision", "WARNING") as logs:
                result = describer(session, vision=vision)(frame(jpeg=b"x" * 11 * 1024), None)
        self.assertIsNone(result)
        self.assertEqual(session.calls, [])
        self.assertEqual(len(logs.records), 1)
        self.assertIn("max_image_kb", logs.output[0])

    def test_a_frame_without_jpeg_is_skipped(self):
        session = FakeSession()
        with self.assertLogs("laika.vision", "WARNING"):
            self.assertIsNone(describer(session)(frame(jpeg=None), None))
        self.assertEqual(session.calls, [])

    def test_the_rate_limit_skips_frames_and_recovers_after_a_minute(self):
        clock = FakeClock(0.0)
        session = FakeSession(*(text_response(json.dumps(REPLY)) for _ in range(3)))
        d = describer(session, vision=VisionConfig(max_per_minute=2), clock=clock)
        self.assertIsNotNone(d(frame(), None))
        self.assertIsNotNone(d(frame(), None))
        with self.assertLogs("laika.vision", "WARNING") as logs:
            self.assertIsNone(d(frame(), None))
        self.assertIn("max_per_minute", logs.output[0])
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(d.skipped, 1)
        clock.advance(61)
        self.assertIsNotNone(d(frame(), None))
        self.assertEqual(len(session.calls), 3)

    @unittest.skipIf(Image is None, "Pillow is not installed")
    def test_pillow_shrinks_a_large_frame_to_fit(self):
        buffer = io.BytesIO()
        Image.effect_noise((1600, 1200), 80).convert("RGB").save(buffer, format="JPEG", quality=95)
        big = buffer.getvalue()
        small = shrink_jpeg(big, 768, 80, 400 * 1024)
        self.assertLess(len(small), len(big))
        self.assertLessEqual(len(small), 400 * 1024)
        self.assertLessEqual(max(Image.open(io.BytesIO(small)).size), 768)
        session = FakeSession(text_response(json.dumps(REPLY)))
        self.assertIsNotNone(describer(session)(frame(jpeg=big), None))
        sent = session.calls[0]["json"]["messages"][1]["content"][1]["image_url"]["url"]
        self.assertEqual(base64.b64decode(sent.split(",", 1)[1]), small)

    def test_shrinking_is_a_no_op_without_pillow(self):
        with mock.patch.dict(sys.modules, {"PIL": None}):
            self.assertIsNone(shrink_jpeg(FAKE_JPEG, 768, 80, 1024))


class OutageTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.events = []
        self.bus.subscribe(self.events.append)
        self.robot = SimulatedRobot(verbose=False)
        self.safety = SafetyController(self.robot, enabled=True, speed=25)
        self.addCleanup(self.safety.close)
        self.safety.start()
        self.monitor = AIEndpointMonitor(
            self.bus,
            on_unavailable=lambda: self.safety.set_ai_available(False),
            on_recovered=lambda: self.safety.set_ai_available(True),
        )

    def test_an_outage_returns_none_and_stops_movement_until_recovery(self):
        session = FakeSession(requests.ConnectionError("down"), text_response(json.dumps(REPLY)))
        d = describer(session, monitor=self.monitor)
        self.assertIsNone(d(frame(), None))
        self.assertFalse(self.monitor.available)
        self.assertFalse(self.safety.ai_available)
        self.assertIsNotNone(d(frame(), None))
        self.assertTrue(self.monitor.available)
        self.assertTrue(self.safety.ai_available)
        self.bus.drain()
        self.assertEqual(
            [e.type for e in self.events],
            [EventType.AI_ENDPOINT_UNAVAILABLE, EventType.AI_ENDPOINT_RECOVERED],
        )

    def test_a_refused_request_is_a_settings_problem_not_an_outage(self):
        # e.g. a model without image support answering 400: driving must not stop for that.
        for status in (400, 401, 404):
            with self.subTest(status=status):
                d = describer(FakeSession(FakeResponse(status)), monitor=self.monitor)
                self.assertIsNone(d(frame(), None))
                self.assertTrue(self.monitor.available)
                self.assertTrue(self.safety.ai_available)

    def test_server_errors_are_an_outage_and_a_bad_reply_is_recovery(self):
        session = FakeSession(FakeResponse(503), text_response("not json"))
        d = describer(session, monitor=self.monitor)
        self.assertIsNone(d(frame(), None))
        self.assertFalse(self.monitor.available)
        with self.assertRaises(InvalidObservation):  # the endpoint answered: it is back
            d(frame(), None)
        self.assertTrue(self.monitor.available)

    def test_works_without_a_monitor(self):
        self.assertIsNone(describer(FakeSession(requests.Timeout("slow")))(frame(), None))


class _Handler(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.seen.append({"path": self.path, "body": body})
        message = {"role": "assistant", "content": json.dumps(REPLY)}
        payload = json.dumps({"choices": [{"message": message, "finish_reason": "stop"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class CameraRobot(SimulatedRobot):
    """A simulated body whose camera returns small fake JPEGs, like the real robot."""

    def capture_image(self):
        pan, tilt = self.head_pose()
        return ImageObservation(
            jpeg=FAKE_JPEG, width=640, height=480, camera_pan=pan, camera_tilt=tilt
        )


class PipelineTests(unittest.TestCase):
    def test_observer_publishes_for_a_real_jpeg_frame(self):
        bus = EventBus()
        bus.start()
        self.addCleanup(bus.close)
        created = []
        bus.subscribe(created.append, EventType.OBSERVATION_CREATED)
        observer = Observer(bus, describe=describer(FakeSession(text_response(json.dumps(REPLY)))))
        self.addCleanup(observer.close)
        image = frame()
        observer.submit(image, "kitchen", "/tmp/img.jpg")
        observation = observer.wait_for(image, 5)
        bus.drain()
        self.assertEqual(observation.image_path, "/tmp/img.jpg")
        (event,) = created
        self.assertEqual(event.data["location"], "kitchen")
        self.assertEqual(event.data["source"], "vision")

    def test_camera_to_world_model_over_real_http(self):
        _Handler.seen = []
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        bus = EventBus()
        bus.start()
        self.addCleanup(bus.close)
        world = WorldModel()
        updater = WorldModelUpdater(world, bus)
        self.addCleanup(updater.close)
        world.enter_room("living_room")
        created = []
        bus.subscribe(created.append, EventType.OBSERVATION_CREATED)
        ai = AIConfig(
            provider="openai-compatible",
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            model="llava",
        )
        vision = VisionConfig(timeout_s=5)
        robot = CameraRobot(verbose=False, house=default_house(), start=Pose(300, 200, 180), events=bus)
        camera = Camera(robot, events=bus, min_interval_s=0)
        observer = Observer(
            bus,
            describe=VisionDescriber(build_vision_client(ai, vision), vision, AIEndpointMonitor(bus)),
            default_location=lambda: world.current_room,
        )
        self.addCleanup(observer.close)
        observer.attach(camera)
        image = camera.capture()
        observation = observer.wait_for(image, 10)
        self.assertIsNotNone(observation)
        bus.drain()

        (request,) = _Handler.seen
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(request["body"]["model"], "llava")
        url = request["body"]["messages"][1]["content"][1]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), FAKE_JPEG)

        self.assertEqual(observation.location, "living_room")
        self.assertEqual(len(created), 1)
        self.assertEqual(world.where_is("sofa").room, "living_room")
        self.assertEqual(world.where_is("lamp").room, "living_room")


class ConfigTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "laika.toml"
            path.write_text(f"[vision]\n{text}\n", encoding="utf-8")
            return load_config(path).vision

    def test_defaults_are_valid_and_match_the_example_file(self):
        validate_vision(VisionConfig())
        self.assertEqual(load_config(EXAMPLE).vision, VisionConfig())

    def test_good_values_load(self):
        vision = self.load('enabled = false\nmodel = "gpt-4o"\ndetail = "high"\nmax_per_minute = 3')
        self.assertEqual(
            (vision.enabled, vision.model, vision.detail, vision.max_per_minute),
            (False, "gpt-4o", "high", 3),
        )

    def test_bad_values_name_the_setting(self):
        for text, name in (
            ('detail = "ultra"', "detail"),
            ("max_image_kb = 0", "max_image_kb"),
            ("max_image_kb = 1.5", "max_image_kb"),
            ("max_side_px = 99999", "max_side_px"),
            ("jpeg_quality = 5", "jpeg_quality"),
            ("timeout_s = 0", "timeout_s"),
            ("max_per_minute = 0", "max_per_minute"),
            ('enabled = "yes"', "enabled"),
            ("model = 3", "model"),
            ("colour = 1", "colour"),
        ):
            with self.subTest(text=text), self.assertRaises(ValueError) as caught:
                self.load(text)
            self.assertIn(name, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
