"""scan_environment: head-only scan, merge rules, tool result and the agent end to end."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from laika.actions import ActionContext, ActionPlayer, available_actions
from laika.agent import Agent
from laika.ai_client import ToolCall
from laika.camera import Camera
from laika.house import Pose, default_house
from laika.observations import Observation, Observer, SeenObject
from laika.prompts import build_tool_system_prompt
from laika.robot import ALL_CAPABILITIES, CAMERA_PAN
from laika.safety import SafetyController
from laika.scan import Scanner, label_for
from laika.simulator import SimulatedRobot
from laika.tools import ToolRegistry, Turn
from laika.world_model import WorldModel
from tests.agent.test_agent import ScriptedClient, calls, text
from tests.fakes import FakeClock

ROOT = Path(__file__).resolve().parents[2]
# Facing west in the living room: the sofa is ahead, the guitars to the left, the backpack right.
FACING_WEST = Pose(300, 200, 180)


class FakeObserver:
    """Returns prepared observations in capture order; records what was waited for."""

    def __init__(self, *observations):
        self.observations = list(observations)
        self.frames = []

    def wait_for(self, image, timeout_s=2.0):
        self.frames.append(image)
        return self.observations.pop(0) if self.observations else None


def obs(*objects, people=0, confidence=1.0, location="living_room"):
    return Observation(
        description="x",
        objects=tuple(o if isinstance(o, SeenObject) else SeenObject(o) for o in objects),
        people=people,
        confidence=confidence,
        location=location,
    )


class ScanTestCase(unittest.TestCase):
    start = FACING_WEST

    def setUp(self):
        self.robot = SimulatedRobot(verbose=False, house=default_house(), start=self.start)
        self.camera = Camera(self.robot, min_interval_s=0.0, save_images=False)
        self.observer = Observer(None)
        self.observer.attach(self.camera)
        self.addCleanup(self.observer.close)

    def scanner(self, observer=None, **kwargs):
        kwargs.setdefault("settle_s", 0.0)
        return Scanner(self.robot, self.camera, observer or self.observer, **kwargs)


class SimulatedScanTests(ScanTestCase):
    def test_labels_follow_the_pan_convention(self):
        labels = [label_for(p) for p in (-60, -5, 0, 5, 60)]
        self.assertEqual(labels, ["left", "centre", "centre", "centre", "right"])

    def test_living_room_objects_are_found_on_the_expected_sides(self):
        scan = self.scanner(observation_wait_s=2.0).scan()
        by_name = {o.name: o.seen_at for o in scan.objects}
        self.assertEqual(by_name["sofa"], ("centre",))
        self.assertEqual(by_name["television"], ("centre",))
        self.assertEqual(by_name["guitars"], ("left",))
        self.assertEqual(by_name["backpack"], ("centre", "right"))
        self.assertNotIn("keyboard", by_name)
        self.assertEqual(scan.location, "living_room")
        self.assertTrue(scan.ok)
        self.assertEqual([p.label for p in scan.positions], ["left", "centre", "right"])

    def test_default_start_sees_the_keyboard_on_the_right(self):
        self.robot = SimulatedRobot(verbose=False, house=default_house())
        self.camera = Camera(self.robot, min_interval_s=0.0, save_images=False)
        self.observer = Observer(None)
        self.observer.attach(self.camera)
        self.addCleanup(self.observer.close)
        scan = self.scanner().scan()
        self.assertEqual({o.name: o.seen_at for o in scan.objects}["keyboard"], ("right",))

    def test_wheels_never_move(self):
        before = self.robot.pose()
        self.scanner().scan()
        self.assertEqual(self.robot.pose(), before)
        self.assertEqual(self.robot.speed, 0)

    def test_head_is_restored_after_a_scan(self):
        self.robot.look(20, -10)
        self.scanner().scan()
        self.assertEqual(self.robot.head_pose(), (20, -10))

    def test_head_is_restored_after_an_exception(self):
        self.robot.look(20, -10)
        observer = mock.Mock()
        observer.wait_for.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            self.scanner(observer).scan()
        self.assertEqual(self.robot.head_pose(), (20, -10))

    def test_all_frames_are_captured_before_waiting_for_observations(self):
        order = []
        real_capture = self.robot.capture_image

        def capture():
            order.append("capture")
            return real_capture()

        self.robot.capture_image = capture
        observer = mock.Mock()
        observer.wait_for.side_effect = lambda *a, **k: order.append("wait")
        self.scanner(observer).scan()
        self.assertEqual(order, ["capture"] * 3 + ["wait"] * 3)

    def test_frames_are_taken_at_the_three_pans(self):
        pans = []
        self.camera.on_capture.append(lambda image, room, path: pans.append(image.camera_pan))
        self.scanner().scan()
        self.assertEqual(pans, [-60, 0, 60])


class FreshFrameTests(ScanTestCase):
    def test_a_limited_camera_is_waited_for_so_no_frame_is_reused(self):
        clock = FakeClock(100.0)
        self.camera = Camera(self.robot, min_interval_s=0.5, save_images=False, clock=clock)
        observer = FakeObserver(obs("a"), obs("b"), obs("c"))
        scanner = Scanner(self.robot, self.camera, observer, settle_s=0.1, sleep=clock.advance)
        scan = scanner.scan()
        self.assertEqual(len({id(f) for f in observer.frames}), 3)
        self.assertEqual(scan.observed, 3)

    def test_a_camera_that_stays_limited_is_reported_not_fatal(self):
        clock = FakeClock(100.0)
        self.camera = Camera(
            self.robot, min_interval_s=0.0, max_per_minute=1, save_images=False, clock=clock
        )
        observer = FakeObserver(obs("a"))
        scan = Scanner(self.robot, self.camera, observer, settle_s=0.0, sleep=clock.advance).scan()
        self.assertEqual(scan.observed, 1)
        reasons = [p.reason for p in scan.positions]
        self.assertEqual(reasons, [None, "camera unavailable", "camera unavailable"])
        self.assertEqual(len(observer.frames), 1)

    def test_a_failing_camera_gives_no_observations(self):
        self.robot.faults.camera_error = True
        scan = self.scanner().scan()
        self.assertFalse(scan.ok)
        self.assertEqual({p.reason for p in scan.positions}, {"camera unavailable"})
        self.assertEqual(self.robot.head_pose(), (0.0, 0.0))


class MergeTests(ScanTestCase):
    def test_objects_merge_with_max_confidence_and_people_are_a_max(self):
        observer = FakeObserver(
            obs(SeenObject("sofa", 0.6, "right"), "lamp", people=1, confidence=0.5),
            obs(SeenObject("sofa", 0.9, "left"), people=2, confidence=1.0),
            obs(people=1, confidence=0.9, location="office"),
        )
        scan = self.scanner(observer).scan()
        sofa = next(o for o in scan.objects if o.name == "sofa")
        self.assertEqual(sofa.seen_at, ("left", "centre"))
        self.assertEqual(sofa.confidence, 0.9)
        self.assertEqual(sofa.in_frame, ("right", "left"))
        self.assertEqual(scan.people, 2)
        self.assertEqual(scan.location, "living_room")
        self.assertAlmostEqual(scan.confidence, 0.8)
        self.assertIn("left: sofa, lamp", scan.summary)
        self.assertIn("People seen: 2", scan.summary)

    def test_a_frame_without_observation_is_reported_not_fatal(self):
        observer = FakeObserver(obs("sofa"), None, obs("lamp"))
        scan = self.scanner(observer).scan()
        self.assertTrue(scan.ok)
        centre = scan.positions[1]
        self.assertIsNone(centre.observation)
        self.assertEqual(centre.reason, "no observation")
        self.assertIn("centre: no observation", scan.summary)
        self.assertEqual(centre.to_dict()["note"], "no observation")


class ScanToolTests(unittest.TestCase):
    capabilities = ALL_CAPABILITIES
    start = FACING_WEST

    def build(self, *script):
        self.robot = SimulatedRobot(
            verbose=False, house=default_house(), start=self.start, capabilities=self.capabilities
        )
        self.safety = SafetyController(self.robot, speed=25)
        self.addCleanup(self.safety.close)
        self.safety.start()
        self.camera = Camera(self.robot, min_interval_s=0.0, save_images=False)
        self.observer = Observer(None)
        self.observer.attach(self.camera)
        self.addCleanup(self.observer.close)
        self.world = WorldModel()
        self.world.enter_room("living_room")
        allowed = available_actions(self.capabilities, True)
        self.actions = ActionPlayer(
            ActionContext(self.robot, self.safety), allowed, talk_motion=False
        )
        self.addCleanup(self.actions.close)
        self.client = ScriptedClient(*script)
        self.registry = ToolRegistry(
            self.robot,
            self.safety,
            world=self.world,
            actions=self.actions,
            camera=self.camera,
            observer=self.observer,
            scan_settle_s=0.0,
            speak=self.robot.speak,
        )
        self.agent = Agent(
            self.client, self.registry, build_tool_system_prompt(allowed, 3), world=self.world
        )

    def run_scan(self):
        call = ToolCall("c1", "scan_environment", {}, "{}")
        with contextlib.redirect_stdout(io.StringIO()):
            return self.registry.run(call)

    def test_tool_result_shape_and_no_image_bytes(self):
        self.build()
        result = self.run_scan()
        self.assertTrue(result["ok"])
        self.assertEqual(result["location"], "living_room")
        self.assertEqual([p["position"] for p in result["positions"]], ["left", "centre", "right"])
        names = {o["name"] for o in result["objects"]}
        self.assertTrue({"sofa", "television", "guitars"} <= names)
        self.assertIn("summary", result)
        self.assertEqual(result["people"], 0)
        self.assertNotIn("jpeg", json.dumps(result))

    def test_total_failure_is_not_ok(self):
        self.build()
        self.robot.faults.camera_error = True
        result = self.run_scan()
        self.assertFalse(result["ok"])
        self.assertIn("camera unavailable", result["error"])

    def test_without_an_observer_nothing_can_be_seen(self):
        self.build()
        self.registry.scanner.observer = None
        result = self.run_scan()
        self.assertFalse(result["ok"])
        self.assertIn("no observation", result["error"])

    def test_not_offered_without_camera_pan(self):
        self.capabilities = ALL_CAPABILITIES - {CAMERA_PAN}
        self.build()
        self.assertNotIn("scan_environment", self.registry.names())
        self.assertFalse(self.run_scan()["ok"])

    def test_takes_no_arguments(self):
        self.build()
        spec = next(s for s in self.registry.specs() if s.name == "scan_environment")
        self.assertEqual(spec.parameters["properties"], {})

    def test_a_stale_turn_does_not_scan(self):
        self.build()
        turn = Turn(0, lambda: 1)
        self.robot.look(10, 0)
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.registry.run(ToolCall("c1", "scan_environment", {}, "{}"), turn)
        self.assertFalse(result["ok"])
        self.assertEqual(self.robot.head_pose(), (10, 0))
        self.assertIsNone(self.camera.last)

    def test_agent_scans_then_speaks_italian(self):
        self.build(
            calls(("scan_environment", {})),
            text("A sinistra ci sono le chitarre, davanti il divano."),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            reply = self.agent.respond("guardati intorno")
        self.assertEqual(reply.tool_calls_made, ["scan_environment"])
        self.assertIn("divano", reply.text)
        tool_messages = [m for m in self.client.requests[1]["messages"] if m["role"] == "tool"]
        payload = json.loads(tool_messages[0]["content"])
        self.assertIn("sofa", {o["name"] for o in payload["objects"]})
        self.assertIn("scan_environment", [t.name for t in self.client.requests[0]["tools"]])

    def test_prompt_mentions_scan_environment(self):
        prompt = build_tool_system_prompt(available_actions(ALL_CAPABILITIES, True), 3)
        self.assertIn("scan_environment", prompt)
        self.assertIn("guardati intorno", prompt)


class _FakeChat(BaseHTTPRequestHandler):
    requests_seen: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests_seen.append(body)
        if any(m["role"] == "tool" for m in body["messages"]):
            message = {"role": "assistant", "content": "Vedo una tastiera a destra."}
            finish = "stop"
        else:
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "scan_environment", "arguments": "{}"},
                    }
                ],
            }
            finish = "tool_calls"
        choice = {"index": 0, "message": message, "finish_reason": finish}
        payload = json.dumps({"choices": [choice]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class SimEndToEndTests(unittest.TestCase):
    def test_run_laika_sim_scans_through_a_fake_chat_server(self):
        _FakeChat.requests_seen = []
        server = HTTPServer(("127.0.0.1", 0), _FakeChat)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as data_dir:
            env = {
                **os.environ,
                "LAIKA_DATA_DIR": data_dir,
                "LAIKA_AI_BASE_URL": f"http://127.0.0.1:{server.server_address[1]}/v1",
                "OPENAI_API_KEY": "x",
                "PYTHONIOENCODING": "utf-8",
            }
            done = subprocess.run(
                [sys.executable, str(ROOT / "run-laika.py"), "--sim"],
                input="ehi laika\nguardati intorno\nbasta\n\n",
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                cwd=ROOT,
                timeout=90,
            )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertGreaterEqual(len(_FakeChat.requests_seen), 2, done.stdout)
        tool_results = [
            m["content"]
            for body in _FakeChat.requests_seen
            for m in body["messages"]
            if m["role"] == "tool"
        ]
        self.assertTrue(tool_results, done.stdout)
        self.assertIn("keyboard", tool_results[0])
        self.assertIn("Vedo una tastiera a destra.", done.stdout)


if __name__ == "__main__":
    unittest.main()
