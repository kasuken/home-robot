"""Looking out of the windows (#48): the watcher, the outside schema, changes, the tool and the agent's context."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path

from laika.agent import Agent
from laika.ai_client import AIUnavailable, AssistantTurn
from laika.battery import BatteryState
from laika.behaviour import Arbiter, Priority
from laika.camera import Camera
from laika.capture_policy import CapturePolicy
from laika.config import CaptureConfig, CuriosityConfig, OutsideConfig, VisionConfig
from laika.curiosity import InterestGate
from laika.events import EventBus, EventType
from laika.house import OutsideView, Pose, default_house
from laika.memory import Memory
from laika.observations import InvalidObservation, Observer
from laika.outside import (
    OUTSIDE_SYSTEM_PROMPT,
    OutsideDescriber,
    OutsideObservation,
    WindowWatcher,
    detect_changes,
    outside_from_scene,
    parse_outside_observation,
)
from laika.robot import ImageObservation
from laika.safety import SafetyController
from laika.simulator import SimulatedRobot
from laika.store import DB_NAME, Store
from laika.tools import Turn, ToolRegistry
from laika.vision import VisionDescriber
from laika.windows import WindowRegistry
from laika.world_model import WorldModel, WorldModelUpdater
from tests.agent.test_agent import ScriptedClient, text
from tests.agent.test_tools import call
from tests.fakes import FakeClock, LookRecordingRobot

LIVING_ROOM = Pose(300, 200, 0)  # the window is on the right, at pan 60
OFFICE = Pose(100, 500, 0)  # no window here
GOOD = {
    "weather": "rain",
    "light": "dusk",
    "description": "Rain falls on a wet street.",
    "events": ["a car passed", "birds on the balcony"],
    "people_outside": 2,
    "confidence": 0.8,
}
FAKE_JPEG = b"\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9"


def image(**kwargs) -> ImageObservation:
    return ImageObservation(jpeg=None, width=640, height=480, camera_pan=60.0, camera_tilt=10.0, **kwargs)


class ParseTests(unittest.TestCase):
    def parse(self, output):
        return parse_outside_observation(output, image=image(), room="Living_Room")

    def test_a_good_answer(self):
        seen = self.parse(json.dumps(GOOD))
        self.assertEqual((seen.weather, seen.light, seen.people_outside), ("rain", "dusk", 2))
        self.assertEqual(seen.events, ("a car passed", "birds on the balcony"))
        self.assertEqual((seen.room, seen.pan, seen.tilt, seen.source), ("living_room", 60.0, 10.0, "vision"))
        self.assertIn("rain, dusk", seen.summary)
        self.assertIn("2 people outside", seen.summary)

    def test_a_fenced_answer_and_a_minimal_one(self):
        minimal = {"weather": "Sunny ", "light": "day", "description": "Blue sky.", "confidence": 1}
        seen = self.parse("```json\n" + json.dumps(minimal) + "\n```")
        self.assertEqual((seen.weather, seen.events, seen.people_outside), ("sunny", (), 0))

    def test_malformed_answers_are_rejected(self):
        broken = [
            "not json at all",
            "[1, 2]",
            {**GOOD, "weather": "overcast"},
            {**GOOD, "light": "evening"},
            {**GOOD, "weather": None},
            {**GOOD, "description": ""},
            {**GOOD, "description": 5},
            {**GOOD, "confidence": 1.5},
            {**GOOD, "events": "a car"},
            {**GOOD, "events": [1]},
            {**GOOD, "events": ["x"] * 6},
            {**GOOD, "events": ["y" * 81]},
            {key: value for key, value in GOOD.items() if key != "confidence"},
            {key: value for key, value in GOOD.items() if key != "weather"},
        ]
        for output in broken:
            with self.subTest(output=output), self.assertRaises(InvalidObservation):
                self.parse(output if isinstance(output, str) else json.dumps(output))

    def test_people_outside_is_a_count_and_nothing_else(self):
        for people in (["a man in a red coat"], "two", 2.5, True, -1, 99, {"count": 2}):
            with self.subTest(people=people), self.assertRaises(InvalidObservation):
                self.parse(json.dumps({**GOOD, "people_outside": people}))

    def test_unknown_keys_are_dropped(self):
        seen = self.parse(json.dumps({**GOOD, "people": [{"name": "Marco", "clothes": "red coat"}]}))
        self.assertNotIn("Marco", json.dumps(seen.to_dict()))

    def test_the_prompt_forbids_identifying_people(self):
        self.assertIn("Never identify anyone", OUTSIDE_SYSTEM_PROMPT)
        self.assertIn("people_outside", OUTSIDE_SYSTEM_PROMPT)

    def test_a_round_trip_through_the_store_form(self):
        seen = self.parse(json.dumps(GOOD))
        again = OutsideObservation.from_dict(json.loads(json.dumps(seen.to_dict())))
        self.assertEqual(again, seen)


class ChangeTests(unittest.TestCase):
    def look(self, weather="sunny", light="day", people=0, events=(), at=1000.0):
        return OutsideObservation(weather, light, "x", tuple(events), people, 1.0, timestamp=at)

    def changes(self, before, after):
        return [text for _, text in detect_changes(before, after)]

    def test_weather_and_light_changes(self):
        self.assertEqual(self.changes(self.look("cloudy"), self.look("rain", at=1060)), ["started raining"])
        self.assertEqual(self.changes(self.look("rain"), self.look("cloudy", at=1060)), ["stopped raining"])
        self.assertEqual(self.changes(self.look("cloudy"), self.look("snow", at=1060)), ["started snowing"])
        self.assertEqual(self.changes(self.look("cloudy"), self.look("sunny", at=1060)), ["the sun came out"])
        self.assertEqual(self.changes(self.look(light="day"), self.look(light="night", at=1060)), ["it got dark"])
        self.assertEqual(self.changes(self.look(light="day"), self.look(light="dusk", at=1060)), ["it is getting dark"])
        self.assertEqual(
            self.changes(self.look("sunny", "day"), self.look("rain", "dusk", at=1060)),
            ["started raining", "it is getting dark"],
        )

    def test_a_busier_street(self):
        self.assertEqual(self.changes(self.look(), self.look(people=2, events=["a bus"], at=1060)), ["the street got busy"])
        self.assertEqual(self.changes(self.look(people=1), self.look(people=2, at=1060)), [])

    def test_nothing_to_compare(self):
        now = self.look("rain", at=1060)
        self.assertEqual(detect_changes(None, now), [])
        self.assertEqual(detect_changes(self.look("unknown", "unknown"), now), [])
        self.assertEqual(detect_changes(self.look("sunny"), self.look("unknown", "unknown", at=1060)), [])
        self.assertEqual(detect_changes(self.look("sunny"), self.look("rain", at=1000 + 7 * 3600)), [])  # too old
        self.assertEqual(detect_changes(self.look("sunny"), self.look("sunny", at=1060)), [])


class WatchCase(unittest.TestCase):
    """The house simulator with the real arbiter, safety, camera and observer."""

    def build(self, *, start=LIVING_ROOM, config=None, clock=None, wall=None, curiosity=None, store=None, **options):
        out = io.StringIO()
        redirect = contextlib.redirect_stdout(out)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.stdout = out
        self.house = default_house()
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.events = []
        self.bus.subscribe(self.events.append)
        self.world = WorldModel()
        self.addCleanup(WorldModelUpdater(self.world, self.bus).close)
        self.memory = Memory()
        self.robot = LookRecordingRobot(house=self.house, start=start)
        self.world.enter_room(self.robot.current_room())
        self.camera = Camera(self.robot, None, self.bus, min_interval_s=0, max_per_minute=0, save_images=False)
        self.observer = Observer(self.bus, default_location=lambda: self.world.current_room)
        self.observer.attach(self.camera)
        self.addCleanup(self.observer.close)
        self.safety = SafetyController(self.robot, speed=25, allow_autonomous=True)
        self.addCleanup(self.safety.close)
        self.safety.start()
        self.arbiter = Arbiter(self.safety, None, lease_timeout_s=2.0)
        self.addCleanup(self.arbiter.close)
        self.windows = WindowRegistry(store, self.bus)
        self.addCleanup(self.windows.close)
        curiosity = curiosity or CuriosityConfig(threshold=0.3, cooldown_s=0.0)
        options.setdefault("settle_s", 0.0)
        self.watcher = WindowWatcher(
            self.arbiter, self.robot, self.camera, self.windows, self.world, self.memory,
            config or OutsideConfig(min_interval_s=0),
            gate=InterestGate(curiosity, self.bus), store=store, clock=clock or time.time,
            **options,
        )
        return self.watcher

    def learn_the_window(self):
        """What a scan does: look at the right, see the window, learn it from the observation."""
        self.robot.look(60, 5)
        self.observer.wait_for(self.camera.capture("scan"))
        self.bus.drain(2.0)
        self.robot.look(0, 0)
        self.robot.poses.clear()
        (window,) = self.windows.in_room(self.world.current_room)
        return window

    def kinds(self, kind):
        return [e for e in self.memory.episodes.recent(50) if e.kind == kind]

    def pictures(self, reason):
        self.bus.drain(2.0)
        return [e for e in self.events if e.type is EventType.IMAGE_CAPTURED and e.data.get("reason") == reason]


class WatcherTests(WatchCase):
    def test_she_watches_the_window_and_records_an_outside_episode(self):
        watcher = self.build()
        window = self.learn_the_window()
        before = len(self.observer.log)
        self.house.set_outside("sunny", "day", ["a car passed"], people=1)
        result = watcher.watch(window)
        self.assertTrue(result.ok, result)
        # The head went to the window, tilted up a little, and came back where it was.
        self.assertEqual(self.robot.poses[0], (60, 15.0))  # tilted up a little from where it was seen
        self.assertEqual(self.robot.poses[-1], (0, 0))
        (picture,) = self.pictures("outside")
        self.assertEqual(picture.data["pan"], 60)
        # The outside frame is read by the watcher, not made into an ordinary observation.
        self.assertEqual(len(self.observer.log), before)
        seen = watcher.latest()
        self.assertEqual((seen.weather, seen.light, seen.people_outside), ("sunny", "day", 1))
        self.assertEqual(seen.events, ("a car passed",))
        self.assertEqual((seen.room, seen.window, seen.source), ("living_room", window.id, "simulator"))
        (episode,) = self.kinds("outside")
        self.assertIn("sunny, day", episode.summary)
        self.assertEqual(episode.location, "living_room")
        self.assertIn("outside", episode.subjects)
        self.assertEqual(self.kinds("outside_change"), [])
        self.assertEqual(self.robot.spoken, [])  # the activity never speaks

    def test_done_when_it_rains_she_notices_the_change_and_it_is_interesting(self):
        watcher = self.build()
        window = self.learn_the_window()
        self.house.set_outside("sunny", "day")
        self.assertTrue(watcher.watch(window).ok)
        self.house.set_outside("rain", "day", ["a car passed"])
        result = watcher.watch(window)
        self.assertTrue(result.ok, result)
        self.assertEqual(result.changes, ["started raining"])
        (change,) = self.kinds("outside_change")
        self.assertEqual(change.summary, "Outside: started raining")
        self.assertEqual(len(self.kinds("outside")), 2)
        self.bus.drain(2.0)
        (event,) = [e for e in self.events if e.type is EventType.INTERESTING_CHANGE_DETECTED]
        self.assertEqual(event.data["reasons"], ["outside: started raining"])
        self.assertEqual(event.data["source"], "outside")
        self.assertEqual(event.data["total"], 0.6)
        self.assertEqual(event.data["location"], "living_room")
        self.assertEqual(self.robot.spoken, [])

    def test_a_change_is_not_interesting_when_the_setting_says_so(self):
        watcher = self.build(config=OutsideConfig(min_interval_s=0, change_interest=0.0))
        window = self.learn_the_window()
        self.house.set_outside("sunny", "day")
        watcher.watch(window)
        self.house.set_outside("rain", "day")
        watcher.watch(window)
        self.assertEqual(len(self.kinds("outside_change")), 1)  # still remembered
        self.bus.drain(2.0)
        self.assertEqual([e for e in self.events if e.type is EventType.INTERESTING_CHANGE_DETECTED], [])

    def test_an_outside_change_does_not_make_the_capture_policy_take_another_picture(self):
        watcher = self.build()
        policy = CapturePolicy(self.camera, self.bus, CaptureConfig(interesting_cooldown_s=0))
        policy.start()
        self.addCleanup(policy.close)
        window = self.learn_the_window()
        self.house.set_outside("sunny", "day")
        watcher.watch(window)
        self.house.set_outside("rain", "day")
        watcher.watch(window)
        self.bus.drain(2.0)
        policy.flush()
        self.assertEqual(policy.stats["interesting"].captured, 0)
        self.assertEqual(len(self.pictures("interesting")), 0)

    def test_light_changes_through_the_scripted_schedule(self):
        watcher = self.build()
        window = self.learn_the_window()
        self.house.set_outside("sunny", "dusk")
        watcher.watch(window)
        self.house.set_outside("sunny", "night")
        self.assertEqual(watcher.watch(window).changes, ["it got dark"])

    def test_rate_limits(self):
        clock = FakeClock(1_000_000.0)
        watcher = self.build(clock=clock, config=OutsideConfig(min_interval_s=1800, max_per_day=3))
        window = self.learn_the_window()
        self.assertEqual(watcher.watch(window).status, "done")
        again = watcher.watch(window)
        self.assertEqual(again.status, "refused")
        self.assertIn("short while", again.reason)
        self.assertEqual(len(self.pictures("outside")), 1)  # nothing was photographed for the refused look
        self.assertFalse(watcher.due(window))
        clock.advance(1801)
        self.assertTrue(watcher.due(window))
        self.assertEqual(watcher.watch(window).status, "done")
        clock.advance(1801)
        self.assertEqual(watcher.watch(window).status, "done")
        clock.advance(1801)
        capped = watcher.watch(window)
        self.assertEqual((capped.status, capped.reason), ("refused", "daily limit reached"))
        self.assertFalse(watcher.due(window))
        clock.advance(86400)
        self.assertEqual(watcher.watch(window).status, "done")

    def test_poll_watches_the_window_of_this_room_when_it_is_due(self):
        watcher = self.build(config=OutsideConfig())
        self.assertIsNone(watcher.poll())  # no window known yet
        self.learn_the_window()
        result = watcher.poll()
        self.assertTrue(result.ok)
        self.assertIsNone(watcher.poll())  # looked a moment ago
        self.assertIsNone(self.arbiter.holder)  # the lease was given back

    def test_no_window_in_this_room(self):
        watcher = self.build(start=OFFICE)
        self.assertEqual(watcher.windows_here(), [])
        self.assertIsNone(watcher.poll())
        self.assertEqual(self.pictures("outside"), [])

    def test_a_window_with_no_outside_in_view_records_nothing(self):
        watcher = self.build()
        window = self.learn_the_window()
        self.windows.note_window("living_room", -60, source="user")  # a told window that faces the wall
        wrong = [w for w in self.windows.in_room("living_room") if w.id != window.id][0]
        result = watcher.watch(wrong)
        self.assertEqual(result.status, "not_seen")
        self.assertEqual(self.kinds("outside"), [])
        self.assertIsNone(watcher.latest())

    def test_a_taught_window_without_a_direction_is_watched_straight_ahead(self):
        watcher = self.build()
        window = self.windows.note_window("living_room", None, source="user")
        result = watcher.watch(window)
        self.assertEqual(result.status, "not_seen")  # nothing in the frame
        self.assertEqual(self.robot.poses[0], (0.0, 10.0))

    def test_pause_stops_it(self):
        watcher = self.build()
        window = self.learn_the_window()
        self.safety.pause_autonomy()
        result = watcher.watch(window)
        self.assertEqual(result.status, "refused")
        self.assertIsNone(watcher.poll())
        self.assertEqual(self.pictures("outside"), [])
        self.assertEqual(self.robot.poses, [])
        self.safety.resume_autonomy()
        self.assertEqual(watcher.watch(window).status, "done")

    def test_a_critical_battery_stops_it(self):
        watcher = self.build()
        window = self.learn_the_window()
        self.safety.set_battery_state(BatteryState.CRITICAL)
        self.assertEqual(watcher.watch(window).status, "refused")
        self.assertEqual(self.pictures("outside"), [])
        self.assertEqual(self.robot.poses, [])
        self.safety.set_battery_state(BatteryState.OK)
        self.assertEqual(watcher.watch(window).status, "done")

    def test_a_conversation_has_the_body_first(self):
        watcher = self.build()
        window = self.learn_the_window()
        chat = self.arbiter.request("conversation", Priority.CONVERSATION)
        self.addCleanup(chat.release)
        self.assertIsNone(watcher.poll())
        self.assertEqual(watcher.watch(window).status, "refused")

    def test_a_lease_taken_while_looking_ends_the_look(self):
        watcher = self.build()
        window = self.learn_the_window()
        lease = self.arbiter.request("test", Priority.CURIOSITY, resumable=False)
        lease.release()  # ended: every check fails
        result = watcher.watch(window, lease)
        self.assertEqual(result.status, "aborted")
        self.assertEqual(self.kinds("outside"), [])

    def test_the_camera_failing_is_not_an_outside_look(self):
        watcher = self.build()
        window = self.learn_the_window()
        self.robot.faults.camera_error = True
        self.assertEqual(watcher.watch(window).status, "no_camera")

    def test_the_latest_look_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / DB_NAME)
            watcher = self.build(store=store, config=OutsideConfig())
            window = self.learn_the_window()
            self.house.set_outside("fog", "night")
            watcher.watch(window)
            self.assertTrue(store.flush())
            store.close()
            again = Store(Path(folder) / DB_NAME)
            revived = WindowWatcher(
                self.arbiter, self.robot, self.camera, self.windows, self.world, Memory(), OutsideConfig(),
                store=again,
            )
            self.assertEqual((revived.latest().weather, revived.latest().light), ("fog", "night"))
            self.assertFalse(revived.due(window))  # the interval did not start again
            again.close()


class VisionPathTests(WatchCase):
    """On the robot: the frame goes to the vision model with the dedicated prompt."""

    class Eyes(LookRecordingRobot):
        """A robot whose camera returns a real-looking frame instead of the simulator's ground truth."""

        def capture_image(self):
            real = super().capture_image()
            return ImageObservation(
                jpeg=FAKE_JPEG, width=real.width, height=real.height,
                camera_pan=real.camera_pan, camera_tilt=real.camera_tilt,
            )

    class Model:
        def __init__(self, *replies):
            self.replies = list(replies)
            self.messages = []

        def complete(self, messages, tools=None, tool_choice="auto", json_mode=False):
            self.messages.append(messages)
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return AssistantTurn(content=reply if isinstance(reply, str) else json.dumps(reply))

    def setUp(self):
        self.model = self.Model()
        self.vision = VisionDescriber(self.model, VisionConfig())
        self.describe = OutsideDescriber(self.vision)

    def watcher_with(self, *replies):
        self.model.replies = list(replies)
        watcher = self.build(describe=self.describe)
        self.robot.__class__ = self.Eyes  # same robot, real-looking frames
        self.world.enter_room("living_room")
        window = self.windows.note_window("living_room", 60, 10, source="vision")
        return watcher, window

    def test_the_dedicated_prompt_and_a_valid_answer(self):
        watcher, window = self.watcher_with(GOOD)
        result = watcher.watch(window)
        self.assertTrue(result.ok, result)
        (messages,) = self.model.messages
        self.assertIn("OUTSIDE the window", messages[0]["content"])
        self.assertIn("living room", json.dumps(messages[1]))
        seen = watcher.latest()
        self.assertEqual((seen.weather, seen.source, seen.people_outside), ("rain", "vision", 2))
        self.assertEqual(len(self.pictures("outside")), 1)
        self.assertEqual(len(self.observer.log), 0)  # one request for the picture, not two

    def test_a_malformed_answer_is_logged_and_not_remembered(self):
        watcher, window = self.watcher_with({**GOOD, "weather": "overcast"})
        result = watcher.watch(window)
        self.assertEqual(result.status, "error")
        self.assertIn("weather", result.reason)
        self.assertIn("weather", watcher.rejections[-1])
        self.assertIsNone(watcher.latest())
        self.assertEqual(self.kinds("outside"), [])
        self.assertIn("outside observation rejected", self.stdout.getvalue())

    def test_a_list_of_people_is_refused_not_stored(self):
        watcher, window = self.watcher_with({**GOOD, "people_outside": ["a woman with a dog"]})
        self.assertEqual(watcher.watch(window).status, "error")
        self.assertIsNone(watcher.latest())

    def test_an_unreachable_model_is_no_answer(self):
        watcher, window = self.watcher_with(AIUnavailable("down"))
        self.assertEqual(watcher.watch(window).status, "no_answer")
        self.assertEqual(self.kinds("outside"), [])

    def test_the_vision_limit_is_shared(self):
        self.vision.config = VisionConfig(max_per_minute=1)
        watcher, window = self.watcher_with(GOOD, GOOD)
        self.assertTrue(watcher.watch(window).ok)
        self.assertEqual(watcher.watch(window).status, "no_answer")  # over max_per_minute, skipped
        self.assertEqual(self.vision.skipped, 1)
        self.assertEqual(len(self.model.messages), 1)

    def test_a_frame_of_a_wall_says_unknown_and_records_nothing(self):
        watcher, window = self.watcher_with({**GOOD, "weather": "unknown", "light": "unknown", "confidence": 0.2})
        self.assertEqual(watcher.watch(window).status, "not_seen")
        self.assertEqual(self.kinds("outside"), [])


class ToolTests(WatchCase):
    def registry(self, watcher):
        return ToolRegistry(
            self.robot, self.safety, world=self.world, memory=self.memory, camera=self.camera,
            observer=self.observer, outside=watcher,
        )

    def test_the_tool_is_offered_only_with_a_watcher(self):
        watcher = self.build()
        self.assertIn("look_outside", self.registry(watcher).names())
        plain = ToolRegistry(self.robot, self.safety, world=self.world)
        self.assertNotIn("look_outside", plain.names())

    def test_with_a_window_here_she_looks_now(self):
        watcher = self.build()
        self.learn_the_window()
        self.house.set_outside("rain", "dusk", ["a car passed"])
        result = self.registry(watcher).run(call("look_outside"))
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["source"], "looked_now")
        self.assertEqual((result["weather"], result["light"]), ("rain", "dusk"))
        self.assertEqual(result["events"], ["a car passed"])
        self.assertIn("ago", result)
        self.assertEqual(len(self.pictures("outside")), 1)
        self.assertEqual(self.robot.spoken, [])

    def test_a_look_from_a_moment_ago_is_reused(self):
        watcher = self.build()
        self.learn_the_window()
        registry = self.registry(watcher)
        first = registry.run(call("look_outside"))
        self.house.set_outside("snow", "night")
        second = registry.run(call("look_outside"))
        self.assertEqual((first["source"], second["source"]), ("looked_now", "latest_observation"))
        self.assertEqual(second["weather"], first["weather"])
        self.assertEqual(len(self.pictures("outside")), 1)

    def test_without_a_window_she_answers_from_the_latest_look_with_its_time(self):
        watcher = self.build()
        window = self.learn_the_window()
        self.house.set_outside("fog", "dusk")
        watcher.watch(window)
        registry = self.registry(watcher)
        self.world.enter_room("office")  # she is in the office now: no window there
        result = registry.run(call("look_outside"))
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["source"], "latest_observation")
        self.assertEqual(result["weather"], "fog")
        self.assertEqual(result["room"], "living_room")
        self.assertIn("when", result)
        self.assertIn("no window", result["note"])
        self.assertEqual(len(self.pictures("outside")), 1)  # nothing new was photographed

    def test_nothing_seen_and_no_window(self):
        watcher = self.build(start=OFFICE)
        result = self.registry(watcher).run(call("look_outside"))
        self.assertFalse(result["ok"])
        self.assertIn("no window", result["error"])

    def test_when_the_daily_limit_stops_the_look_she_says_so_and_answers_from_the_latest(self):
        clock = FakeClock(1_000_000.0)
        watcher = self.build(clock=clock, config=OutsideConfig(max_per_day=1))
        window = self.learn_the_window()
        watcher.watch(window)
        result = self.registry(watcher).run(call("look_outside"))
        self.assertEqual(result["source"], "latest_observation")
        self.assertIn("daily limit", result["note"])

    def test_a_turn_that_read_the_web_cannot_move_the_head_for_it(self):
        watcher = self.build()
        self.learn_the_window()
        turn = Turn()
        turn.tainted = True
        result = self.registry(watcher).run(call("look_outside"), turn)
        self.assertFalse(result["ok"])
        self.assertEqual(self.pictures("outside"), [])


class AgentContextTests(WatchCase):
    def agent(self, watcher, *replies):
        self.client = ScriptedClient(*replies)
        registry = ToolRegistry(self.robot, self.safety, world=self.world, memory=self.memory)
        return Agent(self.client, registry, "SYSTEM", world=self.world, memory=self.memory, outside=watcher)

    def system(self, index):
        return self.client.requests[index]["messages"][0]["content"]

    def test_the_latest_look_is_mentioned_once_with_the_changes(self):
        watcher = self.build()
        window = self.learn_the_window()
        agent = self.agent(watcher, text("Ciao"), text("Bene"), text("Ancora"), text("Di nuovo"))
        agent.respond("Ciao Laika")
        self.assertNotIn("Fuori dalla finestra", self.system(0))  # nothing seen yet
        self.house.set_outside("sunny", "day")
        watcher.watch(window)
        self.house.set_outside("rain", "day", ["a car passed"])
        watcher.watch(window)
        agent.respond("Come stai?")
        prompt = self.system(1)
        self.assertIn("Fuori dalla finestra", prompt)
        self.assertIn("rain, day", prompt)
        self.assertIn("started raining", prompt)
        self.assertIn("solo se", prompt)  # mention it only if it fits
        agent.respond("E adesso?")
        self.assertNotIn("Fuori dalla finestra", self.system(2))  # once
        self.house.set_outside("cloudy", "day")
        watcher.watch(window)
        agent.respond("Ancora?")
        self.assertIn("cloudy, day", self.system(3))  # a newer look is offered again, with what changed
        self.assertIn("stopped raining", self.system(3))
        self.assertNotIn("started raining", self.system(3))

    def test_a_newer_look_is_offered_again_but_an_unanswered_turn_does_not_use_it_up(self):
        # A fake clock: two real looks a few milliseconds apart can share one tick of the
        # Windows clock, and then the second would not be "newer" (minutes apart in real life).
        clock = FakeClock(time.time())
        watcher = self.build(clock=clock)
        window = self.learn_the_window()
        watcher.watch(window)
        agent = self.agent(watcher, AIUnavailable("down"), text("Ciao"), text("Ancora"), text("Fine"))
        with self.assertRaises(AIUnavailable):
            agent.respond("Ciao")
        agent.respond("Ciao di nuovo")
        self.assertIn("Fuori dalla finestra", self.system(1))
        agent.respond("E poi?")
        self.assertNotIn("Fuori dalla finestra", self.system(2))
        self.house.set_outside("snow", "night")
        clock.advance(60)
        watcher.watch(window)
        agent.respond("Nevica?")
        self.assertIn("snow, night", self.system(3))

    def test_an_old_look_is_not_news(self):
        clock = FakeClock(time.time())
        watcher = self.build(clock=clock)
        window = self.learn_the_window()
        watcher.watch(window)
        clock.advance(7 * 24 * 3600)
        agent = self.agent(watcher, text("Ciao"))
        agent.respond("Ciao")
        self.assertNotIn("Fuori dalla finestra", self.system(0))

    def test_the_note_never_speaks_for_her(self):
        watcher = self.build()
        window = self.learn_the_window()
        watcher.watch(window)
        agent = self.agent(watcher, text(""))
        reply = agent.respond("Ciao")
        self.assertEqual(reply.text, "")
        self.assertEqual(self.robot.spoken, [])


class SimulatorTests(unittest.TestCase):
    def test_the_default_house_scripts_the_outside(self):
        house = default_house()
        house.set_outside("rain", "night", ["a bus"], 3)
        self.assertEqual(house.outside_view(), OutsideView("rain", "night", ("a bus",), 3))
        house.clear_outside()
        self.assertEqual(house.outside_view().weather, "sunny")

    def test_the_schedule_follows_the_clock(self):
        import time as _time

        house = default_house()
        lights = {}
        for hour in (3, 7, 12, 19, 22):
            stamp = _time.mktime((2026, 6, 1, hour, 0, 0, 0, 0, -1))
            house.set_outside_schedule(((0.0, OutsideView("sunny", "night")), (7.5, OutsideView("sunny", "day")),
                                        (19.0, OutsideView("rain", "dusk")), (21.0, OutsideView("rain", "night"))),
                                       clock=lambda stamp=stamp: stamp)
            lights[hour] = house.outside_view().light
        self.assertEqual(lights, {3: "night", 7: "night", 12: "day", 19: "dusk", 22: "night"})

    def test_a_frame_with_a_window_in_view_carries_the_outside(self):
        house = default_house()
        house.set_outside("fog", "day")
        robot = SimulatedRobot(verbose=False, house=house, start=LIVING_ROOM)
        robot.look(60, 5)
        frame = robot.capture_image()
        self.assertEqual(frame.outside["weather"], "fog")
        seen = outside_from_scene(frame, "living_room")
        self.assertEqual((seen.weather, seen.light, seen.source), ("fog", "day", "simulator"))
        robot.look(0, 0)
        ahead = robot.capture_image()
        self.assertIsNone(ahead.outside)
        self.assertIsNone(outside_from_scene(ahead, "living_room"))

    def test_the_kitchen_has_a_window_too(self):
        house = default_house()
        robot = SimulatedRobot(verbose=False, house=house, start=Pose(650, 150, 270))
        robot.look(0, 5)
        self.assertIn("window", robot.capture_image().scene)


if __name__ == "__main__":
    unittest.main()
