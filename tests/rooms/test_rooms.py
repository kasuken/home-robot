"""Room recognition from vision (#53): the guess in the schema, the vote, naming rooms and the report."""

import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from laika.config import RoomsConfig, load_config, validate_rooms
from laika.conversation_memory import MemorySelector
from laika.events import EventBus, EventType
from laika.house import Pose, default_house
from laika.house_rules import HouseRules, canonical_room
from laika.memory import Memory
from laika.observations import (
    InvalidObservation,
    Observation,
    Observer,
    SeenObject,
    parse_model_observation,
    parse_room_guess,
)
from laika.robot import ImageObservation
from laika.rooms import RoomTracker, name_current_room, prompt_rooms, resolve_room
from laika.simulator import SimulatedRobot
from laika.store import DB_NAME, MIGRATIONS, Store
from laika.tools import PROACTIVE_TOOLS, Turn
from laika.trial_report import Ev, build_rooms, render_markdown
from laika.vision import SYSTEM_PROMPT, user_prompt
from laika.world_model import WorldModel, WorldModelPersister, WorldModelUpdater
from tests.agent.test_tools import ToolTestCase, call
from tests.vision.test_vision import EXAMPLE, REPLY, FakeSession, describer, frame, text_response

NOW = 1_800_000_000.0


def seen(room=None, conf=0.0, objects=(), t=0.0, object_confidence=0.9):
    """An observation with a room guess, at `t` seconds."""
    return Observation(
        description="a frame",
        objects=tuple(SeenObject(name, object_confidence) for name in objects),
        confidence=0.9,
        timestamp=NOW + t,
        room_guess=room,
        room_confidence=conf,
        source="vision",
    )


class TrackerCase(unittest.TestCase):
    def make(self, **config):
        self.world = WorldModel()
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.entered = []
        self.bus.subscribe(self.entered.append, EventType.ROOM_ENTERED)
        self.tracker = RoomTracker(self.world, self.bus, RoomsConfig(**config))
        return self.tracker

    def feed(self, *observations):
        return [self.tracker.observe(o) for o in observations]

    def events(self):
        self.bus.drain()
        return list(self.entered)


class PromptTests(unittest.TestCase):
    def test_the_prompt_lists_the_known_rooms_and_the_generic_types(self):
        text = user_prompt("kitchen", ["kitchen", "camera_di_leonardo (the household calls it: camera di Leonardo)"])
        self.assertIn("Known room ids: kitchen, camera_di_leonardo", text)
        for generic in ("living_room", "bedroom", "office", "bathroom", "entrance"):
            self.assertIn(generic, text)
        self.assertIn("hallway", text)
        self.assertNotIn("Known room ids", user_prompt(None))

    def test_the_system_prompt_asks_for_a_room_with_a_confidence(self):
        self.assertIn('"room"', SYSTEM_PROMPT)
        self.assertIn('"room_confidence"', SYSTEM_PROMPT)
        self.assertIn("unknown", SYSTEM_PROMPT)

    def test_the_request_carries_the_worlds_rooms(self):
        world = WorldModel()
        world.enter_room("kitchen")
        world.enter_room("bedroom")
        world.add_room_alias("bedroom", "camera di Leonardo")
        session = FakeSession(text_response(json.dumps(REPLY)))
        vision = describer(session)
        vision.known_rooms = lambda: prompt_rooms(world)
        vision(frame(), None)
        text = session.calls[0]["json"]["messages"][1]["content"][0]["text"]
        self.assertIn("kitchen", text)
        self.assertIn("bedroom (the household calls it: camera di Leonardo)", text)

    def test_a_failing_room_list_does_not_stop_the_frame(self):
        session = FakeSession(text_response(json.dumps(REPLY)))
        vision = describer(session)

        def broken():
            raise RuntimeError("boom")

        vision.known_rooms = broken
        with self.assertLogs("laika.vision", "ERROR"):
            self.assertIsNotNone(vision(frame(), None))


class SchemaTests(unittest.TestCase):
    def parse(self, **extra):
        raw = {"description": "A sofa.", "objects": ["sofa"], "confidence": 0.9, **extra}
        return parse_model_observation(raw, image=ImageObservation(jpeg=b"x", width=1, height=1, camera_pan=0.0, camera_tilt=0.0), location=None)

    def test_a_guess_is_normalised(self):
        observation = self.parse(room="Living Room", room_confidence=0.8)
        self.assertEqual((observation.room_guess, observation.room_confidence), ("living_room", 0.8))
        self.assertEqual(self.parse(room="leonardo's room", room_confidence=0.5).room_guess, "leonardos_room")

    def test_unknown_and_missing_mean_no_guess(self):
        for extra in ({}, {"room": "unknown", "room_confidence": 0.9}, {"room": "N/A"}, {"room": None}, {"room": ""}):
            observation = self.parse(**extra)
            self.assertEqual((observation.room_guess, observation.room_confidence), (None, 0.0), extra)

    def test_a_guess_without_a_confidence_is_doubtful(self):
        self.assertEqual(self.parse(room="kitchen").room_confidence, 0.5)

    def test_bad_guesses_are_dropped_but_the_observation_stays(self):
        for extra in (
            {"room": 5},
            {"room": "x" * 31, "room_confidence": 0.9},
            {"room": "kitchen; DROP TABLE", "room_confidence": 0.9},
            {"room": "kitchen", "room_confidence": 1.5},
            {"room": "kitchen", "room_confidence": "high"},
            {"room": "kitchen", "room_confidence": float("nan")},
            {"room": ["kitchen"], "room_confidence": 0.5},
        ):
            with self.assertLogs("laika.observations", "WARNING"):
                observation = self.parse(**extra)
            self.assertIsNone(observation.room_guess, extra)
            self.assertEqual(observation.object_names, ["sofa"])

    def test_parse_room_guess_raises_on_bad_input(self):
        with self.assertRaises(InvalidObservation):
            parse_room_guess("kitchen", -0.1)
        with self.assertRaises(InvalidObservation):
            parse_room_guess({"a": 1}, 0.5)
        self.assertEqual(parse_room_guess("Cucina ", 1), ("cucina", 1.0))

    def test_the_guess_round_trips_and_stored_data_is_strict(self):
        observation = self.parse(room="kitchen", room_confidence=0.7)
        data = observation.to_dict()
        self.assertEqual(data["roomGuess"], {"room": "kitchen", "confidence": 0.7})
        self.assertEqual(Observation.from_dict(data).room_guess, "kitchen")
        self.assertNotIn("roomGuess", self.parse().to_dict())
        with self.assertRaises(InvalidObservation):
            Observation.from_dict({**data, "roomGuess": {"room": "kitchen", "confidence": 2}})
        with self.assertRaises(InvalidObservation):
            Observation.from_dict({**data, "roomGuess": "kitchen"})


class ResolveTests(unittest.TestCase):
    def test_guesses_are_mapped_to_one_id_per_room(self):
        world = WorldModel()
        for raw, room in (
            ("hallway", "entrance"), ("study", "office"), ("cucina", "kitchen"), ("soggiorno", "living_room"),
            ("kitchen", "kitchen"), ("garage", "garage"), ("bedroom", "bedroom"),
        ):
            self.assertEqual(resolve_room(raw, world), room, raw)

    def test_an_alias_wins(self):
        world = WorldModel()
        world.add_room_alias("bedroom", "camera di Leonardo")
        self.assertEqual(resolve_room("camera_di_leonardo", world), "bedroom")


class VoteTests(TrackerCase):
    def test_one_confident_frame_is_not_enough_at_the_start(self):
        self.make()
        (first,) = self.feed(seen("kitchen", 0.9))
        self.assertIsNone(first.room)
        self.assertEqual(first.location, "kitchen")  # the frame's own confident guess, until she is sure
        self.assertIsNone(self.world.current_room)
        self.assertEqual(self.events(), [])

    def test_two_confident_frames_make_the_room(self):
        self.make()
        _, second = self.feed(seen("kitchen", 0.9, t=0), seen("kitchen", 0.9, t=5))
        self.assertTrue(second.changed)
        self.assertEqual((second.room, second.previous), ("kitchen", None))
        self.assertEqual(self.world.current_room, "kitchen")
        self.assertEqual(self.tracker.current_room, "kitchen")

    def test_an_ambiguous_doorway_frame_does_not_flip_the_room(self):
        self.make()
        self.feed(seen("living_room", 0.9, t=0), seen("living_room", 0.9, t=5))
        for frame_ in (seen("entrance", 0.3, t=10), seen("kitchen", 0.4, t=15)):
            (decision,) = self.feed(frame_)
            self.assertEqual(decision.room, "living_room")
        self.assertEqual(self.world.current_room, "living_room")

    def test_one_confident_frame_of_another_room_does_not_flip_it(self):
        self.make()
        self.feed(seen("living_room", 0.9, t=0), seen("living_room", 0.9, t=5), seen("living_room", 0.9, t=10))
        (decision,) = self.feed(seen("kitchen", 0.95, t=15))
        self.assertEqual(decision.room, "living_room")
        self.assertFalse(decision.changed)

    def test_two_confident_kitchen_frames_flip_it(self):
        self.make()
        self.feed(seen("living_room", 0.9, t=0), seen("living_room", 0.9, t=5), seen("living_room", 0.9, t=10))
        first, second = self.feed(seen("kitchen", 0.9, t=15), seen("kitchen", 0.9, t=20))
        self.assertEqual(first.room, "living_room")
        self.assertTrue(second.changed)
        self.assertEqual((second.room, second.previous), ("kitchen", "living_room"))
        self.assertEqual(second.location, "kitchen")  # the frame that changes the room is located in the new one
        self.assertEqual(self.world.current_room, "kitchen")

    def test_a_single_wrong_frame_between_good_ones_changes_nothing(self):
        self.make()
        self.feed(seen("kitchen", 0.9, t=0), seen("kitchen", 0.9, t=5))
        decisions = self.feed(seen("bedroom", 0.9, t=10), seen("kitchen", 0.9, t=15), seen("kitchen", 0.9, t=20))
        self.assertEqual([d.room for d in decisions], ["kitchen"] * 3)
        self.assertEqual(len(self.events()), 1)

    def test_old_frames_expire_so_a_long_stay_does_not_pin_the_room(self):
        self.make(window_s=60)
        self.feed(*[seen("living_room", 0.9, t=i * 5) for i in range(5)])
        # Five minutes later she is somewhere else: the old votes are gone, two frames are enough.
        first, second = self.feed(seen("office", 0.8, t=300), seen("office", 0.8, t=305))
        self.assertEqual((first.room, second.room), ("living_room", "office"))

    def test_unknown_when_nothing_is_confident(self):
        self.make()
        decisions = self.feed(seen(None, t=0), seen("kitchen", 0.2, t=5), seen("bedroom", 0.3, t=10))
        self.assertEqual([d.room for d in decisions], [None, None, None])
        self.assertIsNone(self.world.current_room)
        self.assertEqual(self.tracker.stats.unknown_frames, 3)
        self.assertEqual(self.events(), [])

    def test_she_becomes_unsure_after_frames_that_show_nothing(self):
        self.make()
        self.feed(seen("kitchen", 0.9, t=0), seen("kitchen", 0.9, t=5))
        (one,) = self.feed(seen(None, t=10))
        (two,) = self.feed(seen("bedroom", 0.2, t=15))
        (three,) = self.feed(seen(None, t=16))
        self.assertEqual((one.room, two.room), ("kitchen", "kitchen"))
        self.assertIsNone(three.room)
        two = three
        self.assertIsNone(self.world.current_room)
        self.assertEqual(two.confidence, 0.0)
        # ...and sure again after two confident frames, without a stale room in between.
        self.feed(seen("office", 0.9, t=20))
        self.assertIsNone(self.tracker.current_room)
        (back,) = self.feed(seen("office", 0.9, t=25))
        self.assertEqual(back.room, "office")
        self.assertEqual(self.tracker.stats.to_unknown, 1)

    def test_room_entered_is_published_once_per_change_with_the_vision_source(self):
        self.make()
        self.feed(*[seen("kitchen", 0.9, t=i) for i in range(5)])
        self.feed(*[seen("office", 0.9, t=10 + i) for i in range(4)])
        self.feed(seen(None, t=20), seen(None, t=21), seen(None, t=22))  # unknown: no event
        events = self.events()
        self.assertEqual([e.data["room"] for e in events], ["kitchen", "office"])
        first, second = (e.data for e in events)
        self.assertEqual((first["previous"], first["source"]), (None, "vision"))
        self.assertEqual((second["previous"], second["source"]), ("kitchen", "vision"))
        self.assertGreater(second["confidence"], 0.5)
        self.assertEqual(self.tracker.stats.changes, 2)

    def test_the_world_model_updater_does_not_undo_the_trackers_decision(self):
        self.make()
        updater = WorldModelUpdater(self.world, self.bus)
        self.addCleanup(updater.close)
        self.feed(seen("kitchen", 0.9, t=0), seen("kitchen", 0.9, t=1))
        self.feed(seen(None, t=2), seen(None, t=3), seen(None, t=4))  # unknown again
        self.bus.drain()
        self.assertIsNone(self.world.current_room)

    def test_the_ground_truth_event_still_moves_the_world(self):
        self.make()
        updater = WorldModelUpdater(self.world, self.bus)
        self.addCleanup(updater.close)
        self.bus.emit(EventType.ROOM_ENTERED, room="bedroom", previous=None)
        self.bus.drain()
        self.assertEqual(self.world.current_room, "bedroom")

    def test_the_decision_is_logged_for_the_report(self):
        self.make()
        (decision,) = self.feed(seen("kitchen", 0.9))
        data = decision.to_dict()
        self.assertEqual((data["guess"], data["decided"], data["shadow"]), ("kitchen", None, False))


class LandmarkTests(TrackerCase):
    def known_living_room(self):
        self.world.observe(["sofa", "television", "lamp"], "living_room", timestamp=1.0)
        self.world.observe(["sofa", "television", "lamp"], "living_room", timestamp=2.0)
        self.world.observe(["kettle", "table"], "kitchen", timestamp=3.0)
        self.world.observe(["kettle", "table"], "kitchen", timestamp=4.0)

    def test_furniture_votes_even_when_the_guess_is_weak(self):
        self.make()
        self.known_living_room()
        first, second = self.feed(
            seen("kitchen", 0.3, ["sofa", "television"], t=0), seen("kitchen", 0.3, ["sofa", "television"], t=5)
        )
        self.assertEqual(first.frame_room, "living_room")
        self.assertEqual(second.room, "living_room")

    def test_landmarks_add_to_a_matching_guess(self):
        self.make()
        self.known_living_room()
        (decision,) = self.feed(seen("living_room", 0.4, ["sofa"], t=0))
        self.assertEqual(decision.frame_room, "living_room")  # 0.4 + 0.27 reaches min_confidence

    def test_one_landmark_alone_is_not_enough(self):
        self.make()
        self.known_living_room()
        (decision,) = self.feed(seen(None, 0.0, ["sofa"], t=0))
        self.assertIsNone(decision.frame_room)

    def test_a_confident_guess_beats_landmarks_of_another_room(self):
        self.make()
        self.known_living_room()
        (decision,) = self.feed(seen("kitchen", 0.95, ["sofa", "television"], t=0))
        self.assertEqual(decision.frame_room, "kitchen")

    def test_things_that_move_are_not_landmarks(self):
        self.make()
        self.world.observe(["person", "cat"], "kitchen", timestamp=1.0)
        self.world.observe(["person", "cat"], "kitchen", timestamp=2.0)
        (decision,) = self.feed(seen(None, 0.0, ["person", "cat"], t=0))
        self.assertIsNone(decision.frame_room)

    def test_a_room_an_object_was_seen_in_once_is_not_a_landmark(self):
        self.make()
        self.world.observe(["sofa", "television"], "living_room", timestamp=1.0)
        (decision,) = self.feed(seen(None, 0.0, ["sofa", "television"], t=0))
        self.assertIsNone(decision.frame_room)


class ObserverTests(TrackerCase):
    def setUp(self):
        self.make()
        self.script = []
        self.observer = Observer(
            self.bus, describe=lambda image, location: self.script.pop(0), rooms=self.tracker,
            default_location=lambda: self.world.current_room,
        )
        self.addCleanup(self.observer.close)
        self.created = []
        self.bus.subscribe(self.created.append, EventType.OBSERVATION_CREATED)

    def observe(self, guess):
        self.script.append(guess)
        frame_ = ImageObservation(jpeg=b"\xff\xd8\xff\xd9", width=1, height=1, camera_pan=0.0, camera_tilt=0.0)
        self.observer.submit(frame_, None)
        result = self.observer.wait_for(frame_, 5.0)
        self.assertIsNotNone(result)
        return result

    def test_the_frame_that_changes_the_room_is_located_in_the_new_room(self):
        locations = [
            self.observe(seen("living_room", 0.9, t=0)).location,  # not sure yet: the frame's own guess
            self.observe(seen("living_room", 0.9, t=5)).location,
            self.observe(seen("kitchen", 0.9, t=10)).location,  # one frame: still the living room
            self.observe(seen("kitchen", 0.9, t=15)).location,  # the second one changes the room
        ]
        self.assertEqual(locations, ["living_room", "living_room", "living_room", "kitchen"])
        self.assertEqual(self.world.current_room, "kitchen")
        self.bus.drain()
        self.assertEqual([e.data["location"] for e in self.created], locations)
        self.assertEqual(self.created[-1].data["room"]["decided"], "kitchen")
        self.assertEqual(self.created[-1].data["roomGuess"], {"room": "kitchen", "confidence": 0.9})
        self.assertEqual([e.data["room"] for e in self.events()], ["living_room", "kitchen"])

    def test_the_location_is_dropped_when_the_room_is_unknown(self):
        self.observe(seen("kitchen", 0.9, t=0))
        self.observe(seen("kitchen", 0.9, t=5))
        self.observe(seen(None, t=10))
        self.observe(seen(None, t=12))
        unsure = self.observe(seen(None, t=15))
        self.assertIsNone(unsure.location)
        self.assertIsNone(self.world.current_room)

    def test_a_failing_tracker_does_not_drop_the_observation(self):
        class Broken:
            def observe(self, observation):
                raise RuntimeError("boom")

        self.observer.rooms = Broken()
        with self.assertLogs("laika.observations", "ERROR"):
            result = self.observe(seen("kitchen", 0.9, t=0))
        self.assertEqual(result.room_guess, "kitchen")

    def test_without_a_tracker_nothing_changes(self):
        self.observer.rooms = None
        result = self.observe(replace(seen("kitchen", 0.9), location="hall"))
        self.assertEqual(result.location, "hall")
        self.bus.drain()
        self.assertNotIn("room", self.created[-1].data)


class NamingTests(TrackerCase):
    def setUp(self):
        self.make()
        self.memory = Memory()
        self.selector = MemorySelector(self.world, self.memory, rooms=self.tracker)
        self.rules = HouseRules(self.memory, self.world)

    def settle(self, room):
        self.feed(seen(room, 0.9, t=0), seen(room, 0.9, t=1))
        self.assertEqual(self.world.current_room, room)

    def test_a_room_gets_a_name_by_voice(self):
        self.settle("bedroom")
        stored = self.selector.select("Questa è la camera di Leonardo")
        self.assertEqual(len(stored), 1)
        self.assertEqual(self.world.room_for_alias("la camera di Leonardo"), "bedroom")
        self.assertEqual(self.world.aliases_of("bedroom"), ["camera di Leonardo"])
        self.assertEqual(self.world.current_room, "bedroom")
        self.assertIn("also called camera di Leonardo", self.world.summary())

    def test_other_phrasings_name_the_room_too(self):
        for text, alias in (
            ("Qui è lo studio di Sara", "studio di Sara"),
            ("Questa stanza si chiama taverna", "taverna"),
            ("questa è la stanza dei giochi", "stanza dei giochi"),
        ):
            world = WorldModel()
            world.enter_room("living_room")
            selector = MemorySelector(world, Memory())
            self.assertEqual(len(selector.select(text)), 1, text)
            self.assertEqual(world.aliases_of("living_room"), [alias], text)

    def test_ordinary_sentences_do_not_name_a_room(self):
        self.settle("kitchen")
        for text in ("questa è una bella giornata", "questa è la mia penna", "dove siamo?", "la cucina è grande"):
            self.assertEqual(self.selector.select(text), [], text)
        self.assertEqual(self.world.aliases_of("kitchen"), [])

    def test_a_rule_that_uses_the_alias_becomes_a_no_go_room(self):
        self.settle("bedroom")
        self.selector.select("Questa è la camera di Leonardo")
        self.selector.select("Non entrare nella camera di Leonardo")
        self.assertEqual(self.rules.no_go_rooms(), {"bedroom"})  # not the generic "camera" word: the named room
        self.assertTrue(self.rules.is_no_go("bedroom"))
        self.assertTrue(self.rules.is_no_go("camera di Leonardo"))
        self.assertEqual(self.rules.blocked_room(), "bedroom")

    def test_an_alias_given_after_the_rule_still_applies(self):
        self.memory.facts.remember("Non entrare nella stanza dei giochi", "stanza dei giochi", "rule")
        self.assertEqual(self.rules.no_go_rooms(), {"stanza_dei_giochi"})
        self.settle("office")
        self.selector.select("Questa è la stanza dei giochi")
        self.assertEqual(self.rules.no_go_rooms(), {"office"})

    def test_a_configured_alias_resolves_too(self):
        self.settle("bedroom")
        self.selector.select("Questa è la camera di Leonardo")
        from laika.config import HouseConfig

        rules = HouseRules(self.memory, self.world, HouseConfig(no_go_rooms=["camera di Leonardo"]))
        self.assertEqual(rules.no_go_rooms(), {"bedroom"})

    def test_canonical_room_asks_the_aliases_first(self):
        world = WorldModel()
        world.add_room_alias("garage", "camera di Leonardo")
        self.assertEqual(canonical_room("la camera di Leonardo", world.room_for_alias), "garage")
        self.assertEqual(canonical_room("camera da letto", world.room_for_alias), "bedroom")

    def test_a_room_she_did_not_know_is_created_when_she_does_not_know_where_she_is(self):
        stored = self.selector.select("Questa è la camera di Leonardo")
        self.assertEqual(len(stored), 1)
        self.assertEqual(self.world.current_room, "camera_di_leonardo")
        self.assertEqual(self.tracker.current_room, "camera_di_leonardo")
        self.assertIn("camera_di_leonardo", self.world.rooms())
        self.assertEqual([e.data["source"] for e in self.events()], ["user"])

    def test_a_correction_moves_her_and_the_window_agrees(self):
        self.settle("living_room")
        stored = self.selector.select("No, questa è la cucina")
        self.assertEqual(len(stored), 1)
        self.assertEqual(self.world.current_room, "kitchen")
        self.assertEqual(self.tracker.stats.corrections, 1)
        events = self.events()
        self.assertEqual((events[-1].data["room"], events[-1].data["previous"], events[-1].data["source"]),
                         ("kitchen", "living_room", "user"))
        # One more frame that still looks like the old room does not undo what she was told.
        (decision,) = self.feed(seen("living_room", 0.9, t=10))
        self.assertEqual(decision.room, "kitchen")

    def test_confirming_the_current_room_changes_nothing(self):
        self.settle("kitchen")
        self.assertEqual(self.selector.select("Questa è la cucina"), [])
        naming = name_current_room(self.world, None, self.tracker, "la cucina", NOW)
        self.assertEqual((naming.action, naming.room), ("known", "kitchen"))
        self.assertEqual(self.tracker.stats.corrections, 0)

    def test_a_known_alias_in_another_room_is_a_correction(self):
        self.settle("bedroom")
        self.selector.select("Questa è la camera di Leonardo")
        self.tracker.correct("office", NOW)
        naming = name_current_room(self.world, None, self.tracker, "camera di Leonardo", NOW)
        self.assertEqual((naming.action, naming.room, naming.previous), ("corrected", "bedroom", "office"))
        self.assertEqual(self.world.current_room, "bedroom")

    def test_a_newer_alias_moves_from_the_old_room(self):
        self.world.add_room_alias("bedroom", "camera di Leonardo")
        self.world.add_room_alias("office", "camera di Leonardo")
        self.assertEqual(self.world.room_for_alias("camera di leonardo"), "office")
        self.assertEqual(self.world.aliases_of("bedroom"), [])

    def test_unusable_names_are_refused(self):
        self.settle("kitchen")
        for name in ("", "   ", "123", "x" * 50, "due tre quattro cinque sei sette"):
            with self.assertRaises(ValueError, msg=name):
                name_current_room(self.world, self.memory, self.tracker, name, NOW)

    def test_a_shadow_tracker_never_moves_the_world(self):
        world = WorldModel()
        world.enter_room("living_room")
        shadow = RoomTracker(world, None, RoomsConfig(), drive=False)
        naming = name_current_room(world, None, shadow, "la cucina", NOW)
        self.assertEqual(naming.action, "corrected")
        self.assertEqual(world.current_room, "living_room")
        self.assertEqual(shadow.stats.corrections, 1)


class NoGoTests(TrackerCase):
    def test_house_rules_act_only_on_a_confident_room(self):
        self.make()
        memory = Memory()
        memory.facts.remember("Non entrare in camera da letto", "camera da letto", "rule")
        rules = HouseRules(memory, self.world)
        self.assertIsNone(rules.blocked_room())  # she does not know where she is: nothing is blocked
        self.feed(seen("bedroom", 0.9, t=0))
        self.assertIsNone(rules.blocked_room())  # one frame is not a confident room
        self.feed(seen("bedroom", 0.9, t=1))
        self.assertEqual(rules.blocked_room(), "bedroom")
        self.feed(seen(None, t=2), seen(None, t=3), seen(None, t=4))  # the room is lost again
        self.assertIsNone(rules.blocked_room())

    def test_an_ambiguous_frame_does_not_lift_the_block(self):
        self.make()
        memory = Memory()
        memory.facts.remember("Non entrare in camera da letto", "camera da letto", "rule")
        rules = HouseRules(memory, self.world)
        self.feed(seen("bedroom", 0.9, t=0), seen("bedroom", 0.9, t=1), seen("entrance", 0.3, t=2))
        self.assertEqual(rules.blocked_room(), "bedroom")


class RoomToolTests(ToolTestCase):
    def setUp(self):
        super().setUp()
        self.tracker = RoomTracker(self.world, None, RoomsConfig())
        self.registry.rooms = self.tracker

    def test_name_room_teaches_the_current_room(self):
        result, _ = self.run_tool("name_room", name="camera di Leonardo")
        self.assertTrue(result["ok"], result)
        self.assertEqual((result["action"], result["room"], result["names"]), ("named", "living_room", ["camera di Leonardo"]))
        self.assertEqual(self.world.room_for_alias("camera di leonardo"), "living_room")
        self.assertEqual(len(self.memory_facts()), 1)

    def memory_facts(self):
        return self.registry.memory.facts.search("Leonardo", 5)

    def test_name_room_rejects_an_unusable_name(self):
        result, _ = self.run_tool("name_room", name="1234")
        self.assertFalse(result["ok"])

    def test_name_room_is_blocked_after_web_content(self):
        turn = Turn()
        turn.tainted = True
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.registry.run(call("name_room", name="taverna"), turn)
        self.assertFalse(result["ok"])
        self.assertEqual(self.world.aliases_of("living_room"), [])

    def test_name_room_is_not_offered_to_a_proactive_turn(self):
        self.assertIn("name_room", self.registry.names())
        self.assertNotIn("name_room", PROACTIVE_TOOLS)
        narrowed = self.registry.restricted(PROACTIVE_TOOLS)
        self.assertNotIn("name_room", narrowed.names())
        with contextlib.redirect_stdout(io.StringIO()):
            result = narrowed.run(call("name_room", name="taverna"))
        self.assertFalse(result["ok"])

    def test_recall_finds_a_room_by_its_name(self):
        self.world.observe(["sofa"], "living_room")
        self.world.add_room_alias("living_room", "la sala grande")
        result, _ = self.run_tool("recall", query="sala grande")
        self.assertTrue(result["found"])
        self.assertEqual([o["name"] for o in result["objects"]], ["sofa"])


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "memory" / DB_NAME

    def open(self):
        store = Store(self.path, clock=lambda: NOW)
        self.addCleanup(store.close)
        return store

    def test_aliases_survive_a_restart(self):
        store = self.open()
        world = WorldModel()
        WorldModelPersister(world, store)
        world.enter_room("bedroom", timestamp=1.0)
        world.add_room_alias("bedroom", "camera di Leonardo", 2.0)
        world.add_room_alias("bedroom", "la camera dei ragazzi", 3.0)
        world.add_room_alias("office", "studio di Sara", 4.0)
        world.add_room_alias("office", "camera di Leonardo", 5.0)  # the newest word wins
        world.enter_room("bedroom", timestamp=6.0)  # a later visit must not wipe the aliases
        self.assertTrue(store.flush())
        store.close()

        restored = WorldModel()
        restored.load(*self.open().load_world())
        self.assertEqual(restored.aliases_of("bedroom"), ["camera dei ragazzi"])
        self.assertEqual(restored.aliases_of("office"), ["studio di Sara", "camera di Leonardo"])
        self.assertEqual(restored.room_for_alias("camera di Leonardo"), "office")
        self.assertIsNone(restored.current_room)

    def test_migration_4_adds_the_aliases_to_an_old_database(self):
        self.path.parent.mkdir(parents=True)
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        for number in (1, 2, 3):
            for statement in MIGRATIONS[number - 1]:
                conn.execute(statement)
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (number,))
        conn.execute("INSERT INTO world_rooms (name, first_seen, last_visit) VALUES ('kitchen', 1, 2)")
        conn.close()

        store = self.open()
        (room,) = store.load_world()[1]
        self.assertEqual((room["name"], room["aliases"]), ("kitchen", []))
        world = WorldModel()
        world.load(*store.load_world())
        WorldModelPersister(world, store)
        world.add_room_alias("kitchen", "la cucina di nonna")
        self.assertTrue(store.flush())
        store.close()
        conn = sqlite3.connect(self.path)
        versions = [r[0] for r in conn.execute("SELECT version FROM schema_version ORDER BY version")]
        conn.close()
        self.assertEqual(versions, [1, 2, 3, 4])
        self.assertEqual(self.open().load_world()[1][0]["aliases"], ["cucina di nonna"])


class ConfigTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "laika.toml"
            path.write_text(text, encoding="utf-8")
            return load_config(path)

    def test_defaults(self):
        rooms = load_config(Path("does-not-exist.toml")).rooms
        self.assertEqual(
            (rooms.enabled, rooms.window_s, rooms.min_votes, rooms.switch_margin, rooms.min_confidence, rooms.shadow_in_sim),
            (True, 120.0, 2, 0.4, 0.5, True),
        )

    def test_settings_are_read(self):
        rooms = self.load("[rooms]\nenabled = false\nwindow_s = 60\nmin_votes = 3\nswitch_margin = 1\n"
                          "min_confidence = 0.6\nshadow_in_sim = false\n").rooms
        self.assertEqual((rooms.enabled, rooms.window_s, rooms.min_votes, rooms.shadow_in_sim), (False, 60, 3, False))

    def test_bad_values_are_refused_naming_the_setting(self):
        for body, name in (
            ("window_s = 1", "window_s"), ("window_s = 99999", "window_s"), ("min_votes = 0", "min_votes"),
            ("min_votes = 2.5", "min_votes"), ("min_votes = 6", "min_votes"), ("switch_margin = -1", "switch_margin"),
            ("min_confidence = 0", "min_confidence"), ("min_confidence = 2", "min_confidence"),
            ("enabled = 1", "enabled"), ("shadow_in_sim = \"yes\"", "shadow_in_sim"), ("window_s = true", "window_s"),
        ):
            with self.assertRaises(ValueError, msg=body) as caught:
                self.load(f"[rooms]\n{body}\n")
            self.assertIn(name, str(caught.exception))
        with self.assertRaises(ValueError):
            self.load("[rooms]\nunknown_setting = 1\n")

    def test_the_example_file_documents_the_section(self):
        text = EXAMPLE.read_text(encoding="utf-8")
        self.assertIn("[rooms]", text)
        self.assertEqual(load_config(EXAMPLE).rooms, RoomsConfig())
        validate_rooms(RoomsConfig())


class ShadowTests(unittest.TestCase):
    """The tracker beside the simulator's ground truth, fed by a describer that is right most of the time."""

    SPOTS = {
        "living_room": Pose(300, 200), "kitchen": Pose(650, 100), "office": Pose(300, 550), "bedroom": Pose(650, 550),
    }
    WRONG = {"living_room": "office", "kitchen": "living_room", "office": "bedroom", "bedroom": "kitchen"}

    def test_accuracy_in_the_simulator_and_in_the_report(self):
        world = WorldModel()
        bus = EventBus()
        bus.start()
        self.addCleanup(bus.close)
        events = []
        bus.subscribe(events.append)
        house = default_house()
        state = {"robot": None, "frame": 0}

        def truth():
            return state["robot"].current_room()

        tracker = RoomTracker(world, bus, RoomsConfig(), drive=False, truth=truth)

        def describe(image, location):
            """The truth most of the time; every fifth frame a doubtful wrong guess; the scene is what the robot sees."""
            state["frame"] += 1
            room = truth()
            wrong = state["frame"] % 5 == 0
            objects = [SeenObject(name, 0.9) for name in image.scene]
            return Observation(
                description="a frame", objects=tuple(objects), confidence=0.9, timestamp=NOW + state["frame"] * 10,
                room_guess=self.WRONG[room] if wrong else room, room_confidence=0.55 if wrong else 0.85,
            )

        observer = Observer(bus, describe=describe, rooms=tracker)
        self.addCleanup(observer.close)
        rooms = [*self.SPOTS] * 2  # walk through the house twice
        for room in rooms:
            robot = state["robot"] = SimulatedRobot(verbose=False, house=house, start=self.SPOTS[room])
            for _ in range(8):
                shot = replace(robot.capture_image(), simulated=False, jpeg=b"\xff\xd8\xff\xd9")
                observer.submit(shot, None)
                self.assertIsNotNone(observer.wait_for(shot, 5.0))
        stats = tracker.stats
        self.assertEqual(stats.truth_checked, 64)
        self.assertGreaterEqual(stats.accuracy, 0.6)
        self.assertGreaterEqual(stats.changes, 4)
        self.assertIsNone(world.current_room)  # shadow mode never touches the world
        bus.drain()
        self.assertEqual([e for e in events if e.type is EventType.ROOM_ENTERED], [])

        report = build_rooms([Ev(e.timestamp, e.type.value, e.data) for e in events])
        self.assertEqual(report["frames_tracked"], 64)
        self.assertEqual(report["shadow_checked"], 64)
        self.assertAlmostEqual(report["shadow_accuracy"], stats.accuracy, places=2)
        self.assertEqual(report["room_changes"], 0)  # shadow decisions are not RoomEntered events
        self.assertLessEqual(report["unknown_share"], 0.5)

    def test_the_simulators_ground_truth_stays_in_charge(self):
        """In --sim the tracker (shadow) and the ground-truth RoomEntered do not fight over the world."""
        world = WorldModel()
        bus = EventBus()
        bus.start()
        self.addCleanup(bus.close)
        updater = WorldModelUpdater(world, bus)
        self.addCleanup(updater.close)
        tracker = RoomTracker(world, bus, RoomsConfig(), drive=False)
        for t in range(3):
            tracker.observe(seen("kitchen", 0.9, t=t))
        bus.emit(EventType.ROOM_ENTERED, room="bedroom", previous=None)
        bus.drain()
        self.assertEqual(world.current_room, "bedroom")
        self.assertEqual(tracker.current_room, "kitchen")  # its own opinion, only for the log


class ReportTests(unittest.TestCase):
    def events(self):
        def created(ts, decided, guess, **extra):
            return Ev(ts, "ObservationCreated", {"room": {"guess": guess, "decided": decided, **extra}})

        return [
            created(0, None, "kitchen"),
            created(10, "kitchen", "kitchen"),
            created(20, "kitchen", None),
            created(30, None, None),
            Ev(10, "RoomEntered", {"room": "kitchen", "previous": None, "source": "vision"}),
            Ev(35, "RoomEntered", {"room": "office", "previous": "kitchen", "source": "user"}),
            Ev(36, "RoomEntered", {"room": "bedroom", "previous": "office"}),
        ]

    def test_counts_changes_unknown_time_and_corrections(self):
        report = build_rooms(self.events())
        self.assertEqual(report["frames_tracked"], 4)
        self.assertEqual(report["frames_with_a_guess"], 2)
        self.assertEqual((report["frames_unknown"], report["unknown_share"]), (2, 0.5))
        self.assertEqual(report["unknown_time_share"], round(10 / 30, 3))
        self.assertEqual((report["room_changes"], report["corrected_by_user"], report["ground_truth_entries"]), (1, 1, 1))
        self.assertEqual(report["changes_by_room"], {"kitchen": 1})
        self.assertIsNone(report["shadow_accuracy"])

    def test_shadow_accuracy(self):
        events = [
            Ev(i, "ObservationCreated", {"room": {"decided": decided, "truth": truth, "shadow": True}})
            for i, (decided, truth) in enumerate(
                [("kitchen", "kitchen"), ("kitchen", "kitchen"), ("kitchen", "office"), (None, "office")]
            )
        ]
        report = build_rooms(events)
        self.assertEqual((report["shadow_checked"], report["shadow_accuracy"]), (4, 0.5))
        self.assertEqual(report["shadow_wrong_by_true_room"], {"office": 2})

    def test_an_empty_period_says_so(self):
        report = build_rooms([])
        self.assertEqual((report["frames_tracked"], report["unknown_share"], report["shadow_accuracy"]), (0, None, None))

    def test_the_report_has_a_room_section_and_a_summary_line(self):
        from laika.trial_report import build_report, summary_text
        from tests.trial.synthetic import END, START, write_logs, write_store

        with tempfile.TemporaryDirectory() as tmp:
            write_logs(Path(tmp))
            write_store(Path(tmp))
            report = build_report(Path(tmp), START, END, now=END)
        self.assertEqual(report["rooms"]["frames_tracked"], 0)
        text = render_markdown(report)
        self.assertIn("## Room recognition", text)
        self.assertIn("no room tracking in this period", text)
        self.assertIn("  rooms: ", summary_text(report))


if __name__ == "__main__":
    unittest.main()
