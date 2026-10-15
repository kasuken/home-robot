"""Choosing an activity when Laika is idle (#47): the formula, the choice, and the house simulator over an hour.

The selector runs on fake clocks (the idle time, the time of day and the activity durations are scaled: a fake
sleep moves the clock fast and waits a little real time so the real explorer, watcher and searcher threads get
to work), against the real arbiter, safety controller, camera, observer, world model and memory.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import random
import tempfile
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path

from laika.battery import BatteryState
from laika.behaviour import Arbiter, Priority
from laika.camera import Camera
from laika.changes import Change, ChangeTracker
from laika.config import Config, ExploreConfig, IdleConfig, InteractionConfig, OutsideConfig, PresenceConfig, SearchConfig, load_config
from laika.events import EventBus, EventType, emit
from laika.house import Pose, default_house
from laika.idle import (
    CHECK_HOME,
    EXPLORE,
    FOLLOW_UP,
    LOOK_OUTSIDE,
    NOT_NOW,
    REST,
    IdleContext,
    IdleSelector,
    budget_factor,
    duration_phrase,
    need_factor,
    period_of,
    personality_factor,
    recency_factor,
    time_factor,
)
from laika.interaction_policy import Availability, InteractionPolicy
from laika.investigate import PendingQuestions
from laika.memory import Memory
from laika.observations import Observer
from laika.outside import WindowWatcher
from laika.personality import Personality
from laika.presence import PresenceTracker
from laika.safety import SafetyController
from laika.scan import Scanner
from laika.search import Searcher
from laika.store import DB_NAME, Store
from laika.tools import ToolRegistry
from laika.trial_report import build_report
from laika.windows import WindowRegistry
from laika.world_model import WorldModel, WorldModelUpdater
from laika.explore import Explorer
from tests.agent.test_tools import call
from tests.fakes import FakeClock, LookRecordingRobot

P = Priority
LIVING_ROOM = Pose(300, 200, 0)  # the window is on the right, at pan 60
SPEED = 20.0  # fake seconds per sleep second
SRC = Path(__file__).resolve().parents[2] / "src" / "laika"


def at(hour: int, minute: int = 0) -> float:
    return datetime(2026, 10, 15, hour, minute).timestamp()


def wait_for(condition, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class SpyPolicy:
    def set_exploring(self, exploring):
        pass


class DriveRobot(LookRecordingRobot):
    """Counts the drives that reached the wheels (a speed other than zero)."""

    @property
    def drove(self):
        return any(self.speeds)


class IdleCase(unittest.TestCase):
    def build(
        self,
        *,
        hour=14,
        idle=None,
        allow_autonomous=True,
        quiet_hours="",
        personality=None,
        seed=1,
        window=False,
        outside=None,
        store=None,
        with_searcher=True,
        with_explorer=True,
        driving=None,
    ):
        out = io.StringIO()
        redirect = contextlib.redirect_stdout(out)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.stdout = out
        self.clock = FakeClock(at(hour))
        self.house = default_house()
        self.bus = EventBus()
        self.bus.start()
        self.addCleanup(self.bus.close)
        self.world = WorldModel()
        self.addCleanup(WorldModelUpdater(self.world, self.bus).close)
        self.memory = Memory(store)
        self.robot = DriveRobot(house=self.house, start=LIVING_ROOM)
        self.world.enter_room(self.robot.current_room())
        self.camera = Camera(self.robot, None, self.bus, min_interval_s=0, max_per_minute=0, save_images=False)
        self.observer = Observer(self.bus, default_location=lambda: self.world.current_room)
        self.observer.attach(self.camera)
        self.addCleanup(self.observer.close)
        self.safety = SafetyController(self.robot, speed=25, allow_autonomous=allow_autonomous)
        self.addCleanup(self.safety.close)
        self.safety.start()
        self.arbiter = Arbiter(self.safety, None, lease_timeout_s=2.0)
        self.addCleanup(self.arbiter.close)
        self.presence = PresenceTracker(self.bus, PresenceConfig())
        self.addCleanup(self.presence.close)
        self.policy = InteractionPolicy(
            InteractionConfig(quiet_hours=quiet_hours), self.presence, wall_clock=self.clock, clock=self.clock
        )
        self.windows = WindowRegistry(None, self.bus)
        self.addCleanup(self.windows.close)
        self.watcher = WindowWatcher(
            self.arbiter, self.robot, self.camera, self.windows, self.world, self.memory,
            outside or OutsideConfig(min_interval_s=600), clock=self.clock, settle_s=0.0,
        )
        if window:
            self.learn_the_window()
        scanner = Scanner(self.robot, self.camera, self.observer, settle_s=0.01, observation_wait_s=2.0)
        self.explorer = None
        if with_explorer:
            self.explorer = Explorer(
                self.arbiter, self.safety, scanner, SpyPolicy(),
                config=ExploreConfig(speed=15, step_ms=100, scan_every_steps=3, max_duration_s=600.0, max_steps=10000),
                drive=driving, head_only_interval_s=0.05,
            )
            self.addCleanup(self.explorer.stop)
        self.questions = PendingQuestions()
        self.tracker = ChangeTracker(self.memory.episodes)
        self.searcher = None
        if with_searcher:
            self.searcher = Searcher(
                self.robot, self.safety, self.arbiter,
                Scanner(self.robot, self.camera, self.observer, settle_s=0.01, observation_wait_s=2.0),
                self.explorer, self.world, self.memory, self.presence, self.observer, SearchConfig(),
                settle_s=0.01, observation_wait_s=2.0,
            )
        self.config = idle or IdleConfig(
            idle_after_s=300, between_activities_s=60, max_activity_s=120, max_repeats=2, seed=seed
        )
        self.selector = IdleSelector(
            self.config,
            arbiter=self.arbiter,
            safety=self.safety,
            explorer=self.explorer,
            windows=self.watcher,
            searcher=self.searcher,
            questions=self.questions,
            changes=self.tracker,
            presence=self.presence,
            policy=self.policy,
            world=self.world,
            memory=self.memory,
            personality=personality,
            robot=self.robot,
            events=self.bus,
            clock=self.clock,
            wall_clock=self.clock,
            sleep=self.sleep,
            rng=random.Random(seed),
            search_max_s=20.0,
        )
        self.addCleanup(self.selector.close)
        return self.selector

    def sleep(self, seconds):
        """Fake time runs fast; real time only passes while a real thread (the explorer) has work to do."""
        self.clock.advance(seconds * SPEED)
        working = self.explorer is not None and self.explorer.active
        time.sleep(0.03 if working else 0.001)

    def learn_the_window(self):
        """What a scan does: look at the right, see the window, learn it from the observation."""
        self.robot.look(60, 5)
        self.observer.wait_for(self.camera.capture("scan"))
        self.bus.drain(2.0)
        self.robot.look(0, 0)
        self.robot.poses.clear()
        self.assertEqual(len(self.windows.in_room(self.world.current_room)), 1)

    def idle_for(self, seconds=None):
        self.clock.advance(self.config.idle_after_s if seconds is None else seconds)

    def ctx(self, **overrides):
        return dataclasses.replace(self.selector.context(), **overrides)

    def episodes(self):
        return [e for e in self.memory.episodes.recent(200) if e.kind == "idle"]

    def simulate(self, seconds=3600.0, step=30.0):
        end = self.clock.now + seconds
        runs = []
        while self.clock.now < end:
            run = self.selector.tick()
            if run is None:
                self.clock.advance(step)
            else:
                runs.append(run)
        return runs


# --- the formula -----------------------------------------------------------------------------------


class FormulaTests(IdleCase):
    def test_the_time_of_day_shapes_the_utility(self):
        selector = self.build(window=True)
        self.assertEqual([period_of(h) for h in (2, 6, 14, 20, 23)], ["night", "early_morning", "day", "evening", "night"])
        scores = {}
        for period in ("night", "early_morning", "day", "evening"):
            scores[period] = selector.scores(self.ctx(period=period))
        self.assertGreater(scores["day"][EXPLORE], scores["evening"][EXPLORE])
        self.assertGreater(scores["evening"][EXPLORE], scores["early_morning"][EXPLORE])
        self.assertGreater(scores["early_morning"][EXPLORE], scores["night"][EXPLORE])
        self.assertGreater(scores["night"][REST], scores["day"][REST])
        self.assertGreater(scores["night"][REST], 2 * scores["night"][EXPLORE])
        self.assertGreater(scores["evening"][CHECK_HOME], scores["day"][CHECK_HOME])
        self.assertEqual(time_factor(EXPLORE, "day"), 1.2)
        # Exploring is the choice in the day, rest at night, whatever else is equal.
        self.assertEqual(selector.choose(self.ctx(period="day")), EXPLORE)
        self.assertEqual(selector.choose(self.ctx(period="night")), REST)

    def test_recency_gives_variety_and_recovers_in_an_hour(self):
        self.assertEqual(recency_factor(None), 1.0)
        self.assertEqual(recency_factor(0), 0.25)
        self.assertAlmostEqual(recency_factor(1800), 0.625)
        self.assertEqual(recency_factor(3600), 1.0)
        self.assertEqual(recency_factor(10**6), 1.0)
        selector = self.build()
        fresh = selector.scores(self.ctx(since={}))[EXPLORE]
        just_ran = selector.scores(self.ctx(since={EXPLORE: 0.0}))[EXPLORE]
        half = selector.scores(self.ctx(since={EXPLORE: 1800.0}))[EXPLORE]
        self.assertAlmostEqual(just_ran, fresh * 0.25)
        self.assertGreater(half, just_ran)
        self.assertLess(half, fresh)

    def test_personality_shifts_the_activities_it_names(self):
        self.assertAlmostEqual(personality_factor(EXPLORE, Personality(curiosity=0)), 0.6)
        self.assertAlmostEqual(personality_factor(EXPLORE, Personality(curiosity=50)), 1.0)
        self.assertAlmostEqual(personality_factor(LOOK_OUTSIDE, Personality(curiosity=100)), 1.4)
        self.assertAlmostEqual(personality_factor(CHECK_HOME, Personality(initiative=100)), 1.4)
        self.assertEqual(personality_factor(REST, Personality(curiosity=100, initiative=100)), 1.0)
        selector = self.build(window=True)
        curious, shy = Personality(curiosity=100, initiative=0), Personality(curiosity=0, initiative=100)
        high, low = selector.scores(self.ctx(personality=curious)), selector.scores(self.ctx(personality=shy))
        for name in (EXPLORE, LOOK_OUTSIDE):
            self.assertGreater(high[name], low[name])
        self.assertLess(high[CHECK_HOME], low[CHECK_HOME])
        self.assertEqual(high[REST], low[REST])

    def test_a_pending_question_or_a_change_raises_the_follow_up(self):
        selector = self.build()
        none = selector.scores(self.ctx())
        self.assertEqual(none[FOLLOW_UP], 0.0)  # nothing to follow up: not an option
        one = selector.scores(self.ctx(question_objects=("backpack",)))[FOLLOW_UP]
        two = selector.scores(self.ctx(question_objects=("backpack", "mug")))[FOLLOW_UP]
        both = selector.scores(self.ctx(question_objects=("backpack", "mug"), change_objects=("keys",)))[FOLLOW_UP]
        self.assertGreater(one, 0)
        self.assertGreater(two, one)
        self.assertGreater(both, two)
        self.assertEqual(need_factor(0, 0), 1.0)
        self.assertEqual(need_factor(9, 9), need_factor(3, 3))  # bounded
        only_change = selector.scores(self.ctx(change_objects=("keys",)))[FOLLOW_UP]
        self.assertGreater(only_change, 0)
        self.assertGreater(one, only_change)

    def test_battery_presence_and_driving_change_what_is_possible(self):
        selector = self.build()
        base = selector.scores(self.ctx(driving=True))
        head_only = selector.scores(self.ctx(driving=False))
        self.assertAlmostEqual(head_only[EXPLORE], base[EXPLORE] * 0.5)
        low = selector.scores(self.ctx(battery=BatteryState.LOW))
        self.assertEqual(low[EXPLORE], 0.0)
        self.assertGreater(low[REST], base[REST])
        unknown = selector.scores(self.ctx(battery=BatteryState.UNKNOWN))
        self.assertEqual(unknown[EXPLORE], 0.0)
        critical = selector.scores(self.ctx(battery=BatteryState.CRITICAL))
        self.assertEqual([n for n, s in critical.items() if s > 0], [REST])
        present = selector.scores(self.ctx(driving=True, present=True))
        self.assertEqual((present[EXPLORE], present[CHECK_HOME]), (0.0, 0.0))
        self.assertAlmostEqual(present[REST], base[REST] * 2)
        # Without wheels, someone near does not stop her looking around.
        self.assertGreater(selector.scores(self.ctx(driving=False, present=True))[EXPLORE], 0)
        for state in (Availability.SLEEPING, Availability.DO_NOT_DISTURB, Availability.BUSY):
            with self.subTest(state=state):
                quiet = selector.scores(self.ctx(state=state, question_objects=("cup",)))
                self.assertEqual([n for n, s in quiet.items() if s > 0], [REST])
        self.assertEqual(budget_factor(None), 1.0)
        self.assertEqual(budget_factor(600), 1.0)
        self.assertEqual(budget_factor(30), 0.3)
        self.assertAlmostEqual(budget_factor(90), 0.75)
        tight = selector.scores(self.ctx(driving=True, budget_left_s=30.0))[EXPLORE]
        self.assertAlmostEqual(tight, base[EXPLORE] * 0.3)

    def test_the_weights_and_switches_in_the_config(self):
        config = IdleConfig(explore=False, rest_weight=0.0, look_outside_weight=3.0)
        selector = self.build(idle=config, window=True)
        scores = selector.scores(self.ctx())
        self.assertEqual((scores[EXPLORE], scores[REST]), (0.0, 0.0))
        self.assertGreater(scores[LOOK_OUTSIDE], 2.0)
        self.assertIsNone(selector.choose(self.ctx(question_objects=(), windows_due=0, present=True)))

    def test_at_night_only_rest_is_left_whatever_is_pending(self):
        selector = self.build(window=True)
        ctx = self.ctx(period="night", question_objects=("cup",), change_objects=("keys",), driving=True)
        scores = selector.scores(ctx)
        self.assertEqual([name for name, score in scores.items() if score > 0], [REST])
        for name in (EXPLORE, LOOK_OUTSIDE, FOLLOW_UP, CHECK_HOME):
            self.assertLess(time_factor(name, "night"), NOT_NOW)
        self.assertGreater(time_factor(EXPLORE, "early_morning"), NOT_NOW)
        self.assertEqual(selector.choose(ctx), REST)

    def test_the_same_activity_is_not_chosen_more_than_max_repeats_in_a_row(self):
        selector = self.build()
        ctx = self.ctx(period="day", driving=True)
        self.assertEqual(selector.choose(ctx), EXPLORE)
        self.assertEqual(selector.choose(dataclasses.replace(ctx, streak=(EXPLORE, 1))), EXPLORE)
        other = selector.choose(dataclasses.replace(ctx, streak=(EXPLORE, 2)))
        self.assertNotEqual(other, EXPLORE)
        self.assertIsNotNone(other)
        # Only one thing is possible: she does it again.
        night = dataclasses.replace(ctx, period="night", streak=(REST, 5))
        self.assertEqual(selector.choose(night), REST)

    def test_a_duration_in_words(self):
        self.assertEqual(duration_phrase(120), "2 minutes")
        self.assertEqual(duration_phrase(60), "1 minute")
        self.assertEqual(duration_phrase(30), "30 seconds")
        self.assertEqual(duration_phrase(3 * 3600), "3 hours")


class DeterminismTests(IdleCase):
    def tied(self, seed):
        # Check on the house and rest score the same: 0.4 * 1.0 * 1.0 and 0.5 * 0.8.
        config = IdleConfig(explore=False, look_outside=False, follow_up=False, check_home_weight=0.4, rest_weight=0.5, seed=seed)
        selector = self.build(idle=config, seed=seed, personality=Personality(initiative=50))
        ctx = self.ctx(period="day")
        scores = selector.scores(ctx)
        self.assertAlmostEqual(scores[CHECK_HOME], scores[REST])
        return [selector.choose(ctx) for _ in range(12)]

    def test_the_same_seed_gives_the_same_choices(self):
        first, again = self.tied(7), self.tied(7)
        self.assertEqual(first, again)

    def test_ties_are_broken_by_the_generator(self):
        picks = set()
        for seed in range(1, 12):
            picks.update(self.tied(seed))
        self.assertEqual(picks, {CHECK_HOME, REST})

    def test_the_same_inputs_give_the_same_scores(self):
        selector = self.build(window=True)
        ctx = self.ctx(question_objects=("backpack",))
        self.assertEqual(selector.scores(ctx), selector.scores(ctx))
        self.assertEqual(selector.choose(ctx), selector.choose(ctx))


# --- the simulated hour ---------------------------------------------------------------------------------


class HourTests(IdleCase):
    def test_done_when_an_idle_laika_picks_varied_activities_over_an_hour(self):
        # On the robot everything is head-only until V0.2 is verified: that is the realistic hour.
        self.build(window=True, allow_autonomous=False)
        runs = self.simulate(3600.0)
        names = [r.activity for r in runs if r.outcome.status in ("done", "interrupted")]
        self.assertGreaterEqual(len(set(names)), 3, names)
        for activity in (EXPLORE, LOOK_OUTSIDE, REST):
            self.assertIn(activity, names, names)
        for first in range(len(names) - self.config.max_repeats):
            window = names[first : first + self.config.max_repeats + 1]
            self.assertGreater(len(set(window)), 1, f"{window[0]} more than max_repeats times in a row: {names}")
        # Nothing ever moved the wheels, and she never said a word.
        self.assertFalse(self.robot.drove)
        self.assertEqual(self.robot.spoken, [])
        # Each choice was remembered, and the log says why.
        self.assertTrue(self.episodes())
        self.assertIn("[laika] idle: chose ", self.stdout.getvalue())
        self.assertIn("Idle: looked out of the living room window", [e.summary for e in self.episodes()])

    def test_with_driving_allowed_she_explores_on_her_wheels(self):
        self.build()
        self.idle_for()
        run = self.selector.tick()
        self.assertEqual(run.activity, EXPLORE)
        self.assertEqual(run.outcome.status, "done")
        self.assertTrue(self.robot.drove)
        self.assertEqual(self.episodes()[0].summary, "Idle: explored for 2 minutes")
        self.assertIsNone(self.arbiter.holder)  # the body went back

    def test_autonomous_driving_off_means_exploring_with_the_head_only(self):
        self.build(allow_autonomous=False)
        self.idle_for()
        scores = self.selector.scores(self.selector.context())
        self.assertGreater(scores[EXPLORE], 0)
        run = self.selector.tick()
        self.assertEqual(run.activity, EXPLORE)
        self.assertEqual(run.outcome.status, "done")
        self.assertFalse(self.robot.drove)
        self.assertGreaterEqual(self.explorer.scans, 1)
        self.assertEqual(self.episodes()[0].summary, "Idle: explored with the head only for 2 minutes")

    def test_nothing_starts_before_the_idle_time_and_a_pause_between_activities(self):
        self.build(allow_autonomous=False)
        self.assertIsNone(self.selector.tick())
        self.clock.advance(299)
        self.assertIsNone(self.selector.tick())
        self.clock.advance(2)
        first = self.selector.tick()
        self.assertIsNotNone(first)
        self.assertIsNone(self.selector.tick())  # between_activities_s: not at once
        self.clock.advance(59)
        self.assertIsNone(self.selector.tick())
        self.clock.advance(2)
        self.assertIsNotNone(self.selector.tick())

    def test_a_follow_up_looks_for_the_object_of_a_pending_question_and_asks_nothing(self):
        self.build(allow_autonomous=False)
        self.questions.add("backpack", "living_room")
        self.idle_for()
        run = self.selector.tick()
        self.assertEqual(run.activity, FOLLOW_UP)
        self.assertEqual(run.outcome.status, "done")
        self.assertTrue(run.summary.startswith("Idle: looked for the backpack"), run.summary)
        self.assertEqual(self.robot.spoken, [])
        self.assertEqual(len(self.questions), 1)  # asking is the interaction policy's business
        self.assertFalse(self.robot.drove)
        # The same object is not followed up again at once.
        self.assertEqual(self.selector.context().question_objects, ())

    def test_an_unresolved_change_is_followed_up_too(self):
        self.build(allow_autonomous=False)
        change = Change("missing", "keys", "living_room", None, self.clock.now - 60, "obs-1")
        self.tracker.record([change])
        self.assertEqual(self.selector.context().change_objects, ("keys",))
        self.idle_for()
        run = self.selector.tick()
        self.assertEqual(run.activity, FOLLOW_UP)
        self.assertIn("keys", run.summary)

    def test_checking_on_the_house_counts_people_and_never_moves_the_wheels(self):
        config = IdleConfig(explore=False, look_outside=False, follow_up=False, rest=False, max_activity_s=60)
        self.build(idle=config, hour=19)
        self.idle_for()
        run = self.selector.tick()
        self.assertEqual(run.activity, CHECK_HOME)
        self.assertEqual(run.outcome.status, "done")
        self.assertIn("nobody seen", run.summary)
        self.assertFalse(self.robot.drove)

    def test_the_selector_chooses_at_the_end_of_the_thread_loop_too(self):
        config = IdleConfig(idle_after_s=5, between_activities_s=0, max_activity_s=10)
        selector = self.build(idle=config, allow_autonomous=False)
        selector.poll_s = 0.01
        self.clock.advance(10)
        selector.start()
        self.assertTrue(wait_for(lambda: selector.history))
        selector.close()
        self.assertFalse(selector._thread.is_alive())


# --- what stops her or limits her ------------------------------------------------------------------------


class LimitTests(IdleCase):
    def only_rest(self, run):
        self.assertEqual(run.activity, REST, run.scores)
        self.assertFalse(self.robot.drove)

    def test_quiet_hours_lead_to_rest(self):
        self.build(hour=23, quiet_hours="21:30-07:30")
        self.assertIs(self.policy.state(), Availability.SLEEPING)
        self.questions.add("backpack", "living_room")
        self.idle_for()
        self.only_rest(self.selector.tick())

    def test_do_not_disturb_leads_to_rest(self):
        self.build()
        self.policy.set_do_not_disturb(60)
        self.questions.add("backpack", "living_room")
        self.idle_for()
        self.only_rest(self.selector.tick())

    def test_a_user_who_is_busy_leads_to_rest(self):
        self.build()
        self.policy.set_busy(30)
        self.idle_for()
        self.only_rest(self.selector.tick())

    def test_a_low_battery_means_no_driving_activities(self):
        self.build()
        self.safety.set_battery_state(BatteryState.LOW)
        self.idle_for()
        scores = self.selector.scores(self.selector.context())
        self.assertEqual(scores[EXPLORE], 0.0)
        self.only_rest(self.selector.tick())

    def test_an_unknown_battery_is_treated_like_a_low_one(self):
        self.build()
        self.safety.set_battery_state(BatteryState.UNKNOWN)
        self.idle_for()
        self.only_rest(self.selector.tick())

    def test_a_critical_battery_leaves_rest_only_and_the_head_still(self):
        self.build(window=True)
        self.questions.add("backpack", "living_room")
        self.safety.set_battery_state(BatteryState.CRITICAL)
        self.idle_for()
        self.robot.poses.clear()
        self.only_rest(self.selector.tick())
        self.assertEqual(self.robot.poses, [])

    def test_the_pause_means_nothing_at_all_until_the_user_lifts_it(self):
        self.build()
        self.safety.pause_autonomy()
        self.idle_for(3600)
        self.assertIsNone(self.selector.tick())
        self.idle_for(3600)
        self.assertIsNone(self.selector.tick())
        self.assertEqual(len(self.selector.history), 0)
        self.assertFalse(self.robot.drove)
        self.safety.resume_autonomy()
        self.assertIsNone(self.selector.tick())  # the whole idle time again
        self.idle_for()
        self.assertIsNotNone(self.selector.tick())

    def test_a_conversation_keeps_her_from_choosing_and_the_idle_time_starts_after_it(self):
        self.build(allow_autonomous=False)
        chat = self.arbiter.request("conversation", P.CONVERSATION)
        self.clock.advance(1000)
        self.assertIsNone(self.selector.tick())
        self.arbiter.release(chat)
        self.clock.advance(299)
        self.assertIsNone(self.selector.tick())
        self.clock.advance(2)
        self.assertIsNotNone(self.selector.tick())

    def test_another_behaviour_holding_the_body_is_not_idle(self):
        self.build(allow_autonomous=False)
        lease = self.arbiter.request("investigation", P.CURIOSITY)
        self.clock.advance(1000)
        self.assertIsNone(self.selector.tick())
        lease.release()
        self.assertIsNone(self.selector.tick())

    def test_someone_present_means_no_driving_activity_starts(self):
        self.build()
        self.presence.observe({"people": 1, "confidence": 0.9, "location": "living_room"})
        self.assertTrue(self.presence.present)
        self.idle_for()
        scores = self.selector.scores(self.selector.context())
        self.assertEqual((scores[EXPLORE], scores[CHECK_HOME]), (0.0, 0.0))
        run = self.selector.tick()
        self.assertEqual(run.activity, REST)
        self.assertFalse(self.robot.drove)

    def test_the_switch_turns_the_whole_thing_off(self):
        self.build(idle=IdleConfig(enabled=False))
        self.idle_for(10**6)
        self.assertIsNone(self.selector.tick())
        self.selector.start()
        self.assertIsNone(self.selector._thread)
        self.build(idle=IdleConfig(idle_after_s=0))
        self.idle_for(10**6)
        self.assertIsNone(self.selector.tick())


class InterruptionTests(IdleCase):
    def in_background(self):
        result = []
        thread = threading.Thread(target=lambda: result.append(self.selector.tick()), daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5.0)
        return thread, result

    def test_the_wake_word_interrupts_exploring_at_once(self):
        self.build()
        self.idle_for()
        thread, result = self.in_background()
        self.assertTrue(wait_for(lambda: self.explorer.active))
        self.assertTrue(wait_for(lambda: self.explorer.steps >= 1))
        started = time.monotonic()
        chat = self.arbiter.request("conversation", P.CONVERSATION)
        self.assertIsNotNone(chat)  # it pre-empts, nothing is refused
        thread.join(5.0)
        self.assertFalse(thread.is_alive())
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertEqual(result[0].outcome.status, "interrupted")
        self.assertFalse(self.explorer.active)
        self.assertIs(self.arbiter.holder, chat)
        self.assertTrue(self.episodes()[0].summary.startswith("Idle: explored for"))
        self.assertEqual(self.episodes()[0].subjects[2], "interrupted")
        # She waits the whole idle time again after the conversation, however long it was.
        self.clock.advance(600)
        self.assertIsNone(self.selector.tick())
        self.arbiter.release(chat)
        self.clock.advance(299)
        self.assertIsNone(self.selector.tick())
        self.clock.advance(2)
        self.assertIsNotNone(self.selector.tick())

    def test_the_wake_word_interrupts_resting(self):
        self.build(hour=3, quiet_hours="")
        self.idle_for()
        thread, result = self.in_background()
        self.assertTrue(wait_for(lambda: self.arbiter.holder is not None))
        self.assertEqual(self.arbiter.holder.name, "idle-rest")
        chat = self.arbiter.request("conversation", P.CONVERSATION)
        self.assertIsNotNone(chat)
        thread.join(5.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual((result[0].activity, result[0].outcome.status), (REST, "interrupted"))
        self.assertIs(self.arbiter.holder, chat)

    def test_the_wake_word_event_interrupts_too_and_restarts_the_idle_time(self):
        self.build()
        self.idle_for()
        thread, result = self.in_background()
        self.assertTrue(wait_for(lambda: self.explorer.active))
        emit(self.bus, EventType.INTERACTION_STARTED)
        thread.join(5.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0].outcome.status, "interrupted")
        self.clock.advance(100)
        self.assertIsNone(self.selector.tick())

    def test_safety_stops_the_activity_too(self):
        self.build(hour=3)
        self.idle_for()
        thread, result = self.in_background()
        self.assertTrue(wait_for(lambda: self.arbiter.holder is not None))
        self.safety.emergency_stop("test")
        thread.join(5.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0].outcome.status, "interrupted")
        self.assertIsNone(self.arbiter.holder)


# --- what she remembers ---------------------------------------------------------------------------------------


class RememberingTests(IdleCase):
    def test_a_run_becomes_an_idle_episode_that_recall_and_what_changed_return(self):
        self.build(allow_autonomous=False)
        self.idle_for()
        run = self.selector.tick()
        self.assertEqual(run.activity, EXPLORE)
        (episode,) = self.episodes()
        self.assertEqual(episode.kind, "idle")
        self.assertEqual(episode.subjects[:2], ("idle", "explore"))
        self.assertEqual(episode.location, "living_room")
        self.assertTrue(episode.summary.startswith("Idle: explored"), episode.summary)
        registry = ToolRegistry(self.robot, self.safety, world=self.world, memory=self.memory, clock=self.clock)
        recalled = registry.run(call("recall", query="explore"))
        self.assertTrue(recalled["found"])
        self.assertEqual([e["kind"] for e in recalled["episodes"]], ["idle"])
        changed = registry.run(call("what_changed", since="today"))
        self.assertTrue(changed["ok"])
        self.assertEqual([a["summary"] for a in changed["activities"]], [episode.summary])
        self.assertEqual(changed["count"], 0)  # moved, new, missing objects are another list

    def test_a_run_of_rests_is_one_episode(self):
        self.build(hour=3)
        runs = self.simulate(1500.0, step=60.0)
        self.assertTrue(all(r.activity == REST for r in runs), [r.activity for r in runs])
        self.assertGreaterEqual(len(runs), 4)
        episodes = self.episodes()
        self.assertLess(len(episodes), len(runs))
        self.selector.close()
        episodes = self.episodes()
        self.assertTrue(all(e.summary.startswith("Idle: rested for") for e in episodes))
        total = sum(float(e.subjects[3].rstrip("s")) for e in episodes)
        self.assertAlmostEqual(total, sum(r.outcome.seconds for r in runs), delta=len(runs))

    def test_the_trial_report_counts_the_activities_and_their_time(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "memory").mkdir()
            store = Store(Path(folder) / "memory" / DB_NAME)
            self.build(allow_autonomous=False, window=True, store=store)
            self.questions.add("backpack", "living_room")
            runs = self.simulate(2400.0)
            self.selector.close()
            self.assertTrue(store.flush())
            store.close()
            done = [r for r in runs if r.outcome.status in ("done", "interrupted")]
            start, end = at(0), at(23, 59)
            report = build_report(Path(folder), start, end, now=at(23, 59))
            idle = report["idle"]
            self.assertEqual(idle["count"], len({e.id for e in self.episodes()}))
            self.assertGreaterEqual(idle["count"], 3)
            counted = {name: v["count"] for name, v in idle["by_activity"].items()}
            self.assertEqual(sum(counted.values()), idle["count"])
            self.assertIn(EXPLORE, counted)
            self.assertGreater(idle["by_activity"][EXPLORE]["seconds"], 0)
            self.assertEqual(idle["interrupted"], 0)
            self.assertLessEqual(idle["count"], len(done))

    def test_the_report_has_no_idle_counts_without_a_store(self):
        with tempfile.TemporaryDirectory() as folder:
            report = build_report(Path(folder), at(0), at(23, 59), now=at(23, 59))
            self.assertEqual(report["idle"]["count"], None)


# --- the old timers are gone ----------------------------------------------------------------------------------


class ReplacedTimersTests(unittest.TestCase):
    def test_the_selector_is_the_only_thing_that_starts_idle_activities(self):
        import laika.explore as explore
        import laika.app as app

        self.assertFalse(hasattr(explore, "IdleStarter"))
        self.assertNotIn("IdleStarter", (SRC / "app.py").read_text(encoding="utf-8"))
        self.assertIn("IdleSelector", (SRC / "app.py").read_text(encoding="utf-8"))
        self.assertTrue(hasattr(app, "IdleSelector"))
        for name in ("start", "close", "_run", "poll_s"):
            self.assertFalse(hasattr(WindowWatcher, name), name)
        text = (SRC / "outside.py").read_text(encoding="utf-8")
        self.assertNotIn("laika-window-watch", text)
        self.assertNotIn("window_watcher.start", (SRC / "app.py").read_text(encoding="utf-8"))


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def load(self, text):
        path = Path(self.tmp.name) / "laika.toml"
        path.write_text(text, encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            config = load_config(path)
        return config, out.getvalue()

    def rejected(self, text):
        with self.assertRaises(ValueError) as caught:
            self.load(text)
        return str(caught.exception)

    def test_the_defaults(self):
        config = IdleConfig()
        self.assertEqual(
            (config.enabled, config.idle_after_s, config.between_activities_s, config.max_activity_s, config.max_repeats, config.seed),
            (True, 300.0, 60.0, 180.0, 2, 0),
        )
        self.assertEqual(self.load("")[0].idle, config)

    def test_the_example_file_matches_the_defaults(self):
        example = Path(__file__).resolve().parents[2] / "config" / "laika.example.toml"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            config = load_config(example)
        self.assertEqual(config.idle, IdleConfig())
        self.assertEqual(out.getvalue(), "")

    def test_values_are_loaded(self):
        config, out = self.load("[idle]\nidle_after_s = 120\nrest = false\nexplore_weight = 2.5\nseed = 4\nmax_repeats = 3")
        self.assertEqual(
            (config.idle.idle_after_s, config.idle.rest, config.idle.explore_weight, config.idle.seed, config.idle.max_repeats),
            (120, False, 2.5, 4, 3),
        )
        self.assertEqual(out, "")

    def test_bad_values_are_rejected_naming_the_setting(self):
        for text, name in (
            ("idle_after_s = 1", "idle_after_s"),
            ("idle_after_s = -5", "idle_after_s"),
            ("between_activities_s = -1", "between_activities_s"),
            ("max_activity_s = 1", "max_activity_s"),
            ("max_repeats = 0", "max_repeats"),
            ("max_repeats = 1.5", "max_repeats"),
            ("seed = -1", "seed"),
            ("explore_weight = 11", "explore_weight"),
            ("rest_weight = -1", "rest_weight"),
            ('enabled = "yes"', "enabled"),
            ("follow_up = 1", "follow_up"),
        ):
            with self.subTest(text):
                self.assertIn(name, self.rejected(f"[idle]\n{text}"))
        self.assertIn("Unknown setting", self.rejected("[idle]\nturbo = true"))
        self.assertEqual(self.load("[idle]\nidle_after_s = 0")[0].idle.idle_after_s, 0)

    def test_the_old_explore_idle_after_s_is_read_with_a_note(self):
        config, out = self.load("[explore]\nidle_after_s = 120")
        self.assertEqual(config.idle.idle_after_s, 120)
        self.assertIn("deprecated", out)
        self.assertIn("[idle] idle_after_s = 120", out)

    def test_the_new_setting_wins_over_the_old_one(self):
        config, out = self.load("[explore]\nidle_after_s = 120\n[idle]\nidle_after_s = 600")
        self.assertEqual(config.idle.idle_after_s, 600)
        self.assertIn("deprecated", out)

    def test_the_old_zero_still_means_never(self):
        config, out = self.load("[explore]\nidle_after_s = 0")
        self.assertEqual(config.idle.idle_after_s, 0)
        self.assertIn("deprecated", out)


if __name__ == "__main__":
    unittest.main()
