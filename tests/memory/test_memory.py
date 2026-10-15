from __future__ import annotations

import contextlib
import io
import logging
import sqlite3
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path

from laika.agent import MAX_FACTS_SHOWN, Agent
from laika.events import EventBus, EventType
from laika.memory import (
    EpisodeRecorder,
    EpisodicMemory,
    Memory,
    SemanticMemory,
    iso_minutes,
    relative_time,
)
from laika.observations import Observation, SeenObject, fold
from laika.store import DB_NAME, MIGRATIONS, Store
from laika.tools import ToolRegistry
from laika.world_model import WorldModel, WorldModelPersister, WorldModelUpdater
from tests.agent.test_agent import ScriptedClient, text
from tests.agent.test_tools import ToolTestCase, call

NOW = 1_800_000_000.0
HOUR = 3600.0
DAY = 86400.0


def local(*parts: int) -> float:
    return datetime(*parts).timestamp()


class Clock:
    def __init__(self, now: float = NOW) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float = 1.0) -> None:
        self.now += seconds


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.facts = SemanticMemory(clock=self.clock)

    def test_a_new_location_replaces_the_old_one(self):
        first = self.facts.remember("Lo zaino sta in camera", "zaino", "location")
        self.clock.tick()
        second = self.facts.remember("Lo zaino sta all'ingresso", "Zaino", "location")
        self.assertEqual(len(self.facts), 1)
        self.assertEqual(second.id, first.id)
        self.assertEqual(second.text, "Lo zaino sta all'ingresso")
        self.assertEqual(second.created, first.created)
        self.assertGreater(second.updated, first.updated)

    def test_ownership_replaces_but_other_subjects_and_kinds_stay(self):
        self.facts.remember("La chitarra è di Leo", "chitarra", "ownership")
        self.facts.remember("La chitarra è di Anna", "chitarra", "ownership")
        self.facts.remember("La chitarra sta in salotto", "chitarra", "location")
        self.facts.remember("Il piano è di Anna", "piano", "ownership")
        self.assertEqual(
            {f.text for f in self.facts.all()},
            {"La chitarra sta in salotto", "La chitarra è di Anna", "Il piano è di Anna"},
        )

    def test_a_sighting_does_not_replace_what_the_user_said(self):
        self.facts.remember("Lo zaino sta all'ingresso", "zaino", "location", source="user")
        self.facts.remember("Saw zaino in the office", "zaino", "location", source="investigation")
        self.assertEqual(len(self.facts), 2)
        self.facts.remember("Saw zaino in the kitchen", "zaino", "location", source="investigation")
        self.assertEqual(len(self.facts), 2)

    def test_other_kinds_are_added_and_identical_text_is_deduplicated(self):
        self.facts.remember("Marco ama i dinosauri", "marco", "preference")
        self.facts.remember("Marco odia il broccolo", "marco", "preference")
        self.assertEqual(len(self.facts), 2)
        self.clock.tick()
        again = self.facts.remember("marco  ama i DINOSAURI", "marco", "preference")
        self.assertEqual(len(self.facts), 2)
        self.assertEqual(again.updated, self.clock.now)
        self.facts.remember("Non entrare in camera", None, "rule")
        self.facts.remember("Non entrare in camera", None, "rule")
        self.assertEqual(len(self.facts), 3)

    def test_an_unknown_kind_is_general(self):
        self.assertEqual(self.facts.remember("x", None, "nonsense").kind, "general")

    def test_facts_are_bounded_and_the_least_recently_updated_goes(self):
        facts = SemanticMemory(max_facts=3, clock=self.clock)
        for i in range(3):
            facts.remember(f"fact {i}", None, "general")
            self.clock.tick()
        facts.remember("fact 0", None, "general")  # refreshed: now the newest
        self.clock.tick()
        facts.remember("fact 3", None, "general")
        self.assertEqual(sorted(f.text for f in facts.all()), ["fact 0", "fact 2", "fact 3"])

    def test_the_text_is_cut(self):
        self.assertLessEqual(len(self.facts.remember("x" * 1000).text), 300)

    def test_search_ranks_the_subject_before_the_text(self):
        self.facts.remember("Il gatto dorme sullo zaino nero", "gatto", "general")
        self.clock.tick()
        self.facts.remember("Sta vicino all'ingresso", "zaino nero", "location")
        self.clock.tick()
        self.facts.remember("Lo zaino rosso è di Anna", "zaino rosso", "ownership")
        found = [f.subject for f in self.facts.search("zaino nero")]
        self.assertEqual(found, ["zaino nero", "gatto"])
        self.assertEqual([f.subject for f in self.facts.search("zaino")], ["zaino rosso", "zaino nero", "gatto"])

    def test_search_ignores_case_and_accents(self):
        self.facts.remember("La caffettiera è nel mobile", "caffettiera", "location")
        self.facts.remember("Perché il tè è più buono", None, "general")
        self.assertEqual(len(self.facts.search("CAFFETTIERA")), 1)
        self.assertEqual(len(self.facts.search("te")), 1)
        self.assertEqual(len(self.facts.search("piu buono")), 1)
        self.assertEqual(fold("Living_Room"), "living room")

    def test_search_is_bounded(self):
        for i in range(20):
            self.facts.remember(f"cosa {i}", None)
        self.assertEqual(len(self.facts.search("cosa", 8)), 8)

    def test_mentions(self):
        self.facts.remember("Le chiavi sono vicino alla porta", "chiavi", "location")
        self.assertTrue(self.facts.mentions("chiavi"))
        self.assertTrue(self.facts.mentions("porta"))
        self.assertFalse(self.facts.mentions("lampada"))
        self.assertFalse(self.facts.mentions(""))

    def test_forget_by_subject_or_id(self):
        keep = self.facts.remember("Il piano è di Anna", "piano", "ownership")
        gone = self.facts.remember("Il gatto è di Leo", "gatto", "ownership")
        other = self.facts.remember("Il gatto dorme molto", "gatto", "general")
        self.assertEqual(len(self.facts.forget(other.id)), 1)
        self.assertEqual([f.id for f in self.facts.forget("GATTO")], [gone.id])
        self.assertEqual([f.id for f in self.facts.all()], [keep.id])
        self.assertEqual(self.facts.forget("gatto"), [])

    def test_relevant_facts_share_words_with_the_message(self):
        self.facts.remember("Lo zaino nero sta vicino all'ingresso", "zaino nero", "location")
        self.facts.remember("Marco ama i dinosauri", "marco", "preference")
        self.facts.remember("Il piano è di Anna", "piano", "ownership")
        self.assertEqual([f.subject for f in self.facts.relevant("Dov'è lo zaino?")], ["zaino nero"])
        self.assertEqual([f.subject for f in self.facts.relevant("ti piacciono i dinosauri?")], ["marco"])
        self.assertEqual(self.facts.relevant("ciao come stai"), [])
        self.assertEqual(self.facts.relevant("dov'è, il?"), [])  # only stopwords


class StoreCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "memory" / DB_NAME
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def open(self, **kwargs) -> Store:
        kwargs.setdefault("clock", lambda: NOW)
        store = Store(self.path, **kwargs)
        self.addCleanup(store.close)
        return store


class EpisodeTests(StoreCase):
    def test_the_window_is_bounded_but_the_history_is_searchable(self):
        store = self.open()
        episodes = EpisodicMemory(store, window=3, clock=Clock())
        episodes.record("seen", "Saw cup in the kitchen", ["cup", "kitchen"], "kitchen", NOW - 5 * HOUR)
        for i in range(5):
            episodes.record("seen", f"Saw lamp {i}", ["lamp"], None, NOW - i)
        self.assertEqual(len(episodes), 3)
        self.assertTrue(store.flush())
        (old,) = episodes.about("cup")  # out of the window, found in the store
        self.assertEqual((old.kind, old.location), ("seen", "kitchen"))
        self.assertEqual([e.summary for e in episodes.recent(2)], ["Saw lamp 4", "Saw lamp 3"])  # newest recorded first

    def test_about_matches_accents_case_and_parts_of_a_name(self):
        episodes = EpisodicMemory(self.open(), clock=Clock())
        episodes.record("seen", "Saw the black backpack", ["Zaino Nero", "living_room"])
        for query in ("zaino", "ZAINO NERO", "living room"):
            self.assertEqual(len(episodes.about(query)), 1, query)
        self.assertEqual(episodes.about("lampada"), [])
        self.assertEqual(episodes.about(""), [])

    def test_retention_by_count(self):
        store = self.open(max_episodes=3, retention_every=1)
        episodes = EpisodicMemory(store)
        for i in range(6):
            episodes.record("seen", f"event {i}", ["x"], None, NOW - 100 + i)
        self.assertTrue(store.flush())
        self.assertEqual(store.count_episodes(), 3)
        self.assertEqual([e["summary"] for e in store.recent_episodes()], ["event 3", "event 4", "event 5"])

    def test_retention_by_age_with_a_fake_clock(self):
        clock = Clock()
        store = self.open(max_episode_days=10, retention_every=1, clock=clock)
        episodes = EpisodicMemory(store)
        episodes.record("seen", "old", ["x"], None, NOW - 20 * DAY)
        episodes.record("seen", "recent", ["x"], None, NOW - DAY)
        self.assertTrue(store.flush())
        self.assertEqual([e["summary"] for e in store.recent_episodes()], ["recent"])
        clock.tick(30 * DAY)  # time passes: the next write prunes what has become too old
        episodes.record("seen", "now", ["x"], None, clock.now)
        self.assertTrue(store.flush())
        self.assertEqual([e["summary"] for e in store.recent_episodes()], ["now"])

    def test_episodes_survive_a_restart_into_the_window(self):
        store = self.open()
        Memory(store).episodes.record("answer", "The user said it is a stabilizer", ["stabilizer"], "office", NOW)
        store.close()
        memory = Memory(self.open())
        self.assertEqual(memory.load(), (0, 1))
        (episode,) = memory.episodes.recent()
        self.assertEqual((episode.kind, episode.subjects, episode.location), ("answer", ("stabilizer",), "office"))

    def test_record_moved(self):
        episodes = EpisodicMemory()
        episode = episodes.record_moved("backpack", "living_room", "office", NOW)
        self.assertEqual(episode.kind, "moved")
        self.assertEqual(episode.summary, "backpack moved from the living room to the office")
        self.assertEqual(episode.location, "office")


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.addCleanup(self.bus.close)
        self.bus.start()
        self.episodes = EpisodicMemory(clock=Clock())

    def see(self, timestamp, *objects, room="kitchen"):
        from laika.events import Event

        self.bus.publish(Event(
            EventType.OBSERVATION_CREATED,
            {"objects": list(objects), "location": room},
            timestamp,
        ))
        self.assertTrue(self.bus.drain())

    def test_one_episode_per_object_and_room_per_interval(self):
        EpisodeRecorder(self.episodes, self.bus, interval_s=3600)
        self.see(NOW, "cup", {"name": "Sofa", "confidence": 0.9})
        self.see(NOW + 10, "cup", "sofa")  # a staring camera: nothing new
        self.see(NOW + 3599, "cup")
        self.assertEqual(len(self.episodes), 2)
        self.see(NOW + 3601, "cup", "sofa")
        self.assertEqual(len(self.episodes), 4)
        self.see(NOW + 3602, "cup", room="office")  # another room is another sighting
        self.assertEqual(len(self.episodes), 5)
        latest = self.episodes.recent(1)[0]
        self.assertEqual((latest.kind, latest.summary, latest.location), ("seen", "Saw cup in the office", "office"))
        self.assertEqual(latest.subjects, ("cup", "office"))
        self.assertEqual(latest.timestamp, NOW + 3602)

    def test_a_zero_interval_records_every_sighting(self):
        EpisodeRecorder(self.episodes, self.bus, interval_s=0)
        self.see(NOW, "cup")
        self.see(NOW, "cup")
        self.assertEqual(len(self.episodes), 2)

    def test_close_stops_recording_and_bad_data_is_ignored(self):
        recorder = EpisodeRecorder(self.episodes, self.bus)
        self.see(NOW, "", 5, None, "cup")
        self.assertEqual(len(self.episodes), 1)
        recorder.close()
        self.see(NOW + DAY, "lamp")
        self.assertEqual(len(self.episodes), 1)


class TimeHintTests(unittest.TestCase):
    def test_hints_with_a_fixed_clock(self):
        now = local(2026, 10, 15, 14, 30)
        self.assertEqual(relative_time(now - 20, now), "just now")
        self.assertEqual(relative_time(now - 60, now), "1 minute ago")
        self.assertEqual(relative_time(now - 25 * 60, now), "25 minutes ago")
        self.assertEqual(relative_time(now - 2 * HOUR, now), "2 hours ago")
        self.assertEqual(relative_time(local(2026, 10, 14, 18, 0), now), "yesterday")
        self.assertEqual(relative_time(local(2026, 10, 14, 9, 0), now), "yesterday")
        self.assertEqual(relative_time(local(2026, 10, 12, 9, 0), now), "3 days ago")
        self.assertEqual(relative_time(local(2026, 9, 24, 9, 0), now), "3 weeks ago")
        self.assertEqual(relative_time(local(2026, 4, 1, 9, 0), now), "6 months ago")
        # Shortly after midnight, last evening is hours ago rather than yesterday.
        self.assertEqual(relative_time(local(2026, 10, 14, 22, 0), local(2026, 10, 15, 0, 30)), "2 hours ago")

    def test_iso_minutes_is_absolute_local_time(self):
        stamp = iso_minutes(local(2026, 10, 14, 18, 5, 59))
        self.assertTrue(stamp.startswith("2026-10-14T18:05"), stamp)
        self.assertRegex(stamp, r"[+-]\d\d:\d\d$")


class RegistryCase(ToolTestCase):
    """The memory tools with a real store, as the app wires them."""

    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "memory" / DB_NAME
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        self.clock = Clock(local(2026, 10, 15, 14, 30))
        self.start_session()

    def start_session(self):
        """Open the store and everything on it, like a start of the application."""
        self.store = Store(self.path, clock=self.clock)
        self.memory = Memory(self.store, clock=self.clock)
        self.memory.load()
        self.world = WorldModel()
        self.world.load(*self.store.load_world())
        self.persister = WorldModelPersister(self.world, self.store)
        self.bus = EventBus()
        self.bus.start()
        self.updater = WorldModelUpdater(self.world, self.bus)
        self.recorder = EpisodeRecorder(self.memory.episodes, self.bus)
        self.registry = ToolRegistry(
            self.robot, self.safety, world=self.world, memory=self.memory, clock=self.clock,
            speak=self.spoken.append,
        )

    def end_session(self):
        self.bus.drain()
        self.bus.close()
        self.recorder.close()
        self.updater.close()
        self.persister.close()
        self.store.close()

    def tearDown(self):
        self.end_session()

    def tool(self, name, **arguments):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.registry.run(call(name, **arguments))

    def see(self, timestamp, *objects, room="living_room"):
        from laika.events import Event

        self.bus.publish(Event(
            EventType.OBSERVATION_CREATED, {"objects": list(objects), "location": room}, timestamp
        ))
        self.assertTrue(self.bus.drain())


class PersistenceTests(RegistryCase):
    def test_a_fact_taught_in_an_earlier_session_is_recalled(self):
        taught = self.tool(
            "remember", text="Lo zaino nero sta vicino all'ingresso", subject="zaino nero", kind="location"
        )
        self.assertTrue(taught["ok"])
        self.end_session()

        self.start_session()  # a new store, memory, world and registry on the same file
        result = self.tool("recall", query="zaino nero")
        self.assertTrue(result["found"])
        (fact,) = result["facts"]
        self.assertEqual(fact["text"], "Lo zaino nero sta vicino all'ingresso")
        self.assertEqual((fact["subject"], fact["kind"], fact["source"]), ("zaino nero", "location", "user"))
        self.assertEqual(fact["ago"], "just now")
        self.assertEqual(result["episodes"][0]["kind"], "conversation")

    def test_a_replaced_fact_stays_replaced_after_a_restart(self):
        self.tool("remember", text="Lo zaino sta in camera", subject="zaino", kind="location")
        self.tool("remember", text="Lo zaino sta all'ingresso", subject="zaino", kind="location")
        self.end_session()
        self.start_session()
        self.assertEqual(
            [f["text"] for f in self.tool("recall", query="zaino")["facts"]], ["Lo zaino sta all'ingresso"]
        )

    def test_where_did_you_last_see_it_after_a_restart(self):
        seen_at = self.clock.now - 2 * HOUR
        self.see(seen_at, "zaino nero", "sofa", room="living_room")
        self.see(seen_at + 600, "zaino nero", room="office")
        self.end_session()

        self.clock.tick(HOUR)
        self.start_session()
        result = self.tool("recall", query="zaino nero")
        (known,) = result["objects"]
        self.assertEqual(known["room"], "office")
        self.assertEqual(known["previous_rooms"], ["living_room"])
        self.assertEqual(known["last_seen"], iso_minutes(seen_at + 600))
        self.assertEqual(known["ago"], "2 hours ago")
        episodes = result["episodes"]
        self.assertEqual([e["location"] for e in episodes], ["office", "living_room"])
        self.assertEqual(episodes[0]["summary"], "Saw zaino nero in the office")
        self.assertEqual(episodes[0]["when"], iso_minutes(seen_at + 600))
        self.assertEqual(result["now"], iso_minutes(self.clock.now))

    def test_yesterday_comes_with_a_hint(self):
        yesterday = local(2026, 10, 14, 19, 0)
        self.see(yesterday, "zaino nero", room="kitchen")
        result = self.tool("recall", query="zaino")
        self.assertEqual(result["episodes"][0]["ago"], "yesterday")
        self.assertEqual(result["objects"][0]["ago"], "yesterday")

    def test_forget_removes_the_fact_from_memory_and_the_store(self):
        taught = self.tool("remember", text="Il gatto è di Anna", subject="gatto", kind="ownership")
        self.assertTrue(self.store.flush())
        self.assertEqual(len(self.store.load_facts()), 1)
        self.assertEqual(self.tool("forget", target=taught["stored"]["id"])["forgotten"], 1)
        self.assertEqual(self.tool("recall", query="gatto")["facts"], [])
        self.assertTrue(self.store.flush())
        self.assertEqual(self.store.load_facts(), [])
        self.end_session()
        self.start_session()
        self.assertEqual(len(self.memory), 0)

    def test_eviction_deletes_from_the_store(self):
        facts = SemanticMemory(self.store, max_facts=2, clock=self.clock)
        for i in range(4):
            facts.remember(f"fact {i}")
            self.clock.tick()
        self.assertTrue(self.store.flush())
        self.assertEqual([f["text"] for f in self.store.load_facts()], ["fact 2", "fact 3"])


class RecallTests(RegistryCase):
    def test_subject_matches_come_before_text_matches_and_the_result_is_bounded(self):
        self.tool("remember", text="Il gatto dorme sullo zaino", subject="gatto")
        self.tool("remember", text="Sta all'ingresso", subject="zaino")
        for i in range(30):
            self.tool("remember", text=f"zaino numero {i}")
        result = self.tool("recall", query="Zaino")
        self.assertEqual(result["facts"][0]["subject"], "zaino")
        self.assertLessEqual(len(result["facts"]), 8)
        self.assertLessEqual(len(result["episodes"]), 5)
        self.assertLess(len(str(result)), 6000)

    def test_accents_and_case(self):
        self.tool("remember", text="La caffettiera è nel mobile", subject="Caffettiera")
        self.assertTrue(self.tool("recall", query="CAFFETTIERA")["found"])
        self.tool("remember", text="Il tè è sullo scaffale", subject="tè")
        self.assertTrue(self.tool("recall", query="te")["found"])

    def test_nothing_found(self):
        result = self.tool("recall", query="unicorno")
        self.assertFalse(result["found"])
        self.assertEqual((result["facts"], result["objects"], result["episodes"]), ([], [], []))
        self.assertIn("summary", result)

    def test_remember_with_a_room_teaches_the_world_model(self):
        self.tool("remember", text="Le chiavi stanno in cucina", subject="Le chiavi", kind="location", room="Living Room")
        known = self.world.where_is("le chiavi")
        self.assertEqual((known.room, known.source), ("living_room", "conversation"))

    def test_rules_are_episodes_of_their_own(self):
        self.tool("remember", text="Non entrare in camera da letto", kind="rule")
        self.assertEqual(self.memory.episodes.recent(1)[0].kind, "rule")


class MigrationTests(StoreCase):
    def test_migration_3_applies_to_a_version_2_database(self):
        self.path.parent.mkdir(parents=True)
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        for number in (1, 2):
            for statement in MIGRATIONS[number - 1]:
                conn.execute(statement)
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (number,))
        conn.execute("INSERT INTO world_rooms (name, first_seen, last_visit) VALUES ('kitchen', 1, 2)")
        conn.close()

        store = self.open()
        self.assertEqual(store.load_world()[1][0]["name"], "kitchen")
        self.assertEqual((store.load_facts(), store.recent_episodes()), ([], []))
        memory = Memory(store)
        memory.facts.remember("Lo zaino sta all'ingresso", "zaino", "location")
        memory.episodes.record("seen", "Saw zaino", ["zaino"], None, NOW)
        self.assertTrue(store.flush())
        store.close()

        again = self.open()  # opening again must not re-run migration 3
        self.assertEqual(len(again.load_facts()), 1)
        self.assertEqual(again.count_episodes(), 1)
        conn = sqlite3.connect(self.path)
        versions = [r[0] for r in conn.execute("SELECT version FROM schema_version ORDER BY version")]
        indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
        conn.close()
        self.assertEqual(versions, [1, 2, 3, 4])  # migration 4 (room aliases, #53) follows 3
        self.assertTrue({"idx_facts_subject", "idx_episodes_timestamp"} <= indexes)


class FloodTests(StoreCase):
    def test_memory_is_not_dropped_under_an_observation_flood(self):
        store = self.open(queue_limit=4)
        release, entered = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        original = store._write

        def slow(op):
            entered.set()
            release.wait(10)
            original(op)

        store._write = slow
        store.add_observation(self.observation(0))
        self.assertTrue(entered.wait(5))
        memory = Memory(store)
        memory.facts.remember("Lo zaino sta all'ingresso", "zaino", "location")
        for i in range(1, 60):
            store.add_observation(self.observation(i))
        memory.episodes.record("conversation", "User said something", ["zaino"], None, NOW)
        for i in range(60, 120):
            store.add_observation(self.observation(i))
        release.set()
        self.assertTrue(store.flush(10))
        self.assertGreater(store.stats().dropped, 50)
        self.assertEqual(store.stats().world_dropped, 0)
        self.assertEqual([f["subject"] for f in store.load_facts()], ["zaino"])
        self.assertEqual(store.count_episodes(), 1)

    @staticmethod
    def observation(n: int) -> Observation:
        return Observation(
            description=f"I see {n}", objects=(SeenObject("cup"),), confidence=0.9,
            location="kitchen", timestamp=NOW - 1000 + n, id=f"obs-{n:08x}",
        )


class PromptContextTests(ToolTestCase):
    def build(self, memory):
        self.client = ScriptedClient(text("Ok."))
        self.agent = Agent(
            self.client, self.registry, "Sei Laika.", world=self.world, memory=memory
        )

    def system_prompt(self, message):
        self.agent.respond(message)
        return self.client.requests[0]["messages"][0]["content"]

    def test_the_relevant_facts_are_in_the_prompt(self):
        memory = Memory()
        memory.facts.remember("Lo zaino nero sta vicino all'ingresso", "zaino nero", "location")
        memory.facts.remember("Marco ama i dinosauri", "marco", "preference")
        self.build(memory)
        prompt = self.system_prompt("Dov'è lo zaino nero?")
        self.assertIn("Cosa ricordi", prompt)
        self.assertIn("- Lo zaino nero sta vicino all'ingresso", prompt)
        self.assertNotIn("dinosauri", prompt)
        self.assertIn("Cosa sai della casa", prompt)  # the world summary stays

    def test_nothing_relevant_adds_nothing(self):
        memory = Memory()
        memory.facts.remember("Marco ama i dinosauri", "marco", "preference")
        self.build(memory)
        self.assertNotIn("Cosa ricordi", self.system_prompt("che ore sono?"))

    def test_the_section_is_bounded(self):
        memory = Memory()
        for i in range(40):
            memory.facts.remember(f"{i} il pallone rosso è nella scatola numero {i} " + "x" * 150, f"pallone {i}")
        self.build(memory)
        prompt = self.system_prompt("dov'è il pallone rosso?")
        section = prompt.split("Cosa ricordi")[1]
        self.assertLessEqual(section.count("\n- "), MAX_FACTS_SHOWN)
        self.assertLess(len(section), 800)

    def test_no_memory_no_section(self):
        self.build(None)
        self.assertNotIn("Cosa ricordi", self.system_prompt("zaino"))


if __name__ == "__main__":
    unittest.main()
