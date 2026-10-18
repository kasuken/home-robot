from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

from laika.actions import ACTIONS, ActionContext, ActionPlayer, available_actions
from laika.agent import Agent
from laika.ai_client import AIUnavailable, AssistantTurn, ToolCall
from laika.bedtime import (
    ANOTHER,
    ANSWER_WAIT_S,
    FALLBACK_TITLE,
    MAX_CHUNK_CHARS,
    STORY_PROMPT,
    THINKING,
    BedtimeStories,
    StoryArchive,
    parse_story,
)
from laika.config import VoiceConfig
from laika.conversation import Conversation
from laika.prompts import build_tool_system_prompt
from laika.robot import ALL_CAPABILITIES
from laika.safety import SafetyController
from laika.simulator import SimulatedRobot
from laika.tools import TAINT_BLOCKED, ToolRegistry, Turn
from tests.fakes import LookRecordingRobot

STORY = (
    "**Titolo: La luna di Leo e Ludo**\n"
    "\n"
    "C'era una volta, in un prato verde verde, una tana piccola piccola.\n"
    "Lì vivevano Leo e Ludo.\n"
    "\n"
    "## Una sera, la luna non si vedeva... Shhh!\n"
    "\n"
    "E così, abbracciati, Leo e Ludo si addormentarono felici. Buonanotte!\n"
)


class StoryClient:
    """Only `complete`: returns the story (or raises) and records the messages."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, messages, tools=None, tool_choice="auto", json_mode=False):
        self.requests.append([dict(m) for m in messages])
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return AssistantTurn(content=reply, finish_reason="stop")


class TempDirCase(unittest.TestCase):
    def setUp(self):
        out = io.StringIO()
        redirect = contextlib.redirect_stdout(out)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.archive = StoryArchive(self.root / "stories")

    def files(self):
        return sorted((self.root / "stories").glob("story-*.txt"))


class ParseTests(unittest.TestCase):
    def test_title_and_paragraphs_without_markdown(self):
        story = parse_story(STORY)
        self.assertEqual(story.title, "La luna di Leo e Ludo")
        self.assertEqual(len(story.paragraphs), 3)
        self.assertEqual(story.paragraphs[0], "C'era una volta, in un prato verde verde, una tana piccola piccola. Lì vivevano Leo e Ludo.")
        self.assertNotIn("#", " ".join(story.paragraphs))
        self.assertNotIn("*", story.title)

    def test_a_story_without_a_title_line(self):
        story = parse_story("C'era una volta Leo.\n\nPoi dormì.")
        self.assertEqual(story.title, FALLBACK_TITLE)
        self.assertEqual(story.paragraphs, ("C'era una volta Leo.", "Poi dormì."))

    def test_one_long_block_is_told_in_pieces_of_whole_sentences(self):
        block = " ".join(f"Leo saltò nel prato numero {n}." for n in range(60))
        story = parse_story("Titolo: Salti\n" + block)
        self.assertGreater(len(story.paragraphs), 1)
        for piece in story.paragraphs:
            self.assertLessEqual(len(piece), MAX_CHUNK_CHARS)
            self.assertTrue(piece.endswith("."))
        self.assertEqual(" ".join(story.paragraphs), block)


class ArchiveTests(TempDirCase):
    def test_stories_are_saved_and_recent_titles_come_back_newest_first(self):
        times = iter([1_000_000.0, 1_000_100.0])
        archive = StoryArchive(self.root / "stories", clock=lambda: next(times))
        archive.save(parse_story("Titolo: Prima\n\nUno."))
        archive.save(parse_story("Titolo: Seconda\n\nDue."))
        self.assertEqual(archive.recent_titles(), ["Seconda", "Prima"])
        self.assertEqual(self.files()[0].read_text(encoding="utf-8"), "Prima\n\nUno.\n")

    def test_two_stories_in_the_same_second_both_stay(self):
        archive = StoryArchive(self.root / "stories", clock=lambda: 1_000_000.0)
        archive.save(parse_story("Titolo: A\n\nUno."))
        archive.save(parse_story("Titolo: B\n\nDue."))
        self.assertEqual(len(self.files()), 2)


class TellTests(TempDirCase):
    def stories(self, *replies):
        self.said = []
        self.client = StoryClient(*replies)
        return BedtimeStories(self.client, self.archive, self.said.append, sleep=lambda s: None)

    def test_a_new_story_is_written_saved_told_and_followed_by_the_question(self):
        stories = self.stories(STORY)
        saved_before_telling = []
        stories.speak = lambda text: (saved_before_telling.append(bool(self.files())), self.said.append(text))
        result = stories.tell()
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.said[0], THINKING)
        self.assertIn("La luna di Leo e Ludo", self.said[1])
        self.assertEqual(self.said[2:-1], list(parse_story(STORY).paragraphs))
        self.assertEqual(self.said[-1], ANOTHER)
        self.assertEqual(saved_before_telling, [False, True, True, True, True, True])
        self.assertEqual(len(self.files()), 1)
        # The model gets the facts, not the story to read again.
        self.assertNotIn("prato", str(result))
        self.assertEqual((result["title"], result["paragraphs_told"]), ("La luna di Leo e Ludo", 3))

    def test_the_owners_prompt_is_the_system_message_and_old_titles_are_avoided(self):
        self.archive.save(parse_story("Titolo: Il bosco di Ludo\n\nUno."))
        stories = self.stories(STORY)
        stories.tell(theme="  con un drago gentile  ")
        system, user = self.client.requests[0]
        self.assertEqual((system["role"], system["content"]), ("system", STORY_PROMPT))
        self.assertIn("Leo e Ludo", STORY_PROMPT)
        self.assertIn("Il bosco di Ludo", user["content"])
        self.assertIn("con un drago gentile.", user["content"])
        self.assertIn("Titolo", user["content"])

    def test_a_cancelled_turn_stops_between_paragraphs_and_the_story_is_kept(self):
        stories = self.stories(STORY)
        result = stories.tell(should_stop=lambda: len(self.said) >= 3)
        self.assertEqual((result["ok"], result["error"], result["paragraphs_told"]), (False, "interrupted", 1))
        self.assertNotIn(ANOTHER, self.said)
        self.assertEqual(len(self.files()), 1)
        self.assertEqual(stories.follow_up_s(8.0), 8.0)  # no question, no special wait
        self.assertFalse(stories.take_quiet_reply())

    def test_a_turn_cancelled_while_the_story_was_written_tells_nothing(self):
        stories = self.stories(STORY)
        result = stories.tell(should_stop=lambda: True)
        self.assertEqual((result["ok"], result["error"]), (False, "interrupted"))
        self.assertEqual(self.said, [THINKING])
        self.assertEqual(len(self.files()), 1)  # written, so kept

    def test_without_the_model_nothing_is_told_or_saved(self):
        stories = self.stories(AIUnavailable("down"))
        result = stories.tell()
        self.assertFalse(result["ok"])
        self.assertEqual(self.said, [THINKING])
        self.assertEqual(self.files(), [])

    def test_after_a_story_she_waits_exactly_fifteen_seconds_once(self):
        stories = self.stories(STORY)
        stories.tell()
        self.assertTrue(stories.take_quiet_reply())
        self.assertFalse(stories.take_quiet_reply())
        self.assertEqual(stories.follow_up_s(30.0), ANSWER_WAIT_S)  # exactly, even with a longer default
        self.assertEqual(stories.follow_up_s(30.0), 30.0)
        stories.tell()
        stories.end()
        self.assertEqual(stories.follow_up_s(8.0), 8.0)
        self.assertEqual(ANSWER_WAIT_S, 15.0)


class ToolTests(TempDirCase):
    def setUp(self):
        super().setUp()
        self.robot = LookRecordingRobot(distance_cm=100)
        self.safety = SafetyController(self.robot)
        self.addCleanup(self.safety.close)
        self.said = []
        stories = BedtimeStories(StoryClient(STORY), self.archive, self.said.append, sleep=lambda s: None)
        self.registry = ToolRegistry(self.robot, self.safety, speak=self.said.append, stories=stories)

    def test_the_tool_tells_a_story(self):
        result = self.registry.run(ToolCall("c1", "bedtime_story", {}))
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.said[-1], ANOTHER)

    def test_no_story_client_no_tool(self):
        self.assertNotIn("bedtime_story", ToolRegistry(self.robot, self.safety).names())

    def test_refused_after_web_content(self):
        self.assertIn("bedtime_story", TAINT_BLOCKED)
        turn = Turn()
        turn.tainted = True
        self.assertFalse(self.registry.run(ToolCall("c1", "bedtime_story", {}), turn)["ok"])
        self.assertEqual(self.said, [])

    def test_the_prompt_mentions_stories_only_when_offered(self):
        self.assertIn("usa bedtime_story", build_tool_system_prompt(ACTIONS, 3, True, bedtime=True))
        self.assertNotIn("bedtime_story", build_tool_system_prompt(ACTIONS, 3, True))


class ScriptedEar:
    """Says the scripted answers, then nothing (asleep); records how long each listen waits."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.waits = []

    def listen(self, timeout_s=None):
        self.waits.append(timeout_s)
        return self.answers.pop(0) if self.answers else None


class UnusedAI:
    """The JSON path is not used in tool mode; the conversation only resets it."""

    def reset(self):
        pass


class ScriptedClient:
    def __init__(self, *turns):
        self.turns = list(turns)
        self.requests = []

    def complete(self, messages, tools=None, tool_choice="auto", json_mode=False):
        self.requests.append([dict(m) for m in messages])
        return self.turns.pop(0)


class ConversationTests(TempDirCase):
    def converse(self, ear, *turns):
        self.robot = SimulatedRobot(verbose=False)
        safety = SafetyController(self.robot)
        self.addCleanup(safety.close)
        actions = ActionPlayer(
            ActionContext(self.robot, safety), available_actions(ALL_CAPABILITIES, True), talk_motion=False
        )
        self.addCleanup(actions.close)
        conversation = Conversation(
            ear=ear, robot=self.robot, ai=UnusedAI(), actions=actions, voice=VoiceConfig(follow_up_s=8.0), safety=safety
        )
        stories = BedtimeStories(StoryClient(STORY), self.archive, conversation.speak, sleep=lambda s: None)
        registry = ToolRegistry(self.robot, safety, actions=actions, speak=conversation.speak, stories=stories)
        self.model = ScriptedClient(*turns)
        conversation.agent = Agent(self.model, registry, build_tool_system_prompt(ACTIONS, 3, True, bedtime=True))
        conversation.stories = stories
        conversation.chat("raccontami una storia della buonanotte")

    def test_the_question_is_the_last_thing_she_says_and_she_waits_fifteen_seconds(self):
        ear = ScriptedEar()
        self.converse(
            ear,
            AssistantTurn(tool_calls=(ToolCall("c1", "bedtime_story", {}),)),
            AssistantTurn(content="Che bella storia! Ne vuoi un'altra?", finish_reason="stop"),
        )
        self.assertEqual(self.robot.spoken[-1], ANOTHER)
        self.assertNotIn("Che bella storia! Ne vuoi un'altra?", self.robot.spoken)
        self.assertEqual(ear.waits, [ANSWER_WAIT_S])

    def test_a_bare_yes_still_has_her_question_before_it(self):
        ear = ScriptedEar("sì")
        self.converse(
            ear,
            AssistantTurn(tool_calls=(ToolCall("c1", "bedtime_story", {}),)),
            AssistantTurn(content="", finish_reason="stop"),  # the model said nothing after the story
            AssistantTurn(content="Va bene!", finish_reason="stop"),
        )
        second_turn = self.model.requests[-1]
        self.assertEqual(second_turn[-2], {"role": "assistant", "content": ANOTHER})
        self.assertEqual(second_turn[-1]["content"], "sì")
        self.assertEqual(ear.waits, [ANSWER_WAIT_S, 8.0])  # after the yes, the usual wait


if __name__ == "__main__":
    unittest.main()
