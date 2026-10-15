from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace

from laika.ai_client import AIReply
from laika.config import CuriosityConfig, MemoryConfig, VoiceConfig, load_config
from laika.conversation import Conversation
from laika.conversation_memory import MemorySelector
from laika.curiosity import InterestScorer
from laika.investigate import PendingQuestions
from laika.memory import Memory
from laika.observations import Observation, SeenObject
from laika.prompts import build_tool_system_prompt
from laika.tools import ToolRegistry
from laika.world_model import WorldModel
from tests.agent.test_agent import AgentTestCase, FakeEar, JsonAI, calls, text
from tests.agent.test_tools import call

NOW = 1_800_000_000.0
STABILIZER_ANSWER = "È uno stabilizzatore per la fotocamera"


class SelectorCase(unittest.TestCase):
    def setUp(self):
        self.world = WorldModel()
        self.memory = Memory()
        self.questions = PendingQuestions()
        self.view: list[str] = []
        self.config = MemoryConfig()
        self.printed = io.StringIO()

    def selector(self) -> MemorySelector:
        return MemorySelector(
            self.world, self.memory, self.questions,
            in_view=lambda: self.view, config=self.config, clock=lambda: NOW,
        )

    def select(self, user_text, tools=(), selector=None):
        selector = selector or self.selector()
        selector.begin_turn()
        with contextlib.redirect_stdout(self.printed):
            return selector.select(user_text, tools)

    def facts(self):
        return self.memory.facts.all()


class AnswerTests(SelectorCase):
    """Done when: telling Laika what an object is stores it as a known object."""

    def ask_about_stabilizer(self):
        return self.questions.add("camera_stabilizer", "office")

    def test_the_prd_example_makes_the_object_known(self):
        question = self.ask_about_stabilizer()
        self.assertIsNone(self.world.where_is("camera_stabilizer"))
        (fact,) = self.select(STABILIZER_ANSWER)
        known = self.world.where_is("camera_stabilizer")
        self.assertEqual(known.note, "uno stabilizzatore per la fotocamera")
        self.assertEqual(known.source, "conversation")
        self.assertEqual(known.room, "office")
        self.assertIsNone(self.questions.get(question.id))
        self.assertEqual((fact.kind, fact.subject, fact.source), ("general", "camera_stabilizer", "user"))
        self.assertEqual(fact.text, "camera_stabilizer: uno stabilizzatore per la fotocamera")
        (episode,) = self.memory.episodes.about("camera_stabilizer")
        self.assertEqual((episode.kind, episode.location), ("answer", "office"))
        self.assertIn("[laika] remembered: general camera_stabilizer:", self.printed.getvalue())

    def test_curiosity_no_longer_scores_the_object_as_unknown(self):
        self.ask_about_stabilizer()
        scorer = InterestScorer(self.world, self.memory, CuriosityConfig())
        view = Observation(
            "a desk", objects=(SeenObject("camera_stabilizer", confidence=0.3),), location="office"
        )
        self.assertGreater(scorer.score(view).components["unknown_object"], 0)
        self.select(STABILIZER_ANSWER)
        score = scorer.score(view)
        self.assertEqual(score.components["unknown_object"], 0)
        self.assertNotIn("unknown object: camera_stabilizer", score.reasons)
        self.assertGreater(score.components["user_relevance"], 0)  # and it carries a user note

    def test_the_question_is_asked_about_once_it_is_answered(self):
        self.ask_about_stabilizer()
        self.select(STABILIZER_ANSWER)
        self.assertEqual(self.questions.pending(), [])

    def test_a_short_noun_phrase_answers_the_question(self):
        self.ask_about_stabilizer()
        self.select("Uno stabilizzatore")  # two words: allowed because it answers a question
        self.assertEqual(self.world.where_is("camera_stabilizer").note, "Uno stabilizzatore")

    def test_the_answer_goes_to_the_question_that_is_named(self):
        self.questions.add("lamp", "office")
        self.questions.add("camera_stabilizer", "office")
        self.select("Il camera stabilizer è una cosa per filmare")
        self.assertIsNotNone(self.world.where_is("camera_stabilizer"))
        self.assertIsNone(self.world.where_is("lamp"))

    def test_an_owner_answer_is_an_ownership_fact(self):
        self.ask_about_stabilizer()
        (fact,) = self.select("È di Leonardo")
        self.assertEqual((fact.kind, fact.text), ("ownership", "camera_stabilizer: di Leonardo"))
        self.assertEqual(self.world.where_is("camera_stabilizer").note, "di Leonardo")

    def test_a_dismissal_is_not_an_answer(self):
        question = self.ask_about_stabilizer()
        for reply in ("Non lo so", "Boh", "No", "Niente"):
            self.assertEqual(self.select(reply), [])
        self.assertIsNotNone(self.questions.get(question.id))
        self.assertEqual(self.facts(), [])

    def test_a_sentence_that_is_not_a_noun_phrase_is_not_an_answer(self):
        question = self.ask_about_stabilizer()
        self.assertEqual(self.select("Vai avanti un po"), [])
        self.assertEqual(self.select("La luce sta male"), [])
        self.assertIsNotNone(self.questions.get(question.id))

    def test_a_question_from_the_last_turn_still_counts(self):
        question = self.ask_about_stabilizer()
        selector = self.selector()
        selector.begin_turn()  # shown this turn...
        self.questions.ttl_s = 3600
        selector.begin_turn()  # ...and the next turn it is still there
        with contextlib.redirect_stdout(self.printed):
            self.assertEqual(len(selector.select(STABILIZER_ANSWER)), 1)
        self.assertIsNone(self.questions.get(question.id))

    def test_the_answer_is_ignored_when_the_selector_has_no_questions(self):
        self.ask_about_stabilizer()
        selector = MemorySelector(self.world, self.memory, None, config=self.config)
        self.assertEqual(self.select(STABILIZER_ANSWER, selector=selector), [])


class RulePatternTests(SelectorCase):
    def only_fact(self, user_text):
        (fact,) = self.select(user_text)
        return fact

    def test_ownership_of_a_named_thing(self):
        fact = self.only_fact("Lo zaino nero è di Leonardo")
        self.assertEqual((fact.kind, fact.subject, fact.text), ("ownership", "zaino nero", "Lo zaino nero è di Leonardo"))
        self.assertEqual(fact.source, "user")

    def test_ownership_with_appartiene_a(self):
        fact = self.only_fact("La chitarra appartiene a Anna")
        self.assertEqual((fact.kind, fact.subject), ("ownership", "chitarra"))

    def test_ownership_of_the_thing_in_view_makes_it_known(self):
        self.view = ["backpack"]
        self.world.observe(["backpack"], "living_room", timestamp=1.0)
        fact = self.only_fact("Questo è di Leonardo")
        self.assertEqual((fact.kind, fact.subject), ("ownership", "backpack"))
        self.assertEqual(self.world.where_is("backpack").note, "Questo è di Leonardo")
        self.assertEqual(self.world.where_is("backpack").source, "conversation")

    def test_ownership_without_a_subject_is_not_guessed(self):
        self.assertEqual(self.select("È di Leonardo"), [])

    def test_made_of_is_not_ownership(self):
        self.assertEqual(self.select("Il tavolo è di legno massiccio"), [])

    def test_location_with_a_room(self):
        fact = self.only_fact("Il tablet di solito sta in cucina")
        self.assertEqual((fact.kind, fact.subject), ("location", "tablet"))
        self.assertEqual(self.world.where_is("tablet").room, "kitchen")
        (episode,) = self.memory.episodes.about("tablet")
        self.assertEqual((episode.kind, episode.location), ("conversation", "kitchen"))

    def test_location_without_a_known_room(self):
        fact = self.only_fact("Lo zaino sta vicino all'ingresso")
        self.assertEqual((fact.kind, fact.subject), ("location", "zaino"))
        self.assertIsNone(self.world.where_is("zaino"))

    def test_a_new_usual_place_replaces_the_old_one(self):
        self.select("Lo zaino sta nel soggiorno")
        self.select("Lo zaino sta nella camera da letto")
        (fact,) = self.facts()
        self.assertEqual(fact.text, "Lo zaino sta nella camera da letto")
        self.assertEqual(self.world.where_is("zaino").room, "bedroom")

    def test_where_something_is_now_is_not_where_it_stays(self):
        self.assertEqual(self.select("Il gatto è in cucina"), [])

    def test_usually_is_enough_for_the_verb_to_be(self):
        fact = self.only_fact("Il casco di solito è nel garage")
        self.assertEqual((fact.kind, fact.subject), ("location", "casco"))

    def test_a_house_rule(self):
        fact = self.only_fact("Non entrare in camera")
        self.assertEqual((fact.kind, fact.subject, fact.text), ("rule", "camera", "Non entrare in camera"))
        (episode,) = self.memory.episodes.recent()
        self.assertEqual(episode.kind, "rule")

    def test_a_house_rule_with_modal_and_article(self):
        fact = self.only_fact("Non devi andare nella camera dei bambini")
        self.assertEqual((fact.kind, fact.subject), ("rule", "camera dei bambini"))

    def test_a_preference(self):
        fact = self.only_fact("Mi piace la musica classica")
        self.assertEqual((fact.kind, fact.subject), ("preference", None))
        self.assertEqual(self.only_fact("Preferisco il tè al limone").kind, "preference")

    def test_si_chiama_about_the_thing_in_view(self):
        self.view = ["robot"]
        fact = self.only_fact("Si chiama Rex il robot")
        self.assertEqual((fact.kind, fact.subject), ("general", "robot"))
        self.assertEqual(self.world.where_is("robot").note, "si chiama Rex il robot")

    def test_si_chiama_with_a_named_subject(self):
        fact = self.only_fact("Il gatto si chiama Briciola")
        self.assertEqual((fact.kind, fact.subject), ("general", "gatto"))
        self.assertIsNone(self.world.where_is("gatto"))  # nothing in view said it is an object

    def test_this_is_a_x_about_the_one_thing_in_view(self):
        self.view = ["unknown object"]
        (fact,) = self.select("Quello è un cacciavite elettrico")
        self.assertEqual(fact.text, "unknown object: un cacciavite elettrico")
        self.assertEqual(self.world.where_is("unknown object").note, "un cacciavite elettrico")

    def test_this_is_a_x_needs_one_clear_object(self):
        self.assertEqual(self.select("Questo è un cacciavite elettrico"), [])
        self.view = ["a", "b"]
        self.assertEqual(self.select("Questo è un cacciavite elettrico"), [])

    def test_a_bare_is_a_x_is_too_uncertain_by_default(self):
        self.view = ["lamp"]
        self.assertEqual(self.select("È un bel giorno oggi"), [])
        self.config.min_confidence = 0.4
        self.assertEqual(len(self.select("È un bel giorno oggi")), 1)

    def test_accents_case_and_apostrophes_do_not_matter(self):
        self.assertEqual(self.only_fact("LO ZAINO ROSSO E' DI ANNA").subject, "zaino rosso")
        self.assertEqual(self.only_fact("NON ENTRARE NELLO STUDIO").subject, "studio")
        self.assertEqual(self.only_fact("l'ombrello sta nell'ingresso").subject, "ombrello")

    def test_confidence_is_kept_and_the_threshold_is_respected(self):
        fact = self.only_fact("Lo zaino sta in camera")
        self.assertAlmostEqual(fact.confidence, 0.65)
        self.config.min_confidence = 0.9
        self.assertEqual(self.select("Lo zaino sta in cucina"), [])


class NeverStoredTests(SelectorCase):
    def test_questions_greetings_and_chit_chat(self):
        self.view = ["lamp"]
        for said in (
            "Lo zaino è di Leonardo?",
            "Dove sta lo zaino di solito",  # no question mark, as the microphone writes it
            "Cos'è questo",
            "Ciao Laika come stai",
            "Buongiorno a tutti",
            "Ehi Laika",
            "Grazie mille",
            "Va bene",
            "Che bella giornata oggi",
            "Raccontami una barzelletta",
            "Vai avanti un po",
        ):
            self.assertEqual(self.select(said), [], said)
        self.assertEqual(self.facts(), [])
        self.assertEqual(len(self.memory.episodes), 0)

    def test_short_utterances(self):
        self.assertEqual(self.select("Mi piace"), [])
        self.assertEqual(self.select("Non entrare"), [])

    def test_a_request_to_forget(self):
        self.memory.facts.remember("Lo zaino è di Leonardo", "zaino", "ownership")
        self.assertEqual(self.select("Dimentica che lo zaino è di Leonardo"), [])
        self.assertEqual(self.select("Scordati che il tablet sta in cucina"), [])
        self.assertEqual(len(self.facts()), 1)

    def test_nothing_when_the_model_already_stored_something(self):
        for tool in ("remember", "answer_question", "forget"):
            self.assertEqual(self.select("Lo zaino nero è di Leonardo", [tool, "gesture"]), [], tool)
        self.assertEqual(self.facts(), [])
        # other tools do not count
        self.assertEqual(len(self.select("Lo zaino nero è di Leonardo", ["gesture", "look_left"])), 1)

    def test_nothing_when_auto_select_is_off(self):
        self.config.auto_select = False
        self.questions.add("camera_stabilizer", "office")
        self.assertEqual(self.select("Lo zaino nero è di Leonardo"), [])
        self.assertEqual(self.select(STABILIZER_ANSWER), [])
        self.assertEqual(self.facts(), [])
        self.assertIsNone(self.world.where_is("camera_stabilizer"))

    def test_empty_text(self):
        self.assertEqual(self.select(""), [])
        self.assertEqual(self.select("   "), [])


class ConfigTests(unittest.TestCase):
    def test_defaults_and_validation(self):
        import tempfile
        from pathlib import Path

        self.assertEqual(MemoryConfig(), MemoryConfig(auto_select=True, min_confidence=0.6))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "laika.toml"
            path.write_text("[memory]\nauto_select = false\nmin_confidence = 0.8\n", encoding="utf-8")
            config = load_config(path).memory
            self.assertEqual((config.auto_select, config.min_confidence), (False, 0.8))
            for bad in ("auto_select = 1", "min_confidence = 2", "min_confidence = 'x'", "nope = 1"):
                path.write_text(f"[memory]\n{bad}\n", encoding="utf-8")
                with self.assertRaises(ValueError, msg=bad):
                    load_config(path)

    def test_the_example_file_matches_the_defaults(self):
        from pathlib import Path

        example = Path(__file__).resolve().parents[2] / "config" / "laika.example.toml"
        self.assertEqual(load_config(example).memory, MemoryConfig())


class ModelPathTests(AgentTestCase):
    """The model stores through its tools: the same results, and the selector stays out of the way."""

    def setUp(self):
        self.memory = Memory()
        self.questions = PendingQuestions()

    def build_agent(self, *script):
        agent = self.build(*script, memory=self.memory, questions=self.questions)
        investigator = SimpleNamespace(questions=self.questions, config=SimpleNamespace(enabled=True))
        self.registry = ToolRegistry(
            self.robot, self.safety, world=self.world, actions=self.actions, memory=self.memory,
            speak=self.robot.speak, investigator=investigator,
        )
        agent.registry = self.registry
        return agent

    def test_remember_about_a_known_object_keeps_it_known(self):
        self.build_agent(
            calls(("remember", {
                "text": STABILIZER_ANSWER, "subject": "camera_stabilizer", "kind": "general",
            })),
            text("Utile."),
        )
        self.world.observe(["camera_stabilizer"], "office", timestamp=1.0)
        reply = self.respond_quietly(STABILIZER_ANSWER)
        self.assertEqual(reply.tool_calls_made, ["remember"])
        self.assertEqual(self.world.where_is("camera_stabilizer").note, STABILIZER_ANSWER)
        self.assertEqual(self.world.where_is("camera_stabilizer").source, "conversation")

    def test_remember_about_a_pending_question_answers_it(self):
        question = self.questions.add("camera_stabilizer", "office")
        self.build_agent()
        result = self.registry.run(call(
            "remember", text=STABILIZER_ANSWER, subject="camera_stabilizer", kind="general"
        ))
        self.assertTrue(result["ok"])
        self.assertEqual(self.world.where_is("camera_stabilizer").note, STABILIZER_ANSWER)
        self.assertIsNone(self.questions.get(question.id))
        (episode,) = self.memory.episodes.about("camera_stabilizer")
        self.assertEqual(episode.kind, "answer")

    def test_remember_ownership_of_a_known_object_notes_it_without_overwriting_a_note(self):
        self.build_agent()
        self.world.observe(["backpack"], "living_room", timestamp=1.0)
        self.registry.run(call("remember", text="Lo zaino è di Leonardo", subject="backpack", kind="ownership"))
        self.assertEqual(self.world.where_is("backpack").note, "Lo zaino è di Leonardo")
        self.registry.run(call("remember", text="Lo zaino è di Anna", subject="backpack", kind="ownership"))
        self.assertEqual(self.world.where_is("backpack").note, "Lo zaino è di Leonardo")
        (fact,) = self.memory.facts.search("backpack")
        self.assertEqual(fact.text, "Lo zaino è di Anna")

    def test_remember_about_a_person_or_a_rule_does_not_invent_an_object(self):
        self.build_agent()
        self.registry.run(call("remember", text="Leonardo ama i dinosauri", subject="Leonardo", kind="general"))
        self.registry.run(call("remember", text="Non entrare in camera", subject="camera", kind="rule"))
        self.assertEqual(self.world.to_dict()["objects"], [])

    def test_answer_question_makes_the_object_known(self):
        question = self.questions.add("camera_stabilizer", "office")
        self.build_agent(
            calls(("answer_question", {"question_id": question.id, "answer": "uno stabilizzatore"})),
            text("Utile."),
        )
        self.respond_quietly(STABILIZER_ANSWER)
        self.assertEqual(self.world.where_is("camera_stabilizer").note, "uno stabilizzatore")
        self.assertEqual(self.questions.pending(), [])

    def test_the_prompt_says_what_to_remember_and_what_not(self):
        prompt = build_tool_system_prompt({}, 3)
        for needle in ("remember", "che cos'è un oggetto", "regola di casa", "Non ricordare chiacchiere"):
            self.assertIn(needle, prompt)


class ConversationTests(AgentTestCase):
    """The selector runs after each completed turn, with an agent and without one."""

    def setUp(self):
        self.memory = Memory()
        self.questions = PendingQuestions()

    def conversation(self, *script, with_agent=True, ai=None, config=None):
        agent = self.build(*script, memory=self.memory, questions=self.questions)
        self.robot.verbose = False
        self.registry.memory = self.memory  # the tools write to the same memory the selector uses
        self.selector = MemorySelector(
            self.world, self.memory, self.questions if with_agent else None,
            in_view=lambda: self.view, config=config, clock=lambda: NOW,
        )
        return Conversation(
            ear=FakeEar(), robot=self.robot, ai=ai or JsonAI(AIReply("non usato")),
            actions=self.actions, voice=VoiceConfig(), safety=self.safety,
            agent=agent if with_agent else None, selector=self.selector,
        )

    view: list[str] = []

    def reply(self, conversation, said):
        with contextlib.redirect_stdout(io.StringIO()):
            return conversation.reply(said)

    def test_the_prd_example_in_tool_mode_through_the_selector(self):
        self.questions.add("camera_stabilizer", "office")
        conversation = self.conversation(text("Utile. Pensavo a un sospetto oggetto di plastica."))
        self.reply(conversation, STABILIZER_ANSWER)
        self.assertEqual(self.world.where_is("camera_stabilizer").note, "uno stabilizzatore per la fotocamera")
        self.assertEqual(self.questions.pending(), [])
        self.assertEqual(len(self.memory.facts), 1)

    def test_the_model_storing_it_leaves_the_selector_idle(self):
        conversation = self.conversation(
            calls(("remember", {"text": "Lo zaino è di Leonardo", "subject": "zaino", "kind": "ownership"})),
            text("Ricordato."),
        )
        self.reply(conversation, "Lo zaino è di Leonardo")
        self.assertEqual(len(self.memory.facts), 1)  # one, not two: the selector did not add its own
        self.assertEqual(len(self.memory.episodes), 1)

    def test_no_selection_for_small_talk_in_tool_mode(self):
        conversation = self.conversation(text("Ciao!"))
        self.reply(conversation, "Ciao Laika come stai")
        self.assertEqual(len(self.memory.facts), 0)

    def test_json_mode_stores_through_the_selector(self):
        ai = JsonAI(AIReply("Capito."))
        conversation = self.conversation(with_agent=False, ai=ai)
        self.reply(conversation, "Lo zaino nero è di Leonardo")
        self.reply(conversation, "Non entrare in camera")
        self.reply(conversation, "Raccontami una barzelletta")
        kinds = sorted((f.kind, f.subject) for f in self.memory.facts.all())
        self.assertEqual(kinds, [("ownership", "zaino nero"), ("rule", "camera")])
        self.assertEqual(self.client.requests, [])  # no agent was used

    def test_json_mode_respects_auto_select(self):
        conversation = self.conversation(
            with_agent=False, ai=JsonAI(AIReply("Capito.")), config=MemoryConfig(auto_select=False)
        )
        self.reply(conversation, "Lo zaino nero è di Leonardo")
        self.assertEqual(len(self.memory.facts), 0)

    def test_a_failing_selector_does_not_break_the_turn(self):
        conversation = self.conversation(with_agent=False, ai=JsonAI(AIReply("Capito.")))
        conversation.selector.select = lambda *_: 1 / 0
        reply = self.reply(conversation, "Lo zaino nero è di Leonardo")
        self.assertEqual(reply.say, "Capito.")

    def test_an_unavailable_model_stores_nothing(self):
        from laika.ai_client import AIUnavailable

        conversation = self.conversation(AIUnavailable("down"))
        self.reply(conversation, "Lo zaino nero è di Leonardo")
        self.assertEqual(len(self.memory.facts), 0)


class HistoryBoundTests(AgentTestCase):
    def test_agent_history_keeps_the_newest_turns_only(self):
        self.build(*[text(f"risposta {i}") for i in range(6)], max_history_turns=2)
        for i in range(6):
            self.respond_quietly(f"domanda {i}")
        self.assertEqual(len(self.agent.history), 4)
        self.assertEqual(self.agent.history[0]["content"], "domanda 4")
        self.assertEqual(self.agent.history[-1]["content"], "risposta 5")

    def test_tool_messages_are_not_kept_between_turns(self):
        self.build(calls(("get_distance", {})), text("Vicino."), text("Ok."))
        self.respond_quietly("quanto è lontano?")
        self.respond_quietly("grazie")
        self.assertEqual({m["role"] for m in self.agent.history}, {"user", "assistant"})

    def test_the_end_of_a_chat_forgets_the_history(self):
        self.build(text("Ciao!"))
        self.respond_quietly("ciao")
        self.agent.reset()
        self.assertEqual(len(self.agent.history), 0)

    def test_the_json_mode_client_is_bounded_too(self):
        from laika.ai_client import ChatClient

        client = ChatClient(
            base_url="http://127.0.0.1:9/v1", api_key="x", model="m", system_prompt="s", max_history_turns=3
        )
        self.assertEqual(client.history.maxlen, 6)

    def test_transcripts_are_off_by_default(self):
        from laika.config import StorageConfig

        self.assertFalse(StorageConfig().keep_transcripts)

