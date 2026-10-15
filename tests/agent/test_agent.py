import contextlib
import io
import json
import unittest

from laika.actions import ActionContext, ActionPlayer, available_actions
from laika.agent import BOUNDS_FALLBACK, Agent
from laika.ai_client import AIReply, AIUnavailable, AssistantTurn, ToolCall
from laika.config import VoiceConfig
from laika.conversation import NO_CONNECTION, Conversation
from laika.house import Pose, default_house
from laika.prompts import build_tool_system_prompt
from laika.robot import ALL_CAPABILITIES, DRIVE
from laika.safety import SafetyController
from laika.simulator import SimulatedRobot
from laika.tools import Turn, ToolRegistry
from laika.world_model import WorldModel


def tool_call(tool, call_id="c", **arguments):
    return ToolCall(call_id, tool, arguments, json.dumps(arguments))


def calls(*items):
    """An assistant turn that only proposes tool calls: items are (name, arguments)."""
    return AssistantTurn(
        tool_calls=tuple(tool_call(n, f"c{i}", **a) for i, (n, a) in enumerate(items))
    )


def text(content):
    return AssistantTurn(content=content, finish_reason="stop")


class ScriptedClient:
    """Plays back AssistantTurns (or raises exceptions) in order and records the requests."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests = []

    def complete(self, messages, tools=None, tool_choice="auto", json_mode=False):
        self.requests.append(
            {"messages": [dict(m) for m in messages], "tools": tools, "tool_choice": tool_choice}
        )
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class AgentTestCase(unittest.TestCase):
    capabilities = ALL_CAPABILITIES
    start = Pose(300, 200, 0)

    def build(self, *script, **agent_options):
        self.robot = SimulatedRobot(
            verbose=False, house=default_house(), start=self.start, capabilities=self.capabilities
        )
        self.safety = SafetyController(self.robot, speed=25)
        self.addCleanup(self.safety.close)
        self.safety.start()
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
            speak=self.robot.speak,
        )
        self.agent = Agent(
            self.client,
            self.registry,
            build_tool_system_prompt(allowed, 3),
            world=self.world,
            **agent_options,
        )
        return self.agent

    def respond(self, text_in="ciao"):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.printed = out
            return self.agent.respond(text_in)

    def respond_quietly(self, text_in="ciao"):
        self.printed = io.StringIO()
        with contextlib.redirect_stdout(self.printed):
            return self.agent.respond(text_in)


class TaskTests(AgentTestCase):
    def test_vai_avanti_moves_the_robot_through_a_tool(self):
        self.build(
            calls(("move", {"direction": "forward", "duration_ms": 400})),
            text("Eccomi, sono avanzata."),
        )
        reply = self.respond_quietly("vai avanti")
        self.assertEqual(reply.text, "Eccomi, sono avanzata.")
        self.assertEqual(reply.tool_calls_made, ["move"])
        self.assertGreater(self.robot.pose().x, 300)
        # The second request carries the assistant tool call and its result.
        roles = [m["role"] for m in self.client.requests[1]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool"])
        result = json.loads(self.client.requests[1]["messages"][3]["content"])
        self.assertTrue(result["ok"])

    def test_guarda_a_sinistra(self):
        self.build(calls(("look_left", {})), text("Guardo a sinistra."))
        self.respond_quietly("guarda a sinistra")
        self.assertLess(self.robot.head_pose()[0], 0)

    def test_describe_the_room_with_capture_image(self):
        self.start = Pose(300, 200, 180)
        self.build(calls(("capture_image", {})), text("Vedo un divano."))
        self.respond_quietly("cosa vedi?")
        result = json.loads(self.client.requests[1]["messages"][-1]["content"])
        self.assertIn("sofa", result["scene"])

    def test_remember_then_recall_in_a_later_turn(self):
        self.build(
            calls(("remember", {
                "text": "le chiavi sono sul tavolo della cucina", "subject": "chiavi",
                "kind": "location", "room": "kitchen",
            })),
            text("Ricordo."),
            calls(("recall", {"query": "chiavi"})),
            text("Sono in cucina."),
        )
        self.respond_quietly("le chiavi sono sul tavolo della cucina")
        reply = self.respond_quietly("dove sono le chiavi?")
        self.assertEqual(reply.text, "Sono in cucina.")
        result = json.loads(self.client.requests[3]["messages"][-1]["content"])
        self.assertEqual(result["objects"][0]["room"], "kitchen")
        self.assertEqual(result["facts"][0]["text"], "le chiavi sono sul tavolo della cucina")

    def test_get_distance_returns_a_number(self):
        self.build(calls(("get_distance", {})), text("Cento centimetri."))
        self.respond_quietly("quanto è lontano il muro?")
        result = json.loads(self.client.requests[1]["messages"][-1]["content"])
        self.assertIsInstance(result["distance_cm"], (int, float))

    def test_history_keeps_only_user_and_final_text(self):
        self.build(calls(("look_left", {})), text("Fatto."), text("Di nuovo."))
        self.respond_quietly("uno")
        self.respond_quietly("due")
        roles = [m["role"] for m in self.client.requests[2]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertEqual(list(self.agent.history)[1]["content"], "Fatto.")

    def test_history_is_bounded_and_reset_clears_it(self):
        self.build(*[text(f"r{i}") for i in range(5)], max_history_turns=2)
        for i in range(5):
            self.respond_quietly(f"q{i}")
        self.assertEqual(len(self.agent.history), 4)
        self.agent.reset()
        self.assertEqual(len(self.agent.history), 0)

    def test_system_prompt_includes_the_world_summary(self):
        self.build(text("Ciao."))
        self.respond_quietly()
        system = self.client.requests[0]["messages"][0]["content"]
        self.assertIn("living room", system)
        self.assertIn("Laika", system)

    def test_hidden_tool_is_not_offered_and_a_call_is_rejected(self):
        self.capabilities = ALL_CAPABILITIES - {DRIVE}
        self.build(calls(("move", {"direction": "forward", "duration_ms": 300})), text("Non posso."))
        reply = self.respond_quietly("vai")
        offered = [s.name for s in self.client.requests[0]["tools"]]
        self.assertNotIn("move", offered)
        result = json.loads(self.client.requests[1]["messages"][-1]["content"])
        self.assertFalse(result["ok"])
        self.assertEqual(self.robot.pose().x, 300)
        self.assertIn("tool rejected: move", self.printed.getvalue())
        self.assertEqual(reply.text, "Non posso.")

    def test_invalid_calls_are_rejected_and_reported_to_the_model(self):
        self.build(
            AssistantTurn(
                tool_calls=(
                    ToolCall.parse({"id": "a", "function": {"name": "move", "arguments": "{oops"}}),
                    tool_call("fly", "b"),
                    tool_call("move", "c", direction="forward"),
                )
            ),
            text("Non ci sono riuscita."),
        )
        self.respond_quietly()
        messages = self.client.requests[1]["messages"]
        results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        self.assertEqual([r["ok"] for r in results], [False, False, False])
        self.assertEqual(len(self.registry.rejections), 3)
        self.assertEqual(self.printed.getvalue().count("[laika] tool rejected:"), 3)

    def test_move_blocked_by_an_obstacle_is_reported(self):
        self.build(
            calls(("move", {"direction": "forward", "duration_ms": 400})),
            text("C'è un ostacolo."),
        )
        self.robot.distance = 10
        self.respond_quietly()
        result = json.loads(self.client.requests[1]["messages"][-1]["content"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "obstacle")
        self.assertEqual(self.robot.pose().x, 300)


class StaleTests(AgentTestCase):
    def test_cancel_during_a_turn_rejects_the_following_calls(self):
        self.build(
            calls(
                ("speak", {"text": "Un attimo."}),
                ("look_left", {}),
                ("move", {"direction": "forward", "duration_ms": 400}),
            ),
            text("non detto"),
        )
        # The first call (speak) cancels the turn, like a sleep word arriving mid-turn.
        self.registry._speak = lambda _text: self.agent.cancel()
        reply = self.respond_quietly("vai")
        self.assertTrue(reply.cancelled)
        self.assertEqual(reply.text, "")
        self.assertEqual(self.robot.head_pose(), (0, 0))
        self.assertEqual(self.robot.pose().x, 300)
        reasons = [reason for _, reason in self.registry.rejections]
        self.assertEqual(len(reasons), 2)
        self.assertTrue(all(r.startswith("stale") for r in reasons))
        self.assertIn("tool rejected", self.printed.getvalue())
        self.assertEqual(len(self.client.requests), 1)  # no further model request
        self.assertEqual(len(self.agent.history), 0)

    def test_cancel_while_the_model_is_thinking_runs_nothing(self):
        class CancellingClient(ScriptedClient):
            def complete(inner, *args, **kwargs):
                answer = super().complete(*args, **kwargs)
                self.agent.cancel()  # arrives while the request is in flight
                return answer

        self.build()
        self.agent.client = CancellingClient(calls(("look_left", {})))
        reply = self.respond_quietly()
        self.assertTrue(reply.cancelled)
        self.assertEqual(reply.tool_calls_made, [])
        self.assertEqual(self.robot.head_pose(), (0, 0))

    def test_ai_unavailable_mid_turn_abandons_the_turn(self):
        self.build(calls(("look_left", {})), AIUnavailable("down"))
        old_turn = Turn(self.agent._generation, self.agent._current_generation)
        with self.assertRaises(AIUnavailable):
            self.respond_quietly()
        self.assertEqual(self.robot.head_pose()[0] < 0, True)  # the call before the outage ran
        with contextlib.redirect_stdout(io.StringIO()):
            late = self.registry.run(tool_call("look_right"), old_turn)
        self.assertIn("stale", late["error"])
        self.assertLess(self.robot.head_pose()[0], 0)

    def test_a_new_turn_after_a_cancel_works(self):
        self.build(text("Eccomi."))
        self.agent.cancel()
        reply = self.respond_quietly()
        self.assertFalse(reply.cancelled)
        self.assertEqual(reply.text, "Eccomi.")

    def test_emergency_stop_makes_the_rest_of_the_batch_stale(self):
        self.build(
            calls(
                ("move", {"direction": "forward", "duration_ms": 400}),
                ("look_left", {}),
            ),
            text("Sono ferma."),
        )
        self.safety.emergency_stop("test")
        reply = self.respond_quietly()
        results = [
            json.loads(m["content"])
            for m in self.client.requests[1]["messages"]
            if m["role"] == "tool"
        ]
        self.assertIn("stale", results[1]["error"])
        self.assertEqual(self.robot.head_pose(), (0, 0))
        self.assertEqual(reply.text, "Sono ferma.")


class BoundsTests(AgentTestCase):
    def test_round_limit_returns_the_fallback(self):
        looping = [calls(("get_distance", {})) for _ in range(5)]
        self.build(*looping, max_rounds=3)
        reply = self.respond_quietly()
        self.assertEqual(len(self.client.requests), 3)
        self.assertEqual(self.client.requests[-1]["tool_choice"], "none")
        self.assertEqual(self.client.requests[0]["tool_choice"], "auto")
        self.assertEqual(reply.tool_calls_made, ["get_distance"] * 2)  # last round refused
        self.assertEqual(reply.text, BOUNDS_FALLBACK)

    def test_round_limit_keeps_the_model_text(self):
        turn = AssistantTurn(content="Guardo.", tool_calls=(tool_call("get_distance"),))
        self.build(turn, turn, max_rounds=2)
        self.assertEqual(self.respond_quietly().text, f"Guardo. {BOUNDS_FALLBACK}")

    def test_tool_call_limit(self):
        self.build(
            calls(*[("get_distance", {})] * 5),
            text("Fatto."),
            max_tool_calls=2,
        )
        reply = self.respond_quietly()
        self.assertEqual(reply.tool_calls_made, ["get_distance"] * 2)
        results = [
            json.loads(m["content"])
            for m in self.client.requests[1]["messages"]
            if m["role"] == "tool"
        ]
        self.assertEqual([r["ok"] for r in results], [True, True, False, False, False])
        self.assertIn("too many", results[2]["error"])
        self.assertEqual(reply.text, "Fatto.")

    def test_every_tool_call_gets_a_result_message(self):
        self.build(calls(("look_left", {}), ("look_up", {})), text("ok"))
        self.respond_quietly()
        messages = self.client.requests[1]["messages"]
        assistant = next(m for m in messages if m["role"] == "assistant")
        tool_ids = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
        self.assertEqual(tool_ids, [c["id"] for c in assistant["tool_calls"]])


class FakeEar:
    def listen(self, timeout_s=None):
        return ""


class JsonAI:
    """The old ChatClient.ask path, for the use_tools = false fallback."""

    def __init__(self, reply):
        self.reply = reply
        self.questions = []
        self.resets = 0

    def ask(self, question):
        self.questions.append(question)
        return self.reply

    def reset(self):
        self.resets += 1


class ConversationToolModeTests(AgentTestCase):
    def conversation(self, *script, with_agent=True, ai=None):
        self.build(*script)
        self.robot.verbose = False
        self.agent_client = self.client
        return Conversation(
            ear=FakeEar(),
            robot=self.robot,
            ai=ai or JsonAI(AIReply("non usato")),
            actions=self.actions,
            voice=VoiceConfig(),
            safety=self.safety,
            agent=self.agent if with_agent else None,
        )

    def chat(self, conversation, question):
        with contextlib.redirect_stdout(io.StringIO()):
            conversation.chat(question)

    def test_chat_acts_through_tools_and_speaks_the_final_text(self):
        conversation = self.conversation(
            calls(("gesture", {"name": "nod"}), ("move", {"direction": "forward", "duration_ms": 400})),
            text("Sono avanzata."),
        )
        self.chat(conversation, "vai avanti")
        self.assertGreater(self.robot.pose().x, 300)
        self.assertEqual(self.robot.spoken, ["Sono avanzata."])
        self.assertEqual(conversation.ai.questions, [])  # the JSON path was not used
        self.assertEqual(len(self.agent.history), 0)  # history ends with the chat

    def test_speak_tool_and_final_text_are_both_spoken(self):
        conversation = self.conversation(
            calls(("speak", {"text": "Aspetta, guardo."}), ("look_left", {})),
            text("Vedo il muro."),
        )
        self.registry._speak = conversation.speak
        self.chat(conversation, "guarda a sinistra")
        self.assertEqual(self.robot.spoken, ["Aspetta, guardo.", "Vedo il muro."])

    def test_outage_cancels_the_agent_and_speaks_the_italian_message(self):
        conversation = self.conversation(AIUnavailable("down"))
        generation = self.agent._generation
        with contextlib.redirect_stdout(io.StringIO()):
            reply = conversation.reply("ciao")
        self.assertEqual(self.robot.spoken, [NO_CONNECTION])
        self.assertEqual(reply.say, NO_CONNECTION)
        self.assertFalse(self.safety.ai_available)
        self.assertGreater(self.agent._generation, generation)

    def test_without_an_agent_the_json_mode_still_works(self):
        ai = JsonAI(AIReply("Ciao!", ["nod"]))
        conversation = self.conversation(with_agent=False, ai=ai)
        self.chat(conversation, "ciao")
        self.assertEqual(ai.questions, ["ciao"])
        self.assertEqual(self.robot.spoken, ["Ciao!"])
        self.assertEqual(self.agent_client.requests, [])


class PromptTests(unittest.TestCase):
    def test_prompt_lists_gestures_and_hides_driving_when_off(self):
        allowed = available_actions(ALL_CAPABILITIES, True)
        prompt = build_tool_system_prompt(allowed, 3, driving=False)
        self.assertIn("nod:", prompt)
        self.assertNotIn("- forward:", prompt)
        self.assertNotIn("usa move", prompt.lower())
        self.assertIn("usa move", build_tool_system_prompt(allowed, 3, driving=True).lower())


if __name__ == "__main__":
    unittest.main()
