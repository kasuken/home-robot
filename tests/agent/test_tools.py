import contextlib
import io
import json
import unittest

from laika.actions import ActionContext, ActionPlayer, available_actions
from laika.ai_client import ToolCall
from laika.house import Pose, default_house
from laika.robot import ALL_CAPABILITIES, CAMERA, DRIVE, ULTRASONIC
from laika.safety import SafetyController
from laika.simulator import SimulatedRobot
from laika.tools import Memory, ToolRegistry, Turn, validate_arguments
from laika.world_model import WorldModel


def call(tool, raw=None, **arguments):
    """A tool call as the client would parse it; `raw` sends non-JSON arguments."""
    if raw is not None:
        return ToolCall.parse({"id": "c1", "function": {"name": tool, "arguments": raw}})
    return ToolCall("c1", tool, arguments, json.dumps(arguments))


class ToolTestCase(unittest.TestCase):
    capabilities = ALL_CAPABILITIES
    allow_driving = True
    start = Pose(300, 200, 0)

    def setUp(self):
        self.robot = SimulatedRobot(
            verbose=False,
            house=default_house(),
            start=self.start,
            capabilities=self.capabilities,
        )
        self.safety = SafetyController(self.robot, enabled=self.allow_driving, speed=25)
        self.addCleanup(self.safety.close)
        self.safety.start()
        self.world = WorldModel()
        self.world.enter_room("living_room")
        self.actions = ActionPlayer(
            ActionContext(self.robot, self.safety),
            available_actions(self.capabilities, self.allow_driving),
            talk_motion=False,
        )
        self.addCleanup(self.actions.close)
        self.spoken = []
        self.registry = ToolRegistry(
            self.robot,
            self.safety,
            world=self.world,
            actions=self.actions,
            speak=self.spoken.append,
            allow_driving=self.allow_driving,
        )

    def run_tool(self, tool, **kwargs):
        """Run a call, returning (result, what was printed)."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = self.registry.run(call(tool, **kwargs))
        return result, out.getvalue()


class SchemaTests(unittest.TestCase):
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 5},
            "s": {"type": "string", "enum": ["a", "b"]},
            "x": {"type": "number"},
        },
        "required": ["n"],
        "additionalProperties": False,
    }

    def test_accepts_valid_arguments(self):
        self.assertIsNone(validate_arguments(self.schema, {"n": 3, "s": "a", "x": 1.5}))

    def test_rejects_each_kind_of_mistake(self):
        for arguments in (
            {},  # missing
            {"n": 1, "extra": 1},  # unknown
            {"n": "1"},  # type
            {"n": True},  # bool is not a number
            {"n": 1.5},  # not whole
            {"n": 9},  # range
            {"n": 1, "s": "c"},  # enum
            {"n": 1, "x": float("nan")},  # not finite
        ):
            self.assertIsNotNone(validate_arguments(self.schema, arguments), arguments)


class OfferedToolsTests(ToolTestCase):
    def test_full_robot_gets_every_tool(self):
        self.assertEqual(
            set(self.registry.names()),
            {
                "get_robot_state", "move", "stop", "look", "look_left", "look_right",
                "look_up", "look_down", "capture_image", "scan_environment", "get_distance", "get_sensors",
                "speak", "remember", "recall", "gesture",
            },
        )

    def test_specs_are_json_schemas(self):
        for spec in self.registry.specs():
            self.assertEqual(spec.parameters["type"], "object")
            json.dumps(spec.to_openai())

    def test_gesture_enum_offers_no_driving_action(self):
        gesture = next(s for s in self.registry.specs() if s.name == "gesture")
        names = gesture.parameters["properties"]["name"]["enum"]
        self.assertIn("nod", names)
        self.assertNotIn("forward", names)


class NoDriveTests(ToolTestCase):
    capabilities = ALL_CAPABILITIES - {DRIVE}

    def test_move_is_hidden_and_a_call_is_rejected(self):
        self.assertNotIn("move", self.registry.names())
        result, printed = self.run_tool("move", direction="forward", duration_ms=300)
        self.assertFalse(result["ok"])
        self.assertIn("unknown tool", result["error"])
        self.assertIn("[laika] tool rejected: move", printed)
        self.assertEqual(self.registry.rejections[-1][0], "move")
        self.assertEqual(self.robot.pose().x, 300)


class DrivingDisabledTests(ToolTestCase):
    allow_driving = False

    def test_config_can_hide_move(self):
        self.assertNotIn("move", self.registry.names())
        self.assertFalse(self.run_tool("move", direction="forward", duration_ms=300)[0]["ok"])


class MissingSensorTests(ToolTestCase):
    capabilities = ALL_CAPABILITIES - {CAMERA, ULTRASONIC}

    def test_tools_follow_capabilities(self):
        self.assertNotIn("capture_image", self.registry.names())
        self.assertNotIn("get_distance", self.registry.names())
        self.assertIn("get_sensors", self.registry.names())


class ValidationTests(ToolTestCase):
    def assertRejected(self, result, printed, text):
        self.assertFalse(result["ok"])
        self.assertIn(text, result["error"])
        self.assertIn("[laika] tool rejected:", printed)
        self.assertIn(text, self.registry.rejections[-1][1])
        self.assertEqual(self.robot.pose().x, 300)  # nothing moved

    def test_unknown_tool(self):
        self.assertRejected(*self.run_tool("fly"), "unknown tool")

    def test_bad_json_arguments(self):
        self.assertRejected(*self.run_tool("move", raw="{not json"), "invalid arguments")

    def test_non_object_arguments(self):
        self.assertRejected(*self.run_tool("move", raw="[1]"), "must be a JSON object")

    def test_missing_parameter(self):
        self.assertRejected(*self.run_tool("move", direction="forward"), "missing parameters")

    def test_unknown_parameter(self):
        self.assertRejected(
            *self.run_tool("move", direction="forward", duration_ms=300, turbo=True),
            "unknown parameters",
        )

    def test_out_of_range_duration_and_speed(self):
        self.assertRejected(
            *self.run_tool("move", direction="forward", duration_ms=99999), "duration_ms"
        )
        self.assertRejected(
            *self.run_tool("move", direction="forward", duration_ms=300, speed=100), "speed"
        )

    def test_bad_direction_and_steering(self):
        self.assertRejected(*self.run_tool("move", direction="left", duration_ms=300), "direction")
        self.assertRejected(
            *self.run_tool("move", direction="forward", duration_ms=300, steering=90), "steering"
        )

    def test_look_out_of_range(self):
        self.assertRejected(*self.run_tool("look", pan=500, tilt=0), "pan")

    def test_speak_needs_text(self):
        self.assertRejected(*self.run_tool("speak", text=""), "text")
        self.assertEqual(self.spoken, [])

    def test_an_unavailable_camera_is_reported_not_raised(self):
        self.robot.faults.camera_error = True
        result, printed = self.run_tool("capture_image")
        self.assertFalse(result["ok"])
        self.assertIn("camera", result["error"])
        self.assertIn("[laika] tool capture_image not done", printed)

    def test_stale_turn_is_rejected(self):
        generation = [0]
        turn = Turn(0, lambda: generation[0])
        generation[0] = 1
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = self.registry.run(call("look_left"), turn)
        self.assertIn("stale", result["error"])
        self.assertIn("tool rejected", out.getvalue())
        self.assertEqual(self.robot.head_pose(), (0, 0))


class MovementTests(ToolTestCase):
    def test_move_forward_changes_the_position(self):
        result, _ = self.run_tool("move", direction="forward", duration_ms=300)
        self.assertTrue(result["ok"], result)
        self.assertGreater(self.robot.pose().x, 300)
        self.assertIsInstance(result["distance_cm"], float)

    def test_move_backward(self):
        result, _ = self.run_tool("move", direction="backward", duration_ms=300, steering=10)
        self.assertTrue(result["ok"], result)
        self.assertLess(self.robot.pose().x, 300)

    def test_move_blocked_by_an_obstacle_reports_why(self):
        self.robot.distance = 10
        result, printed = self.run_tool("move", direction="forward", duration_ms=300)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "obstacle")
        self.assertIn("obstacle", result["error"])
        self.assertEqual(self.robot.pose().x, 300)
        self.assertIn("[laika] tool move not done", printed)

    def test_emergency_stop_makes_the_rest_of_the_turn_stale(self):
        self.safety.emergency_stop("test")
        turn = Turn()
        with contextlib.redirect_stdout(io.StringIO()):
            first = self.registry.run(call("move", direction="forward", duration_ms=300), turn)
            second = self.registry.run(call("look_left"), turn)
        self.assertEqual(first["reason"], "emergency_stop")
        self.assertIn("stale", second["error"])

    def test_stop_tool_calls_the_safety_controller(self):
        self.assertTrue(self.run_tool("stop")[0]["ok"])
        self.assertEqual(self.robot.speed, 0)


class LookTests(ToolTestCase):
    def test_look_left_pans_negative(self):
        self.assertTrue(self.run_tool("look_left")[0]["ok"])
        self.assertLess(self.robot.head_pose()[0], 0)

    def test_look_right_up_down(self):
        self.run_tool("look_right")
        self.assertGreater(self.robot.head_pose()[0], 0)
        self.run_tool("look_up")
        self.assertGreater(self.robot.head_pose()[1], 0)
        self.run_tool("look_down")
        self.assertLess(self.robot.head_pose()[1], 0)

    def test_look_with_explicit_angles(self):
        result, _ = self.run_tool("look", pan=-30, tilt=10)
        self.assertEqual((result["pan"], result["tilt"]), (-30, 10))
        self.assertEqual(self.robot.head_pose(), (-30, 10))


class SensingTests(ToolTestCase):
    start = Pose(300, 200, 180)  # in the living room, facing the sofa

    def test_capture_image_sees_the_sofa(self):
        result, _ = self.run_tool("capture_image")
        self.assertTrue(result["ok"])
        self.assertIn("sofa", result["scene"])
        self.assertEqual(result["room"], "living_room")
        self.assertNotIn("note", result)

    def test_capture_image_with_an_empty_scene_says_vision_is_missing(self):
        self.robot.house.things.clear()
        result, _ = self.run_tool("capture_image")
        self.assertEqual(result["scene"], [])
        self.assertIn("not available yet", result["note"])

    def test_get_distance_is_a_number(self):
        result, _ = self.run_tool("get_distance")
        self.assertTrue(result["ok"])
        self.assertIsInstance(result["distance_cm"], (int, float))

    def test_invalid_distance_reading_is_flagged(self):
        self.robot.faults.ultrasonic_invalid = True
        result, _ = self.run_tool("get_distance")
        self.assertIsNone(result["distance_cm"])
        self.assertIn("note", result)

    def test_get_sensors(self):
        result, _ = self.run_tool("get_sensors")
        self.assertEqual(set(result) - {"ok"}, {"distance_cm", "cliff_detected", "grayscale"})

    def test_get_robot_state(self):
        result, _ = self.run_tool("get_robot_state")
        self.assertEqual(result["robot"]["robotId"], "laika-sim")
        self.assertTrue(result["driving_enabled"])
        self.assertEqual(result["state"]["camera"], {"pan": 0.0, "tilt": 0.0})
        self.assertEqual(result["room"], "living_room")
        json.dumps(result)


class SpeakAndGestureTests(ToolTestCase):
    def test_speak_goes_through_the_callable(self):
        self.assertTrue(self.run_tool("speak", text="Aspetta, guardo.")[0]["ok"])
        self.assertEqual(self.spoken, ["Aspetta, guardo."])

    def test_gesture_plays_without_blocking(self):
        self.assertTrue(self.run_tool("gesture", name="nod")[0]["ok"])
        self.assertTrue(self.actions.wait_done())

    def test_unknown_or_driving_gesture_is_rejected(self):
        self.assertFalse(self.run_tool("gesture", name="forward")[0]["ok"])
        self.assertFalse(self.run_tool("gesture", name="moonwalk")[0]["ok"])

    def test_gestures_per_turn_are_capped(self):
        turn = Turn()
        with contextlib.redirect_stdout(io.StringIO()):
            results = [self.registry.run(call("gesture", name="nod"), turn) for _ in range(5)]
        self.assertEqual([r["ok"] for r in results], [True] * 3 + [False] * 2)


class MemoryToolTests(ToolTestCase):
    def test_remember_then_recall_an_object(self):
        result, _ = self.run_tool("remember", name="Le chiavi", room="kitchen", note="sul tavolo")
        self.assertTrue(result["ok"])
        found, _ = self.run_tool("recall", query="le chiavi")
        self.assertTrue(found["found"])
        self.assertEqual(found["objects"][0]["room"], "kitchen")
        self.assertEqual(found["objects"][0]["note"], "sul tavolo")

    def test_recall_by_room_and_partial_name(self):
        self.run_tool("remember", name="chiavi di casa", room="kitchen")
        by_room = self.run_tool("recall", query="kitchen")[0]
        self.assertEqual(by_room["objects"][0]["name"], "chiavi di casa")
        self.assertTrue(self.run_tool("recall", query="chiavi")[0]["found"])

    def test_remember_a_free_fact(self):
        self.run_tool("remember", fact="Marco ama i dinosauri")
        result, _ = self.run_tool("recall", query="dinosauri")
        self.assertEqual(result["facts"], ["Marco ama i dinosauri"])

    def test_remember_needs_something(self):
        self.assertFalse(self.run_tool("remember", room="kitchen")[0]["ok"])

    def test_recall_nothing_returns_the_summary(self):
        result, _ = self.run_tool("recall", query="unicorno")
        self.assertFalse(result["found"])
        self.assertIn("living room", result["summary"])

    def test_facts_are_bounded(self):
        memory = Memory(max_facts=3)
        for i in range(10):
            memory.add(f"fact {i}")
        self.assertEqual(len(memory), 3)
        self.assertEqual(memory.search("fact"), ["fact 9", "fact 8", "fact 7"])


if __name__ == "__main__":
    unittest.main()
