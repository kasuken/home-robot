import json
import unittest

from laika.commands import (
    COMMANDS,
    CommandResult,
    HonkCommand,
    InvalidCommand,
    LookCommand,
    MoveCommand,
    ResultStatus,
    SpeakCommand,
    StopCommand,
    Telemetry,
    parse_command,
)


class RoundTripTests(unittest.TestCase):
    def test_every_command_round_trips_through_json(self):
        commands = [
            MoveCommand(direction="forward", speed=20, duration_ms=500, steering=-10),
            StopCommand(),
            LookCommand(pan=15, tilt=-5),
            HonkCommand(volume=50),
            SpeakCommand(text="Ciao!"),
        ]
        self.assertEqual({c.name for c in commands}, set(COMMANDS))
        for command in commands:
            with self.subTest(command=command.name):
                self.assertEqual(parse_command(command.to_json()), command)
                self.assertEqual(parse_command(command.to_dict()), command)

    def test_prd_move_example_is_valid(self):
        command = parse_command(
            {
                "type": "command",
                "id": "cmd-123",
                "command": "move",
                "parameters": {"direction": "forward", "speed": 20, "durationMs": 500},
            }
        )
        self.assertIsInstance(command, MoveCommand)
        self.assertEqual((command.id, command.speed, command.duration_ms), ("cmd-123", 20, 500))
        self.assertEqual(command.to_dict()["parameters"]["durationMs"], 500)

    def test_ids_are_generated_and_unique(self):
        a, b = StopCommand(), StopCommand()
        self.assertTrue(a.id.startswith("cmd-"))
        self.assertNotEqual(a.id, b.id)
        self.assertGreater(a.created_at, 0)

    def test_created_at_survives_round_trip(self):
        command = StopCommand(created_at=1234.5)
        self.assertEqual(parse_command(command.to_json()).created_at, 1234.5)

    def test_result_round_trips_and_matches_prd(self):
        result = CommandResult("cmd-123", ResultStatus.SUCCESS)
        self.assertEqual(
            result.to_dict(), {"type": "command_completed", "id": "cmd-123", "result": "success"}
        )
        for status in ResultStatus:
            original = CommandResult("cmd-1", status, "because")
            self.assertEqual(CommandResult.from_json(original.to_json()), original)

    def test_telemetry_matches_prd_and_round_trips(self):
        data = {
            "type": "telemetry",
            "distanceCm": 83.4,
            "cliffDetected": False,
            "camera": {"pan": 15, "tilt": -5},
        }
        telemetry = Telemetry.from_dict(data)
        self.assertEqual(telemetry.distance_cm, 83.4)
        self.assertEqual((telemetry.camera_pan, telemetry.camera_tilt), (15, -5))
        self.assertEqual(Telemetry.from_json(telemetry.to_json()), telemetry)
        self.assertIsNone(
            Telemetry.from_json(Telemetry(None, False, 0, 0).to_json()).distance_cm
        )
        json.loads(telemetry.to_json())


class InvalidInputTests(unittest.TestCase):
    def assertRejected(self, data, fragment=""):
        with self.assertRaises(InvalidCommand) as ctx:
            parse_command(data)
        self.assertIn(fragment, str(ctx.exception))

    def move(self, **params):
        base = {"direction": "forward", "speed": 20, "durationMs": 500}
        base.update(params)
        return {"type": "command", "command": "move", "parameters": base}

    def test_malformed_input(self):
        self.assertRejected("{nope", "not valid JSON")
        self.assertRejected("[1]", "must be an object")
        self.assertRejected({"type": "telemetry"}, "type must be 'command'")
        self.assertRejected({"type": "command", "command": "fly"}, "unknown command")

    def test_move_limits(self):
        self.assertRejected(self.move(direction="sideways"), "direction")
        self.assertRejected(self.move(speed=0), "speed")
        self.assertRejected(self.move(speed=41), "speed")
        self.assertRejected(self.move(speed=-5), "speed")
        self.assertRejected(self.move(speed="fast"), "speed")
        self.assertRejected(self.move(speed=True), "speed")
        self.assertRejected(self.move(speed=float("nan")), "speed")
        self.assertRejected(self.move(durationMs=0), "durationMs")
        self.assertRejected(self.move(durationMs=60000), "durationMs")
        self.assertRejected(self.move(steering=45), "steering")

    def test_missing_and_unknown_parameters(self):
        message = self.move()
        del message["parameters"]["durationMs"]
        self.assertRejected(message, "missing parameters: durationMs")
        self.assertRejected(self.move(power=9), "unknown parameters for move: power")
        self.assertRejected({"type": "command", "command": "stop", "parameters": {"x": 1}}, "unknown")

    def test_look_honk_speak_limits(self):
        look = lambda **p: {"type": "command", "command": "look", "parameters": p}
        self.assertRejected(look(pan=91, tilt=0), "pan")
        self.assertRejected(look(pan=0, tilt=-36), "tilt")
        self.assertRejected(look(pan=0), "missing")
        self.assertRejected(
            {"type": "command", "command": "honk", "parameters": {"volume": 101}}, "volume"
        )
        self.assertRejected(
            {"type": "command", "command": "speak", "parameters": {"text": "  "}}, "text"
        )
        self.assertRejected(
            {"type": "command", "command": "speak", "parameters": {"text": "a" * 501}}, "text"
        )

    def test_bad_id_and_timestamp(self):
        self.assertRejected({"type": "command", "command": "stop", "id": ""}, "id")
        self.assertRejected({"type": "command", "command": "stop", "id": 7}, "id")
        self.assertRejected({"type": "command", "command": "stop", "createdAt": "now"}, "createdAt")

    def test_direct_construction_validates_too(self):
        with self.assertRaises(InvalidCommand):
            MoveCommand(direction="forward", speed=99, duration_ms=500)
        with self.assertRaises(InvalidCommand):
            LookCommand(pan=0, tilt=100)

    def test_invalid_result_and_telemetry(self):
        with self.assertRaises(InvalidCommand):
            CommandResult("cmd-1", "maybe")
        with self.assertRaises(InvalidCommand):
            CommandResult.from_dict({"type": "command_completed", "id": "cmd-1"})
        with self.assertRaises(InvalidCommand):
            Telemetry.from_dict({"type": "telemetry", "cliffDetected": "no", "camera": {"pan": 0, "tilt": 0}})
        with self.assertRaises(InvalidCommand):
            Telemetry.from_dict({"type": "telemetry", "cliffDetected": False})


if __name__ == "__main__":
    unittest.main()
