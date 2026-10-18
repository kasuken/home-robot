import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from laika.ai_client import ToolCall
from laika.config import Config, load_config, validate_learning
from laika.learning import (
    CommandResult,
    LearningSettings,
    SkillLearner,
    bump_patch,
    build_prompt,
    check_changes,
    is_protected,
    parse_diff,
    permission_verdict,
    restart_service,
)
from laika.prompts import build_tool_system_prompt
from laika.safety import SafetyController
from laika.simulator import SimulatedRobot
from laika.tools import PROACTIVE_TOOLS, TAINT_BLOCKED, ToolRegistry, Turn

CLEAN_DIFF = """diff --git a/src/laika/dance.py b/src/laika/dance.py
new file mode 100644
--- /dev/null
+++ b/src/laika/dance.py
@@ -0,0 +1,2 @@
+def dance():
+    return 1
"""


def diff_for(path, added=(), removed=()):
    lines = [f"diff --git a/{path} b/{path}", f"--- a/{path}", f"+++ b/{path}", "@@ -1 +1 @@"]
    lines += [f"-{line}" for line in removed] + [f"+{line}" for line in added]
    return "\n".join(lines) + "\n"


class RuleTests(unittest.TestCase):
    def test_protected_paths(self):
        for path in (
            "src/laika/safety.py", "config/laika.toml", "deploy/laika.service", ".git/config",
            "references/sunfounder/picar-x/x.py", "tests/learning/test_learning.py", "src\\laika\\learning.py",
        ):
            self.assertTrue(is_protected(path), path)
        for path in ("src/laika/dance.py", "tests/agent/test_dance.py", "docs/behaviours.md", "src/laika/tools.py"):
            self.assertFalse(is_protected(path), path)

    def test_a_clean_new_module_passes(self):
        self.assertEqual(check_changes(parse_diff(CLEAN_DIFF)), [])

    def test_an_empty_change_is_refused(self):
        self.assertTrue(check_changes([]))

    def test_a_protected_file_is_refused(self):
        problems = check_changes(parse_diff(diff_for("src/laika/safety.py", added=["x = 1"])))
        self.assertIn("src/laika/safety.py is protected", problems)

    def test_a_rename_out_of_a_protected_path_is_refused(self):
        diff = "diff --git a/src/laika/safety.py b/src/laika/other.py\nsimilarity index 90%\n"
        self.assertTrue(check_changes(parse_diff(diff)))

    def test_tools_and_app_may_only_grow(self):
        for path in ("src/laika/tools.py", "src/laika/app.py", "src/laika/store.py"):
            self.assertEqual(check_changes(parse_diff(diff_for(path, added=["    x = 1"]))), [], path)
            self.assertTrue(check_changes(parse_diff(diff_for(path, added=["a"], removed=["b"]))), path)

    def test_the_motion_limits_are_protected(self):
        for path in ("src/laika/commands.py", "src/laika/actions.py"):
            self.assertTrue(check_changes(parse_diff(diff_for(path, added=["x = 1"]))), path)

    def test_the_prompt_tells_copilot_the_rules(self):
        prompt = build_prompt("dance")
        self.assertIn("only ADD lines", prompt)
        self.assertIn("src/laika/safety.py", prompt)
        self.assertNotIn("discover -s tests", prompt)

    def test_store_may_only_grow(self):
        self.assertEqual(check_changes(parse_diff(diff_for("src/laika/store.py", added=["MIGRATION_9 = 1"]))), [])
        self.assertTrue(check_changes(parse_diff(diff_for("src/laika/store.py", added=["a"], removed=["b"]))))

    def test_added_lines_that_reach_the_body_or_system_are_refused(self):
        for line in ("robot.drive(50)", "import subprocess", "from picarx import Picarx", "os.system('x')"):
            with self.subTest(line=line):
                self.assertTrue(check_changes(parse_diff(diff_for("src/laika/dance.py", added=[line]))))
        # tests may mention them
        self.assertEqual(check_changes(parse_diff(diff_for("tests/agent/test_dance.py", added=["robot.drive(50)"]))), [])

    def test_permissions(self):
        repo = Path(tempfile.mkdtemp())
        for command in (
            "git push origin main", "sudo reboot", "pip install x", "curl http://x", "gh release create v1",
            "git commit -am x", "cat config/secrets.toml", "git -C . commit -m x", "git -c a=b push",
            "ls && git reset --hard HEAD~3",
        ):
            self.assertFalse(permission_verdict("shell", command, repo)[0], command)
        for command in ("python -m unittest discover -s tests -t .", "git diff", "ls src", "git status"):
            self.assertTrue(permission_verdict("shell", command, repo)[0], command)
        self.assertTrue(permission_verdict("write", str(repo / "src" / "laika" / "dance.py"), repo)[0])
        self.assertFalse(permission_verdict("write", str(repo / "src" / "laika" / "safety.py"), repo)[0])
        self.assertFalse(permission_verdict("write", str(repo.parent / "elsewhere.py"), repo)[0])
        self.assertFalse(permission_verdict("write", "../x.py", repo)[0])
        self.assertTrue(permission_verdict("read", "src/laika/tools.py", repo)[0])
        self.assertFalse(permission_verdict("read", "config/secrets.toml", repo)[0])
        for kind in ("url", "mcp", "memory", "hook"):
            self.assertFalse(permission_verdict(kind, "x", repo)[0], kind)

    def test_restart_service_only_under_systemd(self):
        self.assertFalse(restart_service({}))
        with mock.patch("laika.learning.subprocess.Popen") as popen:
            self.assertTrue(restart_service({"NOTIFY_SOCKET": "/run/systemd/notify"}))
            self.assertEqual(popen.call_args.args[0], ["systemctl", "restart", "laika"])
            self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_version_bump(self):
        self.assertEqual(bump_patch("0.1.4"), "0.1.5")
        self.assertEqual(bump_patch("weird"), "0.1.0")


class FakeCommands:
    """Records every command; answers from `replies` (a substring of the command -> result)."""

    def __init__(self, diff=CLEAN_DIFF, replies=None):
        self.diff = diff
        self.replies = replies or {}
        self.calls = []

    def __call__(self, args, cwd, timeout_s, env=None):
        self.calls.append((list(args), dict(env or {})))
        text = " ".join(args)
        for needle, result in self.replies.items():
            if needle in text:
                return result
        if "status --porcelain" in text:
            return CommandResult(0, "")
        if "rev-parse --abbrev-ref" in text:
            return CommandResult(0, "main\n")
        if "rev-parse HEAD" in text:
            return CommandResult(0, "abc123\n")
        if "remote get-url" in text:
            return CommandResult(0, "https://github.com/kasuken/home-robot.git\n")
        if "--cached" in text:
            return CommandResult(0, self.diff)
        if "rev-parse -q --verify" in text:
            return CommandResult(1, "")
        return CommandResult(0, "")

    def verbs(self):
        """One short word-run per command: the git subcommand, `tests`, or the program."""
        verbs = []
        for args, _ in self.calls:
            if args[0] == "git":
                rest = [a for a in args[1:] if not a.startswith(("safe.directory", "user."))]
                verbs.append(" ".join(a for a in rest if a != "-c"))
            elif "unittest" in args:
                verbs.append("tests")
            else:
                verbs.append(Path(args[0]).stem)
        return verbs


class LearnerTestCase(unittest.TestCase):
    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        (self.repo / ".git").mkdir()
        (self.repo / "pyproject.toml").write_text('[project]\nname = "laika"\nversion = "0.1.0"\n', encoding="utf-8")
        self.data = Path(tempfile.mkdtemp())
        self.spoken = []
        self.now = 1000.0
        self.asked = []

    def make(self, commands=None, answer="done\nTITLE: Ballo del robot", restart=None, **settings):
        self.commands = commands or FakeCommands()
        self.restarts = []
        self.restart_works = True

        def restart_now():
            self.restarts.append(len(self.spoken))
            return self.restart_works

        def runner(prompt, **kwargs):
            self.asked.append(prompt)
            return answer

        options = {"repo_dir": self.repo, "github_token": "tok-secret"}
        options.update(settings)
        return SkillLearner(
            LearningSettings(**options), self.spoken.append, data_dir=self.data,
            runner=runner, commands=self.commands, clock=lambda: self.now,
            restart=restart or restart_now,
        )

    def learn(self, learner, description="dance when I clap"):
        self.assertTrue(learner.propose(description, object())["ok"])
        result = learner.confirm(object())
        learner.wait(20)
        return result


class FlowTests(LearnerTestCase):
    def test_the_same_turn_cannot_say_yes_for_the_user(self):
        learner = self.make()
        turn = object()
        self.assertTrue(learner.propose("dance when I clap", turn)["ok"])
        self.assertFalse(learner.confirm(turn)["ok"])
        self.assertFalse(learner.running)
        self.assertTrue(learner.confirm(object())["ok"])  # the proposal is still there for the next turn
        learner.wait(20)

    def test_without_a_proposal_or_after_it_expires_nothing_starts(self):
        learner = self.make()
        self.assertFalse(learner.confirm(object())["ok"])
        learner.propose("dance when I clap", object())
        self.now += 601
        self.assertFalse(learner.confirm(object())["ok"])
        self.assertEqual(self.commands.calls, [])

    def test_a_missing_clone_means_no_learning(self):
        learner = self.make(repo_dir=self.repo / "nowhere")
        self.assertFalse(learner.available)
        self.assertFalse(learner.propose("dance when I clap", object())["ok"])

    def test_success_commits_pushes_releases_and_asks_for_the_restart(self):
        learner = self.make()
        self.assertTrue(self.learn(learner)["started"])
        self.assertTrue(learner.last_result["ok"], learner.last_result)
        verbs = self.commands.verbs()
        expected = [
            "status", "pull", "checkout -b", "diff", "tests", "commit", "checkout main", "merge --ff-only",
            "branch -d", "tag", "push -u origin main", "push origin v", "gh",
        ]
        position = -1
        for verb in expected:
            position = next(i for i, v in enumerate(verbs) if i > position and v.startswith(verb))
        self.assertIn('version = "0.1.1"', (self.repo / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertIn("Ballo del robot", self.spoken[-1])
        self.assertFalse(any(v.startswith("checkout -f") for v in verbs))
        self.assertIn("dance when I clap", self.asked[0])

    def test_the_skill_is_merged_into_main_before_the_restart(self):
        learner = self.make()
        self.learn(learner)
        self.assertEqual(learner.last_result["branch"], "main")
        verbs = self.commands.verbs()
        merged = next(i for i, v in enumerate(verbs) if v.startswith("merge --ff-only"))
        tagged = next(i for i, v in enumerate(verbs) if v.startswith("tag"))
        self.assertLess(merged, tagged)
        self.assertEqual(len(self.restarts), 1)
        self.assertFalse(any(v.startswith("checkout -f") for v in verbs))

    def test_a_merge_that_cannot_fast_forward_keeps_the_skill_on_its_branch(self):
        commands = FakeCommands(replies={"merge --ff-only": CommandResult(1, "not possible")})
        learner = self.make(commands)
        self.learn(learner)
        self.assertTrue(learner.last_result["ok"])
        self.assertTrue(learner.last_result["branch"].startswith("skill/"))
        self.assertIn("push -u origin skill/", " | ".join(commands.verbs()))
        self.assertNotIn("branch -d", " | ".join(commands.verbs()))

    def test_success_restarts_by_herself_after_the_last_word(self):
        learner = self.make()
        self.learn(learner)
        self.assertIn("Mi riavvio", self.spoken[-1])
        self.assertEqual(self.restarts, [len(self.spoken)])  # called once, after she had spoken

    def test_a_failed_restart_asks_for_the_command(self):
        learner = self.make()
        self.restart_works = False
        self.learn(learner)
        self.assertIn("sudo systemctl restart laika", self.spoken[-1])

    def test_restart_can_be_switched_off(self):
        learner = self.make(restart=None, **{})
        learner.settings = LearningSettings(repo_dir=self.repo, github_token="tok-secret", restart=False)
        self.learn(learner)
        self.assertIn("sudo systemctl restart laika", self.spoken[-1])
        self.assertEqual(self.restarts, [])

    def test_a_failed_job_never_restarts(self):
        learner = self.make(FakeCommands(replies={"unittest": CommandResult(1, "FAILED")}))
        self.learn(learner)
        self.assertEqual(self.restarts, [])

    def test_the_gate_judges_against_the_commit_the_job_started_from(self):
        learner = self.make()
        self.learn(learner)
        diffs = [args for args, _ in self.commands.calls if "--cached" in args]
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0][-1], "abc123")

    def test_the_release_names_the_repository_so_gh_needs_no_clone(self):
        learner = self.make()
        self.learn(learner)
        release = next(args for args, _ in self.commands.calls if args[0] == "gh")
        self.assertEqual(release[release.index("--repo") + 1], "kasuken/home-robot")

    def test_the_token_never_reaches_the_log_or_the_arguments(self):
        learner = self.make()
        self.learn(learner)
        for args, _ in self.commands.calls:
            self.assertNotIn("tok-secret", " ".join(args))
        logs = list(self.data.glob("skill-*.log"))
        self.assertEqual(len(logs), 1)
        self.assertNotIn("tok-secret", logs[0].read_text(encoding="utf-8"))
        pushes = [env for args, env in self.commands.calls if "push" in args]
        self.assertTrue(pushes)
        self.assertTrue(all("GIT_CONFIG_VALUE_0" in env for env in pushes))

    def test_failing_tests_mean_no_commit_and_a_restored_clone(self):
        commands = FakeCommands(replies={"unittest": CommandResult(1, "FAILED (failures=1)")})
        learner = self.make(commands)
        self.learn(learner)
        self.assertFalse(learner.last_result["ok"])
        verbs = " | ".join(commands.verbs())
        self.assertNotIn("commit", verbs)
        self.assertNotIn("push", verbs)
        self.assertIn("checkout -f main", verbs)
        self.assertIn("clean -fd", verbs)
        self.assertIn("branch -D", verbs)
        self.assertIn("Non sono riuscita", self.spoken[-1])

    def test_a_protected_path_in_the_diff_means_no_tests_and_no_push(self):
        commands = FakeCommands(diff=diff_for("src/laika/safety.py", added=["x = 1"]))
        learner = self.make(commands)
        self.learn(learner)
        self.assertFalse(learner.last_result["ok"])
        self.assertIn("protected", learner.last_result["error"])
        verbs = " | ".join(commands.verbs())
        self.assertNotIn("tests", verbs)
        self.assertNotIn("push", verbs)
        self.assertIn("checkout -f main", verbs)

    def test_a_dirty_clone_is_not_touched(self):
        commands = FakeCommands(replies={"status --porcelain": CommandResult(0, " M src/laika/tools.py\n")})
        learner = self.make(commands)
        self.learn(learner)
        self.assertFalse(learner.last_result["ok"])
        self.assertEqual(self.asked, [])
        self.assertNotIn("checkout -b", " | ".join(commands.verbs()))

    def test_a_missing_sdk_is_a_clean_failure(self):
        learner = self.make()

        def no_sdk(prompt, **kwargs):
            raise ImportError("No module named 'copilot'")

        learner._runner = no_sdk
        self.learn(learner)
        self.assertIn("not installed", learner.last_result["error"])
        self.assertIn("checkout -f main", " | ".join(self.commands.verbs()))

    def test_a_failed_push_keeps_the_skill_but_says_so(self):
        commands = FakeCommands(replies={"push": CommandResult(1, "denied")})
        learner = self.make(commands)
        self.learn(learner)
        self.assertTrue(learner.last_result["ok"])
        self.assertFalse(learner.last_result["published"])
        self.assertIn("pubblicarla", self.spoken[-1])
        self.assertNotIn("gh", commands.verbs())

    def test_a_second_job_is_refused_while_one_runs_and_the_daily_cap_holds(self):
        gate = threading.Event()
        learner = self.make()
        original = learner._runner

        def slow(prompt, **kwargs):
            gate.wait(10)
            return original(prompt, **kwargs)

        learner._runner = slow
        self.assertTrue(learner.propose("dance when I clap", object())["ok"])
        self.assertTrue(learner.confirm(object())["ok"])
        self.assertFalse(learner.propose("sing in english please", object())["ok"])
        gate.set()
        learner.wait(20)
        self.assertTrue(learner.last_result["ok"])
        for _ in range(2):
            self.learn(learner, "another new thing please")
        self.assertFalse(learner.propose("one more thing please", object())["ok"])
        self.now += 90000
        self.assertTrue(learner.propose("one more thing please", object())["ok"])


class ToolTests(LearnerTestCase):
    def setUp(self):
        super().setUp()
        self.robot = SimulatedRobot(verbose=False)
        self.safety = SafetyController(self.robot, enabled=True, speed=25)
        self.addCleanup(self.safety.close)
        self.safety.start()

    def registry(self, learner):
        return ToolRegistry(self.robot, self.safety, speak=self.spoken.append, learner=learner)

    def call(self, registry, name, turn, **arguments):
        return registry.run(ToolCall("c", name, arguments, "{}"), turn)

    def test_the_tools_exist_only_with_a_learner(self):
        self.assertNotIn("learn_skill", self.registry(None).names())
        names = self.registry(self.make()).names()
        self.assertIn("propose_skill", names)
        self.assertIn("learn_skill", names)

    def test_a_proactive_turn_never_offers_them_and_web_content_blocks_them(self):
        for name in ("propose_skill", "learn_skill"):
            self.assertNotIn(name, PROACTIVE_TOOLS)
            self.assertIn(name, TAINT_BLOCKED)

    def test_yes_in_the_same_turn_is_refused_and_in_the_next_it_starts(self):
        learner = self.make()
        registry = self.registry(learner)
        first = Turn()
        self.assertTrue(self.call(registry, "propose_skill", first, description="dance when I clap")["ok"])
        self.assertFalse(self.call(registry, "learn_skill", first)["ok"])
        second = Turn()
        self.assertTrue(self.call(registry, "learn_skill", second)["started"])
        learner.wait(20)

    def test_a_tainted_turn_cannot_learn(self):
        learner = self.make()
        registry = self.registry(learner)
        self.call(registry, "propose_skill", Turn(), description="dance when I clap")
        tainted = Turn()
        tainted.tainted = True
        result = self.call(registry, "learn_skill", tainted)
        self.assertFalse(result["ok"])
        self.assertFalse(learner.running)


class PromptAndConfigTests(unittest.TestCase):
    def test_the_prompt_rule_appears_only_when_learning_is_on(self):
        on = build_tool_system_prompt({}, 3, learning=True)
        off = build_tool_system_prompt({}, 3)
        self.assertIn("propose_skill", on)
        self.assertNotIn("propose_skill", off)

    def test_defaults_validate_and_bad_values_are_named(self):
        validate_learning(Config().learning)
        for name, value in (
            ("max_per_day", 0), ("timeout_s", 1), ("push", "yes"), ("proposal_ttl_s", 99999), ("restart", 1)
        ):
            config = Config()
            setattr(config.learning, name, value)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                validate_learning(config.learning)

    def test_the_token_comes_from_the_environment_or_secrets(self):
        folder = Path(tempfile.mkdtemp())
        settings = folder / "laika.toml"
        settings.write_text("[learning]\nenabled = true\n", encoding="utf-8")
        (folder / "secrets.toml").write_text('[learning]\ngithub_token = "from-file"\n', encoding="utf-8")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(load_config(settings).learning.github_token, "from-file")
        with mock.patch.dict(os.environ, {"LAIKA_GITHUB_TOKEN": "from-env"}, clear=True):
            self.assertEqual(load_config(settings).learning.github_token, "from-env")


if __name__ == "__main__":
    unittest.main()
