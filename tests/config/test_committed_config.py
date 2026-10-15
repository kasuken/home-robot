"""config/laika.toml is committed to a public repository: it must load, hold no keys, and match the defaults."""

import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from laika.config import (
    REPO_ROOT,
    committed_key_warning,
    load_config,
    permission_warning,
)

LIVE = REPO_ROOT / "config" / "laika.toml"
EXAMPLE = REPO_ROOT / "config" / "laika.example.toml"
# Where the committed settings deliberately differ from the defaults.
CHANGED = {("robot", "allow_autonomous_driving"): True}


class CommittedSettingsTests(unittest.TestCase):
    def test_the_committed_settings_hold_no_keys(self):
        self.assertIsNone(committed_key_warning(LIVE))
        data = tomllib.loads(LIVE.read_text(encoding="utf-8"))
        self.assertEqual(data["ai"]["api_key"], "")
        self.assertEqual(data["web"]["brave_api_key"], "")

    def test_they_load_and_turn_on_autonomous_driving(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            config = load_config(LIVE)
        self.assertTrue(config.robot.allow_autonomous_driving)

    def test_they_match_the_example_except_the_marked_changes(self):
        live = tomllib.loads(LIVE.read_text(encoding="utf-8"))
        example = tomllib.loads(EXAMPLE.read_text(encoding="utf-8"))
        for (section, key), value in CHANGED.items():
            self.assertEqual(live[section][key], value)
            live[section][key] = example[section][key]
        self.assertEqual(live, example)


class SecretsTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.settings = self.dir / "laika.toml"
        self.settings.write_text('[ai]\napi_key = ""\n', encoding="utf-8")
        self.secrets = self.dir / "secrets.toml"

    def load(self, env=None):
        with mock.patch.dict(os.environ, env or {}, clear=True):
            return load_config(self.settings)

    def test_keys_come_from_secrets_toml(self):
        self.secrets.write_text('[ai]\napi_key = "sk-secret"\n[web]\nbrave_api_key = "brave"\n', encoding="utf-8")
        config = self.load()
        self.assertEqual((config.ai.api_key, config.web.brave_api_key), ("sk-secret", "brave"))

    def test_environment_variables_win_over_secrets_toml(self):
        self.secrets.write_text('[ai]\napi_key = "sk-file"\n', encoding="utf-8")
        self.assertEqual(self.load({"OPENAI_API_KEY": "sk-env"}).ai.api_key, "sk-env")

    def test_secrets_toml_holds_only_keys(self):
        for text in ('[robot]\nallow_autonomous_driving = true\n', '[ai]\nmodel = "x"\n', '[ai]\napi_key = 5\n'):
            with self.subTest(text=text):
                self.secrets.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.load()

    def test_a_key_in_the_settings_file_is_flagged(self):
        self.settings.write_text('[ai]\napi_key = "sk-oops"\n', encoding="utf-8")
        warning = committed_key_warning(self.settings)
        self.assertIn("[ai] api_key", warning)
        self.assertIn("secrets.toml", warning)

    @unittest.skipUnless(os.name == "posix", "permissions are POSIX only")
    def test_a_world_readable_secrets_file_is_flagged(self):
        self.secrets.write_text('[ai]\napi_key = "sk"\n', encoding="utf-8")
        os.chmod(self.secrets, 0o644)
        self.assertIn("chmod 600", permission_warning(self.settings))
        os.chmod(self.secrets, 0o600)
        self.assertIsNone(permission_warning(self.settings))


if __name__ == "__main__":
    unittest.main()
