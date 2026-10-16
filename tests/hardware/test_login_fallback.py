import os
import unittest
from unittest import mock

from laika.hardware import _login_fallback


class LoginFallbackTests(unittest.TestCase):
    def test_a_working_getlogin_is_left_alone(self):
        original = os.getlogin
        with mock.patch("os.getlogin", return_value="ema") as getlogin:
            _login_fallback()
            self.assertIs(os.getlogin, getlogin)
        self.assertIs(os.getlogin, original)

    def test_without_a_login_session_it_answers_with_a_user(self):
        def no_session():
            raise OSError(-2, "no controlling terminal")

        with mock.patch("os.getlogin", no_session), mock.patch.dict(os.environ, {"SUDO_USER": "ema"}):
            _login_fallback()
            self.assertEqual(os.getlogin(), "ema")


if __name__ == "__main__":
    unittest.main()
