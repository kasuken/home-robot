import unittest

from laika.wake_word import WakeWordMatcher


class WakeWordTests(unittest.TestCase):
    def setUp(self):
        self.matcher = WakeWordMatcher(["laika", "laica"])

    def test_wakes_on_what_vosk_writes_for_laika(self):
        for heard in ["ehi laika", "ehi laica", "hey laica", "e laica", "ei lajka", "laika"]:
            with self.subTest(heard=heard):
                match = self.matcher.match(heard)
                self.assertIsNotNone(match)
                self.assertEqual(match.remainder, "")

    def test_wakes_when_the_name_is_split_in_two_words(self):
        self.assertIsNotNone(self.matcher.match("ehi lai ca"))
        self.assertIsNotNone(self.matcher.match("ehi l'aica"))

    def test_keeps_the_question_said_in_the_same_breath(self):
        match = self.matcher.match("ehi laica quanti anni hai")
        self.assertEqual(match.remainder, "quanti anni hai")

    def test_keeps_words_before_the_name_except_fillers(self):
        match = self.matcher.match("ciao laica")
        self.assertEqual(match.remainder, "ciao")

    def test_ignores_ordinary_speech(self):
        for heard in ["", None, "che ore sono", "la casa", "lei canta", "laghi", "ehi"]:
            with self.subTest(heard=heard):
                self.assertIsNone(self.matcher.match(heard))


if __name__ == "__main__":
    unittest.main()
