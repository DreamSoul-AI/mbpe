import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('tokenizers'), 'optional tokenizers research runtime')
class WikiTextDeltaTests(unittest.TestCase):
    def test_byte_differences_keep_line_boundaries_without_reset(self):
        from wikitext_delta_bpe import ALPHABET, chunks, restore

        raw = 'Aé\r\n\n猫\nZ'.encode('utf-8')
        alphabet = dict(enumerate(ALPHABET))
        raw_parts = list(chunks(raw, False))
        delta_parts = list(chunks(raw, True))
        self.assertEqual([len(part) for part in raw_parts], [len(part) for part in delta_parts])
        self.assertEqual(ord(delta_parts[1][0]) - 256, 0)
        for transformed, parts in ((False, raw_parts), (True, delta_parts)):
            ids = [ord(symbol) - 256 for part in parts for symbol in part]
            self.assertEqual(restore(ids, alphabet, transformed), raw)

    def test_difference_wraps_in_both_directions(self):
        from wikitext_delta_bpe import ALPHABET, chunks, restore

        raw = bytes([255, 0, 255])
        [encoded] = chunks(raw, True)
        ids = [ord(symbol) - 256 for symbol in encoded]
        self.assertEqual(ids, [255, 1, 255])
        self.assertEqual(restore(ids, dict(enumerate(ALPHABET)), True), raw)
