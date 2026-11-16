import unittest

import numpy as np

from loss_guided_bpe_study import (
    FrozenBytePredictor,
    measured_deltas,
    merge,
    self_check as check_multi_rule_mechanics,
)
from loss_guided_pair_study import self_check as check_single_rule


class LossGuidedBPEStudyTests(unittest.TestCase):
    def test_single_rule_count_and_independent_rgb_roundtrip(self):
        check_single_rule()

    def test_nonoverlap_counts_and_suffix_attribution(self):
        check_multi_rule_mechanics()

    def test_causal_delta_matches_direct_prefix_retokenization(self):
        predictor = FrozenBytePredictor([(b"aaabcabbabbaa", b"abcaabbcc")])
        words = [bytes([symbol]) for symbol in range(256)]
        samples = [(0, list(b"aaa"), ord("c")), (0, list(b"aaaa"), ord("b")),
                   (0, list(b"ab"), ord("c")), (0, list(b"acab"), ord("b"))]
        candidates = [(ord("a"), ord("a")), (ord("a"), ord("b"))]
        actual = measured_deltas(samples, candidates, words, predictor, 1)
        self.assertEqual([actual[pair][1] for pair in candidates], [1, 2])
        for size in (1, 2):
            actual = measured_deltas(samples, candidates, words, predictor, size)
            for pair in candidates:
                word = words[pair[0]] + words[pair[1]]
                merged_words = words + [word]
                changes = []
                for _, prefix, target in samples:
                    converted = merge(prefix, pair, 256)
                    before_context = b"".join(words[token] for token in prefix[-size:])
                    after_context = b"".join(merged_words[token] for token in converted[-size:])
                    before = predictor.probability(before_context)[target]
                    after = predictor.probability(after_context)[target]
                    changes.append(np.log(before) - np.log(after))
                self.assertAlmostEqual(actual[pair][0], np.mean(changes))

    def test_predictor_excludes_cross_channel_context(self):
        predictor = FrozenBytePredictor([(b"a", b"b", b"xa")])
        np.testing.assert_array_equal(predictor.occurrences(b"ab"), [])
        np.testing.assert_allclose(predictor.probability(b"ab"), predictor.probability(b"b"))
        self.assertAlmostEqual(predictor.probability(b"xa").sum(), 1)


if __name__ == "__main__":
    unittest.main()
