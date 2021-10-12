import unittest

import pandas as pd

from src.common.scoring import score_generations, score_standard


class TestScorer(unittest.TestCase):
    def test_score_gen_same(self):
        text1 = "This is the first sentence. This is the second sentence. This is the third sentence."
        text2 = "This is the first sentence. This is the second sentence. This is the third sentence."
        test_df = pd.DataFrame([(text1, text2)], columns=["gen_sum", "target_sum"])
        metrics, rouge_df = score_generations(test_df)

        self.assertEqual(metrics["rouge1"]["mean"], 100.)
        self.assertEqual(rouge_df["rouge1"][0], 100.)

    def test_score_gen_different(self):
        text1 = "This is the first sentence. This is the second sentence. This is the third sentence."
        text2 = "One test text. Another test text. Yet another test text."
        test_df = pd.DataFrame([(text1, text2)], columns=["gen_sum", "target_sum"])
        metrics, rouge_df = score_generations(test_df)

        self.assertEqual(metrics["rouge1"]["mean"], 0.)
        self.assertEqual(rouge_df["rouge1"][0], 0.)

    def test_score_standard(self):
        text1 = "This is the first sentence. This is the second sentence. This is the third sentence."
        text2 = "This is the first sentence. This is the second sentence. This is the third sentence."
        text3 = "One test text. Another test text. Yet another test text."
        metrics, rouge_df = score_standard([text1, text3], [text2, text2], [0, 1])

        self.assertEqual(rouge_df["rouge1"][0], 100.)
        self.assertEqual(rouge_df["rouge1"][1], 0.)
