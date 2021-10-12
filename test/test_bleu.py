import unittest

from src.bayesian_summarization.bleu import pair_bleu, analyze_generation_bleuvar


class TestBLEUVar(unittest.TestCase):
    def test_pair_bleu_similar(self):
        text1 = "This is the first sentence. This is the second sentence. This is the third sentence."
        text2 = "This is the first sentence. This is the second sentence. This is the third sentence."
        bleu = pair_bleu(text1, text2)
        self.assertEqual(bleu, 1.)

    def test_pair_bleu_different(self):
        text1 = "This is the first sentence. This is the second sentence. This is the third sentence."
        text2 = "One test text. Another test text. Yet another test text."
        bleu = pair_bleu(text1, text2)
        self.assertLess(bleu, 0.1)

    def test_pair_bleu_empty(self):
        text1 = "This is the first sentence. This is the second sentence. This is the third sentence."
        text2 = ""
        bleu = pair_bleu(text1, text2)
        self.assertEqual(bleu, 0.)

    def test_analyze_gen_bleuvar_similar(self):
        gen_list = [f"This is the a sentence with probability 0" for _ in range(10)]
        bleuvar, min_bleuvar, min_gen_idx, min_gen = analyze_generation_bleuvar(gen_list)

        self.assertEqual(bleuvar, 0.)
        self.assertEqual(min_bleuvar, 0.)
        self.assertEqual(min_gen_idx, 0)

    def test_analyze_gen_bleuvar_different(self):
        gen_list = [f"This is the {i} sentence with probability {i * 255}" for i in range(10)]
        bleuvar, min_bleuvar, min_gen_idx, min_gen = analyze_generation_bleuvar(gen_list)

        self.assertNotEqual(bleuvar, 0.)
        self.assertNotEqual(min_bleuvar, 0.)

    def test_analyze_gen_bleuvar_empty(self):
        gen_list = []
        bleuvar, min_bleuvar, min_gen_idx, min_gen = analyze_generation_bleuvar(gen_list)

        self.assertEqual(bleuvar, 0.)
        self.assertEqual(min_bleuvar, float('inf'))
        self.assertEqual(min_gen_idx, None)
        self.assertEqual(min_gen, "")
