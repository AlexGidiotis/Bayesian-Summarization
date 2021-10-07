import unittest
import mock

from transformers import AutoTokenizer, PegasusForConditionalGeneration, PegasusConfig

from test.testing_common_utils import ids_tensor, floats_tensor, values_tensor
from src.bayesian import BayesianSummarizer


class TestBayesianSummarizer(unittest.TestCase):
    @mock.patch("src.bayesian.convert_bayesian_model")
    def setUp(self, mock_mc_model) -> None:
        self.batch_size = 2
        self.num_beams = 3
        self.sequence_length = 10
        self.vocab_size = 99

        dummy_config = PegasusConfig()
        self.model = PegasusForConditionalGeneration(config=dummy_config)
        self.tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

        mock_mc_model.return_value = self.model
        self.bayesian_summarizer = BayesianSummarizer(model=self.model, tokenizer=self.tokenizer)

    def tearDown(self) -> None:
        pass

    def prepare_inputs(self):
        input_ids = ids_tensor((self.batch_size * self.num_beams, self.sequence_length), self.vocab_size)
        next_tokens = ids_tensor((self.batch_size, 2 * self.num_beams), self.vocab_size).to("cpu")
        next_indices = ids_tensor((self.batch_size, 2 * self.num_beams), self.num_beams).to("cpu")
        next_scores, _ = (-floats_tensor((self.batch_size, 2 * self.num_beams)).to("cpu")).sort(descending=True)
        return input_ids, next_tokens, next_indices, next_scores

    @mock.patch("transformers.PegasusForConditionalGeneration.generate")
    def test_mc_dropout(self, mock_generation):
        input_text = "This is the first sentence. This is the second sentence. This is the third sentence."
        target_text = "This is a generated summary."
        mock_input_ids = [[182, 117, 109, 211, 5577, 107, 182, 117, 109, 1]]
        mock_input_ids_tensor = values_tensor(mock_input_ids)
        mock_gen_ids = [[182, 117, 114, 3943, 5627,  107, 1]]
        mock_gen_ids_tensor = values_tensor(mock_gen_ids)
        batch = {
            "document": input_text,
        }

        mock_generation.return_value = {
            "sequences": mock_gen_ids_tensor,
        }
        generations, input_ids = self.bayesian_summarizer.run_mc_dropout(
            batch,
            "cpu",
            "document",
            self.sequence_length,
            self.num_beams,
            3)

        self.assertEqual(generations[0][0], target_text)
        self.assertListEqual(input_ids[0].tolist(), mock_input_ids_tensor[0].tolist())
