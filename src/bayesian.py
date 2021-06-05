from tqdm import tqdm

from transformers.models.pegasus.modeling_pegasus import PegasusEncoderLayer, PegasusDecoderLayer
from transformers.models.bart.modeling_bart import BartEncoderLayer, BartDecoderLayer

from src.bleu import analyze_generation_bleuvar


class BayesianSummarizer:
    """
    Bayesian Summarizer class

    Converts a given summarization model to a variational model
    by turning dropout "on" during inference.
    The model can then be used to sample summaries with MC Dropout
    and compute model uncertainty.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.convert_bayesian_model(bayesian=True)

    def convert_bayesian_model(self, bayesian=True):
        """
        Convert the given Pegasus model to either a Bayesian
        or deterministic model.
        """
        if bayesian:
            self.model.eval()
            self.model.apply(apply_dropout)
        else:
            self.model.eval()

    def generate_bayesian_summaries(self, test_loader, device, args):
        """
        Run bayesian summary generation given a DataLoader and a Bayesian model.
        During generation, we run generation N times (with MC dropout) in order
        to sample N different summaries. Then the BLEUvar score is computed based
        on the N sampled summaries and the summary that has the lower avg. BLEUvar score
        with all other summaries is selected as the final generated summary.
        """
        target_sums = []
        generated_sums = []
        article_ids = []
        bleuvars = []
        num_articles = 0
        for i, batch in enumerate(tqdm(test_loader)):
            model_inputs = self.tokenizer(
                batch[args.text_column],
                max_length=args.max_source_length,
                truncation=True,
                padding=True,
                return_tensors='pt')

            input_ids = model_inputs['input_ids'].to(device)
            generations = []
            for i_s in range(args.mc_samples):
                sent_outputs = self.model.generate(
                    input_ids,
                    num_beams=args.num_beams,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True)  # only one beam should be equivalent to greedy,
                gen_sum = [
                    self.tokenizer.decode(
                        g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                    sent_outputs["sequences"]]

                generations.append(gen_sum)

            target_sums += batch[args.summary_column]
            try:
                article_ids += batch["article_id"]
            except KeyError:
                article_ids += range(num_articles, num_articles + len(input_ids))

            num_articles += len(input_ids)
            generations_r = [list(x) for x in zip(*generations)]

            for gen_list in generations_r:
                bleuvar, min_bleuvar, min_gen_idx, min_gen = analyze_generation_bleuvar(gen_list)
                generated_sums.append(min_gen)
                bleuvars.append(bleuvar)

        return generated_sums, target_sums, article_ids, bleuvars


def apply_dropout(m):
    """
    Changes all Encoder and Decoder layers to training mode.
    This will essentially turn dropout and layer normalization
    on for MC dropout prediction.
    """
    if type(m) in [PegasusEncoderLayer, PegasusDecoderLayer, BartEncoderLayer, BartDecoderLayer]:
        m.train()
