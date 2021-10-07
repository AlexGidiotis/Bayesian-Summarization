from tqdm import tqdm

from transformers.models.pegasus.modeling_pegasus import PegasusEncoderLayer, PegasusDecoderLayer
from transformers.models.bart.modeling_bart import BartEncoderLayer, BartDecoderLayer

from src.bleu import analyze_generation_bleuvar


class BayesianSummarizer:
    """
    Bayesian summarizer class

    Converts a given summarization model to a variational model
    by turning dropout "on" during inference.
    The model can then be used to sample summaries with MC Dropout
    and compute model uncertainty. During generation, we run generation 
    N times (with MC dropout) in order to sample N different summaries.
    Then the BLEUvar score can be computed based on the N sampled summaries.

    NOTE: The summarization model must be one of (BART and PEGASUS)

    Example:
    ```
    model, tokenizer = load_model(...)
    bayesian_summarizer = BayesianSummarizer(model=model, tokenizer=tokenizer)

    generated_sums = bayesian_summarizer.generate_bayesian_summaries(
        dataloader,
        device='gpu',
        text_column='document',
        max_source_length=128,
        num_beams=3,
        n=10)
        
    mc_dropout_gens = bayesian_summarizer.generate_mc_summaries(
        dataloader,
        device='gpu',
        text_column='document',
        max_source_length=128,
        num_beams=3,
        n=10)
    ```
    """
    def __init__(self, model, tokenizer):
        self.model = convert_bayesian_model(model, bayesian=True)
        self.tokenizer = tokenizer

    def run_mc_dropout(
            self,
            batch,
            device,
            text_column,
            max_source_length=128,
            num_beams=3,
            n=10):
        """
        Runs MC dropout generation given a batch of data and a Bayesian model
        
        Returns: N summaries generated with MC dropout for each input in the batch.
        """
        model_inputs = self.tokenizer(
            batch[text_column],
            max_length=max_source_length,
            truncation=True,
            padding=True,
            return_tensors='pt')

        input_ids = model_inputs['input_ids'].to(device)
        generations = []
        for i_s in range(n):
            sent_outputs = self.model.generate(
                input_ids,
                num_beams=num_beams,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True)  # only one beam should be equivalent to greedy,
            gen_sum = [
                self.tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                sent_outputs["sequences"]]

            generations.append(gen_sum)
            
        return generations, input_ids
    
    def generate_bayesian_summaries(
            self,
            dataloader,
            device,
            text_column,
            summary_column=None,
            max_source_length=128,
            num_beams=3,
            n=10):
        """
        Run bayesian summary generation given a DataLoader and a Bayesian model.
        
        Returns: The summary with the lower avg. BLEUvar score.
        """
        target_sums = []
        generated_sums = []
        article_ids = []
        bleuvars = []
        num_articles = 0
        for i, batch in enumerate(tqdm(dataloader)):
            generations, input_ids = self.run_mc_dropout(
                batch,
                device,
                text_column,
                max_source_length,
                num_beams,
                n)
            
            if summary_column is not None:
                target_sums += batch[summary_column]

            try:
                article_ids += batch["article_id"]
            except KeyError:
                article_ids += range(num_articles, num_articles + len(input_ids))

            num_articles += len(input_ids)
            generations_r = [list(x) for x in zip(*generations)]
            
            for gen_list in generations_r:
                bleuvar, min_bleuvar, min_gen_idx, min_gen = analyze_generation_bleuvar(gen_list, n=n)
                generated_sums.append(min_gen)
                bleuvars.append(bleuvar)

        return generated_sums, target_sums, article_ids, bleuvars
    
    def generate_mc_summaries(
            self,
            dataloader,
            device,
            text_column,
            summary_column=None,
            max_source_length=128,
            num_beams=3,
            n=10):
        """
        Runs MC dropout generation given a DataLoader and a Bayesian model.
        
        Returns: N summaries generated with MC dropout for each input.
        """
        target_sums = []
        generated_sums = []
        article_ids = []
        bleuvars = []
        num_articles = 0
        for i, batch in enumerate(tqdm(dataloader)):
            generations, input_ids = self.run_mc_dropout(
                batch,
                device,
                text_column,
                max_source_length,
                num_beams,
                n)

            if summary_column is not None:
                target_sums += batch[summary_column]

            try:
                article_ids += batch["article_id"]
            except KeyError:
                article_ids += range(num_articles, num_articles + len(input_ids))

            num_articles += len(input_ids)
            generations_r = [list(x) for x in zip(*generations)]

            generated_sums += generations_r

        return generated_sums


def convert_bayesian_model(model, bayesian=True):
    """
    Convert the given Pegasus model to either a Bayesian
    or deterministic model.
    """
    if bayesian:
        model.eval()
        model.apply(apply_dropout)
    else:
        model.eval()
    return model


def apply_dropout(m):
    """
    Changes all Encoder and Decoder layers to training mode.
    This will essentially turn dropout and layer normalization
    on for MC dropout prediction.
    """
    if type(m) in [PegasusEncoderLayer, PegasusDecoderLayer, BartEncoderLayer, BartDecoderLayer]:
        m.train()
