from tqdm import tqdm

from transformers.models.pegasus.modeling_pegasus import PegasusEncoderLayer, PegasusDecoderLayer
from transformers.models.bart.modeling_bart import BartEncoderLayer, BartDecoderLayer

from src.bleu import analyze_generation_bleuvar


def apply_dropout(m):
    """
    Changes all Encoder and Decoder layers to training mode.
    This will essentially turn dropout and layer normalization
    on for MC dropout prediction.
    """
    if type(m) in [PegasusEncoderLayer, PegasusDecoderLayer, BartEncoderLayer, BartDecoderLayer]:
        m.train()


def bayesian_conversion(model, bayesian=True):
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


def bayesian_generate_summaries(test_loader, model, tokenizer, device, args):
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
        model_inputs = tokenizer(
            batch[args.text_column],
            max_length=args.max_source_length,
            truncation=True,
            padding=True,
            return_tensors='pt')

        input_ids = model_inputs['input_ids'].to(device)
        generations = []
        for i_s in range(args.mc_samples):
            sent_outputs = model.generate(
                input_ids,
                num_beams=args.num_beams,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True)  # only one beam should be equivalent to greedy,
            gen_sum = [
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in sent_outputs["sequences"]]

            generations.append(gen_sum)

        target_sums += batch[args.summary_column]
        try:
            article_ids += batch["article_id"]
        except KeyError:
            article_ids += range(num_articles, num_articles+len(input_ids))
            
        num_articles += len(input_ids)
        generations_r = [list(x) for x in zip(*generations)]

        for gen_list in generations_r:
            bleuvar, min_bleuvar, min_gen_idx, min_gen = analyze_generation_bleuvar(gen_list)
            generated_sums.append(min_gen)
            bleuvars.append(bleuvar)

    return generated_sums, target_sums, article_ids, bleuvars
