from tqdm import tqdm


def generate_summaries_batch(
        batch,
        model,
        tokenizer,
        device,
        text_column,
        num_articles,
        max_source_length=128,
        num_beams=3,
):
    """Run summary generation for a batch of data"""
    model_inputs = tokenizer(
        batch[text_column],
        max_length=max_source_length,
        truncation=True,
        padding=True,
        return_tensors='pt')

    input_ids = model_inputs['input_ids'].to(device)
    sent_outputs = model.generate(
        input_ids,
        num_beams=num_beams,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True)  # only one beam should be equivalent to greedy,
    gen_sum = [
        tokenizer.decode(
            g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in sent_outputs["sequences"]]

    try:
        gen_ids = batch["article_id"]
    except KeyError:
        gen_ids = range(num_articles, num_articles + len(input_ids))

    return gen_sum, gen_ids


def generate_summaries(
        dataloader,
        model,
        tokenizer,
        device,
        text_column,
        summary_column=None,
        max_source_length=128,
        num_beams=3
):
    """Run summary generation for a given DataLoader"""
    target_sums = []
    generated_sums = []
    article_ids = []
    num_articles = 0
    for i, batch in enumerate(tqdm(dataloader)):
        generations_r, gen_ids = generate_summaries_batch(
            batch=batch,
            model=model,
            tokenizer=tokenizer,
            device=device,
            text_column=text_column,
            max_source_length=max_source_length,
            num_beams=num_beams,
            num_articles=num_articles)

        if summary_column is not None:
            target_sums += batch[summary_column]

        generated_sums += generations_r
        article_ids += gen_ids
        num_articles += len(gen_ids)

    return generated_sums, target_sums, article_ids
