from tqdm import tqdm


def generate_summaries(test_loader, model, tokenizer, device, args):
    """Run summary generation for a given DataLoader"""
    gen_sums = []
    target_sums = []
    article_ids = []
    num_articles = 0
    for i, batch in enumerate(tqdm(test_loader)):
        model_inputs = tokenizer(
            batch[args.text_column],
            max_length=args.max_source_length,
            truncation=True,
            padding=True,
            return_tensors='pt')

        input_ids = model_inputs['input_ids'].to(device)
        sent_outputs = model.generate(
            input_ids,
            num_beams=args.num_beams,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True)  # only one beam should be equivalent to greedy,
        gen_sum = [
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in sent_outputs["sequences"]]

        gen_sums += gen_sum
        target_sums += batch[args.summary_column]
        
        try:
            article_ids += batch["article_id"]
        except KeyError:
            article_ids += range(num_articles, num_articles+len(input_ids))
            
        num_articles += len(input_ids)

    return gen_sums, target_sums, article_ids
