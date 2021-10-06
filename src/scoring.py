import pandas as pd

from datasets import load_metric


def score_generations(df):
    """Score generations using the python rouge library"""
    rouge = load_metric("rouge")
    
    df["rouge"] = df[["gen_sum", "target_sum"]].apply(
        lambda x: rouge.compute(predictions=[x[0]], references=[x[1]]), axis=1)
    df["rouge"] = df["rouge"].apply(lambda x: {k: round(v.mid.fmeasure * 100, 4) for k, v in x.items()})
    df = pd.concat([df.drop(['rouge'], axis=1), df['rouge'].apply(pd.Series)], axis=1)
    metrics = df[["rouge1", "rouge2", "rougeLsum"]].agg(['mean', 'std'])
    
    return metrics, df


def score_standard(
        gen_sums,
        target_sums,
        article_ids):
    """Score standard summaries"""
    assert len(gen_sums) == len(target_sums) == len(article_ids), f"Input dims must be the equal but got {len(gen_sums)}, {len(target_sums)}, {len(article_ids)}"
    df = pd.DataFrame(
        list(zip(article_ids, target_sums, gen_sums)),
        columns=["article_id", "target_sum", "gen_sum"])

    metrics, mdf = score_generations(df)

    return metrics, mdf
