import re

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
    df = pd.DataFrame(
        list(zip(article_ids, target_sums, gen_sums)),
        columns=["article_id", "target_sum", "gen_sum"])

    metrics, mdf = score_generations(df)

    return metrics, mdf


def process_write(text):
    """Process text for writing to file"""
    proc_text = re.sub("\n", " ", text)
    proc_text = re.sub("<n>", "", proc_text)
    return proc_text


def write_gen(df, out_path):
    """NOT FINISHED"""
    raise NotImplementedError("Writing summaries for the original ROUGE scoring is no supported")
    # out_path = os.path.join(out_path, "generations")
    # hyp_path = os.path.join(out_path, "hyp")
    # ref_path = os.path.join(out_path, "ref")
    # if os.path.exists(out_path):
    #     shutil.rmtree(out_path)
    # os.mkdir(out_path)
    # os.mkdir(hyp_path)
    # os.mkdir(ref_path)
    #
    # for row in tqdm(df.iterrows()):
    #     aid, ref, hyp = row[1]["article_id"], row[1]["target_sum"], row[1]["gen_sum"]
    #     with open(os.path.join(hyp_path, f"hyp_{aid}.txt"), 'w') as hf, open(os.path.join(ref_path, f"ref_{aid}.txt"), 'w') as rf:
    #         hf.write(process_write(hyp))
    #         rf.write(process_write(ref))
