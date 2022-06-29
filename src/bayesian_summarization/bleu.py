import numpy as np

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import sent_tokenize, word_tokenize


def smoothing_function(p_n, references, hypothesis, hyp_len):
    """
    Smooth-BLEU (BLEUS) as proposed in the paper:
    Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
    evaluation metrics for machine translation. COLING 2004.
    """
    smoothed_p_n = []
    for i, p_i in enumerate(p_n, start=1):
        # Smoothing is not applied for unigrams
        if i > 1:
            # If hypothesis length is lower than the current order, its value equals (0 + 1) / (0 + 1) = 0 
            if hyp_len < i:
                assert p_i.denominator == 1
                smoothed_p_n.append(1)
            # Otherwise apply smoothing
            else:
                smoothed_p_i = (p_i.numerator + 1) / (p_i.denominator + 1)
                smoothed_p_n.append(smoothed_p_i)
        else:
            smoothed_p_n.append(p_i)
    return smoothed_p_n


def pair_bleu(text1, text2):
    """
    Compute the bleu score between two given texts.
    A smoothing function is used to avoid zero scores when
    there are no common higher order n-grams between the
    texts.
    """
    tok1 = [word_tokenize(s) for s in sent_tokenize(text1)]
    tok2 = [word_tokenize(s) for s in sent_tokenize(text2)]
    score = 0
    for c_cent in tok2:
        try:
            score += corpus_bleu([tok1], [c_cent], smoothing_function=smoothing_function)
        except KeyError:
            score = 0.
    try:
        score /= len(tok2)
    except ZeroDivisionError:
        score = 0.

    return score


def analyze_generation_bleuvar(gen_list, n=10):
    """
    Given a list of generated texts, computes the pairwise BLEUvar
    between all text pairs. In addition, also finds the generation
    that has the smallest avg. BLEUvar score (most similar)
    with all other generations.
    """
    bleu_scores = np.zeros((n, n), dtype=float)
    bleu_var = 0.
    min_gen_idx = None
    min_bleuvar = float('inf')
    for j, dec_j in enumerate(gen_list):
        for k in range(j + 1, n):
            dec_k = gen_list[k]
            jk_bleu = pair_bleu(dec_j, dec_k)
            kj_bleu = pair_bleu(dec_k, dec_j)

            bleu_var += (1 - jk_bleu) ** 2
            bleu_var += (1 - kj_bleu) ** 2
            bleu_scores[j, k] = 1 - jk_bleu
            bleu_scores[k, j] = 1 - kj_bleu

        mu_bleuvar = np.sum(bleu_scores[j, :]) + np.sum(bleu_scores[:, j])
        if mu_bleuvar < min_bleuvar:
            min_bleuvar = mu_bleuvar
            min_gen_idx = j

    bleu_var /= n * (n - 1)
    try:
        min_gen = gen_list[min_gen_idx]
    except TypeError:
        min_gen = ""

    return bleu_var, min_bleuvar, min_gen_idx, min_gen
