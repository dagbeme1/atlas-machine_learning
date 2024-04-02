#!/usr/bin/env python3
"""
 the function def cumulative_bleu(references, sentence, n):
 that calculates the cumulative n-gram BLEU score for a sentence
"""

import numpy as np


def n_grams(sentence, n):
    """
    Creates the n-grams from a sentence.

    Args:
        sentence (list): A list containing the proposed sentence.
        n (int): The size of the n-gram to use for evaluation.

    Returns:
        list: The n-grams.
    """
    list_grams_cand = []
    i = 0
    while i < len(sentence):
        last = i + n
        begin = i
        if last >= len(sentence) + 1:
            break
        aux = sentence[begin: last]
        result = ' '.join(aux)
        list_grams_cand.append(result)
        i += 1
    return list_grams_cand


def ngram_bleu(references, sentence, n):
    """
    Calculates the unigram BLEU score for a sentence.

    Args:
        references (list): A list of reference translations.
            Each reference translation is a list of words in the translation.
        sentence (list): A list containing the proposed sentence.
        n (int): The size of the n-gram to use for evaluation.

    Returns:
        float: Unigram BLEU score.
    """
    grams = n_grams(sentence, n)
    reference_grams = []
    words_dict = {}

    i = 0
    while i < len(references):
        reference = references[i]
        list_grams = n_grams(reference, n)
        reference_grams.extend(list_grams)
        i += 1

    i = 0
    while i < len(grams):
        word = grams[i]
        if word in reference_grams:
            if word not in words_dict:
                words_dict[word] = 1
            else:
                words_dict[word] += 1
        i += 1

    max_precision = 0.0
    if len(grams) > 0:
        i = 0
        while i < len(words_dict):
            max_precision = max(max_precision,
                                words_dict[grams[i]] / len(grams))
            i += 1

    return max_precision


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.

    Args:
        references (list): A list of reference translations.
            Each reference translation is a list of words in the translation.
        sentence (list): A list containing the proposed sentence.
        n (int): The size of the largest n-gram to use for evaluation.

    Returns:
        float: The cumulative n-gram BLEU score.
    """
    prob = []
    i = 1
    while i <= n:
        result = ngram_bleu(references, sentence, i)
        prob.append(result)
        i += 1

    best_match_tuples = []
    i = 0
    while i < len(references):
        reference = references[i]
        ref_len = len(reference)
        diff = abs(ref_len - len(sentence))
        best_match_tuples.append((diff, ref_len))
        i += 1

    sort_tuples = sorted(best_match_tuples, key=lambda x: x[0])
    best_match = sort_tuples[0][1]

    if len(sentence) > best_match:
        bp = 1
    else:
        bp = np.exp(1 - (best_match / len(sentence)))

    score = bp * np.exp(np.sum(np.log(prob)) / n)
    return score
