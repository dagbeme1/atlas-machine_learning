#!/usr/bin/env python3
"""
 the function def cumulative_bleu(references, sentence, n):
 that calculates the cumulative n-gram BLEU score for a sentence
"""

import numpy as np


def n_grams(sentence, n):
    """
    Generates the n-grams from a sentence.

    Args:
        sentence (list): A list containing the proposed sentence.
        n (int): The size of the n-gram to use for evaluation.

    Returns:
        list: The n-grams.
    """
    list_grams_cand = []
    i = 0
    # Iterate through the sentence to generate n-grams
    while i < len(sentence):
        last = i + n
        begin = i
        # Check if the last index goes beyond the sentence length
        if last >= len(sentence) + 1:
            break
        aux = sentence[begin: last]
        result = ' '.join(aux)
        list_grams_cand.append(result)
        i += 1
    return list_grams_cand


def ngram_bleu(references, sentence, n):
    """
    Computes the unigram BLEU score for a sentence.

    Args:
        references (list): A list of reference translations.
            Each reference translation is a list of words in the translation.
        sentence (list): A list containing the proposed sentence.
        n (int): The size of the n-gram to use for evaluation.

    Returns:
        float: Unigram BLEU score.
    """
    grams = list(set(n_grams(sentence, n)))
    len_g = len(grams)
    reference_grams = []
    words_dict = {}

    i = 0
    # Iterate through reference translations to compute n-grams
    while i < len(references):
        reference = references[i]
        list_grams = n_grams(reference, n)
        reference_grams.append(list_grams)
        i += 1

    i = 0
    # Count the occurrences of n-grams in references
    while i < len(reference_grams):
        ref = reference_grams[i]
        j = 0
        while j < len(ref):
            word = ref[j]
            if word in grams:
                if word not in words_dict.keys():
                    words_dict[word] = ref.count(word)
                else:
                    actual = ref.count(word)
                    prev = words_dict[word]
                    words_dict[word] = max(actual, prev)
            j += 1
        i += 1

    # Calculate the unigram BLEU score
    prob = sum(words_dict.values()) / len_g
    return prob


def cumulative_bleu(references, sentence, n):
    """
    Computes the cumulative n-gram BLEU score for a sentence.

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
    # Iterate through n-grams to calculate unigram BLEU scores
    while i <= n:
        result = ngram_bleu(references, sentence, i)
        prob.append(result)
        i += 1

    best_match_tuples = []
    i = 0
    # Find the best match among reference translations
    while i < len(references):
        reference = references[i]
        ref_len = len(reference)
        diff = abs(ref_len - len(sentence))
        best_match_tuples.append((diff, ref_len))
        i += 1

    # Sort the tuples based on the difference in length
    sort_tuples = sorted(best_match_tuples, key=lambda x: x[0])
    best_match = sort_tuples[0][1]

    # Brevity penalty calculation
    if len(sentence) > best_match:
        bp = 1
    else:
        bp = np.exp(1 - (best_match / len(sentence)))

    # Calculate the cumulative n-gram BLEU score
    score = bp * np.exp(np.sum(np.log(prob)) / n)
    return score
