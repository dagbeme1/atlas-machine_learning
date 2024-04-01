#!/usr/bin/env python3
"""
Determines the unigram BLEU score for a given sentence
"""
import numpy as np


def n_grams(sentence, n):
    """
    Generate n-grams from the given sentence.

    Args:
        sentence (list): A list containing the model proposed sentence.
        n (int): The size of the n-gram to use for evaluation.

    Returns:
        list: A list of n-grams.
    """
    grams_list = []

    i = 0
    while i < len(sentence):
        last = i + n
        start = i

        if last >= len(sentence) + 1:
            break

        n_gram = sentence[start: last]
        result = ' '.join(n_gram)
        grams_list.append(result)

        i += 1

    return grams_list


def ngram_bleu(references, sentence, n):
    """
    Calculate the BLEU score for a sentence.

    Args:
        references (list): A list of reference translations.
            Each reference translation is a list of the words in the translation.
        sentence (list): A list containing the proposed sentence.
        n (int): The size of the n-gram to use for evaluation.

    Returns:
        float: The BLEU score.
    """
    grams = list(set(n_grams(sentence, n)))
    len_g = len(grams)
    reference_grams = []
    words_dict = {}

    j = 0
    while j < len(references):
        ref = references[j]
        ref_n_grams = n_grams(ref, n)
        reference_grams.append(ref_n_grams)
        j += 1

    k = 0
    while k < len(reference_grams):
        ref_grams = reference_grams[k]
        l = 0
        while l < len(ref_grams):
            word = ref_grams[l]
            if word in grams:
                if word not in words_dict.keys():
                    words_dict[word] = ref_grams.count(word)
                else:
                    actual = ref_grams.count(word)
                    prev = words_dict[word]
                    words_dict[word] = max(actual, prev)
            l += 1
        k += 1

    candidate_length = len(sentence)
    prob = sum(words_dict.values()) / len_g

    best_match = []

    m = 0
    while m < len(references):
        ref = references[m]
        ref_length = len(ref)
        diff = abs(ref_length - candidate_length)
        best_match.append((diff, ref_length))
        m += 1

    sort_tuple = sorted(best_match, key=(lambda x: x[0]))
    best_length = sort_tuple[0][1]

    if candidate_length > best_length:
        bleu = 1
    else:
        bleu = np.exp(1 - (best_length / candidate_length))

    score = bleu * np.exp(np.log(prob))
    return score
