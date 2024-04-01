#!/usr/bin/env python3
"""
the function def uni_bleu(references, sentence):
that calculates the unigram BLEU score for a sentence
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculate the unigram BLEU score for a sentence.

    Args:
        references (list): A list of reference translations.
            Each reference translation is a list of the
            words in the translation.
        sentence (list): A list containing the model proposed sentence.

    Returns:
        float: The unigram BLEU score.
    """
    # Get unique words in the candidate sentence
    unique = list(set(sentence))
    # Initialize dictionary to store maximum counts of each word
    words_dict = {}
    # Initialize index for looping through references
    ref_index = 0

    # Loop through each reference translation
    while ref_index < len(references):
        reference = references[ref_index]
        # Initialize index for looping through words in reference
        word_index = 0

        # Loop through each word in the reference
        while word_index < len(reference):
            word = reference[word_index]
            # Check if the word is in the candidate sentence
            if word in unique:
                # Update word count in dictionary
                if word not in words_dict.keys():
                    words_dict[word] = reference.count(word)
                else:
                    actual = reference.count(word)
                    prev = words_dict[word]
                    words_dict[word] = max(actual, prev)
            word_index += 1

        ref_index += 1

    # Calculate total number of words in candidate sentence
    candidate = len(sentence)
    # Calculate precision probability
    prob = sum(words_dict.values()) / candidate

    # Initialize list to store differences in length between references and
    # candidate sentence
    best_match = []
    ref_index = 0

    # Loop through each reference translation again
    while ref_index < len(references):
        reference = references[ref_index]
        # Calculate difference in length between reference and candidate
        # sentence
        ref_len = len(reference)
        diff = abs(ref_len - candidate)
        # Append difference and reference length to list
        best_match.append((diff, ref_len))
        ref_index += 1

    # Sort the list based on difference in length
    sort_tuple = sorted(best_match, key=(lambda x: x[0]))
    # Get the length of the closest reference
    best = sort_tuple[0][1]

    # Calculate BLEU score
    if candidate > best:
        bleu = 1
    else:
        bleu = np.exp(1 - (best / candidate))

    # Compute final BLEU score
    score = bleu * np.exp(np.log(prob))
    return score
