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
    # Initialize an empty list to store n-grams
    grams_list = []

    # Initialize a counter
    i = 0
    # Loop through the sentence
    while i < len(sentence):
        # Define the end index of the n-gram
        last = i + n
        # Define the start index of the n-gram
        start = i

        # Check if the end index exceeds the length of the sentence
        if last >= len(sentence) + 1:
            break

        # Extract the n-gram
        n_gram = sentence[start: last]
        # Join the n-gram to form a string
        result = ' '.join(n_gram)
        # Append the n-gram to the list
        grams_list.append(result)

        # Increment the counter
        i += 1

    return grams_list


def ngram_bleu(references, sentence, n):
    """
    Calculate the BLEU score for a sentence.

    Args:
        references (list): A list of reference translations.
            Each reference translation is a list of the
            words in the translation.
        sentence (list): A list containing the proposed sentence.
        n (int): The size of the n-gram to use for evaluation.

    Returns:
        float: The BLEU score.
    """
    # Generate the set of n-grams in the sentence
    grams = list(set(n_grams(sentence, n)))
    # Compute the length of the n-grams set
    len_g = len(grams)
    # Initialize a list to store reference n-grams
    reference_grams = []
    # Initialize a dictionary to store word counts
    words_dict = {}

    # Initialize a counter
    j = 0
    # Loop through each reference translation
    while j < len(references):
        # Get the j-th reference translation
        ref = references[j]
        # Generate n-grams for the reference translation
        ref_n_grams = n_grams(ref, n)
        # Append the reference n-grams to the list
        reference_grams.append(ref_n_grams)
        # Increment the counter
        j += 1

    # Initialize a counter
    k = 0
    # Loop through each reference n-gram list
    while k < len(reference_grams):
        # Get the k-th reference n-grams
        ref_grams = reference_grams[k]
        # Initialize a counter
        word_index = 0
        # Loop through each word in the reference n-grams
        while word_index < len(ref_grams):
            # Get the l-th word
            word = ref_grams[word_index]
            # Check if the word is in the set of n-grams
            if word in grams:
                # Update word count in the dictionary
                if word not in words_dict.keys():
                    words_dict[word] = ref_grams.count(word)
                else:
                    actual = ref_grams.count(word)
                    prev = words_dict[word]
                    words_dict[word] = max(actual, prev)
            # Increment the counter
            word_index += 1
        # Increment the counter
        k += 1

    # Calculate the length of the candidate sentence
    candidate_length = len(sentence)
    # Calculate the probability
    prob = sum(words_dict.values()) / len_g

    # Initialize a list to store best match lengths
    best_match = []

    # Initialize a counter
    m = 0
    # Loop through each reference translation
    while m < len(references):
        # Get the m-th reference translation
        ref = references[m]
        # Calculate the length of the reference translation
        ref_length = len(ref)
        # Calculate the difference in lengths
        diff = abs(ref_length - candidate_length)
        # Append the tuple of (difference, length) to the list
        best_match.append((diff, ref_length))
        # Increment the counter
        m += 1

    # Sort the list of tuples based on the difference
    sort_tuple = sorted(best_match, key=(lambda x: x[0]))
    # Get the best match length
    best_length = sort_tuple[0][1]

    # Check if candidate length is greater than best match length
    if candidate_length > best_length:
        bleu = 1
    else:
        bleu = np.exp(1 - (best_length / candidate_length))

    # Calculate the BLEU score
    score = bleu * np.exp(np.log(prob))
    return score
