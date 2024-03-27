#!/usr/bin/env python3
"""
Enhanced Bag of Words Embedding
"""

import numpy as np

def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix

    Args:
        sentences (list): A list of sentences to analyze
        vocab (list, optional): A list of the vocabulary words
            to use for the analysis. If None, all unique words
            in sentences will be used.

    Returns:
        tuple: A tuple containing embeddings and features
            embeddings (numpy.ndarray): A numpy array of shape
                (s, f) containing the embeddings
                    s is the number of sentences in sentences
                    f is the number of features analyzed
            features (list): A list of the features used for embeddings
    """
    if vocab is None:
        vocab = set()
        for sentence in sentences:
            words = sentence.split()
            vocab.update(words)

    vocab = sorted(vocab)
    embeddings = []
    for sentence in sentences:
        embedding = [sentence.split().count(word) for word in vocab]
        embeddings.append(embedding)

    embeddings = np.array(embeddings)
    return embeddings, vocab
