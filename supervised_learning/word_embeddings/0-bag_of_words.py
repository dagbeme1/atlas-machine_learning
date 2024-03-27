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
    # Create vocabulary from sentences if vocab is None
    if vocab is None:
        vocab = set()
        for sentence in sentences:
            words = sentence.split()
            vocab.update(words)

    # Map each word to an index
    word_to_index = {word: i for i, word in enumerate(sorted(vocab))}

    # Initialize embeddings matrix
    embeddings = []
    for sentence in sentences:
        # Initialize feature vector for the sentence
        features = [0] * len(vocab)
        for word in sentence.split():
            if word in word_to_index:
                # Increment the count of the word in the feature vector
                features[word_to_index[word]] += 1
        embeddings.append(features)

    # Convert embeddings to numpy array
    embeddings = np.array(embeddings)

    return embeddings, sorted(vocab)
