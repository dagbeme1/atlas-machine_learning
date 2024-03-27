#!/usr/bin/env python3
"""
Enhanced Bag of Words Embedding
"""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix

    Args:
        sentences (list): A list of sentences to analyze
        vocab (list, optional): A list of the vocabulary words
        to use for the analysis

    Returns:
        tuple: A tuple containing embeddings and features
            embeddings (numpy.ndarray): A numpy array of shape
            (s, f) containing the embeddings
                s is the number of sentences in sentences
                f is the number of features analyzed
            features (list): A list of the features used for embeddings
    """
    # This initializes CountVectorizer with optional vocabulary
    vector = CountVectorizer(vocabulary=vocab)

    # fits and transforms the sentences to create the embedding matrix
    X = vector.fit_transform(sentences)

    # retrieves the feature names
    features = vector.get_feature_names()

    # Converts the sparse matrix to a dense array
    embeddings = X.toarray()

    return embeddings, features
