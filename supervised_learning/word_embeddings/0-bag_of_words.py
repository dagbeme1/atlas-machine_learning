#!/usr/bin/env python3
"""
Creates a bag of words embedding matrix
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
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names()
    embeddings = X.toarray()

    return embeddings, features
