#!/usr/bin/env python3
"""
Enhanced Bag of Words Embedding
"""
import numpy as np
import string
from collections import Counter


class CountVectorizer:
    def __init__(self, vocabulary=None):
        self.vocabulary_ = vocabulary

    def fit_transform(self, sentences):
        if self.vocabulary_ is None:
            # If no vocabulary is provided, create it from sentences
            words = [word.strip(string.punctuation).lower()
                     for sentence in sentences for word in sentence.split()]
            self.vocabulary_ = sorted(set(words))

        # Count occurrences of words across all sentences using Counter
        word_counts = Counter()
        for sentence in sentences:
            words = [word.strip(string.punctuation).lower()
                     for word in sentence.split()]
            word_counts.update(words)

        # Transform sentences into a matrix of token counts
        embeddings = []
        for sentence in sentences:
            words = [word.strip(string.punctuation).lower()
                     for word in sentence.split()]
            embedding = [word_counts[word] for word in self.vocabulary_]
            embeddings.append(embedding)

        embeddings = np.array(embeddings)

        return embeddings

    def get_feature_names(self):
        return self.vocabulary_


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

    embeddings = np.array(X)  # this converts list to numpy array
    features = vectorizer.get_feature_names()

    return embeddings, features


def print_matrix(matrix):
    for row in matrix:
        print(" ".join(str(cell) for cell in row))
