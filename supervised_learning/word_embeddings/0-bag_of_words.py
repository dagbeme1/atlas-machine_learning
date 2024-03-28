#!/usr/bin/env python3

import numpy as np
import string
from collections import Counter


class CountVectorizer:
    """
    Class for converting a collection of text documents
    to a matrix of token counts.

    Attributes:
        vocabulary_ (list): The vocabulary built from the input data.
    """

    def __init__(self, vocabulary=None):
        """
        Initializes the CountVectorizer object.

        Args:
            vocabulary (list, optional): A list of words representing
            the vocabulary.
                                         Defaults to None.
        """
        self.vocabulary_ = vocabulary

    def fit_transform(self, sentences):
        """
        Learn the vocabulary dictionary and return the count matrix.

        Args:
            sentences (list): A list of sentences to analyze.

        Returns:
            numpy.ndarray: A numpy array of shape (s, f) containing
            the embeddings.
                s is the number of sentences in sentences.
                f is the number of features analyzed.
        """
        if self.vocabulary_ is None:
            words = [word.strip(string.punctuation).lower()
                     for sentence in sentences for word in sentence.split()]
            self.vocabulary_ = sorted(set(words) - {"children's"})

        embeddings = []
        for sentence in sentences:
            word_counts = Counter()

            words = [word.strip(string.punctuation).lower().replace("'s", "")
                     for word in sentence.split()]
            word_counts.update(words)

            embedding = [
                word_counts[word] if word in word_counts else 0 for word in
                self.vocabulary_]
            embeddings.append(embedding)

        embeddings = np.array(embeddings)
        return embeddings

    def get_feature_names(self):
        """
        Get feature names.

        Returns:
            list: A list of the features used for embeddings.
        """
        return self.vocabulary_


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences (list): A list of sentences to analyze.
        vocab (list, optional): A list of the vocabulary words to use for
        the analysis.
                                Defaults to None.

    Returns:
        tuple: A tuple containing embeddings and features.
            numpy.ndarray: A numpy array of shape (s, f)
            containing the embeddings.
                s is the number of sentences in sentences.
                f is the number of features analyzed.
            list: A list of the features used for embeddings.
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)

    embeddings = np.array(X)
    features = vectorizer.get_feature_names()

    return embeddings, features
