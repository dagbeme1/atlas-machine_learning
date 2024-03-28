#!/usr/bin/env python3

import numpy as np
import string
from collections import Counter


class CountVectorizer:
    def __init__(self, vocabulary=None):
        """
        Initialize the CountVectorizer.

        Args:
            vocabulary (list, optional): A list of vocabulary words.
        """
        self.vocabulary_ = vocabulary

    def fit_transform(self, sentences):
        """
        Fit the CountVectorizer to the given sentences and
        transform them into embeddings.

        Args:
            sentences (list): A list of sentences to analyze.

        Returns:
            numpy.ndarray: A numpy array of shape (s, f) containing
            the embeddings,
                where s is the number of sentences in sentences and
                f is the number of features analyzed.
        """
        if self.vocabulary_ is None:
            # If no vocabulary is provided, create it from sentences
            words = [word.strip(string.punctuation).lower()
                     for sentence in sentences for word in sentence.split()]
            # Exclude "children's" from the vocabulary
            self.vocabulary_ = sorted(set(words) - {"children's"})

        # Transform sentences into a matrix of token counts
        embeddings = []
        for sentence in sentences:
            # Initialize word_counts for each sentence
            word_counts = Counter()

            words = [word.strip(string.punctuation).lower()
                     for word in sentence.split()]
            word_counts.update(words)

            # Generate embedding for the current sentence
            embedding = [
                word_counts[word] if word in word_counts else 0 for word in
                self.vocabulary_]
            embeddings.append(embedding)

        embeddings = np.array(embeddings)

        return embeddings

    def get_feature_names(self):
        """
        Get the feature names used for embeddings.

        Returns:
            list: A list of the features used for embeddings.
        """
        return self.vocabulary_


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences (list): A list of sentences to analyze.
        vocab (list, optional): A list of the vocabulary words
            to use for the analysis.

    Returns:
        tuple: A tuple containing embeddings and features
            embeddings (numpy.ndarray): A numpy array of shape
            (s, f) containing the embeddings,
            where s is the number of sentences in sentences and
            f is the number of features analyzed.
        features (list): A list of the features used for embeddings.
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)

    embeddings = np.array(X)  # Convert list of lists to numpy array
    features = vectorizer.get_feature_names()

    return embeddings, features
