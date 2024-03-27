#!/usr/bin/env python3
"""
TF-IDF Embedding Generation
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Generates a TF-IDF embedding for a given corpus

    Args:
        sentences (list): A list of sentences to analyze
        vocab (list, optional): A list of the vocabulary words to
        use for the analysis
            If None, all words within sentences should be used

    Returns:
        tuple: A tuple containing embeddings and features
            embeddings (numpy.ndarray): A numpy array of shape
            (s, f) containing the embeddings
                s is the number of sentences in sentences
                f is the number of features analyzed
            features (list): A list of the features used for embeddings
    """

    # This initializes TfidfVectorizer with optional vocabulary
    vector_tfidf = TfidfVectorizer(vocabulary=vocab)

    # This also fits and transforms the sentences to create embedding matrix
    X = vector_tfidf.fit_transform(sentences)

    # Also retrieves the feature names
    features = vector_tfidf.get_feature_names()

    # this also converts sparse matrix to a dense array
    embeddings = X.toarray()

    return embeddings, features
