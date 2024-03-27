#!/usr/bin/env python3

import sklearn.feature_extraction.text

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
    if vocab is None:
        all_words = ' '.join(sentences).split()
        vocab = list(set(all_words))

    # Initialize CountVectorizer with custom vocabulary and lowercase=False
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(vocabulary=vocab, lowercase=False)

    # Fit and transform the sentences
    X = vectorizer.fit_transform(sentences)

    # Get the embeddings matrix
    embeddings = X.toarray()

    # Get the feature names from the vocabulary
    features = vectorizer.get_feature_names_out()

    return embeddings, features
