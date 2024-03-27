#!/usr/bin/env python3
"""
Creates and trains a gensim FastText model
"""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates and trains a gensim FastText model

    Args:
        sentences (list): A list of sentences to be trained on
        size (int): The dimensionality of the embedding layer
        min_count (int): The minimum number of occurrences of a word for use in training
        window (int): The maximum distance between the current and predicted word within a sentence
        negative (int): The size of negative sampling
        cbow (bool): Boolean to determine the training type; True is for CBOW, False is for Skip-gram
        iterations (int): The number of iterations over the corpus
        seed (int): The seed for the random number generator
        workers (int): The number of worker threads to train the model

    Returns:
        FastText: The trained FastText model
    """
    sg = 0 if cbow else 1

    model = FastText(window=window, min_count=min_count, workers=workers,
                     sg=sg, negative=negative, seed=seed)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=iterations)

    return model
