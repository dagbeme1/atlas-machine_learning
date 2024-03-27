#!/usr/bin/env python3
"""
3-gensim_to_keras.py
"""
from gensim.models import Word2Vec
import tensorflow.keras as keras


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer

    Args:
        model: A trained Gensim Word2Vec model

    Returns:
        keras.layers.Embedding: A Keras Embedding layer initialized with weights from the Word2Vec model
    """
    # put it on false to Prevent the embedding weights
    # from being updated during training
    return model.wv.get_keras_embedding(train_embeddings=False)
