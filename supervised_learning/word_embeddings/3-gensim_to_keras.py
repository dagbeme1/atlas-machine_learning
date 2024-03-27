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
    # Get the vocabulary size and embedding dimension from the Gensim model
    vocab_size, embedding_dim = model.wv.vectors.shape

    # Create a Keras Embedding layer with weights from the Gensim model
    keras_embedding_layer = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[model.wv.vectors],
        trainable=False  # Prevent the embedding weights from being updated during training
    )

    return keras_embedding_layer
