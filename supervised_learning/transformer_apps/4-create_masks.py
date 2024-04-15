#!/usr/bin/env python3
"""
Defines all masks for training/validation
"""
import tensorflow.compat.v2 as tf

def create_masks(inputs, target):
    """
    Creates all masks for training/validation

    Args:
        inputs (tf.Tensor): A tensor of shape (batch_size, seq_len_in) containing the input sentence.
        target (tf.Tensor): A tensor of shape (batch_size, seq_len_out) containing the target sentence.

    Returns:
        encoder_mask (tf.Tensor): The padding mask for the encoder, shape (batch_size, 1, 1, seq_len_in).
        look_ahead_mask (tf.Tensor): The look-ahead mask for the decoder, shape (batch_size, 1, seq_len_out, seq_len_out).
        decoder_mask (tf.Tensor): The padding mask for the decoder, shape (batch_size, 1, 1, seq_len_in).
    """
    # Create padding mask for encoder
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    # Create padding mask for decoder
    decoder_mask = tf.cast(tf.math.equal(target, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    # Create look-ahead mask for decoder
    size = target.shape[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    # Create combined mask for decoder by taking maximum of decoder padding mask and look-ahead mask
    combined_mask = tf.maximum(decoder_mask, look_ahead_mask)

    return encoder_mask, look_ahead_mask, decoder_mask
