#!/usr/bin/env python3
"""
Create the class Dataset that loads and preps a dataset for machine translation
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

tf.compat.v1.enable_eager_execution()

class Dataset:
    """
    Loads and preps a dataset
    """

    def __init__(self, batch_size, max_len):
        """
        Class constructor.

        Args:
            batch_size (int): The batch size for training.
            max_len (int): The maximum sequence length.
        """
        # Load dataset
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)

        # Split dataset into training and validation sets
        self.data_train, self.data_valid = examples['train'], examples['validation']

        # Tokenize training dataset
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

        # Map tf_encode method to training and validation sets
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        # Filter and cache training dataset
        self.data_train = self.data_train.filter(lambda x, y: self.filter_max_length(x, y, max_len))
        self.data_train = self.data_train.cache()

        # Shuffle training dataset
        shu = metadata.splits['train'].num_examples
        self.data_train = self.data_train.shuffle(shu)

        # Pad and batch training dataset
        pad_shape = ([None], [None])
        self.data_train = self.data_train.padded_batch(batch_size, padded_shapes=pad_shape)

        # Prefetch training dataset
        self.data_train = self.data_train.prefetch(tf.data.experimental.AUTOTUNE)

        # Pad and batch validation dataset
        self.data_valid = self.data_valid.filter(lambda x, y: self.filter_max_length(x, y, max_len))
        self.data_valid = self.data_valid.padded_batch(batch_size, padded_shapes=pad_shape)

    def tokenize_dataset(self, data):  # Method to tokenize dataset
        """
        Creates sub-word tokenizers for the dataset.

        Args:
            data (tf.data.Dataset): A dataset containing tuples (pt, en).

        Returns:
            tfds.deprecated.text.SubwordTextEncoder: The tokenizer for Portuguese text.
            tfds.deprecated.text.SubwordTextEncoder: The tokenizer for English text.
        """
        # Extract Portuguese and English texts from the dataset
        pt_texts = [pt.numpy() for pt, en in data]
        en_texts = [en.numpy() for pt, en in data]

        # Create sub-word tokenizers for Portuguese and English texts
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            pt_texts, target_vocab_size=2 ** 15)
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            en_texts, target_vocab_size=2 ** 15)

        return tokenizer_pt, tokenizer_en  # Return tokenizers for both languages

    def encode(self, pt, en):
        """
        Encodes a translation into tokens.

        Args:
            pt: The tf.Tensor containing the Portuguese sentence.
            en: The tf.Tensor containing the corresponding English sentence.

        Returns:
            pt_tokens: A tf.Tensor containing the Portuguese tokens.
            en_tokens: A tf.Tensor containing the English tokens.
        """
        # Encode Portuguese sentence with start and end tokens
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]
        # Encode English sentence with start and end tokens
        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Acts as a TensorFlow wrapper for the encode instance method.

        Args:
            pt: The tf.Tensor containing the Portuguese sentence.
            en: The tf.Tensor containing the corresponding English sentence.

        Returns:
            result_pt: A tf.Tensor containing the encoded Portuguese sentence.
            result_en: A tf.Tensor containing the encoded English sentence.
        """
        # Use tf.py_function to apply the encode method to pt and en tensors
        result_pt, result_en = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])
        # Set the shapes of the resulting tensors
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en

    def filter_max_length(self, x, y, max_length):
      """
      Filters function based on max_length.

      Args:
          x (tf.Tensor): The input tensor for the source language.
          y (tf.Tensor): The input tensor for the target language.
          max_length (int): The maximum length allowed for sequences.

      Returns:
          bool: A boolean value indicating whether both source and target sequences are within the maximum length.
      """
      return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)
