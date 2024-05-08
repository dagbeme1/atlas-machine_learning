#!/usr/bin/env python3
"""
Define the Dataset class to load and prepare datasets for machine translation
"""
import tensorflow.compat.v2 as tf  # Import TensorFlow
import tensorflow_datasets as tfds  # Import TensorFlow Datasets


class Dataset:
    """
    Class for loading and preparing a dataset
    """

    def __init__(self):  # Constructor method
        """
        Initialize Dataset class.
        """
        examples, metadata = tfds.load(
            'ted_hrlr_translate/pt_to_en',  # Load Portuguese to English translation dataset
            with_info=True,  # Include metadata
            as_supervised=True  # Load as supervised dataset
        )
        # Assign training and validation datasets
        self.data_train, self.data_valid = examples['train'], examples['validation']

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)  # Tokenize training dataset

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