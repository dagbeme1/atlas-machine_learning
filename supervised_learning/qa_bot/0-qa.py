#!/usr/bin/env python3
"""
 a function def question_answer(question, reference):
  that finds a snippet of text within a reference 
  document to answer a question
"""

import tensorflow as tf  # Import TensorFlow library
import tensorflow_hub as hub  # Import TensorFlow Hub library
from transformers import BertTokenizer  # Import BertTokenizer from transformers library

def question_answer(question, reference):
    """
    Finds a snippet of text within a document to answer a question

    Args:
        question (str): A string containing the question to answer.
        reference (str): A string containing the reference document from which to find the answer.

    Returns:
        str: A string containing the answer.
    """
    toks = BertTokenizer.from_pretrained(  # Initialize BERT tokenizer
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')  # Load BERT model from TensorFlow Hub

    question_toks = toks.tokenize(question)  # Tokenize the question
    parag_toks = toks.tokenize(reference)  # Tokenize the reference document
    tokens = ['[CLS]'] + question_toks + ['[SEP]'] + parag_toks + ['[SEP]']  # Combine tokens
    input_ids = toks.convert_tokens_to_ids(tokens)  # Convert tokens to word IDs
    input_mask = [1] * len(input_ids)  # Generate input mask
    type_word_ids = [0] * (1 + len(question_toks) + 1) + [1] * (len(parag_toks) + 1)  # Generate type IDs

    # Convert variables to TensorFlow tensors
    input_ids, input_mask, type_word_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_ids, input_mask, type_word_ids))
    
    outputs = model([input_ids, input_mask, type_word_ids])  # Get model outputs
    get_start = tf.argmax(outputs[0][0][1:]) + 1  # Get the start index of the answer
    get_end = tf.argmax(outputs[1][0][1:]) + 1  # Get the end index of the answer
    answer_toks = tokens[get_start: get_end + 1]  # Extract answer tokens
    answer = toks.convert_tokens_to_string(answer_toks)  # Convert answer tokens to string
    return answer  # Return the answer
