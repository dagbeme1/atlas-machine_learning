#!/usr/bin/env python3
"""
a function def question_answer(coprus_path): that answers questions from multiple reference texts
"""
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from transformers import BertTokenizer

module_online = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
semantic_model = hub.load(module_online)
toks = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad'
)
model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')


def question0_answer(question, reference):
    """
    Finds a snippet of text within a document to answer a question

    Args:
        question (str): A string containing the question to answer.
        reference (str): A string containing the reference document from which to find the answer.

    Returns:
        str: A string containing the answer.
    """

    question_toks = toks.tokenize(question)  # Tokenize the question
    parag_toks = toks.tokenize(reference)  # Tokenize the reference document
    tokens = ['[CLS]'] + question_toks + ['[SEP]'] + \
        parag_toks + ['[SEP]']  # Combine tokens
    input_ids = toks.convert_tokens_to_ids(
        tokens)  # Convert tokens to word IDs
    input_mask = [1] * len(input_ids)  # Generate input mask
    type_word_ids = [0] * (1 + len(question_toks) + 1) + \
        [1] * (len(parag_toks) + 1)  # Generate type IDs

    # Convert variables to TensorFlow tensors
    input_ids, input_mask, type_word_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_ids, input_mask, type_word_ids))

    # Get model outputs
    outputs = model([input_ids, input_mask, type_word_ids])
    # Get the start index of the answer
    get_start = tf.argmax(outputs[0][0][1:]) + 1
    # Get the end index of the answer
    get_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_toks = tokens[get_start: get_end + 1]  # Extract answer tokens
    answer = toks.convert_tokens_to_string(
        answer_toks)  # Convert answer tokens to string
    return answer  # Return the answer


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search to find the most similar document in the corpus to the given sentence.

    Args:
        corpus_path (str): The path to the directory containing the corpus documents.
        sentence (str): The input sentence for which to find the most similar document.

    Returns:
        str: The most similar document to the input sentence.
    """
    # Create a list to store the documents, starting with the input sentence
    documents = [sentence]

    # Iterate through each file in the corpus directory
    for filename in os.listdir(corpus_path):
        # Check if the file is a Markdown file
        if filename.endswith(".md"):
            # Read the contents of the file and append it to the documents list
            with open(os.path.join(corpus_path, filename), "r", encoding="utf-8") as f:
                documents.append(f.read())

    # Embed the documents using the Universal Sentence Encoder
    embedded_doc = model(documents)

    # Calculate the cosine similarity matrix between the embedded documents
    cosine = np.inner(embedded_doc, embedded_doc)

    # Find the index of the document with the highest similarity to the input
    # sentence
    closest_index = np.argmax(cosine[0, 1:]) + 1

    # Retrieve the most similar document based on the closest index
    similarity = documents[closest_index]

    return similarity


def question_answer(corpus_path):
    """
    Perform question answering using a given corpus.

    Args:
    - corpus_path (str): Path to the corpus file containing text data.

    Returns:
    - answers (list): A list of answers corresponding to the questions in the corpus.
    """

    while 1:
        question = input("Q: ")
        words = ["exit", "quit", "goodbye", "bye"]

        if question.lower().strip() in words:
            print("A: Goodbye")
            break
        reference = semantic_search(corpus_path, question)
        answer = question0_answer(question, reference)
        if answer is None or answer is "":
            answer = "Sorry, I do not understand your question."
        print("A: {}".format(answer))
