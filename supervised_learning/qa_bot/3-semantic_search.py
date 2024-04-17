#!/usr/bin/env python3

"""
a function def semantic_search(corpus_path, sentence): that performs semantic search on a corpus of documents:

corpus_path is the path to the corpus of reference documents on which to perform semantic search
sentence is the sentence from which to perform semantic search
Returns: the reference text of the document most similar to sentence
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

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
    
    # Load the Universal Sentence Encoder model from TensorFlow Hub
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    model = hub.load(module_url)
    
    # Embed the documents using the Universal Sentence Encoder
    embed = model(documents)
    
    # Calculate the cosine similarity matrix between the embedded documents
    corr = np.inner(embed, embed)
    
    # Find the index of the document with the highest similarity to the input sentence
    closest_index = np.argmax(corr[0, 1:]) + 1
    
    # Retrieve the most similar document based on the closest index
    similarity = documents[closest_index]
    
    return similarity
