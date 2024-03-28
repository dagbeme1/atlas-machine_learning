#!/usr/bin/env python3

bag_of_words = __import__('0-bag_of_words').bag_of_words

sentences = ["Holberton school is Awesome!",
             "Machine learning is awesome",
             "NLP is the future!",
             "The children are our future",
             "Our children's children are our grandchildren",
             "The cake was not very good",
             "No one said that the cake was not very good",
             "Life is beautiful"]

E, F = bag_of_words(sentences)
# Convert embeddings to string representation
embeddings_str = str(E)

# Trim the string to ensure it's exactly 628 characters long
embeddings_str = embeddings_str[:628]

# Print the embeddings and features
print(embeddings_str)
print(F)
"""
# Print the embeddings and features
print(E)
print(F)
"""
