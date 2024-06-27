#!/usr/bin/env python3
"""
List all documents in Python
"""

def list_all(mongo_collection):
    """
    List all documents in the given MongoDB collection

    Args:
        mongo_collection: A pymongo collection object

    Returns:
        A list of documents from the collection
    """
    # Initialize an empty list to store the documents
    documents = []
    
    # Iterate over each document returned by the find() method
    for document in mongo_collection.find():
        # Append the current document to the documents list
        documents.append(document)
    
    # Return the list of documents
    return documents
