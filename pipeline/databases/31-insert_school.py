#!/usr/bin/env python3
"""
Insert documents into a MongoDB collection in Python
"""

def insert_school(mongo_collection, **kwargs):
    """
    Insert a document into the specified MongoDB collection.

    Args:
        mongo_collection: A pymongo collection object
        **kwargs: Arbitrary keyword arguments representing the fields and values of the document to be inserted

    Returns:
        The ObjectId of the inserted document
    """
    # Insert the document represented by kwargs into the mongo_collection
    result = mongo_collection.insert_one(kwargs)
    
    # Return the ObjectId of the inserted document
    return result.inserted_id
