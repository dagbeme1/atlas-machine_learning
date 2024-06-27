#!/usr/bin/env python3
"""
Update school topics in a MongoDB collection in Python
"""

def update_topics(mongo_collection, name, topics):
    """
    Update the topics for a school document with the specified name.

    Args:
        mongo_collection: A pymongo collection object
        name (str): The name of the school whose topics need to be updated
        topics (list): The list of topics to set for the specified school

    Returns:
        None
    """
    # Create a search query to find the document by school name
    search_query = {"name": name}
    
    # Create the update query to set the new topics
    update_query = {"$set": {"topics": topics}}

    # Perform the update operation on documents matching the search query
    mongo_collection.update_many(search_query, update_query)
