#!/usr/bin/env python3
"""
Fetches and returns a list of schools that have a specific topic in their topics field in a MongoDB collection.
"""

def schools_by_topic(mongo_collection, topic):
    """
    Fetches and returns a list of schools that have a specific topic in their topics field.

    Args:
        mongo_collection: A pymongo collection object.
        topic (str): The topic to search for in the topics field of the schools.

    Returns:
        list: A cursor pointing to the list of schools that have the specified topic.
    """
    # Create a query to search for schools with the specified topic in their topics field
    search_query = {"topics": {"$in": [topic]}}
    
    # Execute the query and return the cursor to the matching documents
    return mongo_collection.find(search_query)
