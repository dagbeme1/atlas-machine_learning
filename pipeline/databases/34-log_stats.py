#!/usr/bin/env python3
"""
Fetches and prints statistics from a MongoDB collection of nginx logs.
"""

# Import necessary module
from pymongo import MongoClient

# Function to fetch and print log statistics
def fetch_log_statistics():
    """
    Connects to MongoDB, fetches log statistics, and prints them.
    """
    # Connect to MongoDB server
    client = MongoClient('mongodb://localhost:27017/')

    try:
        # Access the 'logs' database and 'nginx' collection
        db = client['logs']
        collection = db['nginx']

        # Count total number of logs in the collection
        total_logs = collection.count_documents({})

        # Print total number of logs
        print(f"Total Logs: {total_logs}")

        # Print statistics for each HTTP method
        print("\nMethods:")
        methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
        for method in methods:
            count = collection.count_documents({"method": method})
            print(f"\t{count} logs with method={method}")

        # Count logs with specific method and path
        status_logs_count = collection.count_documents({"method": "GET", "path": "/status"})
        print(f"\n{status_logs_count} logs with method=GET\npath=/status")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Close MongoDB connection
        client.close()
