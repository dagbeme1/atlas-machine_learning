#!/usr/bin/env python3
"""
Fetches and prints statistics from a MongoDB collection of nginx logs.
"""

from pymongo import MongoClient

if __name__ == "__main__":
    # Connect to MongoDB server
    client = MongoClient('mongodb://localhost:27017/')

    try:
        # Access the 'logs' database and 'nginx' collection
        db = client['logs']
        collection = db['nginx']

        # Count total number of logs in the collection
        total_logs = collection.count_documents({})
        print(f"first line: {total_logs} logs where {total_logs} is the number of documents in this collection")

        # Print statistics for each HTTP method
        print("Methods:")
        methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
        for method in methods:
            count = collection.count_documents({"method": method})
            print(f"\tmethod {method}: {count}")

        # Count logs with specific method and path
        status_logs_count = collection.count_documents({"method": "GET", "path": "/status"})
        print(f"{status_logs_count} status check")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Close MongoDB connection
        client.close()
