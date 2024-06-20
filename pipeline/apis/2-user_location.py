#!/usr/bin/env python3
"""
Fetches and prints the location of a specific GitHub user.

Args:
    <GitHub API URL>: The full API URL of the GitHub user whose location you want to fetch.

Returns:
    Prints the location of the GitHub user if found.
    Prints "Not found" if the GitHub user does not exist.
    Prints "Reset in X min" if the GitHub API rate limit is exceeded, where X is the time until reset.

Examples Usage:
    python 2-user_location.py https://api.github.com/users/holbertonschool
    python 2-user_location.py https://api.github.com/users/holberton_ho_no

"""

import requests     # Library for making HTTP requests
import sys          # Library for interacting with the command-line arguments
import time         # Library for handling time-related operations

if __name__ == '__main__':
    # Retrieve the GitHub API URL from the command-line arguments
    url = sys.argv[1]
    
    # Specify headers to accept JSON response from GitHub API
    headers = {'Accept': 'application/vnd.github.v3+json'}
    
    # Send a GET request to the GitHub API URL
    response = requests.get(url, headers=headers)

    # Check the status code of the response
    if response.status_code == 200:
        # If status code is 200, retrieve JSON data from the response
        user_data = response.json()
        
        # Print the user's location if available; otherwise, print default message
        print(user_data.get('location', 'Location not available'))
        
    elif response.status_code == 404:
        # If status code is 404, print "Not found"
        print("Not found")
        
    elif response.status_code == 403:
        # If status code is 403, calculate time until rate limit reset
        rate_limit_reset = int(response.headers['X-Ratelimit-Reset'])
        current_time = int(time.time())
        minutes_remaining = int((rate_limit_reset - current_time) / 60)
        
        # Print the time until rate limit reset in minutes
        print(f"Reset in {minutes_remaining} min")
        
    else:
        # For any other status codes, print an error message with the status code
        print(f"Error: {response.status_code}")
