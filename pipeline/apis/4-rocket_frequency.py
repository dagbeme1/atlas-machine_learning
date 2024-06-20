#!/usr/bin/env python3
"""
a script that displays the number of launches per rocket.

Use this https://api.spacexdata.com/v4/launches to make request
All launches should be taken into consideration
Each line should contain the rocket name and the number of
launches separated by : (format below in the example)
Order the result by the number launches (descending)
If multiple rockets have the same amount of launches, order
them by alphabetic order (A to Z)
Your code should not be executed when the file is
imported (you should use if __name__ == '__main__':)
"""

import requests  # Import the requests library for making HTTP requests

if __name__ == '__main__':
    # Define the SpaceX API endpoints
    launches_url = 'https://api.spacexdata.com/v4/launches'
    rockets_url = 'https://api.spacexdata.com/v4/rockets'

    try:
        # Fetch all launches data
        # Send GET request to fetch launches data
        response_launches = requests.get(launches_url)
        launches = response_launches.json()
        # Convert response JSON to Python dictionary

        # Fetch all rockets data
        # Send GET request to fetch rockets data
        response_rockets = requests.get(rockets_url)
        rockets_data = {rocket['id']: rocket['name']
                        for rocket in response_rockets.json()}
        # Map rocket IDs to names

        # Initialize a dictionary to count launches per rocket
        rocket_launch_counts = {}

        # Count launches per rocket
        for launch in launches:
            rocket_id = launch['rocket']  # Get rocket ID for each launch
            # Fetch rocket name using rocket ID
            rocket_name = rockets_data.get(rocket_id, "Unknown Rocket")

            if rocket_name in rocket_launch_counts:
                # Increment launch count for the rocket
                rocket_launch_counts[rocket_name] += 1
            else:
                # Initialize launch count for the rocket
                rocket_launch_counts[rocket_name] = 1

        # Sort rockets by number of launches (descending), then by name
        # (ascending)
        sorted_rockets = sorted(
            rocket_launch_counts.items(), key=lambda x: (-x[1], x[0]))

        # Print each rocket and its number of launches
        for rocket, count in sorted_rockets:
            # Print rocket name and its launch count
            print(f"{rocket}: {count}")

    except requests.exceptions.RequestException as e:
        # Handle any exceptions related to requests
        # Print error message if an exception occurs
        print(f"An error occurred: {e}")
