#!/usr/bin/env python3
"""
Fetches and prints information about the
next upcoming SpaceX launch.
"""

import requests
from datetime import datetime

if __name__ == '__main__':
    # API endpoint to fetch upcoming launches
    url = "https://api.spacexdata.com/v4/launches/upcoming"

    try:
        # Send GET request to fetch upcoming launches data
        response = requests.get(url)
        response.raise_for_status()
        # Ensure we raise an error for bad responses

        # Initialize variables to find the earliest launch date
        earliest_date_unix = None
        launch_info = None

        # Iterate through each launch entry in the JSON response
        for launch in response.json():
            # Extract necessary information
            date_unix = int(launch["date_unix"])
            launch_name = launch["name"]
            date_local = launch["date_local"]
            rocket_number = launch["rocket"]
            launchpad_number = launch["launchpad"]

            # Determine the earliest launch by comparing date_unix values
            if earliest_date_unix is None or date_unix < earliest_date_unix:
                earliest_date_unix = date_unix
                launch_info = {
                    "name": launch_name,
                    "date": date_local,
                    "rocket": rocket_number,
                    "launchpad": launchpad_number
                }

        # If launch information is found
        if launch_info:
            # Construct rocket_url and launchpad_url using formatted strings
            rocket_base_url = "https://api.spacexdata.com/v4/rockets/"
            rocket_url = "{}{}".format(rocket_base_url, launch_info['rocket'])

            launchpad_base_url = "https://api.spacexdata.com/v4/launchpads/"
            launchpad_url = "{}{}".format(
                launchpad_base_url, launch_info['launchpad'])

            # Send GET request to fetch rocket information
            rocket_response = requests.get(rocket_url)
            rocket_response.raise_for_status()
            # Ensure we raise an error for bad responses
            rocket_name = rocket_response.json()["name"]

            # Send GET request to fetch launchpad information
            launchpad_response = requests.get(launchpad_url)
            # Ensure we raise an error for bad responses
            launchpad_response.raise_for_status()
            launchpad_name = launchpad_response.json()["name"]
            launchpad_locality = launchpad_response.json()["locality"]

            # Format the output string
            output = "{} ({}) {} - {} ({})".format(
                launch_info["name"],
                launch_info["date"],
                rocket_name,
                launchpad_name,
                launchpad_locality
            )

            # Print the formatted output
            print(output)

    except requests.exceptions.RequestException as e:
        print("Error fetching data: {}".format(e))