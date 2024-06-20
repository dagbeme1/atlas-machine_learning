#!/usr/bin/env python3
"""
a method that returns the list of 
ships that can hold a given number of passengers:

Prototype: def availableShips(passengerCount):
Don’t forget the pagination
If no ship available, return an empty list.
"""

import requests  # Import the requests library for making HTTP requests



def availableShips(passengerCount):
    """
    Fetches and returns a list of starships from the
    SWAPI API that can hold at least the given number of passengers.

    Parameters:
        passengerCount (int): The minimum number
        of passengers the starship should be able to carry.

    Returns:
        list: A list of starship names that
        meet the passenger capacity criteria.
    """
    base_url = "https://swapi.dev/api/starships/"
    # Base URL for the SWAPI starships endpoint
    ship_list = []
    # List to store starship names that meet the criteria

    while base_url:
        try:
            # Make an HTTP GET request to the given URL
            response = requests.get(base_url)
            response.raise_for_status()
            # Raise an HTTPError for bad responses (4xx, 5xx)
            data = response.json()
            # Return the response as a JSON dictionary
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            break

        for ship in data.get("results", []):
            try:
                passengers = int(ship.get("passengers", "").replace(",", ""))
                if passengers >= passengerCount:
                    ship_list.append(ship["name"])
            except ValueError:
                continue

        # Get the URL of the next page, None if there are no more pages
        base_url = data.get("next")

    return (
        ship_list  # Return the list of starship names that meet the criteria
    )
