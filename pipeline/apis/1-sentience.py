#!/usr/bin/env python3
"""
create a method that returns the list of names of the home planets of all sentient species.

"""
import requests

def sentientPlanets():
    """
    Fetches and returns the list of names of the home planets of all sentient species from the SWAPI API.
    
    Returns:
        list: A list of planet names that are home to sentient species.
    """
    base_url = "https://swapi.dev/api/species/"  # Base URL for the SWAPI species endpoint
    world_list = []  # List to store names of home planets

    while base_url:
        try:
            response = requests.get(base_url)  # Make an HTTP GET request to the current URL
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx, 5xx)
            data = response.json()  # Return the response as a JSON dictionary
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            break

        for species in data.get("results", []):
            homeworld_url = species.get("homeworld")
            if homeworld_url:
                try:
                    homeworld_response = requests.get(homeworld_url)  # Request the homeworld URL
                    homeworld_response.raise_for_status()
                    homeworld_name = homeworld_response.json().get("name")
                    if homeworld_name:
                        world_list.append(homeworld_name)
                except requests.exceptions.RequestException as e:
                    print(f"Request error for homeworld: {e}")
                    continue

        base_url = data.get("next")  # Get the URL of the next page, None if there are no more pages

    return world_list  # Return the list of home planet names
