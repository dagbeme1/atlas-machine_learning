#!/usr/bin/env python3
""" Saves the instance object to a file in pickle format."""
import pickle

class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        # Constructor implementation
        
    def save(self, filename):
        """
        Saves the instance object to a file in pickle format.

        Args:
            filename: The file to which the object should be saved.

        Returns:
            None
        """
        # Add .pkl extension if not present
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        # Serialize and save the object to a file
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        
    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object from a file.

        Args:
            filename: The file from which the object should be loaded.

        Returns:
            The loaded DeepNeuralNetwork object, or None if filename doesn't exist.
        """
        try:
            # Deserialize the object from the file
            with open(filename, 'rb') as file:
                loaded_object = pickle.load(file)
            
            return loaded_object
        except FileNotFoundError:
            # Handle file not found error
            return None
