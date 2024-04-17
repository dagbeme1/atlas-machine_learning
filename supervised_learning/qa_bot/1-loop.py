#!/usr/bin/env python3
"""
a script that takes in input from the user with the prompt Q:
and prints A: as a response. If the user inputs exit, quit, goodbye, or bye,
case insensitive, print A: Goodbye and exit
"""

while True:
    question = input("Q: ")
    words = ["exit", "quit", "goodbye", "bye"]

    if question.lower().strip() in words:
        print("A: Goodbye")
        break
    else:
        print("A: ")
