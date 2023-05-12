#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

fruit_labels = ['Apples', 'Bananas', 'Oranges', 'Peaches']
person_labels = ['Farrah', 'Fred', 'Felicia']

colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

plt.bar(person_labels, fruit[0], color=colors[0], label=fruit_labels[0])
for i in range(1, len(fruit_labels)):
        plt.bar(person_labels, fruit[i], bottom=np.sum(fruit[:i], axis=0), color=colors[i], label=fruit_labels[i])

        plt.xlabel('Person')
        plt.ylabel('Quantity of Fruit')
        plt.title('Number of Fruit per Person')
        plt.ylim(0, 80)
        plt.yticks(np.arange(0, 81, 10))
        plt.legend()

        plt.show()

