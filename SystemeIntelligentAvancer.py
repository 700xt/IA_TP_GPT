# Question 1 : Librairie
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

# Variable
NUMBER = 100


# Question 2 : Création matrice 
X = []
for i in range(NUMBER):
    for j in range(NUMBER):
        X.append([i, j])
X = np.array(X)
