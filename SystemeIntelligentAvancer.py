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

#Question 3 : Générer la matrice
y = []


for couple in X:
    x = couple[0]
    y_val = couple[1]
    produit = x * y_val
#Vérifie la parité
    if produit % 2 == 1:
        y.append(1)  # impair
    else:
        y.append(0)  # pair

#Convertir en tableau numpy
y = np.array(y)
