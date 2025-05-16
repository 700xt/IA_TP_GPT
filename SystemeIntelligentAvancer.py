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
        y.append(1)
    else:
        y.append(0)

#Convertir en tableau numpy
y = np.array(y)

#Question 4 : Séparation ensemble de test et d'entrainement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Question 5 : Instancier un réseau de neurone

modele = MLPClassifier(hidden_layer_sizes=(15,), max_iter=1000, random_state=42)

#Question 6 : Entrainement

modele.fit(X_train,y_train)

# Question 7 : Affichage le nombre d'époques
print(f"Convergence en {modele.niter} époques")