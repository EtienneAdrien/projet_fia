# Projet Fondamentaux IA

## Contenu

Ce projet vise à implémenter l'algorithme KNN

J'utilise deux Dataset bien connu:
- Iris -> https://archive.ics.uci.edu/ml/datasets/Iris
- Breast Cancer Wisconsin -> https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

Le but de l'algorithme est de prédire des résultats en fonction des données en entrée, par exemple dans le cas
du cancer du sein, en fonction des différentes informations médicales, le programme doit pouvoir trouver si il sagit
d'une tumeur bénigne ou maligne.

Dans le cas du "Breast Cancer Wisconsin", il a fallu nettoyer le dataset en supprimant la colonne ID qui aurait
faussé les résultats car elle n'est pas corrélé à l'état de la tumeur.

L'éxecution du programme génere un graph permettant de visualiser le meilleur nombre de voisins à utiliser et un score
d'accuracy montrant la précision de la prédiction.

## Install

```
pip install -r requirements.txt
```

## Utilisation
```
pyhton iris.py
```

```
pyhton breast_cancer.py
```