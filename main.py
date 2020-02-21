import pandas
from matplotlib import pyplot
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class KNN(object):
    def __init__(self, csv_path, headers, name):
        """
        Simple classe d'abstraction permettant d'effectuer une prédiction en fonction d'un Dataset déjà optimisé
        :param csv_path: Le chemin jusqu'au fichier CSV à charger
        :param headers: Le nom des colonnes
        :param name: Le nom du résultat à prédire
        """
        self.names = headers
        self.df = pandas.read_csv(csv_path, header=None, names=headers)

        x = numpy.array(self.df.iloc[:, 0:len(headers) - 1])  # Les données en entrée
        y = numpy.array(self.df.get(name))  # Les résultats correspondant aux données

        # Découpe les données d'entrainement et les données de test
        # Random state permet de découper les données de la à chaque execution du programme
        # Ici on test 1/3 des données et on s'entraine donc sur les 2/3 restant
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x=x,
            y=y,
            train_size=0.66,
            test_size=0.33,
            random_state=42)

    def predict(self, k):
        """
        Permet de faire une prédiction en fonction du nombre de voisins à prendre en compte
        :param k: Le nombre de voisins
        :return: La précision de la prédiction
        """
        # Instanciation de l'algo de classification
        knn = KNeighborsClassifier(n_neighbors=k)

        # Création du modèle
        knn.fit(self.x_train, self.y_train)

        # Prédiction avec les données de test
        pred = knn.predict(self.x_test)

        return accuracy_score(self.y_test, pred)

    def find_optimal_k(self, min, max):
        """
        Permet de générer un graph montrant le taux de précision des données d'entrainements et de tests en fonction
        du nombres de voisins, cela permet ensuite de choisir le paramètre k optimal
        :param min:
        :param max:
        :return:
        """
        neighbors = numpy.arange(min, max)

        train_accuracy = numpy.empty(len(neighbors))
        test_accuracy = numpy.empty(len(neighbors))

        for index, k in enumerate(neighbors):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.x_train, self.y_train)

            train_accuracy[index] = knn.score(self.x_train, self.y_train)
            test_accuracy[index] = knn.score(self.x_test, self.y_test)

        pyplot.plot(neighbors, test_accuracy, label="Test Accuracy")
        pyplot.plot(neighbors, train_accuracy, label="Train Accuracy")

        pyplot.xlabel('Nombre de voisins')
        pyplot.ylabel('Précision')

        pyplot.legend()
        pyplot.show()
