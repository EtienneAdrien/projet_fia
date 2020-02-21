from main import KNN

names = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]

if __name__ == '__main__':
    knn = KNN(csv_path="./iris.data", headers=names, name="species")
    knn.find_optimal_k(min=2, max=15)
    print(f"Accuracy score: {knn.predict(k=13)}")
