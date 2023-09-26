import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error


def main():
    iris = load_iris()

    # Vul je featurematrix X op basis van de data.
    X = iris.data

    # De uitkomstvector y ga je vullen op basis van target.
    y: np.ndarray = iris.target
    #  Maak deze binair door 0 en 1 allebei 0 te maken en van elke 2 een 1 te maken.
    y[y==1] = False
    y[y==2] = True

    # reshape y from (150,) to (150,1)
    y = np.reshape(y, (-1, 1))

    theta = np.ones((X.shape[1], 1))
    print(X.shape, y.shape, theta.shape)

    for i in range(1500):
        prediction = sigmoid(np.dot(X, theta))
        error = np.subtract(prediction, y)
        gradient = np.dot(X.T, error)
        theta = np.subtract(theta, 0.01 * gradient)
        # calculate the cost
        


def sigmoid(x):
    return 1/(1+np.exp(-x))


if __name__ == "__main__":
    main()
