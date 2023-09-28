import numpy as np
from sklearn.datasets import load_iris

# Download de dataset
iris = load_iris()

# Vul featurematrix X op basis van data
X = iris.data

# Vul uitkomstvector y op basis van target
y = (iris.target == 2).astype(int)  # 1 voor 'virginica', 0 voor de rest


# Definieer de sigmoid-functie
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Initialiseer vector theta met 1.0'en in de juiste shape
# 4 features, dus 4 theta's
theta = np.ones(X.shape[1])
# De learning rate
alpha = 0.01

# Gradient Descent
for _ in range(1500):
    # Bereken de voorspellingen
    predictions = sigmoid(np.dot(X, theta))

    # Bereken de errors
    errors = predictions - y

    # Bereken de gradient
    gradient = np.dot(X.T, errors) / len(y)

    # Pas theta aan
    theta -= alpha * gradient

    # Bereken de kosten (Log Loss)
    cost = -1/len(y) * (np.dot(y, np.log(predictions)) + np.dot(1 - y, np.log(1 - predictions)))
    print(cost)


# Laatste waarde van theta en kosten
print("Laatste waarden:")
print("Theta:", theta)
print("Kosten:", cost)
if __name__ == "__main__":
    pass
