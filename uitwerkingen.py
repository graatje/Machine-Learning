import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# ==== OPGAVE 1 ====
def plot_number(nrVector):
    # Let op: de manier waarop de data is opgesteld vereist dat je gebruik maakt
    # van de Fortran index-volgorde – de eerste index verandert het snelst, de 
    # laatste index het langzaamst; als je dat niet doet, wordt het plaatje 
    # gespiegeld en geroteerd. Zie de documentatie op 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html

    plt.matshow(np.reshape(nrVector, (20,20), order='F'), cmap='gray')
    plt.show()


# ==== OPGAVE 2a ====
def sigmoid(z):
    # Maak de code die de sigmoid van de input z teruggeeft. Zorg er hierbij
    # voor dat de code zowel werkt wanneer z een getal is als wanneer z een
    # vector is.
    # Maak gebruik van de methode exp() in NumPy.
    return 1/(1+np.exp(-z))


# ==== OPGAVE 2b ====
def get_y_matrix(y: np.ndarray, m: int):
    # Gegeven een vector met waarden y_i van 1...x, retourneer een (ijle) matrix
    # van m×x met een 1 op positie y_i en een 0 op de overige posities.
    # Let op: de gegeven vector y is 1-based en de gevraagde matrix is 0-based,
    # dus als y_i=1, dan moet regel i in de matrix [1,0,0, ... 0] zijn, als
    # y_i=10, dan is regel i in de matrix [0,0,...1] (in dit geval is de breedte
    # van de matrix 10 (0-9), maar de methode moet werken voor elke waarde van y en m

    x = len(y)
    y_i = y[:, 0]
    cols = y_i - 1
    rows = np.arange(m)
    width = np.max(cols)
    data = np.ones(x)

    y_vec = csr_matrix((data, (rows, cols)), shape=(len(rows), width+1)).toarray()
    return y_vec

# ==== OPGAVE 2c ==== 
# ===== deel 1: =====
def predict_number(Theta1, Theta2, X):
    # Deze methode moet een matrix teruggeven met de output van het netwerk
    # gegeven de waarden van Theta1 en Theta2. Elke regel in deze matrix
    # is de waarschijnlijkheid dat het sample op die positie (i) het getal
    # is dat met de kolom correspondeert.

    # De matrices Theta1 en Theta2 corresponderen met het gewicht tussen de
    # input-laag en de verborgen laag, en tussen de verborgen laag en de
    # output-laag, respectievelijk.

    # Een mogelijk stappenplan kan zijn:

    #    1. voeg enen toe aan de gegeven matrix X; dit is de input-matrix a1
    #    2. roep de sigmoid-functie van hierboven aan met a1 als actuele
    #       parameter: dit is de variabele a2
    #    3. voeg enen toe aan de matrix a2, dit is de input voor de laatste
    #       laag in het netwerk
    #    4. roep de sigmoid-functie aan op deze a2; dit is het uiteindelijke
    #       resultaat: de output van het netwerk aan de buitenste laag.

    # Voeg enen toe aan het begin van elke stap en reshape de uiteindelijke
    # vector zodat deze dezelfde dimensionaliteit heeft als y in de exercise.

    # Stap 1: Voeg enen toe aan de gegeven matrix X
    a1 = np.insert(X, 0, 1, axis=1)

    # Stap 2: Bereken de activatie van de tweede laag (verborgen laag)
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)

    # Stap 3: Voeg enen toe aan de matrix a2
    a2 = np.insert(a2, 0, 1, axis=1)

    # Stap 4: Bereken de activatie van de derde laag (output laag)
    z3 = np.dot(a2, Theta2.T)
    h = sigmoid(z3)

    return h



# ===== deel 2: =====
def compute_cost(Theta1, Theta2, X, y):
    m = X.shape[0]

    # Zet y om naar een matrix
    y_matrix = get_y_matrix(y, m)

    # Voorspel de getallen
    h = predict_number(Theta1, Theta2, X)

    # Bereken de kost volgens de gegeven formule
    cost_matrix = -y_matrix * np.log(h) - (1 - y_matrix) * np.log(1 - h)
    cost = np.sum(cost_matrix) / m

    return cost


# ==== OPGAVE 3a ====
def sigmoid_gradient(z):
    # Retourneer hier de waarde van de afgeleide van de sigmoïdefunctie.
    # Zie de opgave voor de exacte formule. Zorg ervoor dat deze werkt met
    # scalaire waarden en met vectoren.
    g = 1 / (1 + np.exp(-z))
    gradient = g * (1 - g)
    return gradient

    pass

# ==== OPGAVE 3b ====
def nn_check_gradients(Theta1, Theta2, X, y): 
    # Retourneer de gradiënten van Theta1 en Theta2, gegeven de waarden van X en van y
    # Zie het stappenplan in de opgaven voor een mogelijke uitwerking.

    Delta2 = np.zeros(Theta1.shape)
    Delta3 = np.zeros(Theta2.shape)
    m = 1 #voorbeeldwaarde; dit moet je natuurlijk aanpassen naar de echte waarde van m

    for i in range(m): 
        #YOUR CODE HERE
        pass

    Delta2_grad = Delta2 / m
    Delta3_grad = Delta3 / m
    
    return Delta2_grad, Delta3_grad
