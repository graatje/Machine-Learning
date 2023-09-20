import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.mlab as mlab

def draw_graph(data):
    #OPGAVE 1
    # Maak een scatter-plot van de data die als parameter aan deze functie wordt meegegeven. Deze data
    # is een twee-dimensionale matrix met in de eerste kolom de grootte van de steden, in de tweede
    # kolom de winst van de vervoerder. Zet de eerste kolom op de x-as en de tweede kolom op de y-as.
    # Je kunt hier gebruik maken van de mogelijkheid die Python biedt om direct een waarde toe te kennen
    # aan meerdere variabelen, zoals in het onderstaande voorbeeld:

    #     l = [ 3, 4 ]
    #     x,y = l      ->  x = 3, y = 4

    # Om deze constructie in dit specifieke geval te kunnen gebruiken, moet de data-matrix wel eerst
    # roteren (waarom?).
    # Maak gebruik van pytplot.scatter om dit voor elkaar te krijgen.
    # Je kan de data roteren, want dan krijg je beide kolommen in een aparte array binnen de array. Zoals dit:
    # data = data.T
    # plt.scatter(data[0], data[1])
    # In plaats daarvan hebben we ervoor gekozen om de 1e en de 2e kolom van elk datapunt in de 2-dimensionale array op te halen.
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel('Populatie (10k personen)')
    plt.ylabel('Winst (10k$)')
    plt.show()

def compute_cost(X, y, theta):
    #OPGAVE 2
    # Deze methode berekent de kosten van de huidige waarden van theta, dat wil zeggen de mate waarin de
    # voorspelling (gegeven de specifieke waarde van theta) correspondeert met de werkelijke waarde (die
    # is gegeven in y).

    # Elk datapunt in X wordt hierin vermenigvuldigd met theta (welke dimensies hebben X en dus theta?)
    # en het resultaat daarvan wordt vergeleken met de werkelijke waarde (dus met y). Het verschil tussen
    # deze twee waarden wordt gekwadrateerd en het totaal van al deze kwadraten wordt gedeeld door het
    # aantal data-punten om het gemiddelde te krijgen. Dit gemiddelde moet je retourneren (de variabele
    # J: een getal, kortom).

    # Een stappenplan zou het volgende kunnen zijn:

    # 1. bepaal het aantal datapunten
    m, n = X.shape
    # 2. bepaal de voorspelling (dus elk punt van X maal de huidige waarden van theta)
    prediction = np.dot(X, theta)
    # 3. bereken het verschil tussen deze voorspelling en de werkelijke waarde
    delta = prediction - y
    # 4. kwadrateer dit verschil
    kwadraat = delta ** 2
    # 5. tal al deze kwadraten bij elkaar op en deel dit door twee keer het aantal datapunten
    som = np.sum(kwadraat)
    # 6. deel de som door het aantal rijen keer twee
    value = 1/(2*m)*som

    return value


def gradient_descent(X, y, theta, alpha, num_iters):
    #OPGAVE 3a
    # In deze opgave wordt elke parameter van theta num_iter keer geüpdate om de optimale waarden
    # voor deze parameters te vinden. Per iteratie moet je alle parameters van theta update.

    # Elke parameter van theta wordt verminderd met de som van de fout van alle datapunten
    # vermenigvuldigd met het datapunt zelf (zie Blackboard voor de formule die hierbij hoort).
    # Deze som zelf wordt nog vermenigvuldigd met de 'learning rate' alpha.

    m, n = X.shape
    costs = []

    for _ in range(num_iters):
        # 1. bepaal de voorspelling (dus elk punt van X maal de huidige waarden van theta)
        prediction = np.dot(X, theta.reshape(-1, 1))  # hervorm theta als een kolom vector

        # 2. bepaal het verschil tussen deze voorspelling en de werkelijke waarde
        delta = prediction - y

        # Calculate the gradient (derivative) of the cost function with respect to each theta_j
        # 3. deel de som door het aantal rijen en vermenigvuldig de data met het verschil tussen de voorspelling en de werkelijke waarde
        gradient = (1 / m) * np.dot(X.T, delta)

        # 4. update theta doormiddel van de gradient en de alpha
        theta -= alpha * gradient.T

        # 5. bereken de kosten en voeg het toe aan de lijst
        costs.append(compute_cost(X, y, theta.T))

    # aan het eind van deze loop retourneren we de nieuwe waarde van theta
    # (wat is de dimensionaliteit van theta op dit moment?).
    # Theta is (1, 2)
    # print(theta.shape)

    return theta, costs


def draw_costs(data): 
    # OPGAVE 3b
    # YOUR CODE HERE
    # voeg data toe aan de plot
    plt.plot(data)

    # zet de limit van Y tussen 4 en 7
    plt.ylim(4, 7)

    # zet er een x label en een y label bij
    plt.xlabel('iterations')
    plt.ylabel('J(theta)')

    # laat het plot zien
    plt.show()

def contour_plot(X, y):
    #OPGAVE 4
    # Deze methode tekent een contour plot voor verschillende waarden van theta_0 en theta_1.
    # De infrastructuur en algemene opzet is al gegeven; het enige wat je hoeft te doen is 
    # de matrix J_vals vullen met waarden die je berekent aan de hand van de methode computeCost,
    # die je hierboven hebt gemaakt.
    # Je moet hiervoor door de waarden van t1 en t2 itereren, en deze waarden in een ndarray
    # zetten. Deze ndarray kun je vervolgens meesturen aan de functie computeCost. Bedenk of je nog een
    # transformatie moet toepassen of niet. Let op: je moet computeCost zelf *niet* aanpassen.

    # moest veranderd worden omdat het anders niet werkte
    ax = plt.figure(figsize=(7, 7)).add_subplot(projection='3d')
    plt.get_cmap('jet')

    t1 = np.linspace(-10, 10, 100)
    t2 = np.linspace(-1, 4, 100)
    T1, T2 = np.meshgrid(t1, t2)

    J_vals = np.zeros( (len(t2), len(t2)) )

    #YOUR CODE HERE
    for i in range(len(t1)):
        for j in range(len(t2)):
            # maak een array van de t1 en t2 waarden
            t = np.array([t1[i], t2[j]])
            # vul de J_values doormiddel van het bereken van de kosten
            J_vals[i, j] = compute_cost(X, y, t) / 100 # deel door 100 voor een of andere reden

    ax.plot_surface(T1, T2, J_vals, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlabel(r'$\theta_0$', linespacing=3.2)

    # verander de ticks van de x axis
    ax.set_xticks(np.arange(-10, 12.5, 2.5))
    ax.set_ylabel(r'$\theta_1$', linespacing=3.1)
    ax.set_zlabel(r'$J(\theta_0, \theta_1)$', linespacing=3.4)

    plt.show()
