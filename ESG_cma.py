import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools

# Primera fase

# Leemos los datos
N, mean_returns, std_deviations, correlations = utils.parse_data("./data/port1.txt")
MU = np.atleast_2d(mean_returns).T
SIGMA = correlations

print(MU[:3,:])
print(SIGMA)

# Leemos solucion
sol_returns, sol_variance = utils.parse_solution("./data/portef1.txt")

# Hiperparametros del modelo
GAMMA = 0.025                 # Ratio

# Hiperparametros estrategia
LAMBDA = 20*(N-1)           # Tamaño sampleo
DT = 5.0                    # Desviación típica
CENTROID = np.random.rand(N)
CENTROID = CENTROID / np.linalg.norm(CENTROID, 1)  # Tomamos como punto inicial un punto aleatorio de la superficie
CENTROID = utils.proyection(CENTROID)  # Lo proyectamos obre el plano x1 = 0 para facilitar la expresion de las restricciones


# Definimos la clase de problema
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Definimos la matriz
H = GAMMA * SIGMA - MU@MU.T

# Definimos la funcion a optimizar:   GAMMA * w' * sigma * w - w' * mu * mu' * w = w' * H * w
def evaluate(ind):
    w = utils.map_to_sphere(ind)
    w = np.atleast_2d(w).T
    return (w.T@H@w)[0]


def distance(feasible_ind, original_ind):
    """A distance function to the feasibility region."""
    return sum((f - o)**2 for f, o in zip(feasible_ind, original_ind))

def closest_feasible(individual):
    """A function returning a valid individual from an invalid one."""
    feasible_ind = np.array(individual)
    feasible_ind = np.maximum(0, feasible_ind)
    feasible_ind = feasible_ind / np.linalg.norm(feasible_ind, 1)
    return feasible_ind

def valid(individual):
    """Determines if the individual is valid or not."""
    if any(individual < np.zeros(len(individual))) or np.linalg.norm(np.array(individual),1) > 1:
        return False
    return True

# Definimos registramos la funcion a evaluar y añadimos la penalizacion para los individuos no factibles
toolbox = base.Toolbox()
toolbox.register("evaluate", evaluate)
toolbox.decorate("evaluate", tools.ClosestValidPenalty(valid, closest_feasible, 1.0e+6, distance))


def main():
    # The cma module uses the numpy random number generator
    np.random.seed(128)

    # The CMA-ES algorithm takes a population of one individual as argument
    strategy = cma.Strategy(centroid=CENTROID, sigma=DT)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaGenerateUpdate(toolbox, ngen=250, stats=stats, halloffame=hof)

    return hof[0]

if __name__ == "__main__":
    best_ind = main()
    w = np.atleast_2d(utils.map_to_sphere(best_ind)).T
    return_ = w.T@MU
    variance = w.T@SIGMA@w
    init_return = CENTROID
    print("Mejor individuo: ", best_ind)
    print("Retorno medio: ", return_)
    print("Varianza: ", variance)
    print("Retorno individuo inicial: ", )

    # Representamos los datos
    scale = 1e2
    plt.figure(figsize=(50, 40))
    plt.scatter(return_*scale, variance*scale, color='b', marker='x')
    plt.scatter(sol_returns*scale, sol_variance*scale, color='r', marker='o')
    plt.title('Mean Return vs Variance of Return')
    plt.xlabel('Variance of Return')
    plt.ylabel('Mean Return')
    plt.grid(True)
    plt.axis('equal')
    plt.show()