import numpy as np
import pandas as pd
import utils

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools

# Primera fase

# Cargamos los datos del retorno desde el CSV
returns_data = pd.read_csv('./data/returns_data.csv')
MU = returns_data.values[0]

N = len(MU)  # Numero de assets

# Cargamos los datos del riesgo desde el CSV
risk_data = pd.read_csv('./data/risk_data.csv')
SIGMA = risk_data.values

# Hiperparametros del modelo
GAMMA = 0.5             # Autovalor

# Hiperparametros estrategia
LAMBDA = 20*N           # Tamaño sampleo
DT = 5.0             # Desviación típica
CENTROID = np.random.rand(N)
CENTROID = CENTROID / np.linalg.norm(CENTROID, 1)  # Tomamos como punto inicial un punto aleatorio de la superficie
CENTROID = utils.proyection(CENTROID)  # Lo proyectamos obre el plano x1 = 0 para facilitar la expresion de las restricciones


# Definimos la clase de problema
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Definimos la matriz cuadratica
H = GAMMA * SIGMA - MU*MU.T

# Definimos la funcion a optimizar:   GAMMA * w' * sigma * w - w' * mu * mu' * w
def evaluate(w: np.array) -> float:
    w = utils.map_to_sphere(w)
    return GAMMA * w.T * SIGMA * w - w.T * MU * MU.T * w


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
    if any(individual < np.zeros(len(individual))) or any(np.linalg.norm(np.array(individual),1) > 1):
        return False
    return True

def close_valid(individual):
    """Determines if the individual is close to valid."""
    EPS_BOUND = np.ones(len(individual))*1e-5
    if any(individual < -EPS_BOUND) or any(np.linalg.norm(np.array(individual),1) > 1+EPS_BOUND):
        return False
    return True

# Definimos atributos e individuos
toolbox = base.Toolbox()
toolbox.register("evaluate", evaluate)
toolbox.decorate("evaluate", tools.ClosestValidPenalty(valid, closest_feasible, 1.0e+6, distance))

def main():
    # The cma module uses the numpy random number generator
    np.random.seed(128)

    # The CMA-ES algorithm takes a population of one individual as argument
    # The centroid is set to a vector of 5.0 see http://www.lri.fr/~hansen/cmaes_inmatlab.html
    # for more details about the rastrigin and other tests for CMA-ES    
    strategy = cma.Strategy(centroid=CENTROID, sigma=DT, lambda_=LAMBDA)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # The CMA-ES algorithm converge with good probability with those settings
    algorithms.eaGenerateUpdate(toolbox, ngen=250, stats=stats, halloffame=hof)

    print("Best individual is %s, %s" % (hof[0], hof[0].fitness.values))
    return hof[0].fitness.values[0]

if __name__ == "__main__":
    main()