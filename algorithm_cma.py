import numpy as np
from utils import get_data_by_index, \
    map_from_auxiliary_coordinates, map_to_auxiliary_coordinates
from metrics import *

from deap import algorithms
from deap import base
from deap import cma
from deap import creator
from deap import tools

# We define problem class
# Minimization
creator.create("FitnessMinCMA", base.Fitness, weights=(-1.0,))
creator.create("IndividualCMA", list, fitness=creator.FitnessMinCMA)


def algorithm_cma(sigma, mu, esg, dt, penalization_weight, ngen):

    (n, m) = esg.shape
    mu = np.atleast_2d(mu).T
    centroid = np.random.rand(n)
    centroid = centroid / np.linalg.norm(centroid, 1)
    centroid = map_to_auxiliary_coordinates(centroid)
    centroid = np.append(centroid, np.random.rand())

    # We define evaluation matrix
    def eval_matrix(gamma):
        return gamma * sigma - mu@mu.T

    def penalization(ind):
        """ Penalizes portfolios not summing up to one."""
        w = ind
        norm = np.linalg.norm(w, 1)
        penalization = 0
        if norm > 1:
            penalization += penalization_weight * norm
        for x in w:
            if x < 0:
                penalization += penalization_weight * (-x)
        return penalization

    def evaluate(ind):
        w = ind[:-1]
        gamma = ind[-1]
        p = penalization(w)
        w = map_from_auxiliary_coordinates(w)
        w = np.atleast_2d(w).T
        h = eval_matrix(gamma)
        res = (w.T@h@w)[0]
        return abs(res) + p

    # We register the evaluation function
    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate)

    # The CMA-ES algorithm takes a population of one individual as argument
    strategy = cma.Strategy(centroid=centroid, sigma=dt, lambda_=20*n)
    toolbox.register("generate", strategy.generate, creator.IndividualCMA)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)

    algorithms.eaGenerateUpdate(toolbox, ngen=ngen, halloffame=hof)

    w_p = hof[0][:-1]
    w_p = map_from_auxiliary_coordinates(w_p)
    # gamma_0 = hof[0][-1]

    index_max = np.argmax(w_p.T@esg)

    w_s = np.zeros(m)
    w_s[index_max] = 1

    return w_p, w_s


if __name__ == "__main__":
    sigma, mu, esg = get_data_by_index(0)

    w_p, w_s = algorithm_cma(
        sigma, mu, esg, 1.0, 1000, 100)

    metrics = 
