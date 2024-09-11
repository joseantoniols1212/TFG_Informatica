import numpy as np
from utils import get_data_by_index, \
    map_from_auxiliary_coordinates, map_to_auxiliary_coordinates

from deap import algorithms
from deap import base
from deap import cma
from deap import creator
from deap import tools

from metrics import *

# We define problem class
# Minimization
creator.create("FitnessMaxCMA2", base.Fitness, weights=(1.0,))
creator.create("IndividualCMA2", list, fitness=creator.FitnessMaxCMA2)


def algorithm_cma(sigma, mu, esg, var, penalization_weight, ngen):

    (n, m) = esg.shape
    centroid = np.random.rand(n)
    centroid = centroid / np.linalg.norm(centroid, 1)
    centroid = centroid[:-1]
    mu = np.atleast_2d(mu).T

    def penalization(w):
        norm = np.linalg.norm(w, 1)
        penalization = 0
        if norm > 1:
            penalization += penalization_weight * norm
        for x in w:
            if x < 0:
                penalization += penalization_weight * (-x)
        return penalization

    def evaluate(ind):
        w = ind
        p = penalization(w)
        w = map_from_auxiliary_coordinates(w)
        w = np.atleast_2d(w).T
        res = (w.T@mu / (w.T@sigma@w)**0.5)[0]
        return res - p

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate)

    strategy = cma.Strategy(centroid=centroid, sigma=var)
    toolbox.register("generate", strategy.generate, creator.IndividualCMA2)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)

    algorithms.eaGenerateUpdate(
        toolbox, ngen=ngen, halloffame=hof, verbose=False)

    w_p = hof[0]
    w_p = map_from_auxiliary_coordinates(w_p)

    index_max = np.argmax(w_p.T@esg)
    w_s = np.zeros(m)
    w_s[index_max] = 1

    return w_p, w_s


if __name__ == "__main__":
    VAR = 0.3
    NGEN = 200

    print(VAR)
    print(NGEN)

    sigma, mu, esg = get_data_by_index(0)

    assets = Assets(sigma, mu, esg)

    w_p, w_s = algorithm_cma(sigma, mu, esg, VAR, 1000, NGEN)

    portfolio = Portfolio(w_p, w_s)
    metrics = Metrics(portfolio, assets)

    metrics.report()
