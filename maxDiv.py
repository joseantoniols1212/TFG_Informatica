import numpy as np
from utils import get_data_by_index, \
    map_from_auxiliary_coordinates, map_to_auxiliary_coordinates

from deap import algorithms
from deap import base
from deap import cma
from deap import creator
from deap import tools

# Max
creator.create("FitnessMaxDiv", base.Fitness, weights=(1.0,))
creator.create("IndividualDiv", list, fitness=creator.FitnessMaxDiv)


def maxDiv(sigma, mu, esg, var, penalization_weight, ngen):

    (n, m) = esg.shape
    centroid = np.random.rand(n)
    centroid = centroid / np.linalg.norm(centroid, 1)
    centroid = centroid[:-1]
    centroid = np.append(centroid, np.random.rand())

    dt = np.diagonal(sigma)
    dt = np.atleast_2d(dt).T

    # We define evaluation matrix
    def eval_matrix(gamma):
        return gamma * sigma - dt@dt.T

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
        w = ind[:-1]
        gamma = ind[-1]
        p = penalization(w)
        w = map_from_auxiliary_coordinates(w)
        w = np.atleast_2d(w).T
        h = eval_matrix(gamma)
        res = (w.T@h@w)[0]
        return abs(res) + p

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate)

    strategy = cma.Strategy(centroid=centroid, sigma=var)
    toolbox.register("generate", strategy.generate, creator.IndividualCMA)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)

    algorithms.eaGenerateUpdate(
        toolbox, ngen=ngen, halloffame=hof, verbose=False)

    w_p = hof[0][:-1]
    w_p = map_from_auxiliary_coordinates(w_p)

    index_max = np.argmax(w_p.T@esg)
    w_s = np.zeros(m)
    w_s[index_max] = 1

    return w_p, w_s


if __name__ == "__main__":
    sigma, mu, esg = get_data_by_index(0)

    w_p, w_s = maxDiv(sigma, mu, esg, 1.0, 1000, 100)
