import numpy as np
from utils import get_data_by_index, map_from_auxiliary_coordinates

from deap import algorithms
from deap import base
from deap import cma
from deap import creator
from deap import tools

# Max
creator.create("FitnessMaxSos", base.Fitness, weights=(1.0,))
creator.create("IndividualMaxSos", list, fitness=creator.FitnessMaxSos)


def maxSos(sigma, mu, esg, var, penalization_weight, ngen):
    (n, m) = esg.shape
    w_s = np.random.rand(m)
    w_s = w_s / np.linalg.norm(w_s, 1)
    w_s = w_s[:-1]
    w_p = np.random.rand(n)
    w_p = w_p / np.linalg.norm(w_p, 1)
    w_p = w_p[:-1]

    w = np.append(w_p, w_s)

    def penalization(ind):
        """ Penalizes portfolios not summing up to one."""
        w = ind
        w_p = w[:n-1]
        w_s = w[n-1:]

        norm_p = np.linalg.norm(w_p, 1)
        norm_s = np.linalg.norm(w_s, 1)

        penalization = 0
        if norm_s > 1:
            penalization += penalization_weight * norm_s
        if norm_p > 1:
            penalization += penalization_weight * norm_p
        for x in w:
            if x < 0:
                penalization += penalization_weight * (-x)
        return penalization

    def evaluate(ind):
        w = ind
        w_p = w[:n-1]
        w_p = map_from_auxiliary_coordinates(w_p)
        w_p = np.atleast_2d(w_p)
        w_s = w[n-1:]
        w_s = map_from_auxiliary_coordinates(w_s)
        w_s = np.atleast_2d(w_s)
        p = penalization(w)
        res = (w_p@esg@w_s.T)[0]
        return res - p

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate)
    strategy = cma.Strategy(centroid=w, sigma=var)
    toolbox.register("generate", strategy.generate, creator.IndividualMaxSos)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)
    algorithms.eaGenerateUpdate(
        toolbox, ngen=ngen, halloffame=hof, verbose=False)

    w_p = hof[0][:n-1]
    w_p = map_from_auxiliary_coordinates(w_p)
    w_s = hof[0][n-1:]
    w_s = map_from_auxiliary_coordinates(w_s)

    return w_p, w_s


if __name__ == "__main__":

    sigma, mu, esg = get_data_by_index(0)

    w_p, w_s = maxSos(sigma, mu, esg, 1000, 1., 200)

    return_ = w_p@mu
    risk = w_p@sigma@w_p
    esg_rating = w_p@esg@w_s

    print("Retorno: ", return_)
    print("Riesgo: ", risk)
    print("ESG rating: ", esg_rating)
    print("Norma w_p:", np.linalg.norm(w_p, 1))
    print("Norma w_s:", np.linalg.norm(w_s, 1))
    print("w_p:", w_p)
    print("w_s:", w_s)
