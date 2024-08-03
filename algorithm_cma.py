import numpy as np
from utils import get_data_by_index, \
    map_from_auxiliary_coordinates, map_to_auxiliary_coordinates

from deap import algorithms
from deap import base
from deap import cma
from deap import creator
from deap import tools


def algorithm_cma(sigma, mu, esg,
                  lambda_, dt, penalization_weight, ngen, verbose):

    n = mu.size
    mu = np.atleast_2d(mu).T
    centroid = np.random.rand(n)
    centroid = centroid / np.linalg.norm(centroid, 1)
    centroid = map_to_auxiliary_coordinates(centroid)
    centroid = np.append(centroid, np.random.rand())

    # We define problem class
    # Minimization
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

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
        # return np.pow(res, 2) + p

    # We register the evaluation function
    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate)

    # The CMA-ES algorithm takes a population of one individual as argument
    strategy = cma.Strategy(centroid=centroid, sigma=dt, lambda_=lambda_(n))
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaGenerateUpdate(
        toolbox, ngen=ngen, stats=stats, halloffame=hof, verbose=verbose)

    w_p = hof[0][:-1]
    w_p = map_from_auxiliary_coordinates(w_p)
    gamma_0 = hof[0][-1]

    index_max = np.argmax(w_p.T@esg)

    w_s = np.zeros(n)
    w_s[index_max] = 1

    return w_p, w_s, gamma_0


if __name__ == "__main__":

    sigma, mu, esg = get_data_by_index(0)

    def lambda_(n):
        return 20*n

    dt = 1.0
    penalization_weight = 100
    ngen = 100

    w_p, w_s, gamma_0 = algorithm_cma(sigma, mu, esg, lambda_,
                                      dt, penalization_weight, ngen, False)

    return_ = w_p.T@mu
    risk = w_p.T@sigma@w_p
    fitness = w_p.T@(gamma_0 * sigma - mu@mu.T)@w_p

    print("Portfolio optimo: ", w_p)
    print("Retorno: ", return_)
    print("Riesgo: ", risk)
    print("Norma portfolio:", np.linalg.norm(w_p, 1))
    print("Gamma_0: ", gamma_0)
    print("Fitness: ", fitness)
    print("Vector esg: ", w_s)
