import numpy as np
from utils import get_data_by_index, \
    map_from_auxiliary_coordinates, map_to_auxiliary_coordinates

import operator
import random
import math

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


def algorithm_pso(sigma, mu, esg, abs_speed_max, penalization_weight, ngen):
    
    n = mu.size
    mu = np.atleast_2d(mu).T
    population_size = 10

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Particle", np.ndarray, fitness=creator.FitnessMin, speed=list, best=None)


    def generate(size):
        centroid = np.random.rand(size)
        centroid = centroid / np.linalg.norm(centroid, 1)
        centroid = map_to_auxiliary_coordinates(centroid)
        centroid = np.append(centroid, np.random.rand())
        part = creator.Particle(centroid)
        part.speed = [random.uniform(-abs_speed_max, abs_speed_max) for _ in range(size)]
        part.smin = -abs_speed_max
        part.smax = abs_speed_max
        return part


    def updateParticle(part, best, phi1, phi2):
        u1 = (random.uniform(0, phi1) for _ in range(len(part)))
        u2 = (random.uniform(0, phi2) for _ in range(len(part)))
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
        part.speed = list(map(operator.add, part.speed,
                        map(operator.add, v_u1, v_u2)))
        for i, speed in enumerate(part.speed):
            if abs(speed) < part.smin:
                part.speed[i] = math.copysign(part.smin, speed)
            elif abs(speed) > part.smax:
                part.speed[i] = math.copysign(part.smax, speed)
        part[:] = list(map(operator.add, part, part.speed))

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

    toolbox = base.Toolbox()
    toolbox.register("particle", generate, size=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)  # TODO: Revisar phi
    toolbox.register("evaluate", evaluate)

    pop = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    GEN = ngen
    best = None

    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if part.best is None or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if best is None or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)

    w_p = best[:-1]
    w_p = map_from_auxiliary_coordinates(w_p)
    gamma_0 = best[-1]

    index_max = np.argmax(w_p.T@esg)

    w_s = np.zeros(n)
    w_s[index_max] = 1

    return w_p, w_s, gamma_0
    

if __name__ == "__main__":

    sigma, mu, esg = get_data_by_index(0)

    w_p, w_s, gamma_0 = algorithm_pso(sigma, mu, esg, 0.1, 1000, 200)

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