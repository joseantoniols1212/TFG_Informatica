import numpy as np
import random
import array
import sys

from utils import get_data_by_index, \
    map_from_auxiliary_coordinates, map_to_auxiliary_coordinates

from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMinNSGA2_Pareto", base.Fitness,
               weights=(-1.0, 1.0, 1.0))  # Min risk, max return, Max esg
creator.create("IndividualNSGA2_Pareto", array.array, typecode='d',
               fitness=creator.FitnessMinNSGA2_Pareto)


def imprimir_barra_progreso(iteracion, total, longitud=80):
    progreso = iteracion / total
    bloque = int(longitud * progreso)
    barra = '█' * bloque + '-' * (longitud - bloque)
    porcentaje = progreso * 100
    sys.stdout.write(
        f'\r|{barra}| {porcentaje:.2f}%')
    sys.stdout.flush()


def algorithm_nsga2_pareto(sigma, mu, esg,
                           pop_size, cxpb, eta, penalization_weight, ngen):
    (n, m) = esg.shape
    mu = np.atleast_2d(mu).T
    wp = np.random.rand(n)
    wp = wp / np.linalg.norm(wp, 1)
    wp = map_to_auxiliary_coordinates(wp)
    ws = np.random.rand(m)
    ws = ws / np.linalg.norm(wp, 1)
    ws = map_to_auxiliary_coordinates(wp)

    def penalization(ind):
        """ Penalizes portfolios not summing up to one."""
        wp = ind[:n]
        ws = ind[n:]
        norm_wp = np.linalg.norm(wp, 1)
        norm_ws = np.linalg.norm(ws, 1)
        penalization = 0
        if norm_wp > 1:
            penalization += penalization_weight * norm_wp
        for x in wp:
            if x < 0:
                penalization += penalization_weight * (-x)
        if norm_ws > 1:
            penalization += penalization_weight * norm_ws
        for x in ws:
            if x < 0:
                penalization += penalization_weight * (-x)
        return penalization

    def evaluate(ind):
        wp = ind[:n-1]
        ws = ind[n-1:]
        p = penalization(ind)
        wp = map_from_auxiliary_coordinates(wp)
        ws = map_from_auxiliary_coordinates(ws)
        wp = np.atleast_2d(wp).T
        ws = np.atleast_2d(ws).T
        risk = wp.T@sigma@wp + p
        return_ = wp.T@mu - p
        esg_score = wp.T@esg@ws - p
        return risk[0][0], return_[0][0], esg_score[0][0]

    BOUND_LOW, BOUND_UP = 0.0, 1.0
    ETA = eta
    POPULATION_SIZE = pop_size
    CXPB = cxpb
    NDIM = n+m-2

    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

    toolbox = base.Toolbox()
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initIterate,
                     creator.IndividualNSGA2_Pareto, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=BOUND_LOW, up=BOUND_UP, eta=ETA)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=BOUND_LOW, up=BOUND_UP, eta=ETA, indpb=1.0/NDIM)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=POPULATION_SIZE)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    pareto = tools.ParetoFront()
    pareto.update(pop)

    # Begin the generational process
    for gen in range(1, ngen):
        imprimir_barra_progreso(gen, ngen)

        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, POPULATION_SIZE)
        pareto.update(pop)

    return pareto


if __name__ == "__main__":
    index = int(sys.argv[1])
    sigma, mu, esg = get_data_by_index(index)

    np.random.seed(0)
    pareto = algorithm_nsga2_pareto(
        sigma, mu, esg, pop_size=200, cxpb=0.9, eta=20, penalization_weight=1000, ngen=500)
    pareto_front = np.array([ind.fitness.values for ind in pareto])

    np.save(f'./pareto_fronts/pareto_front_{index}', pareto_front)
