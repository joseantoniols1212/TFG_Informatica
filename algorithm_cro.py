import numpy as np
from cro import *
from cro.fitness import max_ones
from cro.report import plot_results
from utils import get_data_by_index, \
    map_from_auxiliary_coordinates, map_to_auxiliary_coordinates

## ------------------------------------------------------
## Parameters initialization
## ------------------------------------------------------
Ngen = 30                  # Number of generations
N  = 20                    # MxN: reef size
M  = 20                    # MxN: reef size
Fb = 0.7                   # Broadcast prob.
Fa = 0.1                   # Asexual reproduction prob.
Fd = 0.1                   # Fraction of the corals to be eliminated in the depredation operator.
r0 = 0.6                   # Free/total initial proportion
k  = 3                     # Number of opportunities for a new coral to settle in the reef
Pd = 0.1                   # Depredation prob.
opt= 'min'                 # flag: 'max' for maximizing and 'min' for minimizing
ke = 0.2
## ------------------------------------------------------

## ------------------------------------------------------

def algorithm_cro(sigma, mu, esg, penalization_weight, ngen):

    (n, m) = esg.shape
    L = n
    mu = np.atleast_2d(mu).T

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
    
    def generate():
        centroid = np.random.rand(n)
        centroid = centroid / np.linalg.norm(centroid, 1)
        centroid = map_to_auxiliary_coordinates(centroid)
        centroid = np.append(centroid, np.random.rand())
        return centroid

    def initialize(M,N,r0,L, **kargs):
        O = int(np.round(N*M*r0)) # number of occupied reefs 

        A = np.random.zeros((O,L))
        for x in A:
            x = generate()

        B = np.zeros([((N*M)-O), L], int)          
        REEFpob = np.concatenate([A, B]) # Population creation

        REEF = np.array((REEFpob.any(axis=1)),bool) 
        return (REEF, REEFpob)

    cro = CRO(Ngen, N, M, Fb, Fa, Fd, r0, k, Pd, evaluate, opt, L, verbose=True, ke=ke)
    REEF, REEFpob, REEFfitness, ind_best, Bestfitness, Meanfitness = cro.fit()

    w_p = ind_best[:-1]
    w_p = map_from_auxiliary_coordinates(w_p)
    gamma_0 = ind_best[-1]

    index_max = np.argmax(w_p.T@esg)

    w_s = np.zeros(m)
    w_s[index_max] = 1

    return w_p, w_s, gamma_0

if __name__ == "__main__":

    sigma, mu, esg = get_data_by_index(0)

    w_p, w_s, gamma_0 = algorithm_cro(sigma, mu, esg, 1000, 200)

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