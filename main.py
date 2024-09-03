import numpy as np

from metrics import Assets, PortfolioComparison
from utils import get_data_by_index
from algorithm_nsga2 import algorithm_nsga2
from algorithm_cma import algorithm_cma
from algorithm_pso import algorithm_pso
from naive import naive
from maxSos import maxSos
from maxReturn import maxReturn
from minRisk import minRisk


N = 7

assets = []
for i in range(N):
    assets.append(Assets(*get_data_by_index(i)))

naive_params = {}
maxReturn_params = {}
minRisk_params = {}
maxSos_params = {
    "penalization_weight": 1000,
    "dt": 1.5,
    "ngen": 300
}
cma_params = {
    "penalization_weight": 1000,
    "dt": 1.5,
    "ngen": 300
}
nsga2_params = {
    "pop_size": 100,
    "cxpb": 0.9,
    "eta": 20.,
    "penalization_weight": 1000,
    "ngen": 300
}
pso_params = {
    "pop_size": 100,
    "abs_speed_max": 0.25,
    "penalization_weight": 1000,
    "phi1": 1.0,
    "phi2": 1.0,
    "ngen": 300
}

strategies = [
    (naive, naive_params),
    (maxReturn, maxReturn_params),
    (minRisk, minRisk_params),
    (maxSos, maxSos_params),
    (algorithm_cma, cma_params),
    (algorithm_nsga2, nsga2_params),
    (algorithm_pso, pso_params)
]


print("====================================")
print("====== COMENZANDO EJECUCION ========")
print("====================================")

np.random.seed(0)

comparison = PortfolioComparison(assets, strategies).compare()
print(comparison)
comparison.to_csv('results.csv')

print("====================================")
print("====== EJECUCION ACABADA ===========")
print("====================================")
