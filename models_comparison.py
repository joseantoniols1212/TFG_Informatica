import numpy as np

from metrics import Assets, PortfolioComparator
from utils import get_data_by_index

from algorithm_nsga2 import algorithm_nsga2
from algorithm_cma import algorithm_cma
from algorithm_pso import algorithm_pso

from naive import naive
from maxSos import maxSos
from maxReturn import maxReturn
from minRisk import minRisk
from maxDiv import maxDiv

import random

N = 7
NGEN = 1000
PENALIZATION = 1000

assets = []
for i in range(N):
    assets.append(Assets(*get_data_by_index(i)))

naive_params = {}
maxReturn_params = {}
minRisk_params = {}
maxDiv_params = {
    "penalization_weight": PENALIZATION,
    "var": 0.3,
    "ngen": NGEN
}
maxSos_params = {
    "penalization_weight": PENALIZATION,
    "var": 0.3,
    "ngen": NGEN
}
cma_params = {
    "penalization_weight": PENALIZATION,
    "var": 0.3,
    "ngen": NGEN
}
nsga2_params = {
    "pop_size": 100,
    "penalization_weight": PENALIZATION,
    "ngen": NGEN
}
pso_params = {
    "pop_size": 100,
    "abs_speed_max": 0.25,
    "phi1": 1.75,
    "phi2": 1.5,
    "penalization_weight": PENALIZATION,
    "ngen": NGEN
}

strategies = [
    (naive, naive_params),
    (maxReturn, maxReturn_params),
    (minRisk, minRisk_params),
    (maxSos, maxSos_params),
    (maxDiv, maxDiv_params),
    (algorithm_cma, cma_params),
    (algorithm_nsga2, nsga2_params),
    (algorithm_pso, pso_params)
]


print("====================================")
print("====== COMENZANDO EJECUCION ========")
print("====================================")

np.random.seed(0)
random.seed(0)

comparator = PortfolioComparator(assets, strategies)
df = comparator.compare()
comparator.save_portfolios()
df.to_csv('comparison_results.csv', index=False)
print()
print(df)

print("====================================")
print("====== EJECUCION ACABADA ===========")
print("====================================")
