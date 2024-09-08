import numpy as np
from algorithm_nsga2 import algorithm_nsga2
from algorithm_cma import algorithm_cma
from algorithm_pso import algorithm_pso
from utils import get_data_by_index
from metrics import Assets, PortfolioOptimizer, Metrics
from itertools import product
import random
import sys
import json

N = 7

assets_list = []
for i in range(N):
    assets_list.append(Assets(*get_data_by_index(i)))


def optimize_funtion(sharpe_ratio, wp_norm, ws_norm):
    return sharpe_ratio-100*abs(wp_norm-1)-100*abs(wp_norm-1)


# Función para imprimir la barra de progreso
def imprimir_barra_progreso(iteracion, total, current_best, longitud=50):
    progreso = iteracion / total
    bloque = int(longitud * progreso)
    barra = '█' * bloque + '-' * (longitud - bloque)
    porcentaje = progreso * 100
    sys.stdout.write(
        f'\r|{barra}| {porcentaje:.2f}%   *Current best: {current_best}')
    sys.stdout.flush()


cma_params_optimization = {
    "var": np.linspace(0.05, 0.5, 9),
    "penalization_weight": [1000],
    "ngen": [100]
}

pso_params_optimization = {
    "pop_size": [100],
    "abs_speed_max": np.linspace(0.1, 0.5, 5),
    "phi1": np.linspace(1.0, 2.0, 4),
    "phi2": np.linspace(1.0, 2.0, 4),
    "penalization_weight": [1000],
    "ngen": [100]
}

nsga2_params_optimization = {
    "pop_size": [100, 120, 140, 160, 180, 200],
    "penalization_weight": [1000],
    "ngen": [100]
}

random_seed = 0
ALGORITHM = 'nsga2'   # cma, pso, nsga2

if ALGORITHM == "cma":
    algorithm = algorithm_cma
    tested_params = cma_params_optimization
elif ALGORITHM == "pso":
    algorithm = algorithm_pso
    tested_params = pso_params_optimization
elif ALGORITHM == "nsga2":
    algorithm = algorithm_nsga2
    tested_params = nsga2_params_optimization
else:
    raise Exception("ALGORITHM NOT IMPLEMENTED")

print("====================================")
print("====== OPTIMIZANDO =================")
print("====================================")

params_names = tested_params.keys()
params_values = list(product(*tested_params.values()))

test_params = [dict(zip(params_names, combination))
               for combination in params_values]

print("NUMERO DE CONJUNTOS DE PARAMETROS A OPTIMIZAR: ", len(test_params))

best_params = []

best_fitness = 0

file_name = ALGORITHM + "_optimized_params.json"

for i, params in enumerate(test_params):
    fitness = 0
    np.random.seed(random_seed)
    random.seed(random_seed)
    for assets in assets_list:
        imprimir_barra_progreso(i, len(test_params), best_fitness)
        portfolio = PortfolioOptimizer(
            assets, algorithm, **params).optimize()
        metrics = Metrics(portfolio, assets)
        esg = metrics.esg_score()
        sharpe_ratio = metrics.sharpe_ratio()
        wp_norm = metrics.wp_norm()
        ws_norm = metrics.ws_norm()
        fitness += optimize_funtion(sharpe_ratio, wp_norm, ws_norm)
    best_params.append((params, fitness))
    best_fitness = max(best_fitness, fitness)
best_params.sort(reverse=True, key=lambda x: x[1])
with open(file_name, "w") as json_file:
    json.dump(best_params[:10], json_file)


print("\n====================================")
print("========== OPTIMIZADO ==============")
print("====================================")
