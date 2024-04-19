import cma
import numpy as np
import utils
import matplotlib.pyplot as plt

# Primera fase

# Leemos los datos
N, mean_returns, std_deviations, correlations = utils.parse_data("./data/port1.txt")
MU = np.atleast_2d(mean_returns).T
SIGMA = correlations

# Leemos solucion
sol_returns, sol_variance = utils.parse_solution("./data/portef1.txt")

# Hiperparametros del modelo
GAMMA = 0.05                 # Ratio

# Hiperparametros estrategia
DT = 5.0                    # Desviación típica
CENTROID = np.random.rand(N)
CENTROID = CENTROID / np.linalg.norm(CENTROID, 1)  # Tomamos como punto inicial un punto aleatorio de la superficie

print(np.linalg.norm(CENTROID, 1))

# Definimos la matriz
H = GAMMA * SIGMA - MU@MU.T

# Definir la función objetivo
def objective_function(x):
    w = np.atleast_2d(x).T
    return (w.T@H@w)[0,0]

# Definir la función de restricción de igualdad
def equality_constraint(x):
    return np.linalg.norm(x, 1) - 1

def positivity_constraint(x):
    print(np.any(x < 0))
    if np.any(x < 0):
        return 1
    return 0

# Definir la función wrapper que incluye la restricción de igualdad
def fitness_function(x):
    penalty = abs(equality_constraint(x))**2 + 1e20 * positivity_constraint(x)  # Penalización grande si la restricción no se cumple
    return objective_function(x) + penalty

# Configurar el algoritmo CMA-ES
es = cma.CMAEvolutionStrategy(x0=CENTROID, sigma0=DT)

# Ejecutar la optimización
es.optimize(fitness_function)

# Imprimir la mejor solución encontrada
best_sol = np.atleast_2d(es.result.xbest).T
return_ = best_sol.T@MU
variance = best_sol.T@SIGMA@best_sol
init_return = CENTROID
print("Mejor individuo: ", best_sol)
print("Retorno medio: ", return_)
print("Varianza: ", variance)
print("Objetivo: ", fitness_function(es.result.xbest))
print("Norma: ", np.linalg.norm(es.result.xbest,1))

# Representamos los datos
scale = 1e2
plt.figure(figsize=(8, 6))
plt.scatter(return_*scale, variance*scale, color='g', marker='x')
plt.scatter(return_*scale, variance*scale, color='b', marker='x')
plt.scatter(sol_returns*scale, sol_variance*scale, color='r', marker='o')
plt.title('Mean Return vs Variance of Return')
plt.xlabel('Variance of Return')
plt.ylabel('Mean Return')
plt.grid(True)
plt.show()