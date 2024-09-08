import matplotlib.pyplot as plt
import numpy as np
import sys

index = int(sys.argv[1])
algorithm = sys.argv[2]
pareto_front = np.load(f'./pareto_fronts/{algorithm}_pareto_front_{index}.npy')

x = pareto_front[:, 0]
y = pareto_front[:, 1]
z = pareto_front[:, 2]

fig = plt.figure(figsize=(12, 10))

# Retorno-riesgo-esg
ax = fig.add_subplot(221, projection='3d')
sc = ax.scatter(x, y, z, c='b', alpha=0.5, marker="o")
sc = ax.scatter(x, y, min(z), c='grey', marker="x")
sc = ax.scatter(x, 0, z, c='grey', marker="x")
sc = ax.scatter(0, y, z, c='grey', marker="x")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax.xaxis.set_label_coords(-0.5, 0.5)
ax.yaxis.set_label_coords(-0.5, 0.5)
ax.set_xlabel('Retorno')
ax.set_ylabel('Riesgo')
ax.set_zlabel('ESG')
ax.view_init(elev=20, azim=70)
plt.grid()

# Retorno-riesgo

ax = fig.add_subplot(222)
sc = ax.scatter(x, y, c=z, marker="o")
ax.set_xlabel('Retorno')
ax.set_ylabel('Riesgo')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('ESG')
plt.grid()

# Retorno-esg
ax = fig.add_subplot(223)
sc = ax.scatter(x, z, c=y, marker="o")
ax.set_xlabel('Retorno')
ax.set_ylabel('ESG')
cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
cbar.set_label('Riesgo')
plt.grid()

# Riesgo-esg
ax = fig.add_subplot(224)
sc = ax.scatter(y, z, c=x, marker="o")
ax.set_xlabel('Riesgo')
ax.set_ylabel('ESG')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Retorno')
plt.grid()

plt.tight_layout()
fig.savefig(f'./plots/pareto_front_{index}.png', dpi=300)
plt.show()
