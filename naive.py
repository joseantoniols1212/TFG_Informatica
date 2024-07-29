import numpy as np
from utils import get_data_by_index


def naive(index):
    sigma, mu, esg = get_data_by_index(0)
    n = mu.size
    x = np.ones(n) / n
    risk = x.T @ sigma @ x
    return_ = x.T @ mu
    return risk, return_


if __name__ == "__main__":
    risk, return_ = naive(0)
    print("Riesgo: ", risk)
    print("Retorno: ", return_)
