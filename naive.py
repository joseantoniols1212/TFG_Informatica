import numpy as np
from utils import get_data_by_index


def naive(sigma, mu, esg):
    (n, m) = esg.shape
    w_p = np.ones(n) / n
    w_s = np.zeros(m)

    max_esg_index = np.argmax(w_p@esg)
    w_s[max_esg_index] = 1

    return w_p, w_s


if __name__ == "__main__":

    sigma, mu, esg = get_data_by_index(0)

    w_p, w_s = naive(sigma, mu, esg)

    return_ = w_p@mu
    risk = w_p@sigma@w_p
    esg_rating = w_p@esg@w_s

    print("Retorno: ", return_)
    print("Riesgo: ", risk)
    print("ESG rating: ", esg_rating)
    print("Norma w_p:", np.linalg.norm(w_p,1))
    print("Norma w_s:", np.linalg.norm(w_s,1))