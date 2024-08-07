import numpy as np
from utils import get_data_by_index

def maxReturn(sigma, mu, esg):
    (n, m) = esg.shape

    w_p = np.zeros(n)
    w_s = np.zeros(m)

    max_return_index = np.argmax(mu)
    w_p[max_return_index] = 1

    max_esg_index = np.argmax(w_p@esg)
    w_s[max_esg_index] = 1

    return w_p, w_s

if __name__ == "__main__":

    sigma, mu, esg = get_data_by_index(0)

    w_p, w_s = maxReturn(sigma, mu, esg)

    return_ = w_p@mu
    risk = w_p@sigma@w_p
    esg_rating = w_p@esg@w_s

    print("Retorno: ", return_)
    print("Riesgo: ", risk)
    print("ESG rating: ", esg_rating)
    print("Norma w_p:", np.linalg.norm(w_p,1))
    print("Norma w_s:", np.linalg.norm(w_s,1))