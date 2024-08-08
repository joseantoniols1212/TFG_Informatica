import numpy as np


def naive(sigma, mu, esg):
    (n, m) = esg.shape
    w_p = np.ones(n) / n
    w_s = np.zeros(m)

    max_esg_index = np.argmax(w_p@esg)
    w_s[max_esg_index] = 1

    return w_p, w_s
