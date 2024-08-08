import numpy as np


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
    pass
