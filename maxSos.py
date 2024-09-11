import numpy as np
from utils import get_data_by_index


def maxSos(sigma, mu, esg):
    (n, m) = esg.shape
    w_s = np.zeros(m)
    w_p = np.zeros(n)

    flat_index_max = np.argmax(esg)
    (i, j) = np.unravel_index(flat_index_max, esg.shape)

    w_p[i] = 1
    w_s[j] = 1

    return w_p, w_s


if __name__ == "__main__":

    for N in range(8):
        sigma, mu, esg = get_data_by_index(N)

        w_p, w_s = maxSos(sigma, mu, esg)

        return_ = w_p@mu
        risk = w_p@sigma@w_p
        esg_rating = w_p@esg@w_s
        print(f"===== Group {N} ========")
        print("esg matrix: \n", esg)
        print("ESG rating: ", esg_rating)
