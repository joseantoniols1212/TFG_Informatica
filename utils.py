import numpy as np
import pandas as pd


DATAFILES_PATH = "./data/"
DATAFILE_PREFIX = "grupo_pos_"
RETURN_DATAFILE_EXTENSION = ".csv"
ESG_DATAFILE_EXTENSION = ".info"


def get_data(datafile_path: str):
    """"Returns risk and mean returns of a data file by path."""
    ret_data = pd.read_csv(datafile_path+RETURN_DATAFILE_EXTENSION)
    # Drop the first column containing date
    ret_data = ret_data.drop(columns=['Unnamed: 0'])
    risk_matrix = ret_data.cov()
    mean_returns = ret_data.mean()
    esg_data = pd.read_csv(datafile_path+".info")
    esg_data = esg_data.drop(esg_data.columns[[0, 1, 2, 3, 4]], axis=1)
    return risk_matrix.values, mean_returns.values, esg_data.values


def get_data_by_index(index: int):
    """Returns risk and mean returns of a data file by index."""
    datafile_path = DATAFILES_PATH+DATAFILE_PREFIX+str(index)
    ret_data = pd.read_csv(datafile_path+".csv")
    # Drop the first column containing date
    ret_data = ret_data.drop(columns=['Unnamed: 0'])
    esg_data = pd.read_csv(datafile_path+".info")
    esg_data = esg_data.drop(esg_data.columns[[0, 1, 2, 3, 4]], axis=1)
    risk_matrix = ret_data.cov()
    mean_returns = ret_data.mean()
    return risk_matrix.values, mean_returns.values, esg_data.values


def map_from_auxiliary_coordinates(x):
    z = 1 - np.sum(x)
    return np.append(x, z)


def map_to_auxiliary_coordinates(x):
    return x[:-1]


if __name__ == "__main__":
    risk, ret, esg = get_data_by_index(0)
    print("Risk: \n", risk)
    print("Return: \n", ret)
    print("ESG: \n", esg)
