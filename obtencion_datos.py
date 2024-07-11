import numpy as np
import pandas as pd

DATAFILES_PATH = "~/Universidad/TFG_Informatica/data/"
DATAFILE_PREFIX = "grupo_"
DATAFILE_EXTENSION = ".csv"

def datafile_path_from_group_index(index):
    return DATAFILES_PATH+DATAFILE_PREFIX+str(index)+DATAFILE_EXTENSION

def get_return(datafile_path: str):
    data = pd.read_csv(datafile_path)


datafile_path = datafile_path_from_group_index(0)
print(datafile_path)