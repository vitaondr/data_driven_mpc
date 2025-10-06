import yaml
import pandas as pd
import numpy as np

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def get_path(file_path):
    path = pd.read_csv(file_path)
    return np.array(path)


def load_csv_file(file_path):
    data = pd.read_csv(file_path)
    return data
