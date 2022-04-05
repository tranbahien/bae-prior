import os
import importlib

def set_data_path(path):
    file = get_data_path_file()
    with open(file, 'w') as f:
        f.write(path)

DATA_PATH = "./datasets"