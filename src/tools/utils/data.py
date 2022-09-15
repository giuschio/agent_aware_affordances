"""
Base functions dealing with manipulation of files and data
"""
import collections.abc
import os
import yaml

from typing import List


def yaml_to_dict(yaml_path: str):
    with open(yaml_path) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        dictionary = yaml.load(file, Loader=yaml.FullLoader)
    return dictionary


def dict_to_yaml(dictionary: dict, save_path: str):
    with open(save_path, 'w') as file:
        yaml.dump(dictionary, file)


def update_dictionary(original, update):
    # copied from here https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    for k, v in update.items():
        if isinstance(v, collections.abc.Mapping):
            original[k] = update_dictionary(original.get(k, {}), v)
        else:
            original[k] = v
    return original


def get_objects_in_dataset(data_path: str) -> List:
    """
    Returns a list of folders inside the data_path directory
    """
    return os.listdir(data_path)


def get_job_folders(dirpath):
    """
    Given a path <dirpath>, get a list of directories in the path. Clean up the list by removing
    files and empty directories
    """
    # get points
    points = [os.path.relpath(path=x[0], start=dirpath) for x in os.walk(dirpath)]
    res = list()
    for p in points:
        # INIT FLAG
        flag = True
        fpath = os.path.join(dirpath, p)

        # CHECK CONDITIONS
        if 'job' not in str(p):
            flag = False

        if os.path.isdir(fpath):
            if len(os.listdir(fpath)) < 15:
                flag = False
        elif os.path.isfile(fpath):
            # remove if a file
            flag = False

        # FINALLY
        if flag:
            res.append(p)

    return res
