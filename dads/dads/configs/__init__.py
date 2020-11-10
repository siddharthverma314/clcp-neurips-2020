from pathlib import Path


def get_config_path(name):
    return Path(__file__).absolute().parent / "{}.txt".format(name)
