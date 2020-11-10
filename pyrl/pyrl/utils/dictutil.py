from typing import List, Union
from flatten_dict import flatten, unflatten
import torch


def collate(dicts: List[dict]) -> dict:
    keys = flatten(dicts[0]).keys()
    new_dict = {}
    for k in keys:
        new_dict[k] = torch.cat([flatten(d)[k] for d in dicts])
    return unflatten(new_dict)
