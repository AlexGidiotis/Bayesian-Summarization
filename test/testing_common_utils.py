import random

import torch

global_rng = random.Random()


def ids_tensor(shape, vocab_size, rng=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long, device="cpu").view(shape).contiguous()


def floats_tensor(shape, scale=1.0, rng=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return torch.tensor(data=values, dtype=torch.float, device="cpu").view(shape).contiguous()


def values_tensor(values):
    """Creates a tensor from a python list"""
    ids = torch.tensor(data=values, dtype=torch.long, device="cpu")

    return ids.view(ids.shape).contiguous()
