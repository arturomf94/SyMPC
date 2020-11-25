import torch


def modulo(x, config):
    max_value = config.max_value
    min_value = config.min_value
    ring_size = config.ring_size

    mask_pos = x > max_value
    while mask_pos.any():
        mask_pos = mask_pos.long()
        x = x - (mask_pos * ring_size)
        mask_pos = x > max_value

    mask_neg = x < min_value
    while mask_neg.any():
        mask_neg = mask_neg.long()
        x = x + (mask_neg * ring_size)
        mask_neg = x < min_value

    return x
