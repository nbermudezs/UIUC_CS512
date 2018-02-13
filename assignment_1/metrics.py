import numpy as np


def bottom_k(scores, phrases, k=30):
    order = np.argsort(scores)
    return phrases[order[:k]]


def mid_k(scores, phrases, k=30):
    order = np.argsort(scores)
    mid = len(scores) // 2
    start = mid - k // 2
    return phrases[order[start:start + k]]


def top_k(scores, phrases, k=30):
    order = np.argsort(scores)
    return phrases[order[-k:]]
