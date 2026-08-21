"""Hamming distances, subsampling, and noise injection."""
import numpy as np
from scipy.spatial.distance import pdist, squareform


def hamming_matrix(binary):
    """Count of differing segregating sites for every pair."""
    if binary.shape[1] == 0:
        return np.zeros((binary.shape[0], binary.shape[0]))
    return squareform(pdist(binary, metric="hamming") * binary.shape[1])


def subsample(D, ratio, rng):
    n = D.shape[0]
    k = max(2, round(n * ratio))
    idx = rng.choice(n, size=k, replace=False)
    return D[np.ix_(idx, idx)]


def add_noise(D, sigma, rng):
    """Symmetric Gaussian noise that keeps D a valid distance matrix."""
    if sigma <= 0:
        return D.copy()
    n = D.shape[0]
    E = np.zeros((n, n))
    iu = np.triu_indices(n, k=1)
    E[iu] = rng.normal(0.0, sigma, size=iu[0].size)
    out = D + E + E.T
    np.clip(out, 0.0, None, out=out)
    np.fill_diagonal(out, 0.0)
    return out