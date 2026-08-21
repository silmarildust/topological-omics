"""Persistent homology and barcode summaries."""
import numpy as np
from ripser import ripser


def diagrams(D, maxdim=2):
    return ripser(D, maxdim=maxdim, distance_matrix=True)["dgms"]


def features(dgm):
    """(betti, mean barcode length, barcode length variance).

    Bars with infinite death or zero length are excluded. See methods.
    """
    if len(dgm) == 0:
        return 0, 0.0, 0.0
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return 0, 0.0, 0.0
    lengths = finite[:, 1] - finite[:, 0]
    lengths = lengths[lengths > 0]
    if len(lengths) == 0:
        return 0, 0.0, 0.0
    return len(lengths), float(lengths.mean()), float(lengths.var())