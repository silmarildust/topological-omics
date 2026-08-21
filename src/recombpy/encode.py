"""FASTA -> biallelic 0/1 matrix."""
import numpy as np
from Bio import SeqIO

ALPHABET = ("A", "C", "G", "T")


def read_alignment(path):
    records = list(SeqIO.parse(str(path), "fasta"))
    arr = np.array([list(str(r.seq).upper()) for r in records], dtype="<U1")
    return arr, [r.id for r in records]


def to_binary(arr):
    """Keep segregating sites only. Major allele -> 0, all others -> 1."""
    n, L = arr.shape
    cols, kept = [], []
    for j in range(L):
        col = arr[:, j]
        if not np.isin(col, ALPHABET).all():
            continue                      # gap or ambiguity: drop the site
        vals, counts = np.unique(col, return_counts=True)
        if len(vals) == 1:
            continue                      # not segregating
        major = vals[np.argmax(counts)]
        cols.append((col != major).astype(np.uint8))
        kept.append(j)
    if not cols:
        return np.zeros((n, 0), np.uint8), np.array([], int)
    return np.column_stack(cols), np.array(kept)