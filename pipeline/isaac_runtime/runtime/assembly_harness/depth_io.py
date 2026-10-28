"""Lossless depth persistence; read both legacy NPY and compressed NPY."""
import gzip
from pathlib import Path
import numpy as np

def save_depth(path, array):
    path = Path(path)
    if path.suffix != '.gz':
        path = Path(str(path) + '.gz')
    with gzip.open(path, 'wb', compresslevel=1) as stream:
        np.save(stream, array, allow_pickle=False)
    return path

def load_depth(path):
    path = Path(path)
    if not path.exists() and Path(str(path) + '.gz').exists():
        path = Path(str(path) + '.gz')
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as stream:
            return np.load(stream, allow_pickle=False)
    return np.load(path, allow_pickle=False)
