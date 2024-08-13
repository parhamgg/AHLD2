
from scipy.sparse import csr_matrix#, save_npz, load_npz

from .plugin_setup import plugin
from ._format import ModmagsFormat
import numpy as np


# Transformer to convert ModmagsFormat to csr_matrix
# Modmags is saved as a csr_matrix
@plugin.register_transformer
def _1(ff: ModmagsFormat) -> csr_matrix:
    with ff.open() as fh:
        loader = np.load(fh, allow_pickle=True)
        matrix = csr_matrix((loader['data'],
                             loader['indices'],
                             loader['indptr']),
                             shape=loader['shape'])
    return matrix

# Transformer to convert csr_matrix to CSRMatrixFormat
@plugin.register_transformer
def _2(matrix: csr_matrix) -> ModmagsFormat:
    ff = ModmagsFormat()
    with ff.open() as fh:
        np.savez(fh,
                 data=matrix.data,
                 indices=matrix.indices,
                 indptr=matrix.indptr,
                 shape=matrix.shape)
    return ff