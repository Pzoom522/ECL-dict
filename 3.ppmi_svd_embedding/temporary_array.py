
import os
import numpy as np

from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from scipy.sparse import save_npz, load_npz

import tempfile

class TemporaryArray:

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.file_name = tempfile.NamedTemporaryFile().name

        matrix = self.load(erase=True)
        matrix.flush()
        del matrix

    def load(self, erase=False, size=-1, start=0):
        if erase:
            return np.memmap(self.file_name, shape=self.shape, dtype=self.dtype, mode='w+')
        else:
            if size < 0:
                shape = self.shape
            else:
                shape = (size, self.shape[1])
            offset = start * self.shape[1] * self.dtype.itemsize
            return np.memmap(self.file_name, shape=shape, dtype=self.dtype, mode='r+', offset=offset)

    def close(self):
        os.remove(self.file_name)


class NamedSparseArray:

    def __init__(self):
        pass

    def new_one(shape, dtype):
        instance = NamedSparseArray()
        instance.shape = shape
        instance.dtype = dtype

        sshape = "_".join((str(e) for e in shape))
        sdtype = str(dtype)
        instance.name  = tempfile.NamedTemporaryFile().name + '_' + sshape + '_' + sdtype

        instance.m = dok_matrix(shape, dtype=dtype)
        instance.save()

        return instance

    def from_name(name):

        if not os.path.isfile(name + '.npz'):
            raise ValueError("File name '" + name + "' does not exists.")

        instance = NamedSparseArray()
        instance.name = name

        splitted = name.split('_')
        instance.dtype = np.dtype(splitted[-1])
        instance.shape = (int(splitted[-3]), int(splitted[-2]))

        instance.m = load_npz(instance.name + ".npz").todok()

        return instance

    def get_matrix(self):
        return self.m

    def save(self):
        save_npz(self.name + ".npz", self.m.tocoo(), compressed=True)

    def delete(self):
        os.remove(self.name + ".npz")
