# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import numpy as np


class TrainUtils:

    def __init__(self):
        return

    @staticmethod
    #
    # This function returns a Generator, thus using yield keyword
    #
    def mini_batcher(
            X, y, batch_size, shuffle
    ):
        assert len(X) == len(y)

        n_samples = len(X)

        if shuffle:
            idx = np.random.permutation(n_samples)
        else:
            idx = list(range(n_samples))

        for i in range(int(np.ceil(n_samples / batch_size))):
            j = i * batch_size
            k = (i + 1) * batch_size
            # yield instead of return so this function can be treated as a generator
            yield X[idx[j:k]], y[idx[j:k]]


if __name__ == '__main__':
    from nwae.ml.data.DataGenerator import DataGenerator
    n_samples = 99
    data_size = 8
    x = np.array(list(range(n_samples*data_size)))
    x = np.floor(x/data_size)
    x = np.reshape(x, newshape=(n_samples,data_size))
    y = DataGenerator.generate_seq(seq_length=n_samples, min_value=1, max_value=10)
    print(x.shape)
    print(y.shape)

    for mb in TrainUtils.mini_batcher(X=x, y=y, batch_size=5, shuffle=False):
        X_batch = mb[0]
        y_batch = mb[1]
        print(X_batch.shape, y_batch.shape)
        print(X_batch)
