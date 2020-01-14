# -*- coding: utf-8 -*-

import numpy as np
import keras.preprocessing as kerasprep
import keras.layers as keraslay
from keras.models import Sequential
from nwae.lib.lang.preprocessing.BasicPreprocessor import BasicPreprocessor


# Random vectors
data = np.random.random((1000, 5))
# Random labels
labels = np.random.randint(8, size=(1000, 1))

print(data.shape)
print(data[0:10])
print(labels.shape)
print(labels[0:10])
