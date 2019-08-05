# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import threading
import datetime as dt
import mozg.lib.math.ml.TrainingDataModel as tdm
import mozg.common.util.Log as log
from inspect import currentframe, getframeinfo
import mozg.lib.math.ml.ModelInterface as modelIf
import mozg.common.util.ObjectPersistence as objper


class Keras(modelIf.ModelInterface):

    def __init__(
            self,
            # Unique identifier to identify this set of trained data+other files after training
            identifier_string,
            # Directory to keep all our model files
            dir_path_model,
            # Training data in TrainingDataModel class type
            training_data = None,
            do_profiling = True
    ):
        super(Keras,self).__init__(
            identifier_string = identifier_string
        )
        self.identifier_string = identifier_string
        self.dir_path_model = dir_path_model
        self.training_data = training_data
        if self.training_data is not None:
            if type(self.training_data) is not tdm.TrainingDataModel:
                raise Exception(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Training data must be of type "' + str(tdm.TrainingDataModel.__class__)
                    + '", got type "' + str(type(self.training_data))
                    + '" instead from object ' + str(self.training_data) + '.'
                )

        self.do_profiling = do_profiling
        return

    def train(
            self
    ):
        (train_images, train_labels),(test_images, test_labels) = mnist.load_data()

        log.Log.info(
            'Data type train images "' + str(type(train_images)) + '" shape ' + str(train_images.shape)
            + ', train labels "' + str(type(train_labels)) + '", shape ' + str(train_labels.shape)
        )

        network = models.Sequential()
        network.add(
            layers.Dense(
                units=512,
                activation = 'relu',
                input_shape = (28*28,)
            )
        )
        network.add(
            layers.Dense(
                units=10,
                activation='softmax'
            )
        )

        network.compile(
            optimizer = 'rmsprop',
            loss      = 'categorical_crossentropy',
            metrics   = ['accuracy']
        )

        train_images = train_images.reshape((60000, 28*28))
        train_images = train_images.astype('float32') / 255

        test_images = test_images.reshape((10000, 28*28))
        test_images = test_images.astype('float32') / 255

        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        network.fit(
            train_images, train_labels, epochs=5, batch_size=128
        )

        self.persist_training_data_to_storage(network=network)
        return

    def persist_training_data_to_storage(
            self,
            network
    ):
        objper.ObjectPersistence.serialize_object_to_file(
            obj = network,
            obj_file_path  = self.dir_path_model,
            lock_file_path = None
        )

if __name__ == '__main__':
    kr = Keras(
        identifier_string = ''
    )