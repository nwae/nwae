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
        log.Log.info(
            'Training for data, x shape '  + str(self.training_data.get_x().shape)
            + ', train labels with shape ' + str(self.training_data.get_y().shape)
        )

        network = models.Sequential()
        network.add(
            layers.Dense(
                units       = 512,
                activation  = 'relu',
                input_shape = (self.training_data.get_x().shape[1],)
            )
        )

        n_labels = len(list(set(self.training_data.get_y().tolist())))
        log.Log.info('Total unique labels = ' + str(n_labels) + '.')

        network.add(
            layers.Dense(
                units      = n_labels,
                activation = 'softmax'
            )
        )

        network.compile(
            optimizer = 'rmsprop',
            loss      = 'categorical_crossentropy',
            metrics   = ['accuracy']
        )

        train_labels = to_categorical(self.training_data.get_y())

        network.fit(
            self.training_data.get_x(), train_labels, epochs=5, batch_size=128
        )

        self.persist_training_data_to_storage(network=network)
        return

    def persist_training_data_to_storage(
            self,
            network
    ):
        filepath = self.dir_path_model + '/' + self.identifier_string + '.model'
        objper.ObjectPersistence.serialize_object_to_file(
            obj = network,
            obj_file_path  = filepath,
            lock_file_path = None
        )
        log.Log.info(
            str(self.__class__) + str(getframeinfo(currentframe()).lineno)
            + ': Saved network to file "' + filepath + '".'
        )

if __name__ == '__main__':
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_INFO

    # Test data from MNIST
    print('Loading test data from MNIST..')
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    n_samples = train_images.shape[0]
    n_pixels = 1
    i = 1
    while i < train_images.ndim:
        n_pixels *= train_images.shape[i]
        i += 1

    print('Total pixels = ' + str(n_pixels))

    train_images = train_images.reshape((n_samples, n_pixels))
    train_images = train_images.astype('float32') / 255

    print('Using x with shape ' + str(train_images.shape) + ', and y with shape ' + str(train_labels.shape))

    td = tdm.TrainingDataModel(
        x = train_images,
        y = train_labels,
        is_map_points_to_hypersphere = False
    )

    kr = Keras(
        identifier_string = 'keras_image_bw_example',
        dir_path_model    = '/Users/mark.tan/git/mozg/app.data/models',
        training_data     = td
    )
    print('Training started...')
    kr.train()
    print('Training done.')
    exit(0)
