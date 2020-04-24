# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import threading
import nwae.ml.TrainingDataModel as tdm
from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
import nwae.ml.ModelInterface as modelIf
import collections
import matplotlib.pyplot as plt
import numpy as np
try:
    from keras import Sequential
    from keras import models
    from keras import layers
    from keras.utils import to_categorical
    from tensorflow.keras.models import load_model
    # This one will not work in a multi-threaded environment
    #from keras.models import load_model
    from tensorflow import keras
except Exception as ex_keras:
    Log.warning(
        str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
        + ': Exception importing TensorFlow, Keras modules: ' + str(ex_keras)
    )


#
# Sequential data type models
#
# 모델 설명
#  The idea behind this NN is a bit different from the single label output.
#
class SequenceModel(modelIf.ModelInterface):
    MODEL_NAME = 'sequence_model'

    def __init__(
            self,
            # Unique identifier to identify this set of trained data+other files after training
            identifier_string,
            # Directory to keep all our model files
            dir_path_model,
            train_epochs,
            train_batch_size,
            train_optimizer     = 'rmsprop',
            train_loss          = 'categorical_crossentropy',
            evaluate_metrics    = ('accuracy'),
            # Training data in TrainingDataModel class type
            training_data       = None,
            do_profiling        = False,
            # Train only by y/labels and store model files in separate y_id directories
            is_partial_training = False
    ):
        super().__init__(
            model_name          = SequenceModel.MODEL_NAME,
            identifier_string   = identifier_string,
            dir_path_model      = dir_path_model,
            training_data       = training_data,
            is_partial_training = is_partial_training
        )

        self.identifier_string = identifier_string
        self.dir_path_model = dir_path_model
        self.train_epochs = train_epochs
        self.train_batch_size = train_batch_size
        self.train_optimizer = train_optimizer
        self.train_loss = train_loss
        self.evaluate_metrics = evaluate_metrics
        # Keras accepts only list/tuple type
        if type(self.evaluate_metrics) not in (list, tuple):
            self.evaluate_metrics = [self.evaluate_metrics]

        self.training_data = training_data
        self.is_partial_training = is_partial_training

        #
        # TODO: Add option to select our own model, or keras/tf model
        #
        self.lstm_model = self.__load_sample_model()

        if self.is_partial_training:
            # In this case training data must exist
            unique_y = list(set(list(self.training_data.get_y())))
            if len(unique_y) != 1:
                raise Exception(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': [' + str(self.identifier_string)
                    + '] In partial training mode, must only have 1 unique label, but found '
                    + str(unique_y) + '.'
                )
            self.y_id = int(unique_y[0])

        if self.training_data is not None:
            if type(self.training_data) is not tdm.TrainingDataModel:
                raise Exception(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Training data must be of type "' + str(tdm.TrainingDataModel.__class__)
                    + '", got type "' + str(type(self.training_data))
                    + '" instead from object ' + str(self.training_data) + '.'
                )
        # Only train some y/labels and store model files in separate directories by y_id
        self.is_partial_training = is_partial_training

        # Training variables
        self.bot_training_start_time = None
        self.bot_training_end_time = None
        self.is_training_done = False
        self.log_training = []
        self.__mutex_training = threading.Lock()

        return

    #
    # Test model only, not for production use
    #
    def __load_sample_model(
            self,
            embed_input_dim  = 1000,
            embed_output_dim = 64,
            embed_input_len  = 20,
            lstm_units = 128
    ):
        #
        # Layers Design
        #
        lstm_model = keras.Sequential()
        # Add an Embedding layer expecting input vocab of size 1000, and
        # output embedding dimension of size 64.
        lstm_model.add(
            keras.layers.Embedding(
                input_dim    = embed_input_dim,
                output_dim   = embed_output_dim,
                input_length = embed_input_len
            )
        )
        # Add a LSTM layer with 128 internal units.
        lstm_model.add(
            keras.layers.LSTM(
                lstm_units
            )
        )
        # Add a Dense layer with 10 units and softmax activation.
        lstm_model.add(
            keras.layers.Dense(10, activation='softmax')
        )

        # Finally compile the model
        lstm_model.compile(
            optimizer = 'rmsprop',
            loss      = 'categorical_crossentropy',
            metrics   = ['accuracy']
        )
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Model compiled successfully.'
        )

        lstm_model.summary()
        return lstm_model

    #
    # Model interface override
    #
    def is_model_ready(
            self
    ):
        raise Exception(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Template Model only, not a real model!'
        )

    #
    # Model interface override
    #
    def get_model_features(
            self
    ):
        raise Exception(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Template Model only, not a real model!'
        )

    #
    # Model interface override
    #
    def check_if_model_updated(
            self
    ):
        raise Exception(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Template Model only, not a real model!'
        )

    #
    # Model interface override
    #
    # Steps to predict classes
    #
    #  1. Weight by IDF and normalize input x
    #  2. Calculate Euclidean Distance of x to each of the x_ref (or rfv)
    #  3. Calculate Euclidean Distance of x to each of the x_clustered (or rfv)
    #  4. Normalize Euclidean Distance so that it is in the range [0,1]
    #
    def predict_classes(
            self,
            # ndarray type of >= 2 dimensions
            x
    ):
        raise Exception(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Template Model only, not a real model!'
        )

    #
    # Model interface override
    #
    def predict_class(
            self,
            # ndarray type of >= 2 dimensions, single point/row array
            x
    ):
        raise Exception(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Template Model only, not a real model!'
        )

    #
    # Train from partial model files
    #
    def train_from_partial_models(
            self,
            write_model_to_storage=True,
            write_training_data_to_storage=False,
            model_params=None,
            # Log training events
            logs=None
    ):
        raise Exception(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Template Model only, not a real model!'
        )

    #
    # Model interface override
    #
    def train(
            self,
            write_model_to_storage=True,
            write_training_data_to_storage=False,
            model_params=None,
            # Option to train a single y ID/label
            y_id=None
    ):
        #
        #
        #
        raise Exception(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Template Model only, not a real model!'
        )

    def persist_model_to_storage(
            self
    ):
        raise Exception(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Template Model only, not a real model!'
        )

    #
    # Model interface override
    #
    def load_model_parameters(
            self
    ):
        raise Exception(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Template Model only, not a real model!'
        )


if __name__ == '__main__':
    import nwae.config.Config as cf
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config,
        default_config_file = '/usr/local/git/nwae/nwae/app.data/config/default.cf'
    )

    import nwae.lang.nlp.Corpora as corpora
    ret = corpora.Corpora().build_data_set()
    data_set = ret.list_sent_pairs
    max_len = max(ret.max_len_l1, ret.max_len_l2)
    print('Prepared data set of length ' + str(len(data_set)))
    print('Max Length = ' + str(max_len))

    obj = SequenceModel(
        identifier_string = config.get_config(param=cf.Config.PARAM_MODEL_IDENTIFIER),
        dir_path_model    = config.get_config(param=cf.Config.PARAM_MODEL_DIR),
        training_data     = None,
        is_partial_training = False
    )
