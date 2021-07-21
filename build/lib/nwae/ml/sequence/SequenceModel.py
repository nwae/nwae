# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
from nwae.ml.ModelInterface import ModelInterface

# TODO Don't rely on buggy TF/Keras, write our own
try:
    from keras.utils import to_categorical
    from keras.layers import Input, Dense, LSTM
    from tensorflow.keras.models import load_model
    # This one will not work in a multi-threaded environment
    #from keras.models import load_model
except Exception as ex_keras:
    Log.warning(
        str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
        + ': Exception importing Keras modules: ' + str(ex_keras)
    )


#
# Empty template for implementing a new model
#
class Sequence(ModelInterface):

    MODEL_NAME = 'nn_sequence'

    CONFIDENCE_LEVEL_SCORES_DEFAULT = {1: 10, 2: 15, 3: 20, 4:30, 5:40}

    #
    # Overwrite base class if required
    #
    CONFIDENCE_LEVEL_5_SCORE = 50
    CONFIDENCE_LEVEL_4_SCORE = 40
    CONFIDENCE_LEVEL_3_SCORE = 30
    CONFIDENCE_LEVEL_2_SCORE = 20
    CONFIDENCE_LEVEL_1_SCORE = 10

    def __init__(
            self,
            # NN layer configurations, etc.
            model_params,
            # Unique identifier to identify this set of trained data+other files after training
            identifier_string,
            # Directory to keep all our model files
            dir_path_model,
            # Training data in TrainingDataModel class type
            training_data       = None,
            # Train only by y/labels and store model files in separate y_id directories
            is_partial_training = False,
            do_profiling        = False,
    ):
        super().__init__(
            model_name          = Sequence.MODEL_NAME,
            model_params        = model_params,
            identifier_string   = identifier_string,
            dir_path_model      = dir_path_model,
            training_data       = training_data,
            is_partial_training = is_partial_training,
            do_profiling        = do_profiling
        )
        return

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
    def predict_classes(
            self,
            # ndarray type of >= 2 dimensions
            x,
            include_match_details = False,
            top                   = ModelInterface.MATCH_TOP

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
            x,
            include_match_details = False,
            top                   = ModelInterface.MATCH_TOP

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
            write_model_to_storage = True,
            write_training_data_to_storage = False,
            model_params = None,
            # Log training events
            logs = None
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
            write_model_to_storage = True,
            write_training_data_to_storage = False,
            model_params = None,
            # Option to train a single y ID/label
            y_id = None
    ):
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
