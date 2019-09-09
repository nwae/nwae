# -*- coding: utf-8 -*-

import threading
import nwae.lib.math.ml.TrainingDataModel as tdm
import nwae.utils.Log as log
from inspect import currentframe, getframeinfo
import nwae.lib.math.ml.ModelInterface as modelIf


#
# Sequential data (language translation, chatbot, speech recognition) type of models
#
# Model Description
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
            # Training data in TrainingDataModel class type
            training_data,
            # Train only by y/labels and store model files in separate y_id directories
            is_partial_training
    ):
        super(SequenceModel, self).__init__(
            model_name          = SequenceModel.MODEL_NAME,
            identifier_string   = identifier_string,
            dir_path_model      = dir_path_model,
            training_data       = training_data,
            is_partial_training = is_partial_training
        )

        self.identifier_string = identifier_string
        self.dir_path_model = dir_path_model
        self.training_data = training_data
        self.is_partial_training = is_partial_training

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
