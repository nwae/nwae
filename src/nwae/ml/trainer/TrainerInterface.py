# -*- coding: utf-8 -*-

import datetime as dt
import threading
from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo


#
# Base Trainer class to train data using given model
#
class TrainerInterface(threading.Thread):

    #
    # Model Training
    #
    # Train the entire model in one shot
    TRAIN_MODE_MODEL          = 'train_model'
    # In this case the training will loop by y_id and do each partial
    # training one by one, and write only to label specific training files.
    # The purpose is to do incremental training, thus fast
    TRAIN_MODE_MODEL_BY_LABEL = 'train_model_by_label'
    TRAIN_MODE_MODEL_USE_PARTIAL_MODELS = 'train_model_use_partial_models'
    #
    # NLP Training
    #
    TRAIN_MODE_NLP_EIDF = 'train_nlp_eidf'

    def __init__(
            self,
            identifier_string,
            # Where to keep training data model files
            dir_path_model,
            # Can be in TrainingDataModel type or pandas DataFrame type with 3 columns (Intent ID, Intent, Text Segmented)
            training_data,
            # If training data is None, must pass a training_data_source object with method fetch_data() implemented
            training_data_source = None,
            model_name           = None,
            model_params         = None,
            # Either 'train_model' (or None), or 'train_nlp_eidf', etc.
            train_mode           = TRAIN_MODE_MODEL,
            # Train a single y/label ID only, regardless of train mode
            y_id                 = None
    ):
        super().__init__()

        self.identifier_string = identifier_string
        self.dir_path_model = dir_path_model

        #
        # We allow training data to be None, as it may take time to fetch this data.
        # Thus we return this object quickly to caller (to check training logs, etc.).
        #
        self.is_training_data_ready = False
        self.training_data = training_data
        self.training_data_source = training_data_source

        if self.training_data is not None:
            self.is_training_data_ready = True
        else:
            if self.training_data_source is None:
                raise Exception(
                    str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                    + ': Data source must not be None if training data is None!'
                )

        self.model_name = model_name
        self.model_params = model_params

        self.train_mode = train_mode
        self.y_id = y_id

        #
        # TxtDataPreprocessor object passed back to us after fetching and preprocessing
        #
        self.df_training_data_pp = None

        self.__mutex_training = threading.Lock()
        self.bot_training_start_time = None
        self.bot_training_end_time = None
        self.is_training_done = False

        self.log_training = []
        return

    #
    # If training data is not in desired format, we do some conversion
    #
    def preprocess_training_data(
            self
    ):
        raise Exception(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Model "' + str(self.model_name) + '", identifier "' + str(self.identifier_string)
            + '". Must be implemented by derived class "' + str(self.__class__) + '"'
        )

    def train(self):
        raise Exception(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Model "' + str(self.model_name) + '", identifier "' + str(self.identifier_string)
            + '". Must be implemented by derived class "' + str(self.__class__) + '"'
        )

    def run(self):
        self.run_full_train_process()

    def run_full_train_process(self):
        try:
            self.__mutex_training.acquire()
            self.bot_training_start_time = dt.datetime.now()
            self.log_training = []
            self.preprocess_training_data()
            self.train()
        except Exception as ex:
            errmsg = str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Training Identifier ' + str(self.identifier_string) + '" training exception: ' + str(ex) + '.'
            Log.critical(errmsg)
            raise Exception(errmsg)
        finally:
            self.is_training_done = True
            self.bot_training_end_time = dt.datetime.now()
            self.__mutex_training.release()

        Log.important(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Train mode "' + str(self.train_mode)
            + '". Training Identifier ' + str(self.identifier_string) + '" trained successfully.'
        )
        return self.log_training
