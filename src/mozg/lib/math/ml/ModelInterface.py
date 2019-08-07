# -*- coding: utf-8 -*-

import threading
import time
import mozg.common.util.Log as log
from inspect import currentframe, getframeinfo
import pandas as pd
import numpy as np
import mozg.lib.math.ml.TrainingDataModel as tdm


#
# Interfaces that a Model must implement
#
class ModelInterface(threading.Thread):

    # Terms for dataframe, etc.
    TERM_CLASS    = 'class'
    TERM_SCORE    = 'score'
    TERM_DIST     = 'dist'
    TERM_DISTNORM = 'distnorm'
    TERM_RADIUS   = 'radius'

    # Matching
    MATCH_TOP = 10

    class predict_class_retclass:
        def __init__(
                self,
                predicted_classes,
                top_class_distance = None,
                match_details = None
        ):
            self.predicted_classes = predicted_classes
            # The top class and shortest distances (so that we can calculate sum of squared error
            self.top_class_distance = top_class_distance
            self.match_details = match_details
            return

    def __init__(
            self,
            identifier_string,
            dir_path_model
    ):
        super(ModelInterface, self).__init__()

        self.identifier_string = identifier_string
        self.dir_path_model = dir_path_model

        self.stoprequest = threading.Event()

        self.__mutex_load_model = threading.Lock()

        # Training data for testing back only
        self.training_data = None
        prefix = self.dir_path_model + '/' + self.identifier_string
        self.fpath_training_data_x          = prefix + '.training_data.x.csv'
        self.fpath_training_data_x_name     = prefix + '.training_data.x_name.csv'
        self.fpath_training_data_y          = prefix + '.training_data.y.csv'

        return

    def join(self, timeout=None):
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Model Identifier "' + str(self.identifier_string) + '" join called..'
        )
        self.stoprequest.set()
        super(ModelInterface, self).join(timeout=timeout)
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Model Identifier "' + str(self.identifier_string) + '" Background Thread ended..'
        )

    def run(self):
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Model Identifier "' + str(self.identifier_string) + '" Background Thread started..'
        )
        sleep_time = 10
        while True:
            if self.stoprequest.isSet():
                log.Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Model Identifier "' + str(self.identifier_string) + '" Breaking from forever thread...'
                )
                break
            if self.check_if_model_updated():
                try:
                    self.__mutex_load_model.acquire()
                    self.load_model_parameters()
                    if not self.is_model_ready():
                        log.Log.important(
                            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': Model "' + self.identifier_string
                            + '" failed to load. Try again in ' + str(sleep_time) + ' secs..'
                        )
                finally:
                    self.__mutex_load_model.release()
            time.sleep(sleep_time)

    def get_model_features(
            self
    ):
        return None

    def predict_classes(
            self,
            # ndarray type of >= 2 dimensions
            x
    ):
        return

    def predict_class(
            self,
            # ndarray type of >= 2 dimensions, single point/row array
            x
    ):
        return

    def train(
            self
    ):
        return

    def persist_model_to_storage(
            self
    ):
        return

    def load_model_parameters(
            self
    ):
        return

    def is_model_ready(
            self
    ):
        return True

    def check_if_model_updated(
            self
    ):
        return False

    def persist_training_data_to_storage(
            self,
            td
    ):
        #
        # Write back training data to file, for testing back the model only, not needed for the model
        #
        df_td_x = pd.DataFrame(
            data    = td.get_x(),
            columns = td.get_x_name(),
            index   = td.get_y()
        )
        df_td_x.sort_index(inplace=True)
        df_td_x.to_csv(
            path_or_buf = self.fpath_training_data_x,
            index       = True,
            index_label = 'INDEX'
        )
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved Training Data x with shape ' + str(df_td_x.shape)
            + ' filepath "' + self.fpath_training_data_x + '"'
        )

        df_td_x_name = pd.DataFrame(td.get_x_name())
        df_td_x_name.to_csv(
            path_or_buf = self.fpath_training_data_x_name,
            index       = True,
            index_label = 'INDEX'
        )
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved Training Data x_name with shape ' + str(df_td_x_name.shape)
            + ' filepath "' + self.fpath_training_data_x_name + '"'
        )

        df_td_y = pd.DataFrame(
            data  = td.get_y_name(),
            index = td.get_y()
        )
        df_td_y.to_csv(
            path_or_buf = self.fpath_training_data_y,
            index       = True,
            index_label = 'INDEX'
        )
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved Training Data y with shape ' + str(df_td_y.shape)
            + ' filepath "' + self.fpath_training_data_y + '"'
        )
        return

    def load_training_data_from_storage(
            self
    ):
        try:
            df_td_x = pd.read_csv(
                filepath_or_buffer=self.fpath_training_data_x,
                sep=',',
                index_col='INDEX'
            )
            df_td_x_name = pd.read_csv(
                filepath_or_buffer=self.fpath_training_data_x_name,
                sep=',',
                index_col='INDEX'
            )
            df_td_y = pd.read_csv(
                filepath_or_buffer=self.fpath_training_data_y,
                sep=',',
                index_col='INDEX'
            )

            td = tdm.TrainingDataModel(
                x      = np.array(df_td_x.values),
                x_name = np.array(df_td_x_name.values).transpose()[0],
                # y is the index remember, the column is y_name
                y      = np.array(df_td_y.index),
                y_name = np.array(df_td_y.values).transpose()[0],
            )
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Training Data x read ' + str(df_td_x.shape) + ' shape'
                + ', x_name read ' + str(df_td_x_name.shape)
                + '\n\r' + str(td.get_x_name())
                + ', y read ' + str(df_td_y.shape)
                + '\n\r' + str(td.get_y())
            )
            self.training_data = td
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Load training data from file failed for identifier "' + self.identifier_string\
                     + '". Error msg "' + str(ex) + '".'
            log.Log.critical(errmsg)
            raise Exception(errmsg)
