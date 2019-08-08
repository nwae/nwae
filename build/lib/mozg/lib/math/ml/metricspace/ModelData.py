# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import json
import datetime as dt
import os
import threading
import mozg.lib.math.ml.TrainingDataModel as tdm
import mozg.utils.Log as log
from inspect import currentframe, getframeinfo
import mozg.lib.math.Constants as const
import mozg.lib.math.NumpyUtil as npUtil


class ModelData:

    #
    # Sometimes when our dataframe index is in non-string format, they pose an inconsistency
    # and causes problems, so we standardize all index to string type
    #
    CONVERT_DATAFRAME_INDEX_TO_STR = False

    def __init__(
            self,
            # Unique identifier to identify this set of trained data+other files after training
            identifier_string,
            # Directory to keep all our model files
            dir_path_model,
    ):
        self.identifier_string = identifier_string
        self.dir_path_model = dir_path_model

        # Unique classes from y
        self.classes_unique = None

        # Model is loaded or not
        self.model_loaded = False
        self.model_updated_time = None
        self.__model_mutex = threading.Lock()

        #
        # RFVs
        # Original x, y, x_name in self.training_data
        # All np array type unless stated
        #
        # Order follows x_name
        # IDF np array at least 2 dimensional
        # Weights (all 1's by default)
        self.idf = None
        # Make 2D
        self.idf = np.array([self.idf])
        # numpy array, represents a class of y in a single array
        self.x_ref = None
        self.y_ref = None
        # Data Frame type, radius of y_ref
        self.df_y_ref_radius = None
        # Represents a class of y in a few clustered arrays
        self.x_clustered = None
        self.y_clustered = None
        # Data Frame type. Radius of each cluster
        self.y_clustered_radius = None
        # x_name column names np array at least 2 dimensional
        self.x_name = None

        # Unique y (or the unique classes)
        self.y_unique = None

        # First check the existence of the files
        prefix = self.dir_path_model + '/' + self.identifier_string
        self.fpath_updated_file        = prefix + '.lastupdated.txt'
        self.fpath_x_name              = prefix + '.x_name.csv'
        self.fpath_idf                 = prefix + '.idf.csv'
        # y not recorded separately, it is in the index of this dataframe csv
        self.fpath_x_ref               = prefix + '.x_ref.csv'
        self.fpath_x_ref_friendly_json = prefix + '.x_ref_friendly.json'
        # Only for debugging file
        self.fpath_x_ref_friendly_txt  = prefix + '.x_ref_friendly.txt'
        self.fpath_y_ref_radius        = prefix + '.y_ref.radius.csv'
        # y_ref not recorded separately, it is in the index of this dataframe csv
        self.fpath_x_clustered         = prefix + '.x_clustered.csv'
        self.fpath_y_clustered_radius  = prefix + '.y_clustered.radius.csv'
        # Only for debugging file
        self.fpath_x_clustered_friendly_txt = prefix + '.x_clustered_friendly.txt'
        # Training data for testing back only
        self.fpath_training_data_x          = prefix + '.training_data.x.csv'
        self.fpath_training_data_x_friendly = prefix + '.training_data.x_friendly.csv'
        self.fpath_training_data_x_name     = prefix + '.training_data.x_name.csv'
        self.fpath_training_data_y          = prefix + '.training_data.y.csv'

        self.log_training = []
        return

    def is_model_ready(self):
        return self.model_loaded

    def check_if_model_updated(self):
        updated_time = os.path.getmtime(self.fpath_updated_file)
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Model identifier "' + str(self.identifier_string)
            + '" last updated time ' + str(self.model_updated_time)
            + ', updated "' + str(updated_time) + '".'
        )
        if (updated_time > self.model_updated_time):
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Model update time for identifier "' + str(self.identifier_string) + '" - "'
                + str(dt.datetime.fromtimestamp(updated_time)) + '" is newer than "'
                + str(dt.datetime.fromtimestamp(self.model_updated_time))
                + '". Reloading model...'
            )
            try:
                self.__model_mutex.acquire()
                # Reset model flags to not ready
                self.model_loaded = False
                self.model_updated_time = updated_time
            finally:
                self.__model_mutex.release()
            return True
        else:
            return False

    def persist_model_to_storage(
            self
    ):
        self.log_training = []

        # Sort
        df_x_name = pd.DataFrame(data=self.x_name)

        idf_1d = npUtil.NumpyUtil.convert_dimension(arr=self.idf, to_dim=1)
        df_idf = pd.DataFrame(
            data  = npUtil.NumpyUtil.convert_dimension(arr=self.idf, to_dim=1),
            index = self.x_name
        )

        # We use this training data model class to get the friendly representation of the RFV
        xy = tdm.TrainingDataModel(
            x      = self.x_ref,
            y      = self.y_ref,
            x_name = self.x_name
        )
        x_ref_friendly = xy.get_print_friendly_x()

        df_x_ref = pd.DataFrame(
            data    = self.x_ref,
            index   = self.y_ref,
            columns = self.x_name
        ).sort_index()
        self.df_y_ref_radius = self.df_y_ref_radius.sort_index()

        df_x_clustered = pd.DataFrame(
            data    = self.x_clustered,
            index   = self.y_clustered,
            columns = self.x_name
        )

        df_y_clustered_radius = pd.DataFrame(
            data    = self.y_clustered_radius,
            index   = self.y_clustered
        )

        # We use this training data model class to get the friendly representation of the x_clustered
        xy_x_clustered = tdm.TrainingDataModel(
            x = np.array(self.x_clustered),
            y = np.array(self.y_clustered),
            x_name = self.x_name
        )
        x_clustered_friendly = xy_x_clustered.get_print_friendly_x()

        log.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tx_name:\n\r' + str(df_x_name)
            + '\n\r\tIDF:\n\r' + str(df_idf)
            + '\n\r\tRFV:\n\r' + str(df_x_ref)
            + '\n\r\tRFV friendly:\n\r' + str(x_ref_friendly)
            + '\n\r\tRFV Radius:\n\r' + str(self.df_y_ref_radius)
            + '\n\r\tx clustered:\n\r' + str(df_x_clustered)
            + '\n\r\tx clustered friendly:\n\r' + str(x_clustered_friendly)
        )

        #
        # Save to file
        # TODO: This needs to be saved to DB, not file
        #
        df_x_name.to_csv(path_or_buf=self.fpath_x_name, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved x_name shape ' + str(df_x_name.shape) + ', filepath "' + self.fpath_x_name + ']'
            , log_list = self.log_training
        )

        df_idf.to_csv(path_or_buf=self.fpath_idf, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved IDF dimensions ' + str(df_idf.shape) + ' filepath "' + self.fpath_idf + '"'
            , log_list = self.log_training
        )

        df_x_ref.to_csv(path_or_buf=self.fpath_x_ref, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved RFV dimensions ' + str(df_x_ref.shape) + ' filepath "' + self.fpath_x_ref + '"'
            , log_list = self.log_training
        )

        try:
            # This file only for debugging
            f = open(file=self.fpath_x_ref_friendly_txt, mode='w', encoding='utf-8')
            for i in x_ref_friendly.keys():
                line = str(x_ref_friendly[i])
                f.write(str(line) + '\n\r')
            f.close()

            with open(self.fpath_x_ref_friendly_json, 'w', encoding='utf-8') as f:
                json.dump(x_ref_friendly, f, indent=2)
            f.close()
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Saved x_ref friendly ' + str(x_ref_friendly) +  ' to file "' + self.fpath_x_ref_friendly_json + '".'
                , log_list=self.log_training
            )
        except Exception as ex:
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Could not create x_ref friendly file "' + self.fpath_x_ref_friendly_json
                + '". ' + str(ex)
            , log_list = self.log_training
            )

        self.df_y_ref_radius.to_csv(path_or_buf=self.fpath_y_ref_radius, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved RFV radius with dimensions ' + str(self.df_y_ref_radius.shape)
            + ' filepath "' + self.fpath_y_ref_radius + '"'
            , log_list = self.log_training
        )

        df_x_clustered.to_csv(path_or_buf=self.fpath_x_clustered, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved Clustered x with shape ' + str(df_x_clustered.shape) + ' filepath "' + self.fpath_x_clustered + '"'
            , log_list=self.log_training
        )

        df_y_clustered_radius.to_csv(path_or_buf=self.fpath_y_clustered_radius, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved y_clustered_radius with shape ' + str(df_y_clustered_radius.shape)
            + ' filepath "' + self.fpath_y_clustered_radius + '"'
            , log_list=self.log_training
        )

        # This file only for debugging
        try:
            # This file only for debugging
            f = open(file=self.fpath_x_clustered_friendly_txt, mode='w', encoding='utf-8')
            for i in x_clustered_friendly.keys():
                line = str(x_clustered_friendly[i])
                f.write(str(line) + '\n\r')
            f.close()
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Saved x_clustered friendly with keys ' + str(x_clustered_friendly.keys())
                + ' filepath "' + self.fpath_x_clustered_friendly_txt + '"'
                , log_list=self.log_training
            )
        except Exception as ex:
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Could not create x_clustered friendly file "' + self.fpath_x_clustered_friendly_txt
                + '". ' + str(ex)
            , log_list = self.log_training
            )

        # Our servers look to this file to see if RFV has changed
        # It is important to do it last (and fast), after everything is done
        try:
            f = open(file=self.fpath_updated_file, mode='w')
            f.write(str(dt.datetime.now()))
            f.write('\n\r')
            f.close()
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Saved update time file "' + self.fpath_updated_file
                + '" for other processes to detect and restart.'
                , log_list=self.log_training
            )
        except Exception as ex:
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Could not create last updated file "' + self.fpath_updated_file
                + '". ' + str(ex)
            , log_list = self.log_training
            )

    def load_model_parameters_from_storage(
            self
    ):
        # First check the existence of the files
        if not os.path.isfile(self.fpath_updated_file):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Last update file "' + self.fpath_updated_file + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)
        # Keep checking time stamp of this file for changes
        self.last_updated_time_rfv = 0

        #
        # We explicitly put a '_ro' postfix to indicate read only, and should never be changed during the program
        #
        if not os.path.isfile(self.fpath_x_name):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': x_name file "' + self.fpath_x_name + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)

        #
        # We explicitly put a '_ro' postfix to indicate read only, and should never be changed during the program
        #
        if not os.path.isfile(self.fpath_idf):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': IDF file "' + self.fpath_idf + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)

        if not os.path.isfile(self.fpath_x_ref):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': RFV file "' + self.fpath_x_ref + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)

        if not os.path.isfile(self.fpath_x_ref_friendly_json):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': RFV friendly file "' + self.fpath_x_ref_friendly_json + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)

        if not os.path.isfile(self.fpath_y_ref_radius):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': RFV furthest distance file "' + self.fpath_y_ref_radius + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)

        if not os.path.isfile(self.fpath_x_clustered):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': x clustered file "' + self.fpath_x_clustered + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)

        try:
            df_x_name = pd.read_csv(
                filepath_or_buffer = self.fpath_x_name,
                sep       =',',
                index_col = 'INDEX'
            )
            if ModelData.CONVERT_DATAFRAME_INDEX_TO_STR:
                # Convert Index column to string
                df_x_name.index = df_x_name.index.astype(str)
            self.x_name = np.array(df_x_name[df_x_name.columns[0]])
            # Standardize to at least 2-dimensional
            if self.x_name.ndim == 1:
                self.x_name = np.array([self.x_name])

            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': x_name Data: Read ' + str(df_x_name.shape[0]) + ' lines'
                + '\n\r' + str(self.x_name)
            )

            df_idf = pd.read_csv(
                filepath_or_buffer = self.fpath_idf,
                sep       =',',
                index_col = 'INDEX'
            )
            if ModelData.CONVERT_DATAFRAME_INDEX_TO_STR:
                # Convert Index column to string
                df_idf.index = df_idf.index.astype(str)
            self.idf = np.array(df_idf[df_idf.columns[0]])
            if self.idf.ndim == 1:
                self.idf = np.array([self.idf])
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': IDF Data: Read ' + str(df_idf.shape[0]) + ' lines'
                + '\n\r' + str(self.idf)
            )

            df_x_ref = pd.read_csv(
                filepath_or_buffer = self.fpath_x_ref,
                sep       = ',',
                index_col = 'INDEX'
            )
            if ModelData.CONVERT_DATAFRAME_INDEX_TO_STR:
                # Convert Index column to string
                df_x_ref.index = df_x_ref.index.astype(str)
            # Cached the numpy array
            self.y_ref = np.array(df_x_ref.index)
            self.x_ref = np.array(df_x_ref.values)
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': RFV x read ' + str(df_x_ref.shape[0]) + ' lines: '
                + '\n\r' + str(self.x_ref)
                + '\n\rRFV y' + str(self.y_ref)
            )

            self.df_y_ref_radius = pd.read_csv(
                filepath_or_buffer = self.fpath_y_ref_radius,
                sep       = ',',
                index_col = 'INDEX'
            )
            if ModelData.CONVERT_DATAFRAME_INDEX_TO_STR:
                # Convert Index column to string
                self.df_y_ref_radius.index = self.df_y_ref_radius.index.astype(str)
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': RFV Radius Data: Read ' + str(self.df_y_ref_radius.shape[0]) + ' lines'
                + '\n\r' + str(self.df_y_ref_radius)
            )

            df_x_clustered = pd.read_csv(
                filepath_or_buffer = self.fpath_x_clustered,
                sep       = ',',
                index_col = 'INDEX'
            )
            if ModelData.CONVERT_DATAFRAME_INDEX_TO_STR:
                # Convert Index column to string
                df_x_clustered.index = df_x_clustered.index.astype(str)
            self.y_clustered = np.array(df_x_clustered.index)
            self.x_clustered = np.array(df_x_clustered.values)
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': x clustered data: Read ' + str(df_x_clustered.shape[0]) + ' lines\n\r'
                + '\n\r' + str(self.x_clustered)
                + '\n\ry_clustered:\n\r' + str(self.y_clustered)
            )
            self.sanity_check()

            self.model_loaded = True
            self.model_updated_time = os.path.getmtime(self.fpath_updated_file)
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Load RFV from file failed for identifier "' + self.identifier_string\
                     + '". Error msg "' + str(ex) + '".'
            log.Log.critical(errmsg)
            raise Exception(errmsg)

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
            , log_list=self.log_training
        )

        try:
            x_friendly = td.get_print_friendly_x()

            # This file only for debugging
            f = open(file=self.fpath_training_data_x_friendly, mode='w', encoding='utf-8')
            for i in x_friendly.keys():
                line = str(x_friendly[i])
                f.write(str(line) + '\n\r')
            f.close()
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Saved training data x friendly ' + str(x_friendly)
                +  ' to file "' + self.fpath_training_data_x_friendly + '".'
                , log_list=self.log_training
            )
        except Exception as ex:
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Could not create x_ref friendly file "' + self.fpath_x_ref_friendly_json
                + '". ' + str(ex)
            , log_list = self.log_training
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
            , log_list=self.log_training
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
            , log_list=self.log_training
        )

        return

    def load_training_data_from_storage(
            self,
            is_convert_y_label_to_str_type = False
    ):
        try:
            self.__model_mutex.acquire()

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
                is_convert_y_label_to_str_type = is_convert_y_label_to_str_type
            )
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Training Data x read ' + str(df_td_x.shape) + ' shape'
                + ', x_name read ' + str(df_td_x_name.shape)
                + '\n\r' + str(td.get_x_name())
                + ', y read ' + str(df_td_y.shape)
                + '\n\r' + str(td.get_y())
            )
            return td
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Load training data from file failed for identifier "' + self.identifier_string\
                     + '". Error msg "' + str(ex) + '".'
            log.Log.critical(errmsg)
            raise Exception(errmsg)
        finally:
            self.__model_mutex.release()

    def sanity_check(self):
        # Check RFV is normalized
        for i in range(0,self.x_ref.shape[0],1):
            cs = self.y_ref[i]
            rfv = self.x_ref[i]
            if len(rfv.shape) != 1:
                raise Exception(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': RFV vector must be 1-dimensional, got ' + str(len(rfv.shape))
                    + '-dimensional for class ' + str(cs)
                )
            dist = np.sum(np.multiply(rfv,rfv))**0.5
            if abs(dist-1) > const.Constants.SMALL_VALUE:
                log.Log.critical(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Warning: RFV for command [' + str(cs) + '] not 1, ' + str(dist)
                )
                raise Exception('RFV error')

        for i in range(0,self.x_clustered.shape[0],1):
            cs = self.y_clustered[i]
            fv = self.x_clustered[i]
            dist = np.sum(np.multiply(fv,fv))**0.5
            if abs(dist-1) > const.Constants.SMALL_VALUE:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': Warning: x fv error for class "' + str(cs)\
                         + '" at index ' + str(i) + ' not 1, ' + str(dist)
                log.Log.critical(errmsg)
                raise Exception(errmsg)
        return

