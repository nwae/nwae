# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import json
import datetime as dt
import threading
import nwae.ml.TrainingDataModel as tdm
import nwae.ml.ModelInterface as modelIf
import nwae.utils.Log as log
from inspect import currentframe, getframeinfo
import nwae.math.Constants as const
import nwae.math.NumpyUtil as npUtil
import os
import re
from nwae.lang.model.FeatureVect import FeatureVector


class ModelData:

    #
    # Sometimes when our dataframe index is in non-string format, they pose an inconsistency
    # and causes problems, so we standardize all index to string type
    #
    CONVERT_DATAFRAME_INDEX_TO_STR = False

    MODEL_FILES_X_REF_FRIENDLY_TXT_POSTFIX        = '.x_ref_friendly.txt'
    MODEL_FILES_X_REF_FRIENDLY_JSON_POSTFIX       = '.x_ref_friendly.json'
    MODEL_FILES_X_CLUSTERED_FRIENDLY_TXT_POSTFIX  = '.x_clustered_friendly.txt'
    MODEL_FILES_X_CLUSTERED_FRIENDLY_JSON_POSTFIX = '.x_clustered_friendly.json'

    def __init__(
            self,
            model_name,
            # Unique identifier to identify this set of trained data+other files after training
            identifier_string,
            # Directory to keep all our model files
            dir_path_model,
            # Train only by some y/labels and keep model files in separate y_id directories
            is_partial_training,
            y_id = None
    ):
        self.model_name = model_name
        self.identifier_string = identifier_string
        self.dir_path_model = dir_path_model
        self.is_partial_training = is_partial_training
        self.y_id = y_id

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
        self.model_path_prefix = modelIf.ModelInterface.get_model_file_prefix(
            dir_path_model      = self.dir_path_model,
            model_name          = self.model_name,
            identifier_string   = self.identifier_string,
            is_partial_training = self.is_partial_training
        )
        if self.is_partial_training:
            if type(self.y_id) not in (int, str):
                raise Exception(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Cannot do partial training without y_id! Got y_id: ' + str(self.y_id)
                    + ' as type "' + str(type(self.y_id)) + '".'
                )
            self.model_path_prefix = self.model_path_prefix + '/' + str(self.y_id)

        self.fpath_updated_file        = self.model_path_prefix + '.lastupdated.txt'
        self.fpath_x_name              = self.model_path_prefix + '.x_name.csv'
        self.fpath_idf                 = self.model_path_prefix + '.idf.csv'
        # y not recorded separately, it is in the index of this dataframe csv
        self.fpath_x_ref               = self.model_path_prefix + '.x_ref.csv'
        # Only for debugging file
        self.fpath_y_ref_radius        = self.model_path_prefix + '.y_ref.radius.csv'
        # y_ref not recorded separately, it is in the index of this dataframe csv
        self.fpath_x_clustered         = self.model_path_prefix + '.x_clustered.csv'
        self.fpath_y_clustered_radius  = self.model_path_prefix + '.y_clustered.radius.csv'
        #
        # Used for loading back from partial training, which is easy to load (JSON), and verify (TXT)
        # Already contains both x_clustered and y_clustered
        #
        self.fpath_x_ref_friendly_txt        = self.model_path_prefix +\
                                               ModelData.MODEL_FILES_X_REF_FRIENDLY_TXT_POSTFIX
        self.fpath_x_ref_friendly_json       = self.model_path_prefix +\
                                               ModelData.MODEL_FILES_X_REF_FRIENDLY_JSON_POSTFIX
        self.fpath_x_clustered_friendly_txt  = self.model_path_prefix +\
                                               ModelData.MODEL_FILES_X_CLUSTERED_FRIENDLY_TXT_POSTFIX
        self.fpath_x_clustered_friendly_json = self.model_path_prefix +\
                                               ModelData.MODEL_FILES_X_CLUSTERED_FRIENDLY_JSON_POSTFIX
        return

    def is_model_ready(self):
        return self.model_loaded

    def check_if_model_updated(self):
        updated_time = os.path.getmtime(self.fpath_updated_file)
        log.Log.debugdebug(
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

    #
    # TODO Write to .tmp files first, then read back, if ok, then move the .tmp file to desired file name
    #
    def persist_model_to_storage(
            self,
            log_training = None
    ):
        # Sort
        df_x_name = pd.DataFrame(data=self.x_name)

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

        log.Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tx_name:\n\r' + str(df_x_name)
            + '\n\r\tEIDF:\n\r' + str(df_idf)
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
        if not self.is_partial_training:
            modelIf.ModelInterface.safe_dataframe_write(
                df            = df_x_name,
                name_df       = 'x_name',
                include_index = True,
                index_label   = 'INDEX',
                filepath      = self.fpath_x_name,
                log_training  = log_training
            )

            modelIf.ModelInterface.safe_dataframe_write(
                df            = df_idf,
                name_df       = 'eidf',
                include_index = True,
                index_label   = 'INDEX',
                filepath      = self.fpath_idf,
                log_training  = log_training
            )

            modelIf.ModelInterface.safe_dataframe_write(
                df            = df_x_ref,
                name_df       = 'x_ref',
                include_index = True,
                index_label   = 'INDEX',
                filepath      = self.fpath_x_ref,
                log_training  = log_training
            )

        modelIf.ModelInterface.safe_file_write(
            dict_obj      = x_ref_friendly,
            name_dict_obj = 'x_ref_friendly',
            filepath      = self.fpath_x_ref_friendly_txt,
            write_as_json = False,
            log_training  = log_training
        )

        modelIf.ModelInterface.safe_file_write(
            dict_obj      = x_ref_friendly,
            name_dict_obj = 'x_ref_friendly_json',
            filepath      = self.fpath_x_ref_friendly_json,
            write_as_json = True,
            log_training  = log_training
        )

        modelIf.ModelInterface.safe_dataframe_write(
            df            = self.df_y_ref_radius,
            name_df       = 'y_ref_radius',
            include_index = True,
            index_label   = 'INDEX',
            filepath      = self.fpath_y_ref_radius,
            log_training  = log_training
        )

        if not self.is_partial_training:
            modelIf.ModelInterface.safe_dataframe_write(
                df            = df_x_clustered,
                name_df       = 'x_clustered',
                include_index = True,
                index_label   = 'INDEX',
                filepath      = self.fpath_x_clustered,
                log_training  = log_training
            )

        modelIf.ModelInterface.safe_dataframe_write(
            df            = df_y_clustered_radius,
            name_df       = 'y_clustered_radius',
            include_index = True,
            index_label   = 'INDEX',
            filepath      = self.fpath_y_clustered_radius,
            log_training=log_training
        )

        modelIf.ModelInterface.safe_file_write(
            dict_obj      = x_clustered_friendly,
            name_dict_obj = 'x_clustered_friendly',
            filepath      = self.fpath_x_clustered_friendly_txt,
            write_as_json = False,
            log_training  = log_training
        )

        #
        # Only in partial training mode
        #
        if self.is_partial_training:
            modelIf.ModelInterface.safe_file_write(
                dict_obj      = x_clustered_friendly,
                name_dict_obj = 'x_clustered_friendly_json',
                filepath      = self.fpath_x_clustered_friendly_json,
                write_as_json = True,
                log_training  = log_training
            )

        # Our servers look to this file to see if RFV has changed
        # It is important to do it last (and fast), after everything is done
        modelIf.ModelInterface.safe_file_write(
            dict_obj      = {'timenow': str(dt.datetime.now())},
            name_dict_obj = 'model last updated time',
            filepath      = self.fpath_updated_file,
            write_as_json = False,
            log_training = log_training
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
                     + ': EIDF file "' + self.fpath_idf + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)

        if not os.path.isfile(self.fpath_x_ref):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': RFV file "' + self.fpath_x_ref + '" not found!'
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
                + ': x_name Data: Read ' + str(df_x_name.shape[0]) + ' lines: ' + str(self.x_name)
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
                + ': IDF Data: Read ' + str(df_idf.shape[0]) + ' lines: ' + str(self.idf)
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

            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Loading file "' + str(self.fpath_x_clustered) + '...'
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

            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': Model "' + str(self.identifier_string)
                + '" trained at "' + str(self.model_updated_time)
                + '" successfully loaded.'
            )
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Load RFV from file failed for identifier "' + self.identifier_string\
                     + '". Error msg "' + str(ex) + '".'
            log.Log.critical(errmsg)
            raise Exception(errmsg)

    def load_model_from_partial_trainings_data(
            self,
            # Most updated training data from DB/etc.
            td_latest,
            log_training
    ):
        self.x_name = td_latest.get_x_name()
        y_unique_list_latest = list(set(list(td_latest.get_y())))

        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Using x_name: ' + str(self.x_name)
        )

        # We will convert to numpy array later
        self.x_clustered = []
        self.y_clustered = []

        # Get all x_clustered & y_clustered files by y_id
        try:
            file_pattern_regex = '.*' + ModelData.MODEL_FILES_X_CLUSTERED_FRIENDLY_JSON_POSTFIX + '$'
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Loading x_clustered files from folder "' + self.model_path_prefix
                + '/", matching pattern "' + str(file_pattern_regex)
                , log_list = log_training
            )

            # Get files under the partial training folder
            list_files = os.listdir(self.model_path_prefix + '/')
            r = re.compile(pattern = file_pattern_regex)
            x_clustered_fnames = list(filter(r.match, list_files))

            #
            # Now form the general feature vector
            #
            features = list(npUtil.NumpyUtil.convert_dimension(
                arr    = self.x_name,
                to_dim = 1
            ))
            log.Log.debugdebug('Using model features:\n\r' + str(features))

            #
            # Helper object to convert sentence to a mathematical object (feature vector)
            #
            model_fv = FeatureVector()
            model_fv.set_freq_feature_vector_template(
                list_symbols = features
            )

            count = 0
            count_appended = 0
            for fname in x_clustered_fnames:
                x_clstrd_fpath = self.model_path_prefix + '/' + fname
                try:
                    # TODO Need encoding='utf-8'?
                    with open(x_clstrd_fpath) as x_clstrd_handle:
                        d = json.load(x_clstrd_handle)
                    count += 1
                    log.Log.debug(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': ' + str(count) + '. Loaded "' + str(fname) + '" as:\n\r' + str(d)
                    )

                    for k in d.keys():
                        line = d[k]
                        line_x_name = line['x_name']
                        line_x      = line['x']
                        line_y      = line['y']
                        #
                        # Check if we need to keep this y/label, it might be outdated or deleted already
                        #
                        if line_y not in y_unique_list_latest:
                            log.Log.warning(
                                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                + ': ' + str(count) + '. Ignoring y label ' + str(line_y)
                                + ', not in latest training data'
                                , log_list = log_training
                            )
                            continue
                        df_text_counter = pd.DataFrame({
                            FeatureVector.COL_SYMBOL: line_x_name,
                            FeatureVector.COL_FREQUENCY: line_x
                        })
                        # Get feature vector of text
                        try:
                            # TODO When we merge there could be symbols not in feature list
                            df_fv = model_fv.get_freq_feature_vector_df(
                                df_text_counter = df_text_counter
                            )
                        except Exception as ex:
                            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                                     + ': Exception occurred calculating FV for "' + str(df_text_counter) \
                                     + '": Exception "' + str(ex) \
                                     + '\n\rUsing FV Template:\n\r' + str(model_fv.get_fv_template()) \
                                     + ', FV Weights:\n\r' + str(model_fv.get_fv_weights())
                            log.Log.error(errmsg, log_list=log_training)
                            raise Exception(errmsg)

                        # This creates a single row matrix that needs to be transposed before matrix multiplications
                        # ndmin=2 will force numpy to create a 2D matrix instead of a 1D vector
                        # For now we make it 1D first
                        fv_text_1d = np.array(df_fv[FeatureVector.COL_FREQUENCY].values, ndmin=1)
                        fv_text_1d_norm = np.array(df_fv[FeatureVector.COL_FREQ_NORM].values, ndmin=1)
                        if fv_text_1d.ndim != 1:
                            raise Exception(
                                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                + ': Expected a 1D vector, got ' + str(fv_text_1d.ndim) + 'D!'
                            )
                        log.Log.debugdebug(
                            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + '\n\rConverted to:\n\r' + str(fv_text_1d)
                            + '\n\rand:\n\r' + str(fv_text_1d_norm)
                        )

                        self.x_clustered.append(fv_text_1d_norm)
                        self.y_clustered.append(line_y)

                        count_appended += 1

                        log.Log.info(
                            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': Total cluster now=' + str(count_appended) + '. Appended y label ' + str(line_y)
                            + ' to cluster.'
                            , log_list = log_training
                        )
                except Exception as ex:
                    errmsg = \
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                        + ': Could not load JSON from "' + str(x_clstrd_fpath) + '". Exception: ' + str(ex)
                    log.Log.critical(
                        s        = errmsg,
                        log_list = log_training
                    )
                    # Raise exception for this one
                    raise Exception(errmsg)

            self.x_clustered = np.array(self.x_clustered)
            self.y_clustered = np.array(self.y_clustered)

            log.Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Trained x_clustered:\n\r' + str(self.x_clustered)
                + '\n\ry_clustered:\n\r' + str(self.y_clustered)
                , log_list = log_training
            )

            # TODO Store in files the trained model

        except Exception as ex:
            errmsg = \
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': Partial loading exception for "' + str(self.identifier_string)\
                + '" model. Exception message: ' + str(ex)
            log.Log.critical(
                s        = errmsg,
                log_list = log_training
            )
            raise Exception(errmsg)

        raise Exception('Load model from partial trainings data not yet implemented!')

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

