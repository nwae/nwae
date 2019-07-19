import numpy as np
import pandas as pd
import threading
import json
import datetime as dt
import os
import mozg.lib.chat.classification.training.RefFeatureVec as reffv
import mozg.lib.math.ml.TrainingDataModel as tdm
import mozg.common.util.Log as log
from inspect import currentframe, getframeinfo
import mozg.lib.math.Cluster as clstr
import mozg.lib.math.Constants as const


class ModelData:

    def __init__(
            self,
            # Unique identifier to identify this set of trained data+other files after training
            identifier_string,
            # Directory to keep all our model files
            dir_path_model,
    ):
        self.identifier_string = identifier_string
        self.dir_path_model = dir_path_model

        #
        # RFVs
        # Original x, y, x_name in self.training_data
        # All np array type unless stated
        #
        # Order follows x_name
        self.idf = None
        self.x_ref = None
        self.y_ref = None
        self.df_rfv_distance_furthest = None
        self.x_clustered = None
        self.y_clustered = None
        self.x_name = None

        # Unique y
        self.y_unique = None

        # First check the existence of the files
        prefix = self.dir_path_model + '/' + self.identifier_string
        self.fpath_updated_file      = prefix + '.lastupdated.txt'
        self.fpath_x_name            = prefix + '.x_name.csv'
        self.fpath_idf               = prefix + '.idf.csv'
        self.fpath_rfv               = prefix + '.rfv.csv'
        self.fpath_rfv_friendly_json = prefix + '.rfv_friendly.json'
        # Only for debugging file
        self.fpath_rfv_friendly_txt  = prefix + '.rfv_friendly.txt'
        self.fpath_rfv_dist          = prefix + '.rfv.distance.csv'
        self.fpath_x_clustered       = prefix + '.x_clustered.csv'
        # Only for debugging file
        self.fpath_x_clustered_friendly_txt = prefix + '.x_clustered_friendly.txt'
        # Training data for testing back only
        self.fpath_training_data_x        = prefix + '.training_data.x.csv'
        self.fpath_training_data_x_name   = prefix + '.training_data.x_name.csv'
        self.fpath_training_data_y        = prefix + '.training_data.y.csv'

        self.log_training = []
        return

    def persist_model_to_storage(self):
        self.log_training = []

        # Sort
        self.df_x_name = pd.DataFrame(data=self.x_name)
        self.df_idf = pd.DataFrame(data=self.idf, index=self.x_name)
        # We use this training data model class to get the friendly representation of the RFV
        xy = tdm.TrainingDataModel(
            x = np.array(self.df_rfv.values),
            y = np.array(self.df_rfv.index),
            x_name = np.array(self.df_rfv.columns)
        )
        rfv_friendly = xy.get_print_friendly_x()
        #json_rfv_friendly = json.dumps(obj=xy.get_print_friendly_x(), ensure_ascii=False)
        self.df_rfv = self.df_rfv.sort_index()
        self.df_rfv_distance_furthest = self.df_rfv_distance_furthest.sort_index()
        self.df_x_clustered = pd.DataFrame(
            data    = self.x_clustered,
            index   = self.y_clustered,
            columns = self.x_name
        ).sort_index()
        # We use this training data model class to get the friendly representation of the x_clustered
        xy_x_clustered = tdm.TrainingDataModel(
            x = np.array(self.x_clustered),
            y = np.array(self.y_clustered),
            x_name = self.x_name
        )
        x_clustered_friendly = xy_x_clustered.get_print_friendly_x()

        log.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tx_name:\n\r' + str(self.df_x_name)
            + '\n\r\tIDF:\n\r' + str(self.df_idf)
            + '\n\r\tRFV:\n\r' + str(self.df_rfv)
            + '\n\r\tRFV friendly:\n\r' + str(rfv_friendly)
            + '\n\r\tFurthest Distance:\n\r' + str(self.df_rfv_distance_furthest)
            + '\n\r\tx clustered:\n\r' + str(self.df_x_clustered)
            + '\n\r\tx clustered friendly:\n\r' + str(x_clustered_friendly)
        )

        #
        # Save to file
        # TODO: This needs to be saved to DB, not file
        #
        self.df_x_name.to_csv(path_or_buf=self.fpath_x_name, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved x_name shape ' + str(self.df_x_name.shape) + ', filepath "' + self.fpath_x_name + ']'
            , log_list = self.log_training
        )

        self.df_idf.to_csv(path_or_buf=self.fpath_idf, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved IDF dimensions ' + str(self.df_idf.shape) + ' filepath "' + self.fpath_idf + '"'
            , log_list = self.log_training
        )

        self.df_rfv.to_csv(path_or_buf=self.fpath_rfv, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved RFV dimensions ' + str(self.df_rfv.shape) + ' filepath "' + self.fpath_rfv + '"'
            , log_list = self.log_training
        )

        try:
            # This file only for debugging
            f = open(file=self.fpath_rfv_friendly_txt, mode='w', encoding='utf-8')
            for i in rfv_friendly.keys():
                line = str(rfv_friendly[i])
                f.write(str(line) + '\n\r')
            f.close()

            with open(self.fpath_rfv_friendly_json, 'w', encoding='utf-8') as f:
                json.dump(rfv_friendly, f, indent=2)
            f.close()
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Saved rfv friendly ' + str(rfv_friendly) +  ' to file "' + self.fpath_rfv_friendly_json + '".'
                , log_list=self.log_training
            )
        except Exception as ex:
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Could not create rfv friendly file "' + self.fpath_rfv_friendly_json
                + '". ' + str(ex)
            , log_list = self.log_training
            )

        self.df_rfv_distance_furthest.to_csv(path_or_buf=self.fpath_rfv_dist, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved RFV (furthest) dimensions ' + str(self.df_rfv_distance_furthest.shape)
            + ' filepath "' + self.fpath_rfv_dist + '"'
            , log_list = self.log_training
        )

        self.df_x_clustered.to_csv(path_or_buf=self.fpath_x_clustered, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved Clustered x with shape ' + str(self.df_x_clustered.shape) + ' filepath "' + self.fpath_x_clustered + '"'
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

        #
        # Write back training data to file, for testing back the model only, not needed for the model
        #
        df_td_x = pd.DataFrame(self.training_data.get_x())
        df_td_x.to_csv(path_or_buf=self.fpath_training_data_x, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved Training Data x with shape ' + str(df_td_x.shape)
            + ' filepath "' + self.fpath_training_data_x + '"'
            , log_list=self.log_training
        )

        df_td_x_name = pd.DataFrame(self.training_data.get_x_name())
        df_td_x_name.to_csv(path_or_buf=self.fpath_training_data_x_name, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved Training Data x_name with shape ' + str(df_td_x_name.shape)
            + ' filepath "' + self.fpath_training_data_x_name + '"'
            , log_list=self.log_training
        )

        df_td_y = pd.DataFrame(self.training_data.get_y())
        df_td_y.to_csv(path_or_buf=self.fpath_training_data_y, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved Training Data y with shape ' + str(df_td_y.shape)
            + ' filepath "' + self.fpath_training_data_y + '"'
            , log_list=self.log_training
        )
        return

