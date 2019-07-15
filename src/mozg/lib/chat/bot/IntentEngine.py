# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import os
import numpy as np
import pandas as pd
import threading
import time
import json
import datetime as dt
import mozg.lib.chat.bot.IntentEngineThread as intEngThread
import mozg.lib.chat.classification.training.ChatTraining as chatTr
import mozg.lib.lang.model.FeatureVector as fv
import mozg.lib.chat.classification.training.RefFeatureVec as reffv
import mozg.lib.lang.nlp.SynonymList as sl
import mozg.common.util.Profiling as prf
import mozg.common.util.Log as log
from inspect import currentframe, getframeinfo
import mozg.lib.math.ml.metricspace.MetricSpaceModel as msmodel
import mozg.lib.math.ml.TrainingDataModel as tdmodel


#
# Intent AI/NLP Engine
#
# TODO Super important. Reducing features should be done in training, NOT HERE
#
class IntentEngine:

    #
    # At this point using SEARCH_TOPX_RFV=5 gives us an error rate (using all training data) at 4.1%,
    # using SEARCH_TOPX_RFV=4 gives an error rate of 5.1%, and using SEARCH_TOPX_RFV=3 gives 7.0%.
    #
    SEARCH_TOPX_RFV = 5
    # When we measure distance between a random text and all the RFVs, we multiply the furthest distance
    # by this. The final Score will be affected by this
    FURTHEST_DISTANCE_TO_RFV_MULTIPLIER = 1.1
    DEFAULT_SCORE_MIN_THRESHOLD = 5
    # Weight given to distance to RFV & closest sample, to determine the distance to a command
    WEIGHT_RFV = 0.5
    WEIGHT_SAMPLE = 1 - WEIGHT_RFV

    ROUND_BY = 5

    #
    # We convert data frame indexes to string type for standardization and consistency
    #
    CONVERT_COMMAND_INDEX_TO_STR = True

    # Column names for intent detection data frame
    COL_TEXT_NORMALIZED = 'TextNormalized'
    COL_COMMAND = reffv.RefFeatureVector.COL_COMMAND
    COL_DISTANCE_TO_RFV = 'DistToRfv'
    COL_DISTANCE_CLOSEST_SAMPLE = 'DistToSampleClosest'
    COL_DISTANCE_FURTHEST = reffv.RefFeatureVector.COL_DISTANCE_TO_RFV_FURTHEST
    COL_MATCH = 'Match'
    COL_SCORE = 'Score'
    COL_SCORE_CONFIDENCE_LEVEL = 'ScoreConfLevel'

    # From rescoring training data (using SEARCH_TOPX_RFV=5), we find that
    #    5% quartile score  = 55
    #    25% quartile score = 65
    #    50% quartile score = 70
    #    75% quartile score = 75
    #    95% quartile score = 85
    # Using the above information, we set
    CONFIDENCE_LEVEL_5_SCORE = 75
    CONFIDENCE_LEVEL_4_SCORE = 65
    CONFIDENCE_LEVEL_3_SCORE = 55
    # For confidence level 0-2, we run the bot against non-related data and we found
    #    99% quartile score = 32
    #    95% quartile score = 30
    #    75% quartile score = 20
    CONFIDENCE_LEVEL_2_SCORE = 40   # Means <1% of non-related data will go above it
    CONFIDENCE_LEVEL_1_SCORE = 20   # This means 25% of non-related data will go above it

    # TODO Change all lang/brand to botname or id
    def __init__(
            self,
            lang,
            # We do not specify here whether this is bot id or bot name or account-campaign-botname, etc.
            bot_key,
            dir_rfv_commands,
            dirpath_synonymlist,
            do_profiling = False,
            # No loading of heavy things like training data
            minimal = False,
            # Reduce RFV/IDF/etc features
            reduce_features = False,
            verbose = 0
    ):
        ###################################################################################################
        # All variables read-only after all initializations done. Do not modify when program is running.
        ###################################################################################################
        self.lang = lang.lower()
        self.bot_key = bot_key.lower()
        self.dir_rfv_commands = dir_rfv_commands

        self.fpath_updated_file = self.dir_rfv_commands + '/' + self.bot_key + '.lastupdated.txt'
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
        self.fpath_idf = self.dir_rfv_commands + '/' + self.bot_key + '.idf.csv'
        if not os.path.isfile(self.fpath_idf):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': IDF file "' + self.fpath_idf + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)
        self.df_idf_ro = None

        self.fpath_rfv = self.dir_rfv_commands + '/' + self.bot_key + '.rfv.csv'
        if not os.path.isfile(self.fpath_rfv):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': RFV file "' + self.fpath_rfv + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)
        self.df_rfv_ro = None
        # This is the cached data frame version of the RFV in numpy array form
        self.df_rfv_np_array_ro = None

        self.fpath_rfv_friendly_json = self.dir_rfv_commands + '/' + self.bot_key + '.rfv_friendly.json'
        if not os.path.isfile(self.fpath_rfv_friendly_json):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': RFV friendly file "' + self.fpath_rfv_friendly_json + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)
        self.rfv_friendly_json = None

        self.fpath_rfv_dist = self.dir_rfv_commands + '/' + self.bot_key + '.rfv.distance.csv'
        if not os.path.isfile(self.fpath_rfv):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': RFV furthest distance file "' + self.fpath_rfv_dist + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)
        self.df_rfv_dist_furthest_ro = None

        self.fpath_x_clustered = self.dir_rfv_commands + '/' + self.bot_key + '.x_clustered.csv'
        if not os.path.isfile(self.fpath_rfv):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': x clustered file "' + self.fpath_x_clustered + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)
        # Used to zoom into an intent/command group and compare against exact training data in that group
        self.df_x_clustered_ro = None
        self.index_x_clustered_ro = None

        # Used to finally confirm if it is indeed the intent by matching top keywords in the intent category
        # self.df_intent_tf_ro = None

        self.hash_df_rfv_ro = None
        self.hash_df_rfv_np_array_ro = None
        self.hash_df_idf_ro = None
        self.hash_index_x_clustered_ro = None
        # Don't do for training data, it takes too long
        self.hash_df_rfv_dist_furthest_ro = None

        self.synonymlist_ro = sl.SynonymList(
            lang=lang,
            dirpath_synonymlist=dirpath_synonymlist,
            postfix_synonymlist='.synonymlist.txt'
        )
        self.synonymlist_ro.load_synonymlist(verbose=1)

        # Profiling to optimize math code
        self.do_profiling = do_profiling

        # No loading of heavy RAM stuff like training data
        self.minimal = minimal

        # Reduce and optimize features
        # TODO Should be in training not here!
        self.reduce_features = False

        self.count_intent_calls = 0

        ###################################################################################################
        # Read-write Variables
        ###################################################################################################
        # Thread safe
        self.__mutex = threading.Lock()

        ###################################################################################################
        # Initializations
        ###################################################################################################
        # After this initialization, no more modifying of the above class variables
        self.is_rfv_ready = False
        self.is_training_data_ready = False
        self.is_reduced_features_ready = False
        self.background_thread = intEngThread.IntentEngineThread(
            botkey      = self.bot_key,
            intent_self = self
        )
        return

    def reset_ready_flags(self):
        self.is_rfv_ready = False
        self.is_training_data_ready = False
        self.is_reduced_features_ready = False

    def do_background_load(self):
        self.background_thread.start()

    def kill_background_job(self):
        self.background_thread.join()

    def set_rfv_to_ready(self):
        self.is_rfv_ready = True
        return

    def set_training_data_to_ready(self):
        if not self.minimal:
            self.is_training_data_ready = True
        return

    def check_if_rfv_updated(self):
        updated_rfv_time = os.path.getmtime(self.fpath_updated_file)
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': RFV for botkey "' + str(self.bot_key)
            + '" last updated time ' + str(self.last_updated_time_rfv)
            + ', updated "' + str(updated_rfv_time) + '".'
        )
        if (updated_rfv_time > self.last_updated_time_rfv):
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': RFV update time for botkey "' + str(self.bot_key) + '" - "'
                + str(dt.datetime.fromtimestamp(updated_rfv_time)) + '" is newer than "'
                + str(dt.datetime.fromtimestamp(self.last_updated_time_rfv))
                + '". Restarting...'
            )
            self.__mutex.acquire()
            self.reset_ready_flags()
            self.last_updated_time_rfv = updated_rfv_time
            self.__mutex.release()
            return True
        else:
            return False

    #
    # Run this slow load up in background thread
    #
    def background_load_rfv_commands_from_file(self):
        self.__mutex.acquire()

        try:
            # TODO This data actually not needed
            self.df_idf_ro = pd.read_csv(
                filepath_or_buffer = self.fpath_idf,
                sep       =',',
                index_col = 'INDEX'
            )
            if IntentEngine.CONVERT_COMMAND_INDEX_TO_STR:
                # Convert Index column to string
                self.df_idf_ro.index = self.df_idf_ro.index.astype(str)
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': IDF Data: Read ' + str(self.df_idf_ro.shape[0]) + ' lines'
            )
            log.Log.info(self.df_idf_ro)

            self.df_rfv_ro = pd.read_csv(
                filepath_or_buffer = self.fpath_rfv,
                sep       = ',',
                index_col = 'INDEX'
            )
            if IntentEngine.CONVERT_COMMAND_INDEX_TO_STR:
                # Convert Index column to string
                self.df_rfv_ro.index = self.df_rfv_ro.index.astype(str)
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': RFV Data: Read ' + str(self.df_rfv_ro.shape[0]) + ' lines: '
            )
            log.Log.info(self.df_rfv_ro)
            # Cached the numpy array
            self.df_rfv_np_array_ro = np.array(self.df_rfv_ro.values)
            log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Cached huge RFV array from dataframe..')

            # f = open(file=fpath_rfv_friendly, mode='w')
            with open(self.fpath_rfv_friendly_json, 'r') as f:
                self.rfv_friendly_json = json.load(f)
            f.close()
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Rfv friendly read keys ' + str(self.rfv_friendly_json.keys())
                + ' from file "' + self.fpath_rfv_friendly_json + '".'
            )
            log.Log.info(self.rfv_friendly_json)

            self.df_rfv_dist_furthest_ro = pd.read_csv(
                filepath_or_buffer = self.fpath_rfv_dist,
                sep       = ',',
                index_col = 'INDEX'
            )
            if IntentEngine.CONVERT_COMMAND_INDEX_TO_STR:
                # Convert Index column to string
                self.df_rfv_dist_furthest_ro.index = self.df_rfv_dist_furthest_ro.index.astype(str)
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': RFV Furthest Distance Data: Read ' + str(self.df_rfv_dist_furthest_ro.shape[0]) + ' lines'
            )
            log.Log.info(self.df_rfv_dist_furthest_ro)

            #
            # RFV is ready, means we can start detecting intents. But still can't use training data
            #
            self.is_rfv_ready = True

            if not self.minimal:
                self.df_x_clustered_ro = pd.read_csv(
                    filepath_or_buffer = self.fpath_x_clustered,
                    sep       = ',',
                    index_col = 'INDEX'
                )
                if IntentEngine.CONVERT_COMMAND_INDEX_TO_STR:
                    # Convert Index column to string
                    self.df_x_clustered_ro.index = self.df_x_clustered_ro.index.astype(str)
                log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': x clustered data: Read ' + str(self.df_x_clustered_ro.shape[0]) + ' lines')
                log.Log.info(self.df_x_clustered_ro)
                # This is very slow, we do it first! Cache it!
                self.index_command_fv_training_data_ro = np.array(self.get_command_index_from_training_data_index())
                log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                    +': Index of training data by command "'
                                    + str(list(self.index_command_fv_training_data_ro)) + '".')
                log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                   + ': Index original  "' + str(list(self.df_fv_training_data_ro.index)) + '".')

                self.sanity_check()

                #
                # Training data finally ready
                #
                self.is_training_data_ready = True
            else:
                self.sanity_check()
                # Training data is never ready in minimal mode
                self.is_training_data_ready = False

            # TODO This data actually not needed
            # self.fpath_intent_tf = self.dir_rfv_commands + '/' + self.bot_key + '.' + 'chatbot.commands.words.tf.csv'
            # self.df_intent_tf_ro = pd.read_csv(
            #     filepath_or_buffer = self.fpath_intent_tf,
            #     sep       = ',',
            #     index_col = 'INDEX'
            # )
            # Convert Index column to string
            # self.df_intent_tf_ro.index = self.df_intent_tf_ro.index.astype(str)
            # log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            #             + ': TF Data: Read ' + str(self.df_intent_tf_ro.shape[0]) + ' lines')
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Load RFV from file failed for botkey "' + self.bot_key\
                     + '". Error msg "' + str(ex) + '".'
            log.Log.critical(errmsg)
            raise Exception(errmsg)
        finally:
            self.__mutex.release()

        return

    def sanity_check(self):
        # Check RFV is normalized
        for com in list(self.df_rfv_ro.index):
            rfv = np.array(self.df_rfv_ro.loc[com].values)
            if len(rfv.shape) != 1:
                raise Exception(str(self.__class__) + ': RFV vector must be 1-dimensional, got ' + str(len(rfv.shape)) +
                                '-dimensional for intentId ' + str(com) +
                                '!! Possible clash of index (e.g. for number type index column 1234.1 is identical to 1234.10)')
            dist = np.sum(np.multiply(rfv,rfv))**0.5
            if abs(dist-1) > 0.000001:
                log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': Warning: RFV for command [' + str(com) + '] not 1, ' + str(dist))
                raise Exception('RFV error')

        if not self.minimal:
            for idx in list(self.df_fv_training_data_ro.index):
                fv = np.array(self.df_fv_training_data_ro.loc[idx].values)
                if len(fv.shape) != 1:
                    raise Exception(
                        str(self.__class__) + ': Training data vector must be 1-dimensional, got ' + str(len(fv.shape))
                        + '-dimensional for index ' + str(idx)
                        + '!! Possible clash of index (e.g. for number type index column 1234.1 is identical to 1234.10)')
                dist = np.sum(np.multiply(fv,fv))**0.5
                if abs(dist-1) > 0.000001:
                    errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                             + ': Warning: Training data FV Error for index [' + str(idx) + '] not 1, ' + str(dist)
                    log.Log.critical(errmsg)
                    raise Exception(errmsg)
        return

    def get_command_index_from_training_data_index(self):
        # This is very slow, we do it first! Cache it!
        convert_to_type = 'str'
        if not IntentEngine.CONVERT_COMMAND_INDEX_TO_STR:
            convert_to_type = 'int'

        # We derive the intent id or command from the strings '888-1', '888-2',...
        # by removing the ending '-1', '-2', ...
        # This will speed up filtering of training data later by command.
        index_command = \
            chatTr.ChatTraining.retrieve_intent_from_training_data_index(
                index_list      = list(self.df_fv_training_data_ro.index),
                convert_to_type = convert_to_type
            )
        return index_command

    def get_count_intent_calls(self):
        return self.count_intent_calls

    def get_hash_of_readonly_objects(self):
        #self.__mutex.acquire()
        try:
            #
            # Do not convert a data frame into numpy array, then call numpy tostring().
            # These values will change! Not sure why.
            # Use the pandas dataframe ._to_string() method directly
            #
            self.hash_df_rfv_ro = hash(self.df_rfv_ro.to_string())
            self.hash_df_rfv_np_array_ro = hash(self.df_rfv_np_array_ro.tostring())
            self.hash_df_idf_ro = hash(self.df_idf_ro.to_string())
            self.hash_df_rfv_dist_furthest_ro = hash(self.df_rfv_dist_furthest_ro.to_string())
            if not self.minimal:
                self.hash_index_command_fv_training_data_ro = hash(self.index_command_fv_training_data_ro.tostring())
        except Exception as ex:
            raise (ex)
        #finally:
        #    self.__mutex.release()
        return

    def check_hash_of_readonly_objects(self):
        #self.__mutex.acquire()
        errmsg = None
        try:
            if hash(self.df_rfv_ro.to_string()) != self.hash_df_rfv_ro:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': CRITICAL ERROR. self.df_rfv_ro changed!'
            if hash(self.df_rfv_np_array_ro.tostring()) != self.hash_df_rfv_np_array_ro:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': CRITICAL ERROR. self.df_rfv_np_array_ro changed!'
            if hash(self.df_idf_ro.to_string()) != self.hash_df_idf_ro:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': CRITICAL ERROR. self.df_idf_ro changed!'
            if hash(self.df_rfv_dist_furthest_ro.to_string()) != self.hash_df_rfv_dist_furthest_ro:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': CRITICAL ERROR. self.df_rfv_dist_furthest_ro changed! '\
                         + 'New hash "' + str(hash(self.df_rfv_dist_furthest_ro.to_string()))\
                         + ', old hash "' + str(self.hash_df_rfv_dist_furthest_ro) + '".'
            if not self.minimal:
                if hash(
                        self.index_command_fv_training_data_ro.tostring()) != self.hash_index_command_fv_training_data_ro:
                    errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                             + ': CRITICAL ERROR. self.index_command_fv_training_data_ro changed!'
        except Exception as ex:
            raise(ex)
        #finally:
            #self.__mutex.release()

        if errmsg is not None:
            log.Log.critical(errmsg)
            raise Exception(errmsg)
        return

    #
    # This is the external interface for the CRM/etc. bot to call to get text intent/command.
    # Text is assumed to be already word segmented by ' '
    # TODO: Cut down this function task time to <100ms on a 2.8GHz 4 core machine
    # TODO: Can't use ' ' as word boundary for Vietnamese
    # TODO: "No Match" also need to return
    #
    def get_text_class(
            self,
            text_segmented,
            weigh_idf,
            normalized                = True,
            top                       = SEARCH_TOPX_RFV,
            return_match_results_only = True,
            score_min_threshold       = DEFAULT_SCORE_MIN_THRESHOLD,
            verbose                   = 0,
            # WARNING This is for DEBUGGING ONLY to reduce columns, where the math will be wrong
            #remove_zero_columns       = False,
            # Only for debugging, tracking purposes
            chatid                    = None,
            not_necessary_to_use_training_data_samples = True
    ):
        self.count_intent_calls = self.count_intent_calls + 1

        count = 1
        sleep_time_wait_rfv = 0.1
        wait_max_time = 10
        while not (self.is_rfv_ready and
                   (self.is_training_data_ready or self.minimal or not_necessary_to_use_training_data_samples)):
            log.Log.warning(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': RFV not yet ready, sleep for ' + str(count*sleep_time_wait_rfv) + ' secs now..')
            if count*sleep_time_wait_rfv > wait_max_time:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': Waited too long ' + str(count*sleep_time_wait_rfv) + ' secs. Raising exception..'
                raise Exception(errmsg)
            time.sleep(sleep_time_wait_rfv)
            count = count + 1

        start_func = None
        #
        # This routine is thread safe, no writes to class variables, just read.
        #
        if self.do_profiling:
            start_func = prf.Profiling.start()
            log.Log.info('.   '
                        + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
                        + ' PROFILING Intent (reduced features = '
                        + str(self.reduce_features) + ') Start: '
                        + str(start_func))

        a = None
        space_profiling = '      '

        #
        # Replace words with root words
        # This step uses synonyms and replaces say 存钱, 入钱, 入款, all with the standard 存款
        # This will reduce training data without needing to put all versions of the same thing.
        #
        if self.do_profiling:
            a = prf.Profiling.start()
            log.Log.info('.' + space_profiling
                       + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
                       + ' PROFILING Intent (replace root words) Start: ' + str(a))

        text_normalized = self.synonymlist_ro.normalize_text(text_segmented=text_segmented, verbose=verbose)
        text_normalized = text_normalized.lower()
        log.Log.debugdebug('#')
        log.Log.debugdebug('# TEXT NORMALIZATION')
        log.Log.debugdebug('#')
        log.Log.debugdebug('Text [' + text_segmented + '] normalized to [' + text_normalized + ']')
        if self.do_profiling:
            b = prf.Profiling.stop()
            log.Log.info('.' + space_profiling
                        + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
                        + ' PROFILING Intent (replace root words): ' + prf.Profiling.get_time_dif_str(a, b))

        keywords_all = list(self.df_rfv_ro.columns)

        #
        # Convert sentence to a mathematical object (feature vector)
        #
        log.Log.debugdebug('#')
        log.Log.debugdebug('# FEATURE VECTOR & NORMALIZATION')
        log.Log.debugdebug('#')

        if self.do_profiling:
            a = prf.Profiling.start()
            log.Log.info('.' + space_profiling
                       + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
                       + ' PROFILING Intent (FV & Normalization) Start: ' + str(a))

        model_fv = fv.FeatureVector()
        model_fv.set_freq_feature_vector_template(list_symbols=keywords_all)
        if weigh_idf:
            model_fv.set_feature_weights(fw=list(self.df_idf_ro['IDF']))

        # Get feature vector of text
        try:
            df_fv = model_fv.get_freq_feature_vector(text=text_normalized)
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Exception occurred calculating FV for "' + str(text_normalized)\
                     + '": Exception "' + str(ex)\
                     + '. Using FV Template ' + str(model_fv.get_fv_template())\
                     + ', FV Weights ' + str(model_fv.get_fv_weights())
            log.Log.critical(errmsg)
            raise Exception(ex)

        # This creates a single row matrix that needs to be transposed before matrix multiplications
        # ndmin=2 will force numpy to create a 2D matrix instead of a 1D vector
        # For now we make it 1D first
        fv_text_1d = np.array(df_fv['Frequency'].values, ndmin=1)
        if fv_text_1d.ndim != 1:
            raise Exception(str(self.__class__) + ': Expected a 1D vector, got ' + str(fv_text_1d.ndim) + 'D!')
        fv_text_normalized_1d = np.array(df_fv['FrequencyNormalized'].values, ndmin=1)
        if fv_text_normalized_1d.ndim != 1:
            raise Exception(str(self.__class__) + ': Expected a 1D vector, got ' + str(fv_text_normalized_1d.ndim) + 'D!')
        log.Log.debug(fv_text_1d)
        log.Log.debug(fv_text_normalized_1d)

        if self.do_profiling:
            b = prf.Profiling.stop()
            log.Log.info('.' + space_profiling
                        + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
                        + ' PROFILING Intent (FV & Normalization): ' + prf.Profiling.get_time_dif_str(a, b))

        if normalized:
            fv_text_1d = fv_text_normalized_1d

        #
        # TODO At this point, to speed up things, we can cache previously asked questions
        # TODO in some FV form.
        # TODO Then we just compare to that cache, using the same distance measures.
        # TODO Or our method can be other simpler frequency methods also.
        # TODO We can actually do this at the mozg.gbot level also, and not here?
        #

        # Only if we are debugging with remove_zero_columns flag set
        #df_rfv_tmp = None
        #df_fv_training_data_tmp = None

        # Remove columns with 0 value
        # When do this and remove non relevant columns, it is also much easier to troubleshoot
        # if remove_zero_columns:
        #     df_rfv_tmp = self.df_rfv_ro.copy()
        #     df_fv_training_data_tmp = self.df_fv_training_data_ro.copy()
        #
        #     if self.do_profiling:
        #         a = prf.Profiling.start()
        #         if verbose>=2: log.Log.log('.' + space_profiling
        #                                    + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
        #                                    + ' PROFILING Intent (Remove Zero Columns) Start: ' + str(a))
        #     cols_non_zero = np.array(list(df_rfv_tmp.columns))
        #     cols_non_zero = list(cols_non_zero[fv_text_1d>0])
        #     if verbose >= 3:
        #         log.Log.log(cols_non_zero)
        #     fv_text_1d = fv_text_1d[fv_text_1d>0]
        #     if verbose >= 3:
        #         log.Log.log('cols_non_zero:')
        #         log.Log.log(cols_non_zero)
        #         log.Log.log('fv_text_1d')
        #         log.Log.log(fv_text_1d)
        #
        #     # Do the same for RFV, IDF, Training Data
        #     df_rfv_tmp = df_rfv_tmp.loc[:, cols_non_zero]
        #     if verbose >= 3:
        #         log.Log.log('df_rfv_tmp:')
        #         log.Log.log(df_rfv_tmp)
        #
        #     df_fv_training_data_tmp = df_fv_training_data_tmp.loc[:, cols_non_zero]
        #     if verbose >= 3:
        #         log.Log.log('df_fv_training_data_tmp:')
        #         log.Log.log(df_fv_training_data_tmp)
        #
        #     if self.do_profiling:
        #         b = prf.Profiling.stop()
        #         log.Log.log('.' + space_profiling
        #                     + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
        #                     + ' PROFILING Intent (Remove Zero Columns): ' + prf.Profiling.get_time_dif_str(a, b))


        # Convert back to 2D array of dimension (1,x)
        ori_len = fv_text_1d.size
        fv_text_2d = np.array(fv_text_1d, ndmin=2)
        if fv_text_2d.shape != (1,ori_len):
            raise Exception(str(self.__class__) + ': Expected dimension (1,' + str(ori_len) + ') , got ' + str(fv_text_2d.shape))
        log.Log.debug('fv_text_2d:')
        log.Log.debug(fv_text_2d)

        #fv_text_feature_presence = (fv_text > 0) * 1

        #
        # Step 1:
        #    Pre-filter and remove intent RFVs which have no common features.
        #    This speeds up processing by a lot, more than half.
        # By right, if we don't do a 2 step processing, we can directly compare with the entire training
        # data set FV, which will not be slow due to matrix operations.
        # However it is mathematically inelegant and maybe not extensible in the future as training data grows.
        #
        if self.do_profiling:
            a = prf.Profiling.start()
            log.Log.info('.' + space_profiling
                       + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
                       + ' PROFILING Intent (RFV Simplification) Start: ' + str(a))
        log.Log.debug('#')
        log.Log.debug('# RFV SIMPLIFICATION')
        log.Log.debug('#')

        # Make sure cached RFV array is same dimension with RFV
        if self.df_rfv_np_array_ro.shape != self.df_rfv_ro.shape:
            raise Exception(str(self.__class__) + ': Cached RFV array dimensions '
                            + str(self.df_rfv_np_array_ro.shape) + ', different from RFV data frame '
                            + str(self.df_rfv_ro.shape) + '!')
        rfv_matrix = self.df_rfv_np_array_ro
        # if remove_zero_columns:
        #     rfv_matrix = np.array(df_rfv_tmp.values)
        # else:
        #     rfv_matrix = self.df_rfv_np_array_ro
        log.Log.debug('rfv_matrix:')
        log.Log.debug(rfv_matrix)
        # Multiply matrices to see which rows are zero, returns a single column matrix
        rfv_text_mul = np.matmul(rfv_matrix, fv_text_2d.transpose())
        #print(rfv_text_mul)
        # Get non-zero intents, those that have at least a single intersecting feature
        non_zero_intents = rfv_text_mul.transpose() > 0
        non_zero_intents = non_zero_intents.tolist()
        # Only select the RFVs that have common features with the text
        df_rfv_nonzero = self.df_rfv_ro.loc[non_zero_intents[0]]
        # if remove_zero_columns:
        #     df_rfv_nonzero = df_rfv_tmp.loc[non_zero_intents[0]]
        # else:
        #     df_rfv_nonzero = self.df_rfv_ro.loc[non_zero_intents[0]]
        log.Log.debug('df_rfv_nonzero:')
        log.Log.debug(df_rfv_nonzero)

        if self.do_profiling:
            b = prf.Profiling.stop()
            log.Log.info('.' + space_profiling
                        + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
                        + ' PROFILING Intent (RFV Simplification): ' + prf.Profiling.get_time_dif_str(a, b))

        # If no matches
        if df_rfv_nonzero.shape[0] == 0:
            log.Log.warning('No common features with RFV!!')
            return None

        #
        # FV-RFV Distance Calculation
        #

        if self.do_profiling:
            a = prf.Profiling.start()
            log.Log.info('.' + space_profiling
                       + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
                       + ' PROFILING Intent (FV-RFV Distance Calculation) Start: ' + str(a))

        # Create a matrix of similar rows (fv_text_2d)
        text_matrix = np.repeat(a=fv_text_2d, repeats=df_rfv_nonzero.shape[0], axis=0)
        log.Log.debug('text_matrix:')
        log.Log.debug(text_matrix)
        rfv_nonzero_matrix = np.array(df_rfv_nonzero)

        dif_rfv_text = rfv_nonzero_matrix - text_matrix
        log.Log.debug('dif_rfv_text:')
        log.Log.debug(dif_rfv_text)
        # Square every element in the matrix
        dif_rfv_text2 = np.power(dif_rfv_text, 2)
        log.Log.debug('dif_rfv_text2:')
        log.Log.debug(dif_rfv_text2)
        # Sum every row to create a single column matrix
        dif_rfv_text2_sum = dif_rfv_text2.sum(axis=1)
        log.Log.debug('dif_rfv_text2_sum:')
        log.Log.debug(dif_rfv_text2_sum)
        # Take the square root of every element in the single column matrix as distance
        distance_rfv_text = np.power(dif_rfv_text2_sum, 0.5)
        log.Log.debug('distance_rfv_text:')
        log.Log.debug(distance_rfv_text)
        # Convert to a single row matrix
        distance_rfv_text = distance_rfv_text.transpose()
        log.Log.debug('distance_rfv_text (transpose):')
        log.Log.debug(distance_rfv_text)
        #distance_rfv_text = distance_rfv_text.tolist()
        #if verbose >= 3:
        #    log.Log.log('distance_rfv_text (to list):')
        #    log.Log.log(distance_rfv_text)

        #
        # Code below expects this distance to RFV as a 2D array of dimension (1,N)
        #
        if distance_rfv_text.ndim == 1:
            distance_rfv_text = np.array(distance_rfv_text, ndmin=2)
        log.Log.debug('distance_rfv_text (2D):')
        log.Log.debug(distance_rfv_text)

        if distance_rfv_text.shape[0] != 1:
            raise Exception(str(self.__class__) + ': Expected a single row for distance to RFV matrix, got ' +
                            str(distance_rfv_text.shape))

        if self.do_profiling:
            b = prf.Profiling.stop()
            log.Log.info('.' + space_profiling
                        + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
                        + ' PROFILING Intent (FV-RFV Distance Calculation): '
                        + prf.Profiling.get_time_dif_str(a, b))

        #
        # Keep top X matches
        #   1) Distance criteria of given text FV to all intent RFV
        #   2) Presence of features, how many matches in proportion
        #
        if self.do_profiling:
            a = prf.Profiling.start()
            log.Log.info('.' + space_profiling
                       + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
                       + ' PROFILING Intent (Data Frame Preparation) Start: ' + str(a))

        close_commands = list(df_rfv_nonzero.index)

        # Furthest distances to RFV in training data
        condition = self.df_rfv_dist_furthest_ro[IntentEngine.COL_COMMAND].isin(close_commands)
        df_distance_furthest = self.df_rfv_dist_furthest_ro[condition]
        close_commands_distance_furthest = df_distance_furthest[IntentEngine.COL_DISTANCE_FURTHEST].tolist()

        tmp_len = len(close_commands)
        df_dist_to_classes = pd.DataFrame(data={
            IntentEngine.COL_TEXT_NORMALIZED: [text_normalized]*tmp_len,
            IntentEngine.COL_COMMAND: close_commands,
            IntentEngine.COL_DISTANCE_TO_RFV: np.round(distance_rfv_text[0], IntentEngine.ROUND_BY),
            # We will replace this later
            IntentEngine.COL_DISTANCE_CLOSEST_SAMPLE: [99999999]*tmp_len,
            IntentEngine.COL_DISTANCE_FURTHEST: np.round(close_commands_distance_furthest, IntentEngine.ROUND_BY),
            IntentEngine.COL_MATCH: [0]*tmp_len,
            IntentEngine.COL_SCORE: [0]*tmp_len,
            IntentEngine.COL_SCORE_CONFIDENCE_LEVEL: [0]*tmp_len
        })

        # Furthest distance to RFV of input text
        max_distance_inputtext_to_rfv = max(df_dist_to_classes[IntentEngine.COL_DISTANCE_TO_RFV])
        log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': Max distance of input text to RFV = ' + str(max_distance_inputtext_to_rfv))

        # Sort distance ascending
        df_dist_to_classes = df_dist_to_classes.sort_values(by=[IntentEngine.COL_DISTANCE_TO_RFV], ascending=True)
        # Take only top X
        df_dist_to_classes = df_dist_to_classes.loc[
            df_dist_to_classes.index[0:min(top, df_dist_to_classes.shape[0])]
        ]
        #df_dist_to_classes = df_dist_to_classes.reset_index(drop=True)
        df_dist_to_classes = df_dist_to_classes.set_index(IntentEngine.COL_COMMAND)
        log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                           + ': Result Data Frame')
        # We sort again by command, so that later we can maintain consistency
        df_dist_to_classes = df_dist_to_classes.sort_values(by=[IntentEngine.COL_COMMAND], ascending=True)
        log.Log.debugdebug(df_dist_to_classes)

        if self.do_profiling:
            b = prf.Profiling.stop()
            log.Log.info('.' + space_profiling
                        + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
                        + ' END PROFILING Intent (Data Frame Preparation): ' + prf.Profiling.get_time_dif_str(a, b))

        #
        # Doing only for top intents save time, speeds up the process by a lot
        # We zoom into all training data of each intent now
        #
        is_using_training_data = self.is_training_data_ready
        if not is_using_training_data:
            log.Log.warning(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': Not using training data for chat id "' + str(chatid)
                            + '", text "' + str(text_segmented) + '".')
        else:
            #
            # Zooming into the intent category
            # We do more detailed calculations now, to see which of the top intents is the best match.
            #
            if self.do_profiling:
                a = prf.Profiling.start()
                log.Log.info('.' + space_profiling
                             + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
                             + ' START PROFILING Intent (Zoom Into Training Data) Start: ' + str(a))

            top_intents = df_dist_to_classes.index.tolist()
            # log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            #                    + ': Top intents: ' + str(top_intents) + '.')
            # log.Log.debugdebug(df_dist_to_classes)
            #
            # log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            #                    + ': Before Zoom Into Training Data..')
            # log.Log.debugdebug(df_dist_to_classes)

            df_shortest_distance_to_samples = self.get_closest_distance_to_training_sample(
                inputtext_fv_2d = fv_text_2d,
                intents_list    = top_intents
            )
            USE_JOIN = True
            if USE_JOIN:
                # Join data frames by command.
                # By using join, we ensure it is correct even if the return data frame is out of original sort order
                # But the problem is that we cannot set the column initially, thus we will have to drop it now
                df_dist_to_classes = df_dist_to_classes.drop([IntentEngine.COL_DISTANCE_CLOSEST_SAMPLE], axis=1)
                df_dist_to_classes = df_dist_to_classes.join(
                    other = df_shortest_distance_to_samples
                )
            else:
                # This assumes sorting is done properly
                df_dist_to_classes[IntentEngine.COL_DISTANCE_CLOSEST_SAMPLE] = \
                    df_shortest_distance_to_samples[IntentEngine.COL_DISTANCE_CLOSEST_SAMPLE]
            # Replace NaN with some large value
            df_dist_to_classes[IntentEngine.COL_DISTANCE_CLOSEST_SAMPLE].fillna(99999999, inplace=True)
            log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                               + ': After Zoom Into Training Data..')
            log.Log.debugdebug(df_dist_to_classes)

            if self.do_profiling:
                b = prf.Profiling.stop()
                log.Log.info('.' + space_profiling
                            + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
                            + ' END PROFILING Intent (Zoom Into Training Data): '
                             + prf.Profiling.get_time_dif_str(a, b))

        #
        # Now for the most important calculations: MATCH & SCORE
        #
        if self.do_profiling:
            a = prf.Profiling.start()
            log.Log.info('.' + space_profiling
                       + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
                       + ' PROFILING Intent (Match & Score) Start: ' + str(a))

        # Be CAREFUL. This value of 1.1 will affect score, so don't simply change it
        # Just in case there is only 1 value of distance to RFV (unlikely), we multiply by 1.1 to avoid a 0 score
        distance_threshold_for_match =\
            IntentEngine.FURTHEST_DISTANCE_TO_RFV_MULTIPLIER * max_distance_inputtext_to_rfv
        # Minimum of the threshold is 1.0
        # TODO Don't hardcode this min distance threshold of 1.0, use some proper math
        distance_threshold_for_match = max(1, distance_threshold_for_match)
        log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Using distance threshold for MATCH ' + str(distance_threshold_for_match))

        # Calculate the distance to Intent using weighted sum of distance to RFV and distance to closest sample

        dist_min_vec = np.array(df_dist_to_classes[IntentEngine.COL_DISTANCE_TO_RFV]) * IntentEngine.WEIGHT_RFV +\
                       np.array(df_dist_to_classes[IntentEngine.COL_DISTANCE_CLOSEST_SAMPLE]) * IntentEngine.WEIGHT_SAMPLE
        if not is_using_training_data:
            dist_min_vec = np.array(df_dist_to_classes[IntentEngine.COL_DISTANCE_TO_RFV])
        log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Distance Min Vector: ' + str(dist_min_vec))
        #
        # A match to an intent is determined using the following factors:
        # - Distance (either distance to RFV or to closest sample) is less than threshold set
        #
        intent_match_vec = np.array( (dist_min_vec <= distance_threshold_for_match) * 1 )
        log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Intent Match Vector: ' + str(intent_match_vec))
        # Assign to data frame
        df_dist_to_classes[IntentEngine.COL_MATCH] = intent_match_vec
        # Calculate our measure of "Score", which is somewhat arbitrary
        diff_vec = np.maximum(distance_threshold_for_match - dist_min_vec, 0)
        range_allowed = max(distance_threshold_for_match, 0.00001)
        diff_normalized_vec = np.minimum(1, np.maximum(0, (diff_vec / range_allowed)))
        #
        # FINALLY THE SCORE:
        # We want to create a curve that is y=1 when x=0, and y=0 when x=1, but decreases very slowly.
        # Thus we use cos(x)^k function, where k<1
        score_vec = np.round(100 * diff_normalized_vec, 2) * intent_match_vec
        # round(100 * (math.cos(math.pi*diff_normalized/2)**0.2) * diff_normalized, 2)
        df_dist_to_classes[IntentEngine.COL_SCORE] = score_vec
        # Maximum confidence level is 5, minimum 0
        score_confidence_level_vec = \
            (score_vec >= IntentEngine.CONFIDENCE_LEVEL_1_SCORE) * 1 + \
            (score_vec >= IntentEngine.CONFIDENCE_LEVEL_2_SCORE) * 1 + \
            (score_vec >= IntentEngine.CONFIDENCE_LEVEL_3_SCORE) * 1 + \
            (score_vec >= IntentEngine.CONFIDENCE_LEVEL_4_SCORE) * 1 + \
            (score_vec >= IntentEngine.CONFIDENCE_LEVEL_5_SCORE) * 1
        df_dist_to_classes[IntentEngine.COL_SCORE_CONFIDENCE_LEVEL] = score_confidence_level_vec
        log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                               + ': After calculating score: ')
        log.Log.debugdebug(df_dist_to_classes.columns)
        log.Log.debugdebug(df_dist_to_classes.values)

        # Sort score descending
        df_dist_to_classes = df_dist_to_classes.sort_values(by=[IntentEngine.COL_SCORE], ascending=False)
        log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                               + ': After calculating score (sorted): ')
        log.Log.debugdebug(df_dist_to_classes.columns)
        log.Log.debugdebug(df_dist_to_classes.values)
        # Extract only matches
        if return_match_results_only:
            df_dist_to_classes = df_dist_to_classes[df_dist_to_classes[IntentEngine.COL_MATCH]==1]
        log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                               + ': After getting matches: ')
        log.Log.debugdebug(df_dist_to_classes.columns)
        log.Log.debugdebug(df_dist_to_classes.values)

        # Extract only scores greater than threshold
        df_dist_to_classes = df_dist_to_classes[df_dist_to_classes[IntentEngine.COL_SCORE] >= score_min_threshold]
        log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                               + ': After extracting matches: ')
        log.Log.debugdebug(df_dist_to_classes)
        log.Log.debugdebug(df_dist_to_classes.values)

        # Reset indexes
        # df_dist_to_classes = df_dist_to_classes.reset_index(drop=True)

        if self.do_profiling:
            b = prf.Profiling.stop()
            log.Log.critical('.' + space_profiling
                        + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
                        + ' PROFILING Intent (Match & Score): ' + prf.Profiling.get_time_dif_str(a, b))

        if self.do_profiling:
            b = prf.Profiling.stop()
            log.Log.critical('.   '
                        + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
                        + ' PROFILING Intent (reduced features = '
                        + str(self.reduce_features) + ') End: '
                        + prf.Profiling.get_time_dif_str(start_func, b))

        if df_dist_to_classes.shape[0] >= 1:
            col_command = df_dist_to_classes.index
            df_return = df_dist_to_classes.copy()
            df_return.index = range(0, df_return.shape[0], 1)
            df_return[IntentEngine.COL_COMMAND] = col_command
            return df_return[0:min(top, df_return.shape[0])]
        else:
            return None

    def get_closest_distance_to_training_sample(
            self,
            inputtext_fv_2d,
            intents_list
    ):
        if (intents_list is None) or (len(intents_list) == 0):
            return None

        a = prf.Profiling.start()

        log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                           + ': Intents List "' + str(intents_list) + '".')

        condition = np.isin(self.index_command_fv_training_data_ro, intents_list)
        df_fv_td_intents = self.df_fv_training_data_ro.loc[condition]
        # Make sure to change the index to the command or intent ID, otherwise we can't do groupby() later
        index_td_intents = self.index_command_fv_training_data_ro[condition]
        df_fv_td_intents.index = index_td_intents
        log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                           + ': Indexes of filtered training data for intents list ' + str(intents_list))
        log.Log.debugdebug(index_td_intents)

        if df_fv_td_intents.shape[0] == 0:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Warning! For intents ' + str(intents_list) + ' - No Training Data Filtered!'
            log.Log.warning(errmsg)
            raise Exception(errmsg)

        # Create a matrix of similar rows (fv_text)
        tmp_text_matrix = np.repeat(a=inputtext_fv_2d, repeats=df_fv_td_intents.shape[0], axis=0)
        fv_td_intent_matrix = np.array(df_fv_td_intents)
        log.Log.debugdebug(fv_td_intent_matrix)

        # Get difference
        dif_fv_text = fv_td_intent_matrix - tmp_text_matrix
        log.Log.debugdebug(dif_fv_text)

        # Square every element in the matrix
        dif_fv_text2 = np.power(dif_fv_text, 2)
        # Sum every row to create a single vector
        dif_fv_text2_sum = dif_fv_text2.sum(axis=1)
        log.Log.debugdebug(dif_fv_text2_sum)

        # Take the square root of every element in the single column matrix as distance
        distance_fv_text = np.power(dif_fv_text2_sum, 0.5)
        # Convert to a single row matrix
        distance_fv_text = distance_fv_text.transpose()
        distance_fv_text = distance_fv_text.tolist()
        log.Log.debugdebug(distance_fv_text)

        df_distance_fv_text = pd.DataFrame({
            IntentEngine.COL_COMMAND:                 df_fv_td_intents.index,
            IntentEngine.COL_DISTANCE_CLOSEST_SAMPLE: np.round(distance_fv_text, IntentEngine.ROUND_BY)
        })
        log.Log.debugdebug(df_distance_fv_text)
        df_distance_fv_text = df_distance_fv_text.sort_values(
            by        = [IntentEngine.COL_COMMAND, IntentEngine.COL_DISTANCE_CLOSEST_SAMPLE],
            ascending = True
        )
        log.Log.debugdebug(df_distance_fv_text)

        pd_series_min = df_distance_fv_text.groupby([IntentEngine.COL_COMMAND])[
            IntentEngine.COL_DISTANCE_CLOSEST_SAMPLE
        ].min()

        df_min = pd.DataFrame(data={
            IntentEngine.COL_COMMAND                : pd_series_min.index.tolist(),
            IntentEngine.COL_DISTANCE_CLOSEST_SAMPLE: pd_series_min.tolist()
        })
        df_min = df_min.sort_values(by=[IntentEngine.COL_COMMAND], ascending=True)
        df_min = df_min.set_index(IntentEngine.COL_COMMAND)
        log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                           + ' Min Distance to samples by commands:')
        log.Log.debugdebug(df_min)

        if self.do_profiling:
            b = prf.Profiling.stop()
            log.Log.info('. END PROFILING Intent (get_closest_distance_to_training_sample()): '
                         + prf.Profiling.get_time_dif_str(a, b))

        return df_min

    #
    # We try to reduce the size of the RFV here, by removing the least significant columns
    #
    def reduce_and_optimize_features(
            self
    ):
        #
        # Find features/columns with,
        #   1. The most zeros
        #   2. The least sum
        #
        rfv_ori_dim = self.df_rfv_ro.shape
        n_rfv = rfv_ori_dim[0]
        n_features = rfv_ori_dim[1]
        # We remove those features that cover less than threshold_feature_coverage_percent% of the RFV vectors
        threshold_feature_coverage_percent = 0.5
        threshold_feature_coverage = (threshold_feature_coverage_percent / 100) * n_rfv
        log.Log.important('Start reduce & optimize RFV...')
        log.Log.important('RFV original dimensions ' + str(rfv_ori_dim))
        log.Log.important('RFV count is ' + str(n_rfv) + ' with ' + str(n_features) + ' features.')
        log.Log.important('Preparing to remove features covering less than '
                    + str(threshold_feature_coverage_percent) + '% or count less than '
                    + str(threshold_feature_coverage) + ' of RFVs...')

        #
        # Count how many non-zeros in columns
        #
        features_nonzerocount = 1*(self.df_rfv_ro>0)
        features_nonzerocount = features_nonzerocount.sum(axis=0)
        features_nonzerocount = features_nonzerocount.sort_values(axis=0, ascending=False)
        # Remove features/columns that cover less than ?% of the RFV vectors
        log.Log.debug('Length of series = ' + str(len(features_nonzerocount)))
        features_keep = features_nonzerocount[features_nonzerocount > threshold_feature_coverage]

        log.Log.debug('Features:')
        log.Log.debug(features_keep.to_dict())
        log.Log.debug('Length of kept features series = ' + str(len(features_keep)))

        features_keep_list = features_keep.index.tolist()
        log.Log.debug('Features to keep:')
        log.Log.debug(features_keep_list)

        #
        # Now we modify the "read-only" variables (make to to get a mutex lock)
        #
        self.__mutex.acquire()

        try:
            #
            # Removing features may cause some RFV to become 0, need to deal with that also
            #
            self.df_rfv_ro = self.df_rfv_ro.loc[:, features_keep_list]
            if not self.minimal:
                self.df_fv_training_data_ro = self.df_fv_training_data_ro.loc[:, features_keep_list]

            # Check which RFVs became 0
            df_rfv_zeroes = self.df_rfv_ro[ self.df_rfv_ro.sum(axis=1)==0 ]
            rfvs_to_throw = df_rfv_zeroes.index.tolist()
            log.Log.warning('RFVs that became zero:')
            log.Log.warning(df_rfv_zeroes.index)
            log.Log.warning(df_rfv_zeroes)
            log.Log.warning(rfvs_to_throw)

            # Drop the RFVs with 0 vector
            self.df_rfv_ro = self.df_rfv_ro.drop(rfvs_to_throw)

            # Which training data to remain kept
            df_td = pd.DataFrame()

            # Renormalize RFV, loop through indexes
            for idx in self.df_rfv_ro.index:
                log.Log.debug('.   Doing index ' + str(idx) + '...')

                if not self.minimal:
                    filter_intent = str(idx) + '-' + '[0-9]+.*'
                    df_tmp = self.df_fv_training_data_ro.filter(regex=filter_intent, axis=0)
                    if df_tmp.shape[0] == 0:
                        log.Log.warning('Warning! For intent [' + str(idx) + '] - No Training Data Filtered!')
                        continue
                    df_td = df_td.append(df_tmp)

                v = np.array(self.df_rfv_ro.loc[idx].values)
                # Normalize v
                mag = np.multiply(v, v)
                mag = np.sum(mag) ** 0.5
                v = v / mag

                mag_check = np.multiply(v, v)
                mag_check = np.sum(mag_check) ** 0.5
                if abs(mag_check - 1) > 0.000000001:
                    raise Exception('Not normalized!')

                log.Log.debug(v)
                self.df_rfv_ro.at[idx] = v

            log.Log.important('Renormalized RFV:')
            log.Log.important('New RFV dimensions: ' + str(self.df_rfv_ro.shape))
            log.Log.debug(self.df_rfv_ro)

            # Now do the same for IDF
            self.df_idf_ro = self.df_idf_ro[self.df_idf_ro['Word'].isin(features_keep_list)]
            log.Log.important('Renormalized IDF:')
            log.Log.important('New IDF dimensions: ' + str(self.df_rfv_ro.shape))
            log.Log.debug(self.df_idf_ro)

            # Remember to update also the cached numpy array
            self.df_rfv_np_array_ro = np.array(self.df_rfv_ro.values)

            # Now do the same for training data
            if not self.minimal:
                self.df_fv_training_data_ro = df_td.copy()
                # Renormalize TD, loop through indexes
                for idx in self.df_fv_training_data_ro.index:
                    v = np.array(self.df_fv_training_data_ro.loc[idx].values)
                    # Normalize v
                    mag = np.multiply(v, v)
                    mag = np.sum(mag) ** 0.5
                    if mag < 0.000000001:
                        log.Log.warning(
                            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': Training data at index ' + str(idx) + ' became 0. Removing..'
                        )
                        df_td = df_td.drop(idx)
                        continue
                    v = v / mag

                    mag_check = np.multiply(v, v)
                    mag_check = np.sum(mag_check) ** 0.5
                    if abs(mag_check - 1) > 0.000000001:
                        raise Exception('Not normalized!')

                    log.Log.debug(v)
                    df_td.at[idx] = v

                # Reassign back df_td
                self.df_fv_training_data_ro = df_td

                log.Log.important('New Training Data dimensions: ' + str(self.df_fv_training_data_ro.shape))
                # This is very slow, we do it first! Cache it!
                self.index_command_fv_training_data_ro = np.array(self.get_command_index_from_training_data_index())
                log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                   + ': Reduced Features Index of training data by command "'
                                   + str(list(self.index_command_fv_training_data_ro)) + '".')
                log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                   + ': Reduced Features Index original  "'
                                   + str(list(self.df_fv_training_data_ro.index)) + '".')

            # Now for RFV Furthest Distance of sample
            log.Log.debug('Nothing to do for RFV Furthest Distance:')
            log.Log.debug('RFV Furthest Distance dimensions: ' + str(self.df_rfv_dist_furthest_ro.shape))

            #
            # 2. Remove columns with sums < ...
            #
            # This is s pandas series
            colsums = self.df_rfv_ro.sum(axis=0)
            colsums = colsums.sort_values(axis=0, ascending=False)
            #log.Log.log(colsums)

            self.sanity_check()

            self.is_reduced_features_ready = True
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Botkey "' + self.bot_key + '" failed to load reduced features. Exception ' + str(ex)
            log.Log.critical(errmsg)
            raise Exception(errmsg)
        finally:
            self.__mutex.release()

        return True


if __name__ == '__main__':
    topdir = '/Users/mark.tan/git/mozg'
    db_profile = 'mario2'
    account_id = 3
    bot_id = 4
    bot_lang = 'cn'

    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_DEBUG_1

    it = IntentEngine(
        lang = 'cn',
        bot_key = 'demo_msmodel_testdata',
        # bot_key = botia.BotIntentAnswer.get_bot_key(
        #     db_profile = db_profile,
        #     account_id = account_id,
        #     bot_id     = bot_id,
        #     lang       = bot_lang
        # ),
        dir_rfv_commands = topdir + '/app.data/models',
        dirpath_synonymlist = topdir + '/nlp.data/app/chats',
        do_profiling = True,
        minimal = False,
        verbose = 2
    )
    it.do_background_load()

    txt_segm = '怎么 吗'
    #txt_segm = '你 是不是 野种'
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_DEBUG_2

    res = it.get_text_class(
        text_segmented = txt_segm,
        weigh_idf      = True,
        verbose        = 3,
        not_necessary_to_use_training_data_samples = False
    )
    print(res.values)
    exit(0)

    res = it.get_text_class(
        text_segmented = txt_segm,
        weigh_idf      = True,
        verbose        = 3,
        not_necessary_to_use_training_data_samples = False
    )
    print(res.values)

    exit(0)
