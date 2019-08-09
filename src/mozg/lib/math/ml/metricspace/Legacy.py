# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import threading
import datetime as dt
import mozg.lib.math.ml.TrainingDataModel as tdm
import mozg.utils.Log as log
from inspect import currentframe, getframeinfo
import mozg.lib.math.Constants as const
import mozg.lib.math.ml.ModelInterface as modelIf
import mozg.lib.math.NumpyUtil as npUtil
import mozg.utils.Profiling as prf


class Legacy:

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
    COL_COMMAND = 'Command'
    COL_DISTANCE_TO_RFV = 'DistToRfv'
    COL_DISTANCE_CLOSEST_SAMPLE = 'DistToSampleClosest'
    COL_DISTANCE_FURTHEST = 'DistToRfvThreshold'
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

    #
    # Radius min/max
    # TODO For certain classes, all points are different, and this min cluster will not work
    #
    CLUSTER_RADIUS_MAX = 0.5
    N_CLUSTER_MAX = 5

    def __init__(
            self,
            # Unique identifier to identify this set of trained data+other files after training
            identifier_string,
            # Directory to keep all our model files
            dir_path_model,
            # Training data in TrainingDataModel class type
            training_data = None,
            # No loading of heavy things like training data
            minimal = False,
            # From all the initial features, how many we should remove by quartile. If 0 means remove nothing.
            key_features_remove_quartile = 0,
            # Initial features to remove, should be an array of numbers (0 index) indicating column to delete in training data
            stop_features = (),
            # If we will create an "IDF" based on the initial features
            weigh_idf = False,
            do_profiling = False
    ):
        super(Legacy, self).__init__(
            identifier_string = identifier_string,
            dir_path_model    = dir_path_model
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

        self.minimal = minimal
        self.key_features_remove_quartile = key_features_remove_quartile
        self.stop_features = stop_features
        self.weigh_idf = weigh_idf
        self.do_profiling = do_profiling

        self.fpath_updated_file = self.dir_path_model + '/' + self.identifier_string + '.lastupdated.txt'
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
        self.fpath_idf = self.dir_path_model + '/' + self.identifier_string + '.' + 'chatbot.words.idf.csv'
        if not os.path.isfile(self.fpath_idf):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': IDF file "' + self.fpath_idf + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)
        self.df_idf_ro = None

        self.fpath_rfv = self.dir_path_model + '/' + self.identifier_string + '.' + 'chatbot.commands.rfv.csv'
        if not os.path.isfile(self.fpath_rfv):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': RFV file "' + self.fpath_rfv + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)
        self.df_rfv_ro = None
        # This is the cached data frame version of the RFV in numpy array form
        self.df_rfv_np_array_ro = None

        self.fpath_rfv_dist = self.dir_path_model + '/' + self.identifier_string + '.' + 'chatbot.commands.rfv.distance.csv'
        if not os.path.isfile(self.fpath_rfv):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': RFV furthest distance file "' + self.fpath_rfv_dist + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)
        self.df_rfv_dist_furthest_ro = None

        self.fpath_fv_all = self.dir_path_model + '/' + self.identifier_string + '.' + 'chatbot.fv.all.csv'
        if not os.path.isfile(self.fpath_rfv):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Training Data file "' + self.fpath_fv_all + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)
        # Used to zoom into an intent/command group and compare against exact training data in that group
        self.df_fv_training_data_ro = None
        self.index_command_fv_training_data_ro = None

        # Used to finally confirm if it is indeed the intent by matching top keywords in the intent category
        # self.df_intent_tf_ro = None

        self.hash_df_rfv_ro = None
        self.hash_df_rfv_np_array_ro = None
        self.hash_df_idf_ro = None
        self.hash_index_command_fv_training_data_ro = None
        # Don't do for training data, it takes too long
        self.hash_df_rfv_dist_furthest_ro = None

        self.count_intent_calls = 0

        ###################################################################################################
        # Initializations
        ###################################################################################################
        # After this initialization, no more modifying of the above class variables
        self.is_rfv_ready = False
        self.is_training_data_ready = False
        self.is_reduced_features_ready = False

        self.bot_training_start_time = None
        self.bot_training_end_time = None
        self.logs_training = None
        self.is_training_done = False
        self.__mutex_training = threading.Lock()

        return

    #
    # Model interface override
    #
    def run(self):
        self.__mutex_training.acquire()
        try:
            self.bot_training_start_time = dt.datetime.now()
            self.log_training = []
            self.train()
            self.bot_training_end_time = dt.datetime.now()
        except Exception as ex:
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Identifier ' + str(self.identifier_string) + '" training exception: ' + str(ex) + '.'
            )
        finally:
            self.is_training_done = True
            self.__mutex_training.release()

        log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Identifier ' + str(self.identifier_string) + '" trained successfully.')

    def reset_ready_flags(self):
        # When everything is ready
        self.model_loaded = False

        # Below in more detailed breakdown by RFV, training data, reduced features
        self.is_rfv_ready = False
        self.is_training_data_ready = False
        self.is_reduced_features_ready = False

    #
    # Model interface override
    #
    def is_model_ready(
            self
    ):
        return self.model_loaded

    #
    # Model interface override
    #
    def get_model_features(
            self
    ):
        return npUtil.NumpyUtil.convert_dimension(arr=self.trai.x_name, to_dim=1)

    #
    # Model interface override
    #
    def check_if_model_updated(
            self
    ):
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
                self.__mutex_training.acquire()
                # Reset model flags to not ready
                self.reset_ready_flags()
                self.model_updated_time = updated_time
            finally:
                self.__mutex_training.release()
            return True
        else:
            return False

    #
    # Given our training data x, we get the IDF of the columns x_name
    #
    @staticmethod
    def get_feature_weight_idf(
            x,
            y,
            x_name,
            feature_presence_only_in_label_training_data = True
    ):
        df_tmp = pd.DataFrame(data=x, index=y)

        # Group by the labels y, as they are not unique
        df_agg_sum = df_tmp.groupby(df_tmp.index).sum()
        np_agg_sum = df_agg_sum.values

        # Get presence only by cell, then sum up by columns to get total presence by document
        np_feature_presence = np_agg_sum
        if feature_presence_only_in_label_training_data:
            np_feature_presence = (np_agg_sum>0)*1

        # Sum by column axis=0
        np_feature_presence_sum = np.sum(np_feature_presence, axis=0)
        log.Log.debug(
            str(Legacy.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tAggregated sum by labels:\n\r' + str(np_agg_sum)
            + '\n\r\tPresence array:\n\r' + str(np_feature_presence)
            + '\n\r\tPresence sum:\n\r' + str(np_feature_presence_sum)
            + '\n\r\tx_names: ' + str(x_name) + '.'
        )

        # Total document count
        n_documents = np_feature_presence.shape[0]
        log.Log.important(
            str(Legacy.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Total unique documents/intents to calculate IDF = ' + str(n_documents)
        )

        # If using outdated np.matrix, this IDF will be a (1,n) array, but if using np.array, this will be 1-dimensional vector
        # TODO RuntimeWarning: divide by zero encountered in true_divide
        idf = np.log(n_documents / np_feature_presence_sum)
        # Replace infinity with 1 count or log(n_documents)
        idf[idf==np.inf] = np.log(n_documents)
        # If only 1 document, all IDF will be zero, we will handle below
        if n_documents <= 1:
            log.Log.warning(
                str(Legacy.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Only ' + str(n_documents) + ' document in IDF calculation. Setting IDF to 1.'
            )
            idf = np.array([1]*len(x.shape[1]))
        log.Log.debug(
            str(Legacy.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tWeight IDF:\n\r' + str(idf)
        )
        return idf

    #
    # Get all class proximity scores to a point
    #
    def calc_proximity_class_score_to_point(
            self,
            # ndarray type of >= 2 dimensions, with 1 row (or 1st dimension length == 1)
            x_distance,
            y_label,
            top = modelIf.ModelInterface.MATCH_TOP
    ):
        prf_start = prf.Profiling.start()

        if ( type(x_distance) is not np.ndarray ):
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Wrong type "' + type(x_distance) + '" to predict classes. Not ndarray.'
            )

        if x_distance.ndim > 1:
            if x_distance.shape[0] != 1:
                raise Exception(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Expected x has only 1 row got c shape ' + str(x_distance.shape)
                    + '". x = ' + str(x_distance)
                )
            else:
                x_distance = x_distance[0]

        log.Log.debugdebug('x_distance: ' + str(x_distance) + ', y_label ' + str(y_label))

        #
        # Normalize distance to between 0 and 1
        #
        x_distance_norm = x_distance / MetricSpaceModel.HPS_MAX_EUCL_DIST
        log.Log.debugdebug('distance normalized: ' + str(x_distance_norm))

        # Theoretical Inequality check
        check_less_than_max = np.sum(1 * (x_distance_norm > 1+const.Constants.SMALL_VALUE))
        check_greater_than_min = np.sum(1 * (x_distance_norm < 0-const.Constants.SMALL_VALUE))

        if (check_less_than_max > 0) or (check_greater_than_min > 0):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Distance ' + str(x_distance) + ' fail theoretical inequality test.' \
                     + ' Distance normalized:\n\r' + str(x_distance_norm)
            log.Log.critical(errmsg)
            raise Exception(errmsg)

        # x_score = np.round(100 - x_distance_norm*100, 1)

        df_score = pd.DataFrame({
            MetricSpaceModel.TERM_CLASS: y_label,
            # MetricSpaceModel.TERM_SCORE: x_score,
            MetricSpaceModel.TERM_DIST:  x_distance,
            MetricSpaceModel.TERM_DISTNORM: x_distance_norm
        })
        # Sort distances
        # df_score.sort_values(by=[MetricSpaceModel.TERM_DIST], ascending=True, inplace=True)
        # df_score = df_score[0:top]
        # df_score.reset_index(drop=True, inplace=True)
        # log.Log.debugdebug('DF SCORE 1:\n\r' + str(df_score))

        # Aggregate class by min distance, don't make class index.
        df_score = df_score.groupby(by=[MetricSpaceModel.TERM_CLASS], as_index=False, axis=0).min()
        # log.Log.debugdebug('DF SCORE 2:\n\r' + str(df_score))

        # Put score last (because we need to do groupby().min() above, which will screw up the values
        # as score is in the reverse order with distances) and sort scores
        np_distnorm = np.array(df_score[MetricSpaceModel.TERM_DISTNORM])
        df_score[MetricSpaceModel.TERM_SCORE] = np.round(100 - np_distnorm*100, 1)
        df_score.sort_values(by=[MetricSpaceModel.TERM_SCORE], ascending=False, inplace=True)
        # Make sure indexes are conventional 0,1,2,...
        df_score = df_score[0:min(top,df_score.shape[0])]
        df_score.reset_index(drop=True, inplace=True)

        log.Log.debugdebug('x_score:\n\r' + str(df_score))

        if self.do_profiling:
            prf_dur = prf.Profiling.get_time_dif(prf_start, prf.Profiling.stop())
            log.Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING calc_proximity_class_score_to_point(): ' + str(round(1000*prf_dur,0))
                + ' milliseconds.'
            )

        return df_score

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
            x,
            include_rfv = False,
            # This will slow down by a whopping 20ms!!
            include_match_details = False,
            top = modelIf.ModelInterface.MATCH_TOP
    ):
        prf_start = prf.Profiling.start()

        if type(x) is not np.ndarray:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Wrong type "' + type(x) + '" to predict classes. Not ndarray.'
            )

        if x.ndim < 2:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Expected x dimension >= 2, got ' + str(x.ndim) + '".'
            )

        if x.shape[0] < 1:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Expected x has at least 1 row got c shape ' + str(x.shape) + '".'
            )

        log.Log.debugdebug('x:\n\r' + str(x))

        match_details = {}
        x_classes = []
        top_class_distance = []
        mse = 0
        mse_norm = 0

        #
        # Calculate distance to x_ref & x_clustered for all the points in the array passed in
        #
        for i in range(x.shape[0]):
            v = npUtil.NumpyUtil.convert_dimension(arr=x[i], to_dim=2)
            predict_result = self.predict_class(
                x           = v,
                include_rfv = include_rfv,
                include_match_details = include_match_details
            )
            x_classes.append(list(predict_result.predicted_classes))
            top_class_distance.append(predict_result.top_class_distance)
            if include_match_details:
                match_details[i] = predict_result.match_details
            metric = predict_result.top_class_distance
            metric_norm = metric / MetricSpaceModel.HPS_MAX_EUCL_DIST
            mse += metric ** 2
            mse_norm += metric_norm ** 2

        # Mean square error MSE and MSE normalized
        top_class_distance = np.array(top_class_distance)

        class retclass:
            def __init__(
                    self,
                    predicted_classes,
                    top_class_distance,
                    match_details,
                    mse,
                    mse_norm
            ):
                self.predicted_classes = predicted_classes
                # The top class and shortest distances (so that we can calculate sum of squared error
                self.top_class_distance = top_class_distance
                self.match_details = match_details
                self.mse = mse
                self.mse_norm = mse_norm
                return

        retval = retclass(
            predicted_classes  = x_classes,
            top_class_distance = top_class_distance,
            match_details      = match_details,
            mse                = mse,
            mse_norm           = mse_norm
        )

        if self.do_profiling:
            prf_dur = prf.Profiling.get_time_dif(prf_start, prf.Profiling.stop())
            # Duration per prediction
            dpp = round(1000 * prf_dur / x.shape[0], 0)
            log.Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING predict_classes(): ' + str(prf_dur)
                + ', time per prediction = ' + str(dpp) + ' milliseconds.'
            )

        return retval

    #
    # Model interface override
    #
    def predict_class(
            self,
            # ndarray type of >= 2 dimensions, single point/row array
            x,
            include_rfv = False,
            # This will slow down by a whopping 20ms!!
            include_match_details = False,
            top = modelIf.ModelInterface.MATCH_TOP
    ):
        prf_start = prf.Profiling.start()

        if type(x) is not np.ndarray:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Wrong type "' + type(x) + '" to predict classes. Not ndarray.'
            )

        if x.ndim < 2:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Expected x dimension >= 2, got ' + str(x.ndim) + '".'
            )

        if x.shape[0] != 1:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Expected x has 1 row got c shape ' + str(x.shape) + '".'
            )

        log.Log.debug('x:\n\r' + str(x))

        #
        # Weigh x with idf
        #
        x_weighted = x * self.model_data.idf
        log.Log.debugdebug('x_weighted:\n\r' + str(x_weighted))

        v = x_weighted.copy()

        #
        # Normalize x_weighted
        #
        mag = np.sum(np.multiply(v,v)**0.5)
        #
        # In the case of 0 magnitude, we put the point right in the center of the hypersphere at 0
        #
        if mag < const.Constants.SMALL_VALUE:
            v = np.multiply(v, 0)
        else:
            v = v / mag
        log.Log.debugdebug('v normalized:\n\r' + str(v))

        #
        # Calculate distance to x_ref & x_clustered for all the points in the array passed in
        #
        # Returns absolute distance
        distance_x_ref = None
        y_ref_rel = None
        if include_rfv:
            retobj = npUtil.NumpyUtil.calc_distance_of_point_to_x_ref(
                v=v, x_ref=self.model_data.x_ref, y_ref=self.model_data.y_ref, do_profiling=self.do_profiling)
            distance_x_ref = retobj.distance_x_rel
            y_ref_rel = retobj.y_rel
        retobj = npUtil.NumpyUtil.calc_distance_of_point_to_x_ref(
            v=v, x_ref=self.model_data.x_clustered, y_ref=self.model_data.y_clustered, do_profiling=self.do_profiling)
        distance_x_clustered = retobj.distance_x_rel
        y_clustered_rel = retobj.y_rel

        # We combine all the reference points, or sub-classes of the classes. Thus each class
        # is represented by more than one point, reference sub_classes.
        if include_rfv:
            x_distance = np.append(distance_x_ref, distance_x_clustered)
            y_distance = np.append(y_ref_rel, y_clustered_rel)
        else:
            x_distance = distance_x_clustered
            y_distance = y_clustered_rel
        log.Log.debugdebug('x_distance combined:\n\r' + str(x_distance))
        log.Log.debugdebug('y_distance combined:\n\r' + str(y_distance))

        # Get the score of point relative to all classes.
        df_class_score = self.calc_proximity_class_score_to_point(
            x_distance = x_distance,
            y_label    = y_distance,
            top        = top
        )
        log.Log.debugdebug('df_class_score:\n\r' + str(df_class_score))

        top_classes_label = list(df_class_score[MetricSpaceModel.TERM_CLASS])
        top_class_distance = df_class_score[MetricSpaceModel.TERM_DIST].loc[df_class_score.index[0]]

        # Get the top class
        log.Log.debugdebug('x_classes:\n\r' + str(top_classes_label))
        log.Log.debugdebug('Class for point:\n\r' + str(top_classes_label))
        log.Log.debugdebug('distance to rfv:\n\r' + str(distance_x_ref))
        log.Log.debugdebug('distance to x_clustered:\n\r' + str(distance_x_clustered))
        log.Log.debugdebug('top class distance:\n\r' + str(top_class_distance))

        # Mean square error MSE and MSE normalized
        top_class_distance = np.array(top_class_distance)

        retval = MetricSpaceModel.predict_class_retclass(
            predicted_classes  = np.array(top_classes_label),
            top_class_distance = top_class_distance,
            match_details      = df_class_score,
        )

        if self.do_profiling:
            prf_dur = prf.Profiling.get_time_dif(prf_start, prf.Profiling.stop())
            # Duration per prediction
            dpp = round(1000 * prf_dur / x.shape[0], 0)
            log.Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING predict_classes(): ' + str(prf_dur)
                + ', time per prediction = ' + str(dpp) + ' milliseconds.'
            )

        return retval
    #
    # Model interface override
    #
    # TODO: Include training/optimization of vector weights to best define the category and differentiate with other categories.
    # TODO: Currently uses static IDF weights.
    #
    def train(
            self
    ):
        prf_start = prf.Profiling.start()

        if self.training_data is None:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Cannot train without training data for identifier "' + self.identifier_string + '"'
            )

        self.__mutex_training.acquire()
        try:
            self.log_training = []

            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Training for identifier=' + self.identifier_string
                + '. Using key features remove quartile = ' + str(self.key_features_remove_quartile)
                + ', stop features = [' + str(self.stop_features) + ']'
                + ', weigh by IDF = ' + str(self.weigh_idf)
                , log_list = self.log_training
            )

            #
            # Here training data must be prepared in the correct format already
            # Значит что множество свойств уже объединено как одно (unified features)
            #
            log.Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + '\n\r\tTraining data:\n\r' + str(self.training_data.get_x())
                + '\n\r\tx names: ' + str(self.training_data.get_x_name())
                + '\n\r\ty labels: ' + str(self.training_data.get_y())
            )

            #
            # Get IDF first
            # The function of these weights are nothing more than dimension reduction
            # TODO: IDF may not be the ideal weights, design an optimal one.
            #
            if self.weigh_idf:
                # Sum x by class
                self.model_data.idf = MetricSpaceModel.get_feature_weight_idf(
                    x      = self.training_data.get_x(),
                    y      = self.training_data.get_y(),
                    x_name = self.training_data.get_x_name()
                )
                # Standardize to at least 2-dimensional, easier when weighting x
                self.model_data.idf = npUtil.NumpyUtil.convert_dimension(
                    arr    = self.model_data.idf,
                    to_dim = 2
                )

                log.Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + '\n\r\tIDF values:\n\r' + str(self.model_data.idf)
                )

                # This will change the x in self.training data
                self.training_data.weigh_x(w=self.model_data.idf[0])
            else:
                self.model_data.idf = np.array([1]*self.training_data.get_x_name().shape[0])
                # Standardize to at least 2-dimensional, easier when weighting x
                self.model_data.idf = npUtil.NumpyUtil.convert_dimension(
                    arr    = self.model_data.idf,
                    to_dim = 2
                )

            #
            # Initizalize model data
            #
            # Refetch again after weigh
            x = self.training_data.get_x()
            y = self.training_data.get_y()
            self.model_data.x_name = self.training_data.get_x_name()
            self.model_data.idf = self.training_data.get_w()

            # Unique y or classes
            # We do this again because after weighing, it will remove bad rows, which might cause some y
            # to disappear
            self.model_data.y_unique = np.array(list(set(y)))

            log.Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + '\n\r\tx weighted by idf and renormalized:\n\r' + str(x)
                + '\n\r\ty\n\r' + str(y)
                + '\n\r\tx_name\n\r' + str(self.model_data.x_name)
                , log_list=self.log_training
            )

            #
            # Get RFV for every command/intent, representative feature vectors by command type
            #

            # 1. Cluster training data of the same intent.
            #    Instead of a single RFV to represent a single intent, we should have multiple.
            xy_clstr = MetricSpaceModel.get_clusters(
                x      = x,
                y      = y,
                x_name = self.model_data.x_name
            )
            self.model_data.x_clustered = xy_clstr.x_cluster
            self.model_data.y_clustered = xy_clstr.y_cluster
            self.model_data.y_clustered_radius = xy_clstr.y_cluster_radius

            #
            # RFV Derivation
            #
            m = np.zeros((len(self.model_data.y_unique), len(self.model_data.x_name)))
            # Temporary only this data frame
            df_x_ref = pd.DataFrame(
                m,
                columns = self.model_data.x_name,
                index   = self.model_data.y_unique
            )
            self.model_data.df_y_ref_radius = pd.DataFrame(
                {
                    MetricSpaceModel.TERM_CLASS: list(self.model_data.y_unique),
                    MetricSpaceModel.TERM_RADIUS: [MetricSpaceModel.HPS_MAX_EUCL_DIST]*len(self.model_data.y_unique),
                },
                index = self.model_data.y_unique
            )

            #
            # Derive x_ref and y_ref
            #
            for cs in self.model_data.y_unique:
                log.Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Doing class [' + str(cs) + ']'
                    , log_list = self.log_training
                )
                # Extract class points
                class_points = x[y==cs]
                #
                # Reference feature vector for the command is the average of all feature vectors
                #
                rfv = np.sum(class_points, axis=0) / class_points.shape[0]
                # Renormalize it again
                # At this point we don't have to check if it is a 0 vector, etc. as it was already done in TrainingDataModel
                # after weighing process
                normalize_factor = np.sum(np.multiply(rfv, rfv)) ** 0.5
                if normalize_factor < const.Constants.SMALL_VALUE:
                    raise Exception(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Normalize factor for rfv in class "' + str(cs) + '" is 0.'
                    )
                rfv = rfv / normalize_factor
                # A single array will be created as a column dataframe, thus we have to name the index and not columns
                df_x_ref.at[cs] = rfv

                check_normalized = np.sum(np.multiply(rfv,rfv))**0.5
                if abs(check_normalized-1) > const.Constants.SMALL_VALUE:
                    errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                             + ': Warning! RFV for class [' + str(cs) + '] not 1, but [' + str(check_normalized) + '].'
                    raise Exception(errmsg)
                else:
                    log.Log.info(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Check RFV class "' + str(cs) + '" normalized ok [' + str(check_normalized) + '].'
                    )

                #
                # Get furthest point of classification to rfv
                # This will be used to accept or reject a classified point to a particular class,
                # once the nearest class is found (in which no class is found then).
                #
                # Minimum value of threshold, don't allow 0's
                radius_max = -1
                for i in range(0, class_points.shape[0], 1):
                    p = class_points[i]
                    dist_vec = rfv - p
                    dist = np.sum(np.multiply(dist_vec, dist_vec)) ** 0.5
                    log.Log.debugdebug(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + '   Class ' + str(cs) + ' check point ' + str(i)
                        + ', distance= ' + str(dist) + '. Point ' + str(class_points[i])
                        + ' with RFV ' + str(rfv)
                    )
                    if dist > radius_max:
                        radius_max = dist
                        self.model_data.df_y_ref_radius[MetricSpaceModel.TERM_RADIUS].at[cs] = dist

                log.Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Class "' + str(cs) + '". Max Radius = '
                    + str(self.model_data.df_y_ref_radius[MetricSpaceModel.TERM_RADIUS].loc[cs])
                )
            df_x_ref.sort_index(inplace=True)
            self.model_data.y_ref = np.array(df_x_ref.index)
            self.model_data.x_ref = np.array(df_x_ref.values)
            log.Log.debug('**************** ' + str(self.model_data.y_ref))

            if self.do_profiling:
                log.Log.important(
                    str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                    + ' PROFILING train(): ' + prf.Profiling.get_time_dif_str(prf_start, prf.Profiling.stop())
                )

            self.persist_model_to_storage()
            self.persist_training_data_to_storage(td=self.training_data)
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Training exception for identifier "' + str(self.identifier_string) + '".'\
                     + ' Exception message ' + str(ex) + '.'
            log.Log.error(errmsg)
            raise ex
        finally:
            self.__mutex_training.release()

        return

    def persist_model_to_storage(
            self
    ):
        prf_start = prf.Profiling.start()
        self.model_data.persist_model_to_storage()
        if self.do_profiling:
            log.Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING persist_model_to_storage(): '
                + prf.Profiling.get_time_dif_str(prf_start, prf.Profiling.stop())
            )
        return

    #
    # When we index FV of all training data, we keep in the format 'intentId-1', 'intentId-2,...
    # So we just filter off the '-\d+'
    #
    @staticmethod
    def retrieve_intent_from_training_data_index(
            index_list,
            convert_to_type = 'str'
    ):
        for i in range(0,len(index_list),1):
            v = str(index_list[i])
            v = re.sub(
                pattern = '[-]\d+.*',
                repl    = '',
                string  = v
            )
            if convert_to_type == 'int':
                index_list[i] = int(v)
            else:
                index_list[i] = v
        return index_list

    def get_command_index_from_training_data_index(self):
        # This is very slow, we do it first! Cache it!
        convert_to_type = 'str'
        if not Legacy.CONVERT_COMMAND_INDEX_TO_STR:
            convert_to_type = 'int'

        # We derive the intent id or command from the strings '888-1', '888-2',...
        # by removing the ending '-1', '-2', ...
        # This will speed up filtering of training data later by command.
        index_command = \
            Legacy.retrieve_intent_from_training_data_index(
                index_list      = list(self.df_fv_training_data_ro.index),
                convert_to_type = convert_to_type
            )
        return index_command

    #
    # Model interface override
    #
    def load_model_parameters(
            self
    ):
        prf_start = prf.Profiling.start()

        try:
            self.__mutex_training.acquire()

            # TODO This data actually not needed
            self.df_idf_ro = pd.read_csv(
                filepath_or_buffer = self.fpath_idf,
                sep       =',',
                index_col = 'INDEX'
            )
            if Legacy.CONVERT_COMMAND_INDEX_TO_STR:
                # Convert Index column to string
                self.df_idf_ro.index = self.df_idf_ro.index.astype(str)
            log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': IDF Data: Read ' + str(self.df_idf_ro.shape[0]) + ' lines')

            self.df_rfv_ro = pd.read_csv(
                filepath_or_buffer = self.fpath_rfv,
                sep       = ',',
                index_col = 'INDEX'
            )
            if Legacy.CONVERT_COMMAND_INDEX_TO_STR:
                # Convert Index column to string
                self.df_rfv_ro.index = self.df_rfv_ro.index.astype(str)
            log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': RFV Data: Read ' + str(self.df_rfv_ro.shape[0]) + ' lines: ')
            log.Log.important(self.df_rfv_ro.loc[self.df_rfv_ro.index[0:10]])
            # Cached the numpy array
            self.df_rfv_np_array_ro = np.array(self.df_rfv_ro.values)
            log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Cached huge RFV array from dataframe..')

            self.df_rfv_dist_furthest_ro = pd.read_csv(
                filepath_or_buffer = self.fpath_rfv_dist,
                sep       = ',',
                index_col = 'INDEX'
            )
            if Legacy.CONVERT_COMMAND_INDEX_TO_STR:
                # Convert Index column to string
                self.df_rfv_dist_furthest_ro.index = self.df_rfv_dist_furthest_ro.index.astype(str)
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': RFV Furthest Distance Data: Read ' + str(self.df_rfv_dist_furthest_ro.shape[0]) + ' lines'
            )

            #
            # RFV is ready, means we can start detecting intents. But still can't use training data
            #
            self.is_rfv_ready = True

            # TODO When we go live to production, training data is no longer in file but in DB. How will we do?
            if not self.minimal:
                self.df_fv_training_data_ro = pd.read_csv(
                    filepath_or_buffer = self.fpath_fv_all,
                    sep       = ',',
                    index_col = 'INDEX'
                )
                if Legacy.CONVERT_COMMAND_INDEX_TO_STR:
                    # Convert Index column to string
                    self.df_fv_training_data_ro.index = self.df_fv_training_data_ro.index.astype(str)
                log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': FV Training Data: Read ' + str(self.df_fv_training_data_ro.shape[0]) + ' lines')
                log.Log.info(self.df_fv_training_data_ro[0:5])
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
            self.__mutex_training.release()

        if self.do_profiling:
            log.Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING load_model_parameters_from_storage(): '
                + prf.Profiling.get_time_dif_str(prf_start, prf.Profiling.stop())
            )
        return

    def persist_training_data_to_storage(
            self,
            td
    ):
        prf_start = prf.Profiling.start()
        # For debugging only, not required by model
        self.model_data.persist_training_data_to_storage(td=self.training_data)
        if self.do_profiling:
            log.Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING persist_training_data_to_storage(): '
                + prf.Profiling.get_time_dif_str(prf_start, prf.Profiling.stop())
            )
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

    def load_training_data_from_storage(
            self
    ):
        prf_start = prf.Profiling.start()

        try:
            self.__mutex_training.acquire()
            self.training_data = self.model_data.load_training_data_from_storage()
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Failed to load training data for identifier "' + self.identifier_string\
                     + '". Exception message: ' + str(ex) + '.'
            log.Log.critical(errmsg)
            raise Exception(errmsg)
        finally:
            self.__mutex_training.release()

        if self.do_profiling:
            log.Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING load_training_data_from_storage(): '
                + prf.Profiling.get_time_dif_str(prf_start, prf.Profiling.stop())
            )
        return

