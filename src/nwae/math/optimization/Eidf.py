# -*- coding: utf-8 -*-

import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo
import numpy as np
import pandas as pd
import nwae.math.NumpyUtil as nputil
import random as rd
import nwae.math.Cluster as cl
import datetime as dt
import re
import os


#
# Enhanced IDF (EIDF)
#
# Given a set of vectors v1, v2, ..., vn with features f1, f2, ..., fn
# We try to find weights w1, w2, ..., wn, such that the separation
# (default we are using angle) between the vectors
# v1, v2, ... vn by some metric (default metric is average angle or the
# 61.8% quantile) is maximum when projected onto a unit hypersphere.
#
class Eidf:

    #
    # Storage columns
    #
    STORAGE_COL_X_NAME = 'x_name'
    STORAGE_COL_EIDF   = 'eidf'

    # Use 10 to make the weights smaller
    EIDF_BASE_LOG = np.log(np.e)
    # Roughly 100 documents
    MAX_EIDF_WEIGHT = np.log(100) / EIDF_BASE_LOG

    # When nan after merging x_name, we replace with this, assuming
    # roughly 100 documents or log(100/1)
    # DEFAULT_EIDF_IF_NAN = 4.605170185988092
    DEFAULT_EIDF_IF_NAN = MAX_EIDF_WEIGHT


    #
    # We maximize by the 61.8% quantile, so that given all distance pairs
    # in the vector set, the 61.8% quantile is optimized at maximum
    #
    MAXIMIZE_QUANTILE = 2/(1+5**0.5)
    # Max weight movements, When doing gradient ascent /descent
    MAXIMUM_IDF_WEIGHT_MOVEMENT = 0.8
    # Don't set to 0.0 as this might cause vectors to become 0.0
    MINIMUM_WEIGHT_IDF = 0.01
    # delta % of target function start value
    DELTA_PERCENT_OF_TARGET_FUNCTION_START_VALUE = 0.005

    #
    # Monte Carlo start points quick start, instead of starting from unit weights vector
    #
    MONTE_CARLO_SAMPLES_N = 200

    #
    # This is the fast closed formula calculation of target function value
    # otherwise the double looping exact calculation is unusable in production
    # when data is too huge.
    #
    TARGET_FUNCTION_AS_SUM_COSINE = True

    #
    # If too many rows in the array, the normalize() function will be too slow
    # so we cluster them
    #
    MAX_X_ROWS_BEFORE_CLUSTER = 500

    @staticmethod
    def get_file_path_eidf(
            dir_path_model,
            identifier_string
    ):
        return str(dir_path_model) + '/nlp.eidf.' + str(identifier_string) + '.csv'

    @staticmethod
    def get_file_path_eidf_info(
            dir_path_model,
            identifier_string
    ):
        return str(dir_path_model) + '/nlp.eidf.' + str(identifier_string) + '.info.csv'

    #
    # Given our training data x, we get the IDF of the columns x_name.
    # TODO Generalize this into a NN Layer instead
    # TODO Optimal values are when "separation" (by distance in space or angle in space) is maximum
    #
    @staticmethod
    def get_feature_weight_idf_default(
            x,
            # Class label, if None then all vectors are different class
            y,
            # Feature name, if None then we use default numbers with 0 index
            x_name,
            #
            # The intuitive meaning is this. If set to False, then we ignore y
            # and treat all samples as separate documents.
            # If set to True, a single word appearing in a group only once will
            # not be weighted too heavily. However if a word appears only very
            # infrequently in every group will have very low weight.
            #
            feature_presence_only_in_label_training_data = False
    ):
        if feature_presence_only_in_label_training_data:
            df_tmp = pd.DataFrame(data=x, index=y)
            # Group by the labels y, as they are not unique
            df_agg_sum = df_tmp.groupby(df_tmp.index).sum()
            # No need a copy, the dataframe will create a new copy already from original x
            np_agg_sum = df_agg_sum.values
        else:
            # No need to group, each row is it's own document
            np_agg_sum = x.copy()

        # Just overwrite inline, don't copy
        np.nan_to_num(np_agg_sum, copy=False)

        # Get presence only by cell, then sum up by columns to get total presence by document
        np_feature_presence = (np_agg_sum > 0) * 1
        # Sum by column axis=0
        np_idf = np.sum(np_feature_presence, axis=0)
        # Don't allow 0's
        np_idf[np_idf<=1] = 1

        lg.Log.debugdebug(
            str(Eidf.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tAggregated sum by labels (' + str(np_agg_sum.shape[0]) + ' rows):\n\r' + str(np_agg_sum)
            + '\n\r\tPresence array (' + str(np_feature_presence.shape[0]) + ' rows):\n\r' + str(np_feature_presence)
            + '\n\r\tArray for IDF presence/normalized sum (' + str(np_idf) + ' rows):\n\r' + str(np_idf)
            + '\n\r\tx_names: ' + str(x_name) + '.'
        )

        # Total document count
        n_documents = np_agg_sum.shape[0]
        lg.Log.important(
            str(Eidf.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': EIDF: Total unique documents/intents to calculate IDF = ' + str(n_documents)
        )

        # If using outdated np.matrix, this IDF will be a (1,n) array, but if using np.array, this will be 1-dimensional vector
        # TODO RuntimeWarning: divide by zero encountered in true_divide
        idf = np.log(n_documents / np_idf) / Eidf.EIDF_BASE_LOG
        # Replace infinity with 1 count or log(n_documents)
        idf[idf==np.inf] = np.log(n_documents) / Eidf.EIDF_BASE_LOG
        # If only 1 document, all IDF will be zero, we will handle below
        if n_documents <= 1:
            lg.Log.warning(
                str(Eidf.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Only ' + str(n_documents) + ' document in IDF calculation. Setting IDF to 1.'
            )
            idf = np.array([1]*x.shape[1])

        log_percentiles = (5, 10, 25, 50, 75, 90, 95)
        lg.Log.important(
            str(Eidf.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': EIDF : Quantiles ' + str(log_percentiles)
            + str(np.percentile(idf, log_percentiles))
        )
        lg.Log.debugdebug(
            str(Eidf.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tWeight IDF:\n\r' + str(idf)
        )
        return idf

    def __init__(
            self,
            # numpy array 2 dimensions
            x,
            # Class label, if None then all vectors are different class
            y = None,
            # Feature name, if None then we use default numbers with 0 index
            x_name = None,
            feature_presence_only_in_label_training_data = False
    ):
        if type(x) is not np.ndarray:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Wrong type x "' + str(type(x)) + '". Must be numpy ndarray type.'
            )

        if x.ndim != 2:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Wrong dimensions x "' + str(x.shape) + '". Must be 2 dimensions.'
            )

        self.x = x
        self.y = y
        self.x_name = x_name
        self.feature_presence_only_in_label_training_data = feature_presence_only_in_label_training_data

        if self.y is None:
            # Default to all vectors are different class
            self.y = np.array(range(0, x.shape[0], 1), dtype=int)
        if type(self.y) is not np.ndarray:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Wrong type y "' + str(type(self.y)) + '". must be numpy ndarray type.'
            )
        if self.y.ndim != 1:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Wrong dimensions y "' + str(self.y.shape) + '". Must be 1 dimension.'
            )
        if self.y.shape[0] != self.x.shape[0]:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Length of y ' + str(self.y.shape[0]) + ' must be equal to rows of x ' + str(self.x.shape[0])
            )

        if self.x_name is None:
            self.x_name = np.array(range(0, x.shape[1], 1), dtype=int)
        if self.x_name.ndim != 1:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Wrong dimensions x_name "' + str(self.x_name.shape) + '". Must be 1 dimension.'
            )
        if self.x_name.shape[0] != self.x.shape[1]:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Length of x_name ' + str(self.x_name.shape[0])
                + ' must be equal to columns of x ' + str(self.x.shape[1])
            )

        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Normalizing x..'
        )
        # TODO This is unusable in production, too slow!
        # Normalized version of vectors on the hypersphere
        self.xh = nputil.NumpyUtil.normalize(x=self.x)

        #
        # Start with default 1 weights
        #
        self.w_start = np.array([1.0]*self.x.shape[1], dtype=float)
        # We want to opimize these weights to make the separation of angles
        # between vectors maximum
        self.w = self.w_start.copy()

        self.log_training = []
        self.optimize_info = ''

        lg.Log.debugdebug(
            str(Eidf.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tIDF Initialization, x:\n\r' + str(self.x)
            + '\n\r\ty:\n\r' + str(self.y)
            + '\n\r\tx_name:\n\r' + str(self.x_name)
        )
        return

    def get_w(self):
        return self.w.copy()

    #
    # This is the target function to maximize the predetermined quantile MAXIMIZE_QUANTILE
    # TODO Unusable in production too slow. Optimize!
    #
    @staticmethod
    def target_ml_function(
            x_input
    ):
        #
        # TODO
        #  This normalize is unusable when x has too many rows,
        #  This is why we should cluster x first, then only optimize.
        #
        x_n = nputil.NumpyUtil.normalize(x = x_input)

        #
        # Fast calculation closed formula, compared to double-looping below that cannot be used in most cases
        #
        if Eidf.TARGET_FUNCTION_AS_SUM_COSINE:
            #
            # If already normalized, then a concise formula for sum of cosine of angles are just:
            #
            # 0.5 * [ (x_11 + x_21 +... + x_n1)^2 - (x_11^2 + x_21^2 + ... x_n1^2)
            #         (x_12 + x_22 +... + x_n2)^2 - (x_12^2 + x_22^2 + ... x_n2^2)
            #         ...
            #         (x_1n + x_2n +... + x_nn)^2 - (x_1n^2 + x_2n^2 + ... x_nn^2)
            #       ]
            #
            # Can be seen that the above formula takes care of all pairs = n_row*(n_row-1)/2.
            #
            # All vectors we assume are positive values only thus cosine values are in the range [0,1]
            #
            n_els = x_n.shape[0]
            n_pairs = n_els*(n_els-1)/2
            sum_cols = np.sum(x_n, axis=0)
            lg.Log.debugdebug('************** Sum Cols (' + str(n_els)  + 'rows ):\n\r' + str(sum_cols))
            sum_cols_square = np.sum(sum_cols**2)
            lg.Log.debugdebug('************** Sum Cols Square: ' + str(sum_cols_square))
            sum_els_square = np.sum(x_n**2)
            lg.Log.debugdebug('************** Sum Elements Square: ' + str(sum_els_square))
            sum_cosine = 0.5 * (sum_cols_square - sum_els_square)
            lg.Log.debugdebug('************** Sum Cosine: ' + str(sum_cosine))
            # Average of the cosine sum
            avg_cosine_sum = sum_cosine/n_pairs
            lg.Log.debugdebug('************** Avg Cosine Sum (of ' + str(n_pairs) + ' pairs): ' + str(avg_cosine_sum))
            avg_cosine_sum = max(0.0, min(1.0, avg_cosine_sum))
            lg.Log.debugdebug('************** Avg Cosine Sum (of ' + str(n_pairs) + ' pairs): ' + str(avg_cosine_sum))
            # Return average angle as this has meaning when troubleshooting or analyzing
            angle = np.arccos(avg_cosine_sum)*180/np.pi
            lg.Log.debugdebug('************** Angle: ' + str(angle))
            return angle

        # Get total angle squared between all points on the hypersphere
        quantile_angle_x = 0
        angle_list = []
        #
        # TODO
        #  The code below is quite unusable as it is super slow.
        #  This double looping must be eliminated to no loops.
        #
        for i in range(0, x_n.shape[0], 1):
            for j in range(i+1, x_n.shape[0], 1):
                if i == j:
                    continue
                # Get
                v1 = x_n[i]
                v2 = x_n[j]
                # It is possible after iterations that vectors become 0 due to the weights
                if (np.linalg.norm(v1) == 0) or (np.linalg.norm(v2) == 0):
                    lg.Log.warning(
                        str(Eidf.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                        + ': Vector zerorized from iterations.'
                    )
                    continue
                #
                # The vectors are already normalized
                #
                cos_angle = np.dot(v1, v2)
                #
                # This value can be >1 due to computer roundings and after that everything will be nan
                #
                if np.isnan(cos_angle):
                    errmsg = str(Eidf.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                             + ': Cosine Angle between v1=' + str(v1) + ' and v2=' + str(v2) + ' is nan!!'
                    lg.Log.critical(errmsg)
                    raise Exception(errmsg)
                if cos_angle > 1.0:
                    cos_angle = 1.0
                elif cos_angle < 0.0:
                    cos_angle = 0.0
                angle = abs(np.arcsin((1 - cos_angle**2)**0.5))
                if np.isnan(angle):
                    errmsg = str(Eidf.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                             + ': Angle between v1=' + str(v1) + ' and v2=' + str(v2) + ' is nan!!'\
                             + ' Cosine of angle = ' + str(cos_angle) + '.'
                    lg.Log.critical(errmsg)
                    raise Exception(errmsg)
                lg.Log.debugdebug(
                    'Angle between v1=' + str(v1) + ' and v2=' + str(v2) + ' is ' + str(180 * angle / np.pi)
                )
                angle_list.append(angle)

        quantile_angle_x = np.quantile(a=angle_list, q=[Eidf.MAXIMIZE_QUANTILE])[0]
        values_in_quantile = np.array(angle_list)
        values_in_quantile = values_in_quantile[values_in_quantile<=quantile_angle_x]
        sum_square_values_in_q = np.sum(values_in_quantile**2)
        if np.isnan(quantile_angle_x):
            errmsg = str(Eidf.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Final quantile angle =' + str(quantile_angle_x)\
                     + ' for x_input:\n\r' + str(x_input) + '.'
            lg.Log.error(errmsg)
            raise Exception(errmsg)

        lg.Log.debugdebug(
            str(Eidf.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Angle = ' + str(np.sort(angle_list))
            + '\n\rQuantile ' + str(100*Eidf.MAXIMIZE_QUANTILE) + '% = ' + str(quantile_angle_x)
            + '\n\rSum square values in quantile = ' + str(sum_square_values_in_q)
        )
        return sum_square_values_in_q

    #
    # Differentiation of target function with respect to weights.
    # Returns a vector same dimensions as w
    #
    @staticmethod
    def differentiate_dml_dw(
            x_input,
            w_vec,
            delta = 0.000001
    ):
        # Take dw
        l = w_vec.shape[0]
        dw_diag = np.diag(np.array([delta]*l, dtype=float))
        # The return value
        dml_dw = np.zeros(l, dtype=float)
        for i in range(l):
            dw_i = dw_diag[i]
            dm_dwi = Eidf.target_ml_function(x_input = np.multiply(x_input, w_vec + dw_i)) -\
                Eidf.target_ml_function(x_input = np.multiply(x_input, w_vec))
            dm_dwi = dm_dwi / delta
            lg.Log.debugdebug(
                'Differentiation with respect to w' + str(i) + ' = ' + str(dm_dwi)
            )
            dml_dw[i] = dm_dwi

        lg.Log.debugdebug(
            'Differentiation with respect to w = ' + str(dml_dw)
        )

        return dml_dw

    #
    # Opimize by gradient ascent, on predetermined quantile MAXIMIZE_QUANTILE
    #
    def optimize(
            self,
            # If we don't start with standard IDF=log(present_in_how_many_documents/total_documents),
            # then we Monte Carlo some start points and choose the best one
            initial_w_as_standard_idf = False,
            max_iter = 10,
            # Log training events
            logs = None
    ):
        if type(logs) is list:
            self.log_training = logs
        else:
            self.log_training = []

        x_vecs = self.xh.copy()
        y_vecs = self.y.copy()

        #
        # If too many rows, we will have problem calculating normalize() after weighing vectors
        #
        if x_vecs.shape[0] > Eidf.MAX_X_ROWS_BEFORE_CLUSTER:
            logmsg = ': Too many rows ' + str(x_vecs.shape[0]) + ' > ' + str(Eidf.MAX_X_ROWS_BEFORE_CLUSTER)\
                     + '. Clustering to ' + str(Eidf.MAX_X_ROWS_BEFORE_CLUSTER) + ' rows..'
            lg.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + logmsg
            )
            self.log_training_messages(msg=logmsg)

            cl_result = cl.Cluster.cluster(
                matx = x_vecs,
                ncenters = Eidf.MAX_X_ROWS_BEFORE_CLUSTER
            )
            x_vecs = cl_result.np_cluster_centers
            lg.Log.debug(
                'New x after clustering:\n\r' + str(x_vecs)
            )
            y_vecs = np.array(range(x_vecs.shape[0]))

        ml_start = Eidf.target_ml_function(x_input = x_vecs)
        ml_final = ml_start
        # The delta of limit increase in target function to stop iteration
        delta = ml_start * Eidf.DELTA_PERCENT_OF_TARGET_FUNCTION_START_VALUE

        logmsg = ': Start target function value = ' + str(ml_start) + ', using delta = ' + str(delta)\
                 + ', quantile used = ' + str(Eidf.MAXIMIZE_QUANTILE)
        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + logmsg
        )
        self.log_training_messages(msg=logmsg)

        iter = 1

        ml_prev = ml_start

        #
        # Find best initial start weights for iteration, either using standard IDF or via MC
        #
        if initial_w_as_standard_idf:
            # Start with standard IDF values
            self.w_start = Eidf.get_feature_weight_idf_default(
                x = x_vecs,
                y = y_vecs,
                x_name = self.x_name,
                feature_presence_only_in_label_training_data = self.feature_presence_only_in_label_training_data
            )
        else:
            # Monte Carlo the weights for some start points to see which is best
            tf_val_best = -np.inf
            for i in range(Eidf.MONTE_CARLO_SAMPLES_N):
                rd_vec = np.array([rd.uniform(-0.5, 0.5) for i in range(self.w.shape[0])])
                w_mc = self.w + rd_vec
                lg.Log.debugdebug(
                    'MC random w = ' + str(w_mc)
                )
                x_weighted = nputil.NumpyUtil.normalize(x=np.multiply(x_vecs, w_mc))
                tf_val = Eidf.target_ml_function(x_input=x_weighted)
                if tf_val > tf_val_best:
                    tf_val_best = tf_val
                    self.w_start = w_mc
                    lg.Log.debugdebug(
                        'Update best MC w to ' + str(self.w_start)
                        + ', target function value = ' + str(tf_val_best)
                    )
            lg.Log.debugdebug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + 'Best MC w: ' + str(self.w_start) + ', target function value = ' + str(tf_val_best)
            )
        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Start weights:\n\r' + str(self.w_start)
        )

        w_iter_test = self.w_start.copy()
        w_iter_test = np.minimum(w_iter_test, Eidf.MAX_EIDF_WEIGHT)
        while True:
            logmsg = 'ITERATION #' + str(iter) + ', using test weights:\n\r' + str(w_iter_test)\
                     + '\n\rexisting old weights:\n\r' + str(self.w)
            lg.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + logmsg
            )
            self.log_training_messages(msg=logmsg)

            # Get new vectors after weightage
            x_weighted = nputil.NumpyUtil.normalize(x=np.multiply(x_vecs, w_iter_test))
            # Get new separation we are trying to maximize (if using sum cosine is minimize)
            ml_cur = Eidf.target_ml_function(x_input = x_weighted)
            ml_increase = ml_cur - ml_prev
            if ml_cur - ml_prev > 0:
                ml_final = ml_cur
                self.w = w_iter_test.copy()
            # Update old ML value to current
            ml_prev = ml_cur

            logmsg = ': ITERATION #' + str(iter) + '. ML Increase = ' + str(ml_increase)\
                     + ', delta =' + str(delta) + ', ML = ' + str(ml_cur)\
                     + ', updated weights:\n\r' + str(self.w)
            lg.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + logmsg
            )
            self.log_training_messages(msg=logmsg)

            if ml_increase < delta:
                # Gradient ascent done, max local optimal reached
                break
            if iter > max_iter:
                # Gradient ascent done, too many iterations already
                break
            iter += 1

            #
            # Find the dw we need to move to
            #
            # Get delta of target function d_ml
            dml_dw = Eidf.differentiate_dml_dw(
                x_input = x_vecs,
                w_vec   = w_iter_test
            )

            lg.Log.debugdebug(
                'dml/dw = ' + str(dml_dw)
            )
            # Adjust weights
            l = w_iter_test.shape[0]
            max_movement_w = np.array([Eidf.MAXIMUM_IDF_WEIGHT_MOVEMENT] * l)
            min_movement_w = -max_movement_w

            w_iter_test = w_iter_test + np.maximum(np.minimum(dml_dw*0.1, max_movement_w), min_movement_w)
            # Don't allow negative weights
            w_iter_test = np.maximum(w_iter_test, Eidf.MINIMUM_WEIGHT_IDF)
            # Don't allow too big
            w_iter_test = np.minimum(w_iter_test, Eidf.MAX_EIDF_WEIGHT)
            lg.Log.debugdebug(
                'Iter ' + str(iter) + ': New weights:\n\r' + str(w_iter_test)
            )

        self.optimize_info =\
            'Train time: ' + str(dt.datetime.now()) + '\n\r' \
            + 'Using standard IDF as start EIDF weights = ' + str(initial_w_as_standard_idf) + '\n\r'\
            + 'Total Iterations = ' + str(iter) + '\n\r'\
            + 'Start ML = ' + str(ml_start) + ', End ML = ' + str(ml_final) + '\n\r' \
            + 'Start weights:\n\r' + str(self.w_start.tolist()) + '\n\r' \
            + 'End weights:\n\r' + str(self.w.tolist()) + '\n\r' \
            + 'x_name:\n\r' + str(self.x_name.tolist())
        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': ' + self.optimize_info
        )
        self.log_training_messages(msg=self.optimize_info)
        return self.optimize_info

    @staticmethod
    def read_eidf_from_storage(
            # Option to pass in a DataFrame
            data_pd_dataframe = None,
            # Option to read from directory and identifier string
            dir_path_model    = None,
            identifier_string = None,
            # We put in the same order as x_name passed in
            x_name            = None,
            log_training      = None
    ):
        try:
            if type(data_pd_dataframe) is pd.DataFrame:
                df_eidf_file = data_pd_dataframe
            else:
                fpath_eidf = Eidf.get_file_path_eidf(
                    dir_path_model    = dir_path_model,
                    identifier_string = identifier_string
                )
                df_eidf_file = pd.read_csv(
                    filepath_or_buffer = fpath_eidf,
                    sep                = ','
                )

            if Eidf.STORAGE_COL_X_NAME not in df_eidf_file.columns:
                raise Exception('Column "' + str(Eidf.STORAGE_COL_X_NAME) + '" not in dataframe!')
            if Eidf.STORAGE_COL_EIDF not in df_eidf_file.columns:
                raise Exception('Column "' + str(Eidf.STORAGE_COL_EIDF) + '" not in dataframe!')

            if type(x_name) is np.ndarray:
                # Put in the same order as x_name passed in
                df_eidf = pd.DataFrame({
                    Eidf.STORAGE_COL_X_NAME: x_name
                })
                df_eidf = df_eidf.merge(
                    right    = df_eidf_file,
                    how      = 'left',
                    left_on  = [Eidf.STORAGE_COL_X_NAME],
                    right_on = [Eidf.STORAGE_COL_X_NAME]
                )
                # Log those nan values
                w_eidf = np.array(df_eidf[Eidf.STORAGE_COL_EIDF])
                cond_nan = np.isnan(w_eidf)
                x_name_nan = x_name[cond_nan]
                if x_name_nan.shape[0] > 0:
                    lg.Log.warning(
                        str(Eidf.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': EIDF needs to update. Symbols missing as follows: ' + str(x_name_nan.tolist())
                        + '. Replaced NANs with ' + str(Eidf.DEFAULT_EIDF_IF_NAN) + '.'
                        , log_list = log_training
                    )

                df_eidf = df_eidf.fillna(
                    value = {
                        Eidf.STORAGE_COL_EIDF: Eidf.DEFAULT_EIDF_IF_NAN
                    }
                )
                return df_eidf
            else:
                return df_eidf_file
        except Exception as ex:
            errmsg =\
                str(Eidf.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                + ': Error reading EIDF from file, exception ' + str(ex)
            lg.Log.error(errmsg, log_list=log_training)
            raise Exception(errmsg)

    def persist_eidf_to_storage(
            self,
            dir_path_model,
            identifier_string
    ):
        try:
            df_eidf = pd.DataFrame({
                Eidf.STORAGE_COL_X_NAME: self.x_name,
                Eidf.STORAGE_COL_EIDF: self.w
            })
            fpath_eidf = Eidf.get_file_path_eidf(
                dir_path_model    = dir_path_model,
                identifier_string = identifier_string
            )
            df_eidf.to_csv(
                path_or_buf = fpath_eidf,
                index       = True
            )
            logmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Successfully saved EIDF to file "' + str(fpath_eidf)
            lg.Log.important(logmsg)
            self.log_training_messages(logmsg)

            #
            # Now write some info
            #
            fpath_eidf_info = Eidf.get_file_path_eidf_info(
                dir_path_model    = dir_path_model,
                identifier_string = identifier_string
            )
            f = None
            f = open(file=fpath_eidf_info, mode='w', encoding='utf-8')
            f.write(self.optimize_info)
            f.close()
        except Exception as ex:
            errmsg =\
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                + ': Error persisting EIDF to file, exception ' + str(ex)
            lg.Log.error(errmsg)
            self.log_training_messages(msg=errmsg)
            raise Exception(errmsg)

    def log_training_messages(
            self,
            msg
    ):
        msg = re.sub(pattern='[\n\r]', repl='<br>', string=msg)
        self.log_training.append(msg)


if __name__ == '__main__':
    lg.Log.LOGLEVEL = lg.Log.LOG_LEVEL_DEBUG_1
    x = np.array([
        [1.0, 0.0, 1.0], #0
        [0.9, 0.8, 1.0], #0
        [0.5, 0.0, 0.0], #1
        [0.1, 1.0, 0.0], #2
        [0.0, 1.0, 0.0]  #3
    ])
    y = np.array([0, 0, 1, 2, 3])
    obj = Eidf(
        x = x,
        y = y,
        feature_presence_only_in_label_training_data = True
    )
    obj.optimize(
        initial_w_as_standard_idf=True
    )

    obj = Eidf(
        x = x,
        y = y,
        feature_presence_only_in_label_training_data = False
    )
    obj.optimize(
        initial_w_as_standard_idf = True
    )

    obj = Eidf(
        x = x,
        y = y,
        feature_presence_only_in_label_training_data = False
    )
    obj.optimize(
        initial_w_as_standard_idf = False
    )
