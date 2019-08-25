# -*- coding: utf-8 -*-

import mozg.utils.Log as lg
from inspect import currentframe, getframeinfo
import numpy as np
import pandas as pd
import mozg.lib.math.NumpyUtil as nputil


#
# Given a set of vectors v1, v2, ..., vn with features f1, f2, ..., fn
# We try to find weights w1, w2, ..., wn or in NLP notation known as IDF,
# such that the separation between the vectors v1, v2, ... vn by some metric
# is maximum when projected onto a unit hypersphere.
#
class Idf:

    #
    # Given our training data x, we get the IDF of the columns x_name.
    # TODO Generalize this into a NN Layer instead
    # TODO Optimal values are when "separation" (by distance in space or angle in space) is maximum
    #
    @staticmethod
    def get_feature_weight_idf_default(
            x,
            # Class label, if None then all vectors are different class
            y = None,
            # Feature name, if None then we use default numbers with 0 index
            x_name = None,
            feature_presence_only_in_label_training_data = True
    ):
        if y is None:
            # Default to all vectors are different class
            y = np.array(range(0, x.shape[0], 1), dtype=int)

        if x_name is None:
            x_name = np.array(range(0, x.shape[1], 1), dtype=int)

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
        lg.Log.debug(
            str(Idf.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tAggregated sum by labels:\n\r' + str(np_agg_sum)
            + '\n\r\tPresence array:\n\r' + str(np_feature_presence)
            + '\n\r\tPresence sum:\n\r' + str(np_feature_presence_sum)
            + '\n\r\tx_names: ' + str(x_name) + '.'
        )

        # Total document count
        n_documents = np_feature_presence.shape[0]
        lg.Log.important(
            str(Idf.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Total unique documents/intents to calculate IDF = ' + str(n_documents)
        )

        # If using outdated np.matrix, this IDF will be a (1,n) array, but if using np.array, this will be 1-dimensional vector
        # TODO RuntimeWarning: divide by zero encountered in true_divide
        idf = np.log(n_documents / np_feature_presence_sum)
        # Replace infinity with 1 count or log(n_documents)
        idf[idf==np.inf] = np.log(n_documents)
        # If only 1 document, all IDF will be zero, we will handle below
        if n_documents <= 1:
            lg.Log.warning(
                str(Idf.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Only ' + str(n_documents) + ' document in IDF calculation. Setting IDF to 1.'
            )
            idf = np.array([1]*x.shape[1])
        lg.Log.debug(
            str(Idf.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tWeight IDF:\n\r' + str(idf)
        )
        return idf

    def __init__(
            self,
            # numpy array 2 dimensions
            x
    ):
        if type(x) is not np.ndarray:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Wrong type "' + str(type(x)) + '". Must be numpy ndarray type.'
            )

        if x.ndim != 2:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Wrong dimensions "' + str(x.shape) + '". Must be 2 dimensions.'
            )

        self.x = x
        self.w = np.zeros(shape=(self.x.shape[1]))

        # Normalized version of vectors on the hypersphere
        self.xh = nputil.NumpyUtil.normalize(x=self.x)

        #
        # Start with standard IDF values
        #
        self.w_start = Idf.get_feature_weight_idf_default(
            x = self.xh
        )
        # We want to opimize these weights to make the separation of angles
        # between vectors maximum
        self.w = self.w_start.copy()
        return

    #
    # This is the target function to maximize
    #
    def target_ml_function(
            self,
            x_input
    ):
        # Get total angle squared between all points on the hypersphere
        sum_angle_2 = 0
        for i in range(0, x_input.shape[0], 1):
            for j in range(i+1, x_input.shape[0], 1):
                if i == j:
                    continue
                # Get
                v1 = x_input[i]
                v2 = x_input[j]
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = abs(np.arcsin((1 - cos_angle**2)**0.5))
                lg.Log.debugdebug(
                    'Angle between v1=' + str(v1) + ' and v2=' + str(v2) + ' is ' + str(180 * angle / np.pi)
                )
                sum_angle_2 += angle**2
        lg.Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Sum square of angle = ' + str(sum_angle_2)
        )
        return sum_angle_2

    #
    # Differentiation of target function with respect to weights.
    # Returns a vector same dimensions as w
    #
    def differentiate_dml_dw(
            self,
            delta = 0.000001
    ):
        # Take dw
        l = self.w.shape[0]
        dw_diag = np.diag(np.array([delta]*l, dtype=float))
        # The return value
        dml_dw = np.zeros(l, dtype=float)
        for i in range(l):
            dw_i = dw_diag[i]
            dm_dwi = self.target_ml_function(x_input = np.multiply(self.xh, self.w + dw_i)) -\
                self.target_ml_function(x_input = np.multiply(self.xh, self.w))
            dm_dwi = dm_dwi / delta
            lg.Log.debugdebug(
                'Differentiation with respect to w' + str(i) + ' = ' + str(dm_dwi)
            )
            dml_dw[i] = dm_dwi

        return dml_dw

    def optimize(
            self,
            delta = 0.1,
            max_iter = 10
    ):
        ml_old = self.target_ml_function(x_input = self.xh)
        iter = 1

        while True:
            lg.Log.debugdebug(
                'Iteration #' + str(iter)
            )
            # Get new vectors after weightage
            x_weighted = nputil.NumpyUtil.normalize(x=np.multiply(self.xh, self.w))
            # Get new separation we are trying to maximize
            ml_new = self.target_ml_function(x_input = x_weighted)
            ml_increase = ml_new - ml_old
            if ml_new - ml_old > 0:
                lg.Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Iteration #' + str(iter) + ': Increase from ' + str(ml_old) + ' to ' + str(ml_new)
                    + ' with weights ' + str(self.w)
                )
                # Update the new normalized vectors
                self.xh = nputil.NumpyUtil.normalize(x=self.x)
                ml_old = ml_new

            # If the increase in target function is small enough already, we are done
            if ml_increase < delta:
                break

            iter += 1
            if iter > max_iter:
                break

            #
            # Find the dw we need to move to
            #
            # Get delta of target function d_ml
            dml_dw = self.differentiate_dml_dw()
            lg.Log.debugdebug(
                'dml/dw = ' + str(dml_dw)
            )
            # Adjust weights
            l = self.w.shape[0]
            max_movement_w = np.array([0.1]*l)
            min_movement_w = -max_movement_w
            self.w = self.w + np.maximum(np.minimum(dml_dw*0.1, max_movement_w), min_movement_w)
            # Don't allow negative weights
            #self.w = np.maximum(self.w, np.array([0.001]*l))
            lg.Log.debug(
                'Iter ' + str(iter) + ': New weights: ' + str(self.w)
            )

        return


if __name__ == '__main__':
    lg.Log.LOGLEVEL = lg.Log.LOG_LEVEL_DEBUG_1
    x = np.array([
        [0.9, 0.8, 1.0],
        [0.5, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    obj = Idf(
        x = x
    )
    obj.optimize()
    print(obj.x)
    print(obj.w)
