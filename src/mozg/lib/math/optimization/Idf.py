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
    def get_feature_weight_idf(
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

        self.xh = nputil.NumpyUtil.normalize(x=self.x)
        return

    def get_separation(self):
        # Get total angle squared between all points on the hypersphere
        for i in range(0, self.xh.shape[0], 1):
            for j in range(i+1, self.xh.shape[0], 1):
                if i == j:
                    continue
                # Get
                v1 = self.xh[i]
                v2 = self.xh[j]
                cross_prd = np.cross(v1, v2)
                angle = np.arcsin(1 - cross_prd**2)
                lg.Log.debugdebug(
                    'Angle between v1=' + str(v1) + ' and v2=' + str(v2) + ' is ' + str(180 * angle / np.pi)
                )

    def optimize(self):
        #
        # Start with standard IDF values
        #
        self.get_separation()
        self.w = Idf.get_feature_weight_idf(
            x = self.xh
        )

        return


if __name__ == '__main__':
    lg.Log.LOGLEVEL = lg.Log.LOG_LEVEL_DEBUG_2
    x = np.array([
        [0.9, 0.8],
        [0.5, 0.0],
        [0.0, 1.0],
        [0.0, 1.0]
    ])
    obj = Idf(
        x = x
    )
    obj.optimize()
    print(obj.x)
    print(obj.w)
