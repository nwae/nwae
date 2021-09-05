#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import collections as col
import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo


#
# TODO: The ideal feature vector for a piece of text will have
# TODO: [ (word1, postag1), (word2, postag2), .. ]
# TODO: The same word may appear multiple times with different postags.
# POS tags are especially important for languages with no conjugations like Chinese,
# where the same word can be a noun, verb, adjective, etc.
#
class FeatureVector:

    COL_NO = 'No'
    COL_SYMBOL = 'Symbol'
    COL_FREQUENCY = 'Frequency'
    COL_FREQ_NORM = 'FrequencyNormalized'
    COL_FREQ_WEIGHTED = 'FrequencyWeighted'

    def __init__(self):
        self.fv_template = None
        self.fv_weights = None
        return

    #
    # Set features for word frequency fv
    #
    def set_freq_feature_vector_template(
            self,
            list_symbols
    ):
        lg.Log.warning(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': DEPRECATED. Use nwae.lang.model.FeatureVect.py instead'
        )
        # This number will become default vector ordering in all feature vectors
        len_symbols = len(list_symbols)
        no = range(1, len_symbols+1, 1)

        self.fv_template = pd.DataFrame({
            FeatureVector.COL_NO:     no,
            FeatureVector.COL_SYMBOL: list_symbols
        })
        # Default feature weights to 1
        self.set_feature_weights( [1]*len_symbols )
        return

    def get_fv_template(self):
        return self.fv_template

    def get_fv_weights(self):
        return self.fv_weights

    #
    # Set feature weights, this can be some IDF measure or something along that line.
    # TODO: Should we put this here or somewhere else?
    # TODO: Putting here is only useful when we already know in advance the weight (e.g. IDF)
    # TODO: Usually we need the FV first before calculating the weights, however weights
    # TODO: can be pre-calculated and set here for convenience.
    #
    def set_feature_weights(self, fw):
        self.fv_weights = np.array(fw)
        lg.Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ' Feature weights set to ' + str(self.fv_weights) + '.'
        )
        return

    #
    # Given a string, creates a word frequency fv based on set template.
    # If feature_as_presence_only=True, then only presence is considered (means frequency is 0 or 1 only)
    #
    def get_freq_feature_vector(
            self,
            # A word array. e.g. ['this','is','a','sentence','or','just','any','word','array','.']
            text_list,
            feature_as_presence_only = False,
    ):
        counter = col.Counter(text_list)
        # Order the counter
        counter = counter.most_common()

        symbols = [x[0] for x in counter]
        freqs = np.array( [x[1] for x in counter] )
        # lg.Log.debugdebug(
        #     str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
        #     + ': Symbols ' + str(symbols)
        #     + ', Frequencies ' + str(freqs)
        #     + ', Presence ' + str(presence)
        # )

        # If <feature_as_presence_only> flag set, we don't count frequency, but presence
        if feature_as_presence_only:
            presence = (freqs >= 1) * 1
            freqs = presence
        df_counter = pd.DataFrame({
            FeatureVector.COL_SYMBOL: symbols,
            FeatureVector.COL_FREQUENCY: freqs
        })
        #lg.Log.debugdebug(df_counter.values)

        df_merge = self.get_freq_feature_vector_df(
            df_text_counter = df_counter
        )
        return df_merge

    def get_freq_feature_vector_df(
            self,
            # Data frame of columns 'Symbol', 'Frequency'
            df_text_counter
    ):
        # Merge feature vector template with counter
        df_merge = pd.merge(
            self.fv_template,
            df_text_counter,
            how = 'left',
            on  = [FeatureVector.COL_SYMBOL]
        )
        df_merge = df_merge.sort_values(by=[FeatureVector.COL_NO], ascending=[True])
        # Replace NaN with 0's
        df_merge[FeatureVector.COL_FREQUENCY].fillna(0, inplace=True)
        #lg.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
        #                  + ': Merged with FV template: ')

        # Just a simple list multiplication
        if df_merge.shape[0] != len(self.fv_weights):
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Length of merged frequency ' + str(df_merge.shape)
                + ' differs from length of FV weights ' + str(len(self.fv_weights))
                + '. df_merge ' + str(df_merge) + ', fv weights ' + str(self.fv_weights)
            )
        df_merge[FeatureVector.COL_FREQ_WEIGHTED] =\
            np.multiply(df_merge[FeatureVector.COL_FREQUENCY].values, self.fv_weights)

        # Normalize vector
        freq_weighted = np.array( df_merge[FeatureVector.COL_FREQ_WEIGHTED] )
        # TF (Term Frequency)
        if sum(freq_weighted) > 0.000000001:
            normalize_factor = sum(np.multiply(freq_weighted, freq_weighted)) ** 0.5
            df_merge[FeatureVector.COL_FREQ_NORM] = freq_weighted / normalize_factor
            # Normalization factor can be 0
            df_merge[FeatureVector.COL_FREQ_NORM].fillna(0, inplace=True)
            df_merge['TF'] = freq_weighted / sum(freq_weighted)
        else:
            df_merge[FeatureVector.COL_FREQ_NORM] = 0
            df_merge['TF'] = 0
            lg.Log.warning(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Zero sum of weighted frequency for df_text_counter "' + str(df_text_counter) + '".'
            )
            df_merge['TF'] = 0
        # TF Normalized is just the same as frequency normalized
        # normalize_factor = (sum(df_merge['TF'].as_matrix()*df_merge['TF'].as_matrix()) ** 0.5)
        # df_merge['TFNormalized'] = df_merge['TF'] / normalize_factor

        return df_merge


if __name__ == '__main__':
    lg.Log.LOGLEVEL = lg.Log.LOG_LEVEL_DEBUG_2

    sb = ['我', '帮', '崔', 'I', '确实']
    f = FeatureVector()
    f.set_freq_feature_vector_template(sb)
    print(f.fv_template)

    # Use word frequency
    txt_list = ['确实 有 在 帮 我 崔 吧 帮 我']
    df_fv = f.get_freq_feature_vector(text_list=txt_list, feature_as_presence_only=False)
    print(df_fv)
    # Now try with different weights
    f.set_feature_weights([1,2,3,4,5])
    df_fv = f.get_freq_feature_vector(text_list=txt_list, feature_as_presence_only=False)
    print(df_fv)

    # Use word presence
    txt_list = ['确实 有 在 帮 我 崔 吧 帮 我']
    f.set_feature_weights([1,1,1,1,1])
    df_fv = f.get_freq_feature_vector(text_list=txt_list, feature_as_presence_only=True)
    print(df_fv)
    # Now try with different weights
    f.set_feature_weights([1,2,3,4,5])
    df_fv = f.get_freq_feature_vector(text_list=txt_list, feature_as_presence_only=True)
    print(df_fv)

    txt_list = ['为什么 无法 兑换 商品 ？']
    f.set_feature_weights([1,1,1,1,1])
    df_fv = f.get_freq_feature_vector(text_list=txt_list, feature_as_presence_only=True)
    print(df_fv)
