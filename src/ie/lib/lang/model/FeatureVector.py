#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import collections as col


#
# TODO: The ideal feature vector for a piece of text will have
# TODO: [ (word1, postag1), (word2, postag2), .. ]
# TODO: The same word may appear multiple times with different postags.
# POS tags are especially important for languages with no conjugations like Chinese,
# where the same word can be a noun, verb, adjective, etc.
#
class FeatureVector:

    def __init__(self):
        self.fv_template = None
        self.fv_weights = None
        return

    #
    # Set features for word frequency fv
    #
    def set_freq_feature_vector_template(self, list_symbols):
        # This number will become default vector ordering in all feature vectors
        len_symbols = len(list_symbols)
        no = range(1, len_symbols+1, 1)
        self.fv_template = pd.DataFrame({ 'No': no, 'Symbol':list_symbols })
        # Default feature weights to 1
        self.set_feature_weights( [1]*len_symbols )
        return

    #
    # Set feature weights, this can be some IDF measure or something along that line.
    # TODO: Should we put this here or somewhere else?
    # TODO: Putting here is only useful when we already know in advance the weight (e.g. IDF)
    # TODO: Usually we need the FV first before calculating the weights, however weights
    # TODO: can be pre-calculated and set here for convenience.
    #
    def set_feature_weights(self, fw):
        self.fv_weights = fw
        return

    #
    # Given a string, creates a word frequency fv based on set template.
    # If feature_as_presence_only=True, then only presence is considered (means frequency is 0 or 1 only)
    #
    def get_freq_feature_vector(self, str, feature_as_presence_only=False, split_by=' '):
        str_list = str.split(split_by)
        #print(str_list)

        counter = col.Counter(str_list)
        # Order the counter
        counter = counter.most_common()

        symbols = [x[0] for x in counter]
        freqs = [x[1] for x in counter]
        # If <feature_as_presence_only> flag set, we don't count frequency, but presence
        if feature_as_presence_only:
            for i in range(0, len(freqs), 1):
                if freqs[i] > 1:
                    freqs[i] = 1
        df_counter = pd.DataFrame({'Symbol': symbols, 'Frequency': freqs})
        #print(df_counter)

        # Merge feature vector template with counter
        df_merge = pd.merge(self.fv_template, df_counter, how='left', on=['Symbol'])
        df_merge = df_merge.sort_values(by=['No'], ascending=[True])
        # Replace NaN with 0's
        df_merge['Frequency'].fillna(0, inplace=True)
        #print(df_merge)
        #print(self.fv_weights)
        #print(df_merge['Frequency'].values)
        # Just a simple list multiplication
        df_merge['FrequencyWeighted'] = df_merge['Frequency'].values * self.fv_weights
        #print(df_merge['FrequencyWeighted'])

        # Normalize vector
        freq_col = list( df_merge['FrequencyWeighted'] )
        normalize_factor = sum(np.multiply(freq_col, freq_col)) ** 0.5
        df_merge['FrequencyNormalized'] = df_merge['FrequencyWeighted'] / normalize_factor
        # Normalization factor can be 0
        df_merge['FrequencyNormalized'].fillna(0, inplace=True)

        # TF (Term Frequency)
        df_merge['TF'] = df_merge['FrequencyWeighted'] / sum(freq_col)
        # TF Normalized is just the same as frequency normalized
        # normalize_factor = (sum(df_merge['TF'].as_matrix()*df_merge['TF'].as_matrix()) ** 0.5)
        # df_merge['TFNormalized'] = df_merge['TF'] / normalize_factor

        return df_merge


def demo_1():
    sb = ['我', '帮', '崔', 'I', '确实']
    f = FeatureVector()
    f.set_freq_feature_vector_template(sb)
    print(f.fv_template)

    # Use word frequency
    str = '确实 有 在 帮 我 崔 吧 帮 我'
    df_fv = f.get_freq_feature_vector(str, feature_as_presence_only=False)
    print(df_fv)
    # Now try with different weights
    f.set_feature_weights([1,2,3,4,5])
    df_fv = f.get_freq_feature_vector(str, feature_as_presence_only=False)
    print(df_fv)

    # Use word presence
    str = '确实 有 在 帮 我 崔 吧 帮 我'
    f.set_feature_weights([1,1,1,1,1])
    df_fv = f.get_freq_feature_vector(str, feature_as_presence_only=True)
    print(df_fv)
    # Now try with different weights
    f.set_feature_weights([1,2,3,4,5])
    df_fv = f.get_freq_feature_vector(str, feature_as_presence_only=True)
    print(df_fv)

    return


#demo_1()
