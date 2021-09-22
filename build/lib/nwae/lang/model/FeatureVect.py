#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import collections as col
from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
from nwae.utils.UnitTest import ResultObj, UnitTest

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


"""
To replace FeatureVector.py
"""
class FeatureVector:

    COL_NO                = 'No'
    COL_SYMBOL            = 'Symbol'
    COL_FREQUENCY         = 'Freq'
    COL_FREQ_WEIGHTED     = 'FreqWeighted'
    COL_FREQ_NORM         = 'FreqNormalized'
    COL_FREQ_PROB         = 'FreqProbability'
    # Logarithm, Not weighted
    COL_LOG_FREQ          = 'LogFreq'
    COL_LOG_FREQ_NORM     = 'LogFreqNormalized'
    COL_LOG_FREQ_PROB     = 'LogFreqProbability'
    # Sigmoid, not weighted
    COL_SIGMOID_FREQ      = 'SigmoidFreq'
    COL_SIGMOID_FREQ_NORM = 'SigmoidFreqNormalized'
    COL_SIGMOID_FREQ_PROB = 'SigmoidFreqProbability'

    DEFAULT_LOG_BASE = 10

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
        # This number will become default vector ordering in all feature vectors
        len_symbols = len(list_symbols)
        no = range(1, len_symbols+1, 1)

        self.fv_template = pd.DataFrame({
            self.COL_NO:     no,
            self.COL_SYMBOL: list_symbols
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
    #
    def set_feature_weights(self, fw):
        self.fv_weights = np.array(fw)
        Log.debugdebug(
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
            # Log base has no effect on LogFreqNormalized & LogFreqProbability as it is just a constant factor
            log_base = DEFAULT_LOG_BASE,
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
            self.COL_SYMBOL: symbols,
            self.COL_FREQUENCY: freqs
        })
        Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Converted text "' + str(text_list) + '" to ' +  str(df_counter.values)
        )

        df_merge = self.get_freq_feature_vector_df(
            df_text_counter = df_counter,
            log_base = log_base,
        )
        return df_merge

    """
    Для LOG и SIGMOID веса не имеют влияние на вычисление, так как LOG/SIGMOID сами является взвешенными мерами
    """
    def get_freq_feature_vector_df(
            self,
            # Data frame of columns 'Symbol', 'Frequency'
            df_text_counter,
            log_base = DEFAULT_LOG_BASE,
    ):
        # Merge feature vector template with counter
        df_merge = pd.merge(
            self.fv_template,
            df_text_counter,
            how = 'left',
            on  = [self.COL_SYMBOL]
        )
        df_merge = df_merge.sort_values(by=[self.COL_NO], ascending=[True])
        Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': After merge ' + str(df_text_counter) + ' to ' + str(df_merge)
        )
        # Replace NaN with 0's
        df_merge[self.COL_FREQUENCY].fillna(0, inplace=True)
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
        df_merge[self.COL_FREQ_WEIGHTED] =\
            np.multiply(df_merge[self.COL_FREQUENCY].values, self.fv_weights)

        # Normalize vector & create probability column
        self.__normalize(
            df = df_merge,
            freq_colname = self.COL_FREQ_WEIGHTED,
            norm_colname = self.COL_FREQ_NORM,
            prob_colname = self.COL_FREQ_PROB,
        )

        """
        log(1+freq) so that when freq=0, log(1+0)=0 
        >>> x=np.array(range(10))
        >>> x
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> np.log(x+1)
        array([0.        , 0.69314718, 1.09861229, 1.38629436, 1.60943791,
               1.79175947, 1.94591015, 2.07944154, 2.19722458, 2.30258509])
        As we can see, the growth becomes much slower as frequency increases, thus controlling stopwords effect
        without having to filter out stopwords
        *** Для LOG и SIGMOID веса не имеют влияние на вычисление, так как LOG/SIGMOID сами является взвешенными мерами
        """
        df_merge[self.COL_LOG_FREQ] = np.log(1.0 + df_merge[self.COL_FREQUENCY]) / np.log(log_base)
        self.__normalize(
            df = df_merge,
            freq_colname = self.COL_LOG_FREQ,
            norm_colname = self.COL_LOG_FREQ_NORM,
            prob_colname = self.COL_LOG_FREQ_PROB,
        )

        """*** Для LOG и SIGMOID веса не имеют влияние на вычисление, так как LOG/SIGMOID сами является взвешенными мерами"""
        df_merge[self.COL_SIGMOID_FREQ] = 1 / (1 + np.exp(- df_merge[self.COL_FREQUENCY]))
        # So that 0 frequency is 0, and multiply by 2 to normalize back to 1.0 as max value
        df_merge[self.COL_SIGMOID_FREQ] = 2 * ( df_merge[self.COL_SIGMOID_FREQ] - 0.5 )
        self.__normalize(
            df = df_merge,
            freq_colname = self.COL_SIGMOID_FREQ,
            norm_colname = self.COL_SIGMOID_FREQ_NORM,
            prob_colname = self.COL_SIGMOID_FREQ_PROB,
        )

        return df_merge

    def __normalize(self, df, freq_colname, norm_colname, prob_colname):
        # Normalize vector & create probability column
        freq_weighted = np.array( df[freq_colname] )
        sum_freq_weighted = np.sum(freq_weighted)
        if sum(freq_weighted) > 0.000000001:
            normalize_factor = np.sum(np.multiply(freq_weighted, freq_weighted)) ** 0.5
        else:
            normalize_factor = 1.0
            sum_freq_weighted = 1.0
        df[norm_colname] = freq_weighted / normalize_factor
        # Normalization factor can be 0
        df[norm_colname].fillna(0, inplace=True)
        df[prob_colname] = freq_weighted / sum_freq_weighted


class FeatureVectorUnitTest:
    def __init__(self, ut_params = None):
        return

    def check(
            self,
            name,
            df,
            expected_freq,
            base,
            unit_test_res,
            feature_weights = None,
            feature_as_presence_only = False,
    ):
        v_freq = np.array( df[FeatureVector.COL_FREQUENCY] )
        if feature_as_presence_only:
            v_freq = 1 * (v_freq > 0)
        v_freq_w = np.array( df[FeatureVector.COL_FREQ_WEIGHTED] )
        v_log_freq = np.array( df[FeatureVector.COL_LOG_FREQ] )
        v_sigm_freq = np.array( df[FeatureVector.COL_SIGMOID_FREQ] )

        expected_v_freq = np.array(expected_freq)
        if feature_as_presence_only:
            expected_v_freq = 1.0 * (v_freq > 0)
        expected_v_freq_w = np.array((v_freq_w))
        if feature_weights:
            expected_v_freq_w = expected_v_freq * np.array(feature_weights)
        expected_v_log_freq = np.log(1 + v_freq) / np.log(base)
        expected_v_sigm_freq = 2 * ( ( 1 / ( 1 + np.exp(-v_freq) ) ) - 0.5 )

        for (tname, observed, expected) in (
                ('freq', v_freq, expected_v_freq),
                ('freq_w', v_freq_w, expected_v_freq_w),
                ('log_freq', v_log_freq, expected_v_log_freq),
                ('sigm_freq', v_sigm_freq, expected_v_sigm_freq)
        ):
            ok = UnitTest.assert_true(
                observed = observed.tolist(),
                expected = expected.tolist(),
                test_comment = 'Test "' + str(name) + ':' + str(tname) + '"'
            )
            unit_test_res.update_bool(res_bool=ok)


    def run_unit_test(self):
        res = ResultObj(count_ok=0, count_fail=0)

        sb = ['我', '帮', '崔', 'I', '确实']
        base = 10
        f = FeatureVector()
        f.set_freq_feature_vector_template(sb)
        # print(f.fv_template)

        # Use word frequency
        txt_list = '确实 有 在 帮 我 崔 吧 帮 我'.split(sep=' ')
        df_fv = f.get_freq_feature_vector(text_list=txt_list, feature_as_presence_only=False, log_base=base)
        # print(df_fv)
        self.check(name=1, df=df_fv, expected_freq=[2.0, 2.0, 1.0, 0.0, 1.0], base=base, unit_test_res=res)

        # Now try with different weights
        f.set_feature_weights([1, 2, 3, 4, 5])
        df_fv = f.get_freq_feature_vector(text_list=txt_list, feature_as_presence_only=False, log_base=base)
        # print(df_fv)
        self.check(name=1, df=df_fv, expected_freq=[2.0, 2.0, 1.0, 0.0, 1.0], base=base, unit_test_res=res, feature_weights=[1, 2, 3, 4, 5])

        # Use word presence
        txt_list = '确实 有 在 帮 我 崔 吧 帮 我'.split(' ')
        f.set_feature_weights([1, 1, 1, 1, 1])
        df_fv = f.get_freq_feature_vector(text_list=txt_list, feature_as_presence_only=True, log_base=base)
        # print(df_fv)
        self.check(name=3, df=df_fv, expected_freq=[2.0, 2.0, 1.0, 0.0, 1.0], base=base, unit_test_res=res, feature_as_presence_only=True)

        # Now try with different weights
        f.set_feature_weights([1, 2, 3, 4, 5])
        df_fv = f.get_freq_feature_vector(text_list=txt_list, feature_as_presence_only=True, log_base=base)
        # print(df_fv)
        self.check(name=4, df=df_fv, expected_freq=[2.0, 2.0, 1.0, 0.0, 1.0], base=base, unit_test_res=res, feature_as_presence_only=True)

        txt_list = '为什么 无法 兑换 商品 ？'.split(' ')
        f.set_feature_weights([1, 1, 1, 1, 1])
        df_fv = f.get_freq_feature_vector(text_list=txt_list, feature_as_presence_only=True, log_base=base)
        # print(df_fv)
        self.check(name=5, df=df_fv, expected_freq=[0.0, 0.0, 0.0, 0.0, 0.0], base=base, unit_test_res=res, feature_as_presence_only=True)

        return res


if __name__ == '__main__':
    Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1
    res = FeatureVectorUnitTest().run_unit_test()
    exit(res.count_fail)
