# -*- coding: utf-8 -*-

import numpy as np
import mozg.common.util.Log as log
from inspect import currentframe, getframeinfo
import time
import mozg.common.util.Profiling as prf
import mozg.common.util.StringUtils as su
import mozg.lib.lang.model.FeatureVector as fv
import mozg.lib.lang.nlp.WordSegmentation as ws
import mozg.lib.lang.nlp.SynonymList as sl
import mozg.lib.math.NumpyUtil as npUtil
import mozg.lib.math.ml.ModelInterface as modelIf


#
# Given a model, predicts the point class
#
class PredictClass:

    #
    # This is to decide how many top answers to keep.
    # If this value is say 70%, and our top scores are 70, 60, 40, 20, then
    # 70% * 70 is 49, thus only scores 70, 60 will be kept as it is higher than 49
    #
    CONSTANT_PERCENT_WITHIN_TOP_SCORE = 0.6
    MAX_QUESTION_LENGTH = 100

    # Default match top X
    MATCH_TOP = 10

    def __init__(
            self,
            # This is the model with standard model interface that implements the basic methods
            model_interface,
            lang,
            dirpath_synonymlist,
            dir_wordlist,
            postfix_wordlist,
            dir_wordlist_app,
            postfix_wordlist_app,
            do_profiling = True
    ):
        self.model = model_interface
        self.lang = lang
        self.dirpath_synonymlist = dirpath_synonymlist
        self.dir_wordlist = dir_wordlist
        self.postfix_wordlist = postfix_wordlist
        self.dir_wordlist_app = dir_wordlist_app
        self.postfix_wordlist_app = postfix_wordlist_app
        self.do_profiling = do_profiling

        self.synonymlist = sl.SynonymList(
            lang                = self.lang,
            dirpath_synonymlist = self.dirpath_synonymlist,
            postfix_synonymlist = '.synonymlist.txt'
        )
        self.synonymlist.load_synonymlist()

        self.wseg = ws.WordSegmentation(
            lang = self.lang,
            dirpath_wordlist = self.dir_wordlist,
            postfix_wordlist = self.postfix_wordlist,
            do_profiling = self.do_profiling
        )
        # Add application wordlist
        self.wseg.add_wordlist(
            dirpath=self.dir_wordlist_app,
            postfix=self.postfix_wordlist_app
        )

        # Add synonym list to wordlist (just in case they are not synched)
        len_before = self.wseg.lang_wordlist.wordlist.shape[0]
        self.wseg.add_wordlist(
            dirpath     = None,
            postfix     = None,
            array_words = list(self.synonymlist.synonymlist['Word'])
        )
        len_after = self.wseg.lang_wordlist.wordlist.shape[0]
        if len_after - len_before > 0:
            words_not_synched = self.wseg.lang_wordlist.wordlist['Word'][len_before:len_after]
            log.Log.warning(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ": Warning. These words not in word list but in synonym list:\n\r" + str(words_not_synched)
            )

        self.count_intent_calls = 0
        return

    #
    # A helper class to predict class given text sentence instead of a nice array
    #
    def predict_class_text_features(
            self,
            inputtext,
            top = MATCH_TOP,
            match_pct_within_top_score=CONSTANT_PERCENT_WITHIN_TOP_SCORE,
            include_match_details = False,
            chatid = None
    ):
        starttime_prf = prf.Profiling.start()
        space_profiling = '      '

        # Segment words first
        inputtext_trim = su.StringUtils.trim(inputtext)
        text_segmented = self.wseg.segment_words(
            text = su.StringUtils.trim(inputtext_trim)
        )

        #
        # Replace words with root words
        # This step uses synonyms and replaces say 存钱, 入钱, 入款, all with the standard 存款
        # This will reduce training data without needing to put all versions of the same thing.
        #
        text_normalized = self.synonymlist.normalize_text(text_segmented=text_segmented)
        text_normalized = text_normalized.lower()
        log.Log.debugdebug('#')
        log.Log.debugdebug('# TEXT NORMALIZATION')
        log.Log.debugdebug('#')
        log.Log.debugdebug('Text [' + text_segmented + '] normalized to [' + text_normalized + ']')
        if self.do_profiling:
            log.Log.info(
                '.' + space_profiling
                + 'Chat ID="' + str(chatid) + '", Txt="' + text_segmented + '"'
                + ' PROFILING Intent (replace root words): '
                + prf.Profiling.get_time_dif_str(starttime_prf, prf.Profiling.stop())
            )

        return self.predict_class_features(
            v_feature_segmented = text_normalized,
            id                  = chatid,
            top                 = top,
            match_pct_within_top_score = match_pct_within_top_score,
            include_match_details = include_match_details
        )

    #
    # A helper class to predict class given features instead of a nice array
    #
    def predict_class_features(
            self,
            # This is the point given in feature format, instead of standard array format
            v_feature_segmented,
            top = MATCH_TOP,
            match_pct_within_top_score = CONSTANT_PERCENT_WITHIN_TOP_SCORE,
            include_match_details = False,
            # Any relevant ID for logging purpose only
            id = None
    ):
        self.count_intent_calls = self.count_intent_calls + 1

        count = 1
        sleep_time_wait_model = 0.1
        wait_max_time = 10
        while not self.model.is_model_ready():
            log.Log.warning(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Model not yet ready, sleep for ' + str(count * sleep_time_wait_model) + ' secs now..'
            )
            if count * sleep_time_wait_model > wait_max_time:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                         + ': Waited too long ' + str(count * sleep_time_wait_model) + ' secs. Raising exception..'
                raise Exception(errmsg)
            time.sleep(sleep_time_wait_model)
            count = count + 1

        starttime_predict_class = prf.Profiling.start()

        features_model = list(self.model.get_model_features())
        log.Log.debugdebug('Using model features:\n\r' + str(features_model))

        #
        # Convert sentence to a mathematical object (feature vector)
        #
        model_fv = fv.FeatureVector()
        model_fv.set_freq_feature_vector_template(list_symbols=features_model)

        # Get feature vector of text
        try:
            df_fv = model_fv.get_freq_feature_vector(text=v_feature_segmented)
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Exception occurred calculating FV for "' + str(v_feature_segmented) \
                     + '": Exception "' + str(ex) \
                     + '\n\rUsing FV Template:\n\r' + str(model_fv.get_fv_template()) \
                     + ', FV Weights:\n\r' + str(model_fv.get_fv_weights())
            log.Log.critical(errmsg)
            raise Exception(errmsg)

        # This creates a single row matrix that needs to be transposed before matrix multiplications
        # ndmin=2 will force numpy to create a 2D matrix instead of a 1D vector
        # For now we make it 1D first
        fv_text_1d = np.array(df_fv['Frequency'].values, ndmin=1)
        if fv_text_1d.ndim != 1:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Expected a 1D vector, got ' + str(fv_text_1d.ndim) + 'D!'
            )
        log.Log.debug(fv_text_1d)

        v = npUtil.NumpyUtil.convert_dimension(arr=fv_text_1d, to_dim=2)
        log.Log.debugdebug('v dims ' + str(v.shape))
        predict_result = self.model.predict_class(
            x             = v,
            top           = top,
            include_match_details = include_match_details
        )

        #
        # Choose which scores to keep.
        #
        if include_match_details:
            df_match = predict_result.match_details
            top_score = float(df_match[modelIf.ModelInterface.TERM_SCORE].loc[df_match.index[0]])
            df_match_keep = df_match[
                df_match[modelIf.ModelInterface.TERM_SCORE] >= top_score*match_pct_within_top_score
            ]
            df_match_keep = df_match_keep.reset_index(drop=True)
            # Overwrite data frame
            predict_result.match_details = df_match_keep

        y_observed = predict_result.predicted_classes
        top_class_distance = predict_result.top_class_distance

        log.Log.info(
            str(self.__class__) + str(getframeinfo(currentframe()).lineno)
            + ': Point v ' + str(v) + '\n\rObserved Class: ' + str(y_observed)
            + ', Top Class Distance: ' + str(top_class_distance)
        )

        if self.do_profiling:
            log.Log.info(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ': ID="' + str(id) + '", Txt="' + str(v_feature_segmented) + '"'
                + ' PROFILING predict class: '
                + prf.Profiling.get_time_dif_str(starttime_predict_class, prf.Profiling.stop())
            )
        return predict_result


if __name__ == '__main__':
    #
    # Now read back params and predict classes
    #
    topdir = '/Users/mark.tan/git/mozg'
    import mozg.lib.math.ml.metricspace.MetricSpaceModel as msModel
    ms_pc = msModel.MetricSpaceModel(
        identifier_string = 'demo_msmodel_accid4_botid22',
        # Directory to keep all our model files
        dir_path_model    = topdir + '/app.data/models',
        do_profiling      = True
    )
    ms_pc.load_model_parameters()

    pc = PredictClass(
        model_interface      = ms_pc,
        lang                 = 'cn',
        dirpath_synonymlist  = topdir + '/nlp.data/app/chats',
        dir_wordlist         = topdir + '/nlp.data/wordlist',
        postfix_wordlist     = '-wordlist.txt',
        dir_wordlist_app     = topdir + '/nlp.data/app/chats',
        postfix_wordlist_app = '.wordlist.app.txt',
        do_profiling         = True
    )

    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_INFO
    # Return all results in the top 5
    res = pc.predict_class_text_features(
        inputtext="存款",
        match_pct_within_top_score = 0,
        include_match_details      = True,
        top = 5
    )
    print(res.match_details)

    # Return only those results with score at least 70% of top score
    res = pc.predict_class_text_features(
        inputtext="存款",
        match_pct_within_top_score = 0.7,
        include_match_details      = True,
        top = 5
    )
    print(res.match_details)