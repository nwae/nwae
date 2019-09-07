# -*- coding: utf-8 -*-

import numpy as np
import nwae.utils.Log as log
from inspect import currentframe, getframeinfo
import time
import nwae.utils.Profiling as prf
import nwae.utils.StringUtils as su
import nwae.lib.lang.model.FeatureVector as fv
import nwae.lib.lang.nlp.WordSegmentation as ws
import nwae.lib.lang.nlp.SynonymList as sl
import nwae.lib.math.NumpyUtil as npUtil
import nwae.lib.math.ml.ModelInterface as modelIf
import nwae.lib.math.ml.ModelHelper as modelHelper
import threading


#
# Given a model, predicts the point class
#
class PredictClass(threading.Thread):

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
            model_name,
            identifier_string,
            dir_path_model,
            lang,
            dirpath_synonymlist,
            postfix_synonymlist,
            dir_wordlist,
            postfix_wordlist,
            dir_wordlist_app,
            postfix_wordlist_app,
            do_profiling = False
    ):
        super(PredictClass, self).__init__()

        self.model_name = model_name
        self.identifier_string = identifier_string
        self.dir_path_model = dir_path_model

        self.lang = lang
        self.dirpath_synonymlist = dirpath_synonymlist
        self.postfix_synonymlist = postfix_synonymlist
        self.dir_wordlist = dir_wordlist
        self.postfix_wordlist = postfix_wordlist
        self.dir_wordlist_app = dir_wordlist_app
        self.postfix_wordlist_app = postfix_wordlist_app
        self.do_profiling = do_profiling

        self.model = modelHelper.ModelHelper.get_model(
            model_name        = self.model_name,
            identifier_string = self.identifier_string,
            dir_path_model    = self.dir_path_model,
            training_data     = None,
            do_profiling      = self.do_profiling
        )
        self.model.start()

        self.synonymlist = sl.SynonymList(
            lang                = self.lang,
            dirpath_synonymlist = self.dirpath_synonymlist,
            postfix_synonymlist = self.postfix_synonymlist
        )

        self.wseg = ws.WordSegmentation(
            lang             = self.lang,
            dirpath_wordlist = self.dir_wordlist,
            postfix_wordlist = self.postfix_wordlist,
            do_profiling     = self.do_profiling
        )
        # Add application wordlist
        self.wseg.add_wordlist(
            dirpath = self.dir_wordlist_app,
            postfix = self.postfix_wordlist_app
        )
        self.count_intent_calls = 0

        # Wait for model to be ready to load synonym & word lists
        self.start()
        return

    def run(self):
        try:
            self.wait_for_model_to_be_ready(
                wait_max_time = 30
            )
        except Exception as ex:
            errmsg =\
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                + ': Waited 30secs for model to be ready but failed! ' + str(ex)
            log.Log.critical(errmsg)
            raise Exception(errmsg)

        try:
            log.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ": Model " + str(self.model_name) + '" ready. Loading synonym & word lists..'
            )
            # Model ready we load the other things
            self.synonymlist.load_synonymlist(
                list_main_words = self.model.get_model_features().tolist()
            )

            # Add synonym list to wordlist (just in case they are not synched)
            len_before = self.wseg.lang_wordlist.wordlist.shape[0]
            self.wseg.add_wordlist(
                dirpath     = None,
                postfix     = None,
                array_words = list(self.synonymlist.synonymlist[sl.SynonymList.COL_WORD])
            )
            len_after = self.wseg.lang_wordlist.wordlist.shape[0]
            if len_after - len_before > 0:
                words_not_synched = self.wseg.lang_wordlist.wordlist[sl.SynonymList.COL_WORD][len_before:len_after]
                log.Log.warning(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ": Warning. These words not in word list but in synonym list:\n\r" + str(words_not_synched)
                )
        except Exception as ex:
            errmsg = \
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': Exception initializing synonym & word lists: ' + str(ex)
            log.Log.critical(errmsg)
            raise Exception(errmsg)

    #
    # Two things need to be ready, the model and our synonym list that depends on x_name from the model
    #
    def wait_for_model_to_be_ready(
            self,
            wait_max_time = 10
    ):
        if self.model.is_model_ready():
            return

        count = 1
        sleep_time_wait_model = 0.1
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
        self.wait_for_model_to_be_ready()

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
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Text "' + str(inputtext) + '" segmented to "' + str(text_segmented)
            + '", normalized to "' + str(text_normalized) + '"'
        )
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
        self.wait_for_model_to_be_ready()

        self.count_intent_calls = self.count_intent_calls + 1

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
        log.Log.debugdebug(fv_text_1d)

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

        log.Log.debugdebug(
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
    import mozg.ConfigFile as cf
    cf.ConfigFile.get_cmdline_params_and_init_config()
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_INFO

    pc = PredictClass(
        model_name           = modelHelper.ModelHelper.MODEL_NAME_HYPERSPHERE_METRICSPACE,
        identifier_string    = '',
        dir_path_model       = '',
        lang                 = 'cn',
        dirpath_synonymlist  = cf.ConfigFile.DIR_SYNONYMLIST,
        postfix_synonymlist  = cf.ConfigFile.POSTFIX_SYNONYMLIST,
        dir_wordlist         = cf.ConfigFile.DIR_WORDLIST,
        postfix_wordlist     = cf.ConfigFile.POSTFIX_WORDLIST,
        dir_wordlist_app     = cf.ConfigFile.DIR_APP_WORDLIST,
        postfix_wordlist_app = cf.ConfigFile.POSTFIX_APP_WORDLIST,
        do_profiling         = True
    )

    # Return all results in the top 5
    res = pc.predict_class_text_features(
        inputtext="我爱你",
        match_pct_within_top_score = 0,
        include_match_details      = True,
        top = 5
    )
    print(res.match_details)

    # Return only those results with score at least 70% of top score
    res = pc.predict_class_text_features(
        inputtext="我爱你",
        match_pct_within_top_score = 0.7,
        include_match_details      = True,
        top = 5
    )
    print(res.match_details)
