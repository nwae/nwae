# -*- coding: utf-8 -*-

import mozg.common.util.Log as log
from inspect import currentframe, getframeinfo
import time
import mozg.common.util.Profiling as prf
import mozg.lib.lang.model.FeatureVector as fv


#
# Given a model, predicts the point class
#
class PredictClass:

    def __init__(
            self,
            # This is the model with standard model interface that implements the basic methods
            model_interface,
            do_profiling = True
    ):
        self.model = model_interface
        self.do_profiling = do_profiling

        self.count_intent_calls = 0
        return

    def predict_class(
            self,
            # This is the point given in feature format, instead of standard array format
            v_feature_segmented
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
        space_profiling = '      '

        #
        # Replace words with root words
        # This step uses synonyms and replaces say 存钱, 入钱, 入款, all with the standard 存款
        # This will reduce training data without needing to put all versions of the same thing.
        #
        text_normalized = self.synonymlist_ro.normalize_text(text_segmented=v_feature_segmented)
        text_normalized = text_normalized.lower()
        log.Log.debugdebug('#')
        log.Log.debugdebug('# TEXT NORMALIZATION')
        log.Log.debugdebug('#')
        log.Log.debugdebug('Text [' + v_feature_segmented + '] normalized to [' + text_normalized + ']')
        if self.do_profiling:
            b = prf.Profiling.stop()
            log.Log.info(
                '.' + space_profiling
                + 'Txt=' + v_feature_segmented + ']'
                + ' PROFILING Intent (replace root words): ' + prf.Profiling.get_time_dif_str(a, b)
            )

        features_model = list(self.df_rfv_ro.columns)

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
        model_fv.set_freq_feature_vector_template(list_symbols=features_model)

        # Get feature vector of text
        try:
            df_fv = model_fv.get_freq_feature_vector(text=text_normalized)
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Exception occurred calculating FV for "' + str(text_normalized) \
                     + '": Exception "' + str(ex) \
                     + '. Using FV Template ' + str(model_fv.get_fv_template()) \
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
            raise Exception(
                str(self.__class__) + ': Expected a 1D vector, got ' + str(fv_text_normalized_1d.ndim) + 'D!')
        log.Log.debug(fv_text_1d)
        log.Log.debug(fv_text_normalized_1d)

        if self.do_profiling:
            b = prf.Profiling.stop()
            log.Log.info('.' + space_profiling
                         + '[ChatID=' + str(chatid) + ', Txt=' + text_segmented + ']'
                         + ' PROFILING Intent (FV & Normalization): ' + prf.Profiling.get_time_dif_str(a, b))



        v = npUtil.NumpyUtil.convert_dimension(arr=reordered_test_x[i], to_dim=2)
        predict_result = self.model.predict_class(
            x           = v
        )
        y_observed = predict_result.predicted_classes
        all_y_observed_top.append(y_observed[0])
        all_y_observed.append(y_observed)
        top_class_distance = predict_result.top_class_distance
        match_details = predict_result.match_details

        print('Point v ' + str(v) + '\n\rTop Class Distance: ' + str(top_class_distance))