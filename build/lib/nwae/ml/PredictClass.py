# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
import time
import nwae.utils.Profiling as prf
from nwae.lang.LangFeatures import LangFeatures
from nwae.ml.ModelInterface import ModelInterface
from nwae.ml.modelhelper.ModelHelper import ModelHelper
import threading
from nwae.lang.preprocessing.TxtPreprocessor import TxtPreprocessor
from nwae.lang.detect.LangDetect import LangDetect
import nwae.utils.UnitTest as ut
from nwae.lang.model.FeatureVect import FeatureVector


#
# Given a model, predicts the point class
#
class PredictClass(threading.Thread):

    #
    # This is to decide how many top answers to keep.
    # If this value is say 70%, and our top scores are 70, 60, 40, 20, then
    # 70% * 70 is 49, thus only scores 70, 60 will be kept as it is higher than 49.
    # This value should not be very high as it is the first level filtering, as
    # applications might apply their own filtering some more.
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
            word_freq_model         = FeatureVector.COL_FREQUENCY,
            confidence_level_scores = None,
            do_spelling_correction  = False,
            do_word_stemming        = True,
            do_profiling            = False,
            lang_additional         = ()
    ):
        super(PredictClass, self).__init__()

        self.model_name = model_name
        self.identifier_string = identifier_string
        self.dir_path_model = dir_path_model

        self.lang_main = lang
        self.dirpath_synonymlist = dirpath_synonymlist
        self.postfix_synonymlist = postfix_synonymlist
        self.dir_wordlist = dir_wordlist
        self.postfix_wordlist = postfix_wordlist
        self.dir_wordlist_app = dir_wordlist_app
        self.postfix_wordlist_app = postfix_wordlist_app
        self.word_freq_model = word_freq_model
        self.do_spelling_correction = do_spelling_correction
        self.do_word_stemming = do_word_stemming
        self.do_profiling = do_profiling

        if lang_additional is None:
            lang_additional = ()
        self.lang_additional = [
            LangFeatures.map_to_lang_code_iso639_1(lang_code=l) for l in lang_additional
        ]
        try:
            self.lang_additional.remove(self.lang_main)
        except ValueError:
            pass
        self.lang_additional = list(set(self.lang_additional))

        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Predictor class initialization using model "' + str(self.identifier_string)
            + '", word freq model "' + str(self.word_freq_model)
            + '", main language "' + str(self.lang_main)
            + '", additional languages: ' + str(self.lang_additional)
        )

        self.model = ModelHelper.get_model(
            model_name              = self.model_name,
            # We will load the model params from file/etc from trained model
            model_params            = None,
            identifier_string       = self.identifier_string,
            dir_path_model          = self.dir_path_model,
            training_data           = None,
            confidence_level_scores = confidence_level_scores,
            do_profiling            = self.do_profiling
        )
        self.model.start()
        # Keep track if model reloaded. This counter is manually updated by this class.
        self.model_last_reloaded_counter = 0
        self.load_text_processor_mutex = threading.Lock()

        # After loading model, we still need to load word lists, etc.
        self.is_all_initializations_done = False

        #
        # We initialize word segmenter and synonym list after the model is ready
        # because it requires the model features so that root words of synonym lists
        # are only from the model features
        #
        self.predict_class_txt_processor = None
        self.lang_detect = None

        self.count_predict_calls = 0

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
            Log.critical(errmsg)
            raise Exception(errmsg)

        self.load_text_processor()
        return

    def stop_model_thread(self):
        # Kill any background jobs
        try:
            Log.info(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ': "' + str(self.identifier_string) + '" Stopping model background job..'
            )
            self.model.stoprequest.set()
        except Exception as ex:
            Log.error(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ': "' + str(self.identifier_string) + '" Stop model background job exception: ' + str(ex)
            )

    #
    # Когда модель перезагрузит, текстовый процессор также надо перезагрузить
    # Должен опять загрузить потому что класс TxtPreprocessor нужны данные из модели
    # Этот процесс занимает много времени (больше чем 5 секунд до десяти)
    # TODO Переместить эту функцию в фон
    #
    def load_text_processor(self):
        try:
            self.load_text_processor_mutex.acquire()
            # Don't allow to load again
            if self.model_last_reloaded_counter == self.model.get_model_reloaded_counter():
                Log.warning(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Model "' + str(self.identifier_string) + '" not reloading PredictClassTxtProcessor.'
                )
                return

            Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Model "' + str(self.model_name) + '" ready. Loading synonym & word lists..'
            )

            self.lang_detect = LangDetect()

            self.predict_class_txt_processor = {}
            for uh in [self.lang_main] + self.lang_additional:
                try:
                    model_features_list = self.model.get_model_features().tolist()
                except Exception as ex_feature_list:
                    Log.error(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Model "' + str(self.model_name) + '" identifier "' + str(self.identifier_string)
                        + '" model feature list empty!'
                    )
                    model_features_list = None

                self.predict_class_txt_processor[uh] = TxtPreprocessor(
                    identifier_string      = self.identifier_string,
                    dir_path_model         = self.dir_path_model,
                    model_features_list    = model_features_list,
                    lang                   = uh,
                    dirpath_synonymlist    = self.dirpath_synonymlist,
                    postfix_synonymlist    = self.postfix_synonymlist,
                    dir_wordlist           = self.dir_wordlist,
                    postfix_wordlist       = self.postfix_wordlist,
                    dir_wordlist_app       = self.dir_wordlist_app,
                    postfix_wordlist_app   = self.postfix_wordlist_app,
                    # TODO For certain languages like English, it is essential to include this
                    #   But at the same time must be very careful. By adding manual rules, for
                    #   example we include words 'it', 'is'.. But "It is" could be a very valid
                    #   training data that becomes excluded wrongly.
                    stopwords_list         = None,
                    do_spelling_correction = self.do_spelling_correction,
                    do_word_stemming       = self.do_word_stemming,
                    do_profiling           = self.do_profiling
                )

            self.is_all_initializations_done = True
            # Manually update this model last reloaded counter
            self.model_last_reloaded_counter = self.model.get_model_reloaded_counter()
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Model name "' + str(self.model_name)
                + '", identifier "' + str(self.identifier_string)
                + '" All initializations done for model "' + str(self.identifier_string)
                + '". Model Reload counter = ' + str(self.model_last_reloaded_counter)
            )
        except Exception as ex:
            errmsg = \
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': Model name "' + str(self.model_name) \
                + '", identifier "' + str(self.identifier_string) \
                + '" Exception initializing synonym & word lists: ' + str(ex)
            Log.critical(errmsg)
            raise Exception(errmsg)
        finally:
            self.load_text_processor_mutex.release()

    #
    # Two things need to be ready, the model and our synonym list that depends on x_name from the model
    #
    def wait_for_model_to_be_ready(
            self,
            wait_max_time = 10
    ):
        #
        # Model reloaded without us knowing, e.g. user trained it, etc.
        #
        if self.model_last_reloaded_counter != self.model.get_model_reloaded_counter():
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + 'Model "' + str(self.identifier_string) + '" last counter '
                + str(self.model_last_reloaded_counter) + ' not equal to model counter '
                + str(self.model.get_model_reloaded_counter())
                + '. Model updated, thus we must update our text processor.'
            )
            #
            # Должен опять загрузить потому что класс TxtPreprocessor нужны данные из модели
            #
            self.load_text_processor()

        if self.model.is_model_ready():
            return

        count = 1
        sleep_time_wait_model = 0.1
        while not self.model.is_model_ready():
            Log.warning(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Model "' + str(self.identifier_string) + '" not yet ready, sleep for '
                + str(count * sleep_time_wait_model) + ' secs now..'
            )
            if count * sleep_time_wait_model > wait_max_time:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                         + ': Waited for model "' + str(self.identifier_string)\
                         + '" too long ' + str(count * sleep_time_wait_model) + ' secs. Raising exception..'
                raise Exception(errmsg)
            time.sleep(sleep_time_wait_model)
            count = count + 1
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Model "' + str(self.identifier_string) + '" READY.'
        )
        return

    def wait_for_all_initializations_to_be_done(
            self,
            wait_max_time = 10
    ):
        if self.is_all_initializations_done:
            return

        count = 1
        sleep_time_wait_initializations = 0.1
        while not self.is_all_initializations_done:
            Log.warning(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Model not yet fully initialized, sleep for '
                + str(count * sleep_time_wait_initializations) + ' secs now..'
            )
            if count * sleep_time_wait_initializations > wait_max_time:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                         + ': Waited too long ' + str(count * sleep_time_wait_initializations)\
                         + ' secs. Raising exception..'
                raise Exception(errmsg)
            time.sleep(sleep_time_wait_initializations)
            count = count + 1
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Initializations all done for model "' + str(self.identifier_string) + '" READY.'
        )
        return

    #
    # A helper class to predict class given text sentence instead of a nice array
    #
    def predict_class_text_features(
            self,
            inputtext,
            top = MATCH_TOP,
            match_pct_within_top_score = CONSTANT_PERCENT_WITHIN_TOP_SCORE,
            include_match_details = False,
            chatid = None,
    ):
        self.wait_for_model_to_be_ready()
        self.wait_for_all_initializations_to_be_done()

        #
        # Step 1: Detect Language
        #
        possible_langs = self.lang_detect.detect(
            text = inputtext
        )
        # Empty list
        if not possible_langs:
            lang_detected = self.lang_main
        else:
            lang_detected = possible_langs[0]
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Detected lang "' + str(lang_detected) + '" for "' + str(inputtext) + '"'
        )

        # If detected language not supported
        if lang_detected not in [self.lang_main] + self.lang_additional:
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': For "' + str(self.identifier_string)
                + '", detected lang "' + str(lang_detected) + '" not in languages supported'
            )
            lang_detected = self.lang_main

        #
        # Step 2: Process Sentence
        #   Replace with special symbols, split sentence, lemmatization, spell check, etc.
        #
        processed_txt_array = self.predict_class_txt_processor[lang_detected].process_text(
            inputtext = inputtext
        )
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Processed "' + str(inputtext) + '" to "' + str(processed_txt_array) + '"'
        )

        transformed_txt_array = self.model.transform_input_for_model(
            x_input = processed_txt_array,
            word_freq_model = self.word_freq_model,
        )
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Use word model "' + str(self.word_freq_model) + '" Transformed "' + str(processed_txt_array)
            + '" to "' + str(transformed_txt_array) + '"'
        )

        #
        # Step 3: Predict
        #   The text will be converted to suitable vector or one-hot vector types
        #   depending on the underlying model, then only fed to model.
        #
        predict_result = self.predict_class_features(
            x_transformed       = transformed_txt_array,
            id                  = chatid,
            top                 = top,
            match_pct_within_top_score = match_pct_within_top_score,
            include_match_details = include_match_details
        )

        class retclass:
            def __init__(
                    self,
                    predict_result,
                    processed_text_arr,
                    transformed_text_arr
            ):
                # ModelInterface PredictReturnClass
                self.predict_result = predict_result
                self.processed_text_arr = processed_text_arr
                self.transformed_text_arr = transformed_text_arr

        retobj = retclass(
            predict_result       = predict_result,
            processed_text_arr   = processed_txt_array,
            transformed_text_arr = transformed_txt_array
        )
        return retobj

    #
    # A helper class to predict class given features instead of a nice array
    #
    def predict_class_features(
            self,
            # This is the point given in feature format, instead of standard array format
            x_transformed,
            top = MATCH_TOP,
            match_pct_within_top_score = CONSTANT_PERCENT_WITHIN_TOP_SCORE,
            include_match_details = False,
            # Any relevant ID for logging purpose only
            id = None
    ):
        self.wait_for_model_to_be_ready()
        self.wait_for_all_initializations_to_be_done()

        self.count_predict_calls = self.count_predict_calls + 1

        starttime_predict_class = prf.Profiling.start()

        predict_result = self.model.predict_class(
            x             = x_transformed,
            top           = top,
            include_match_details = include_match_details
        )

        #
        # Choose which scores to keep, we only have scores if we included the match details
        #
        if include_match_details:
            df_match = predict_result.match_details
            if df_match is not None:
                top_score = float(df_match[ModelInterface.TERM_SCORE].loc[df_match.index[0]])
                df_match_keep = df_match[
                    df_match[ModelInterface.TERM_SCORE] >= top_score*match_pct_within_top_score
                ]
                df_match_keep = df_match_keep.reset_index(drop=True)
                # Overwrite data frame
                predict_result.match_details = df_match_keep

        y_observed = predict_result.predicted_classes
        top_class_distance = predict_result.top_class_distance

        Log.debug(
            str(self.__class__) + str(getframeinfo(currentframe()).lineno)
            + ': Input x: ' + str(x_transformed) + ', observed class: ' + str(y_observed)
            + ', top distance: ' + str(top_class_distance)
        )

        if self.do_profiling:
            Log.debug(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ': ID="' + str(id) + '", x="' + str(x_transformed) + '"'
                + ' PROFILING predict class: '
                + prf.Profiling.get_time_dif_str(starttime_predict_class, prf.Profiling.stop())
            )

        return predict_result


class PredictClassUnitTest:
    def __init__(self, ut_params):
        self.ut_params = ut_params

    def run_unit_test(self):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)

        from nwae.ml.metricspace.ut.UtMetricSpaceModel import UnitTestMetricSpaceModel
        x_text = UnitTestMetricSpaceModel.DATA_TEXTS
        y = UnitTestMetricSpaceModel.DATA_Y

        predict = PredictClass(
            model_name             = ModelHelper.MODEL_NAME_HYPERSPHERE_METRICSPACE,
            identifier_string      = UnitTestMetricSpaceModel.IDENTIFIER_STRING,
            dir_path_model         = self.ut_params.dirpath_model,
            lang                   = LangFeatures.LANG_KO,
            dir_wordlist           = self.ut_params.dirpath_wordlist,
            postfix_wordlist       = self.ut_params.postfix_wordlist,
            dir_wordlist_app       = self.ut_params.dirpath_app_wordlist,
            postfix_wordlist_app   = self.ut_params.postfix_app_wordlist,
            dirpath_synonymlist    = self.ut_params.dirpath_synonymlist,
            postfix_synonymlist    = self.ut_params.postfix_synonymlist,
            # чуть-чуть не правильно, потому-что мы используем модель тестировки из UtMetricSpaceModel.py
            # и там последним тестом был сигмоид. но если бы это было не так, тест все равно бы прошел
            word_freq_model        = FeatureVector.COL_SIGMOID_FREQ,
            do_spelling_correction = False,
            do_profiling           = True
        )

        for i in range(len(x_text)):
            label = y[i]
            text_arr = x_text[i]
            text = ' '.join(text_arr)
            # Return all results in the top 5
            res = predict.predict_class_text_features(
                inputtext                  = text,
                match_pct_within_top_score = 0,
                include_match_details      = True,
                top                        = 5,
            )
            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = res.predict_result.predicted_classes[0],
                expected = label,
                test_comment = 'Test "' + str(text) + '" label ' + str(label)
            ))
            Log.info(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ': Match Details' + str(res.predict_result.match_details)
            )

        # Kill any background jobs
        predict.stop_model_thread()

        return res_final


if __name__ == '__main__':
    import nwae.ml.config.Config as cf
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config,
        default_config_file = cf.Config.CONFIG_FILE_DEFAULT
    )
    Log.LOGLEVEL = Log.LOG_LEVEL_INFO

    ut_params = ut.UnitTestParams(
        dirpath_wordlist     = config.get_config(param=cf.Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist     = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_WORDLIST),
        dirpath_app_wordlist = config.get_config(param=cf.Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_app_wordlist = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        dirpath_synonymlist  = config.get_config(param=cf.Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist  = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        dirpath_model        = config.get_config(param=cf.Config.PARAM_MODEL_DIR)
    )
    Log.important('Unit Test Params: ' + str(ut_params.to_string()))

    res_ut = PredictClassUnitTest(ut_params=ut_params).run_unit_test()
    print('***** PASS ' + str(res_ut.count_ok) + ', FAIL ' + str(res_ut.count_fail))
    exit(res_ut.count_fail)
