# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from nwae.ml.modelhelper.ModelHelper import ModelHelper
from nwae.ml.trainer.TextTrainer import TextTrainer
import nwae.ml.TrainingDataModel as tdm
from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
import nwae.math.NumpyUtil as npUtil
from nwae.ml.config.Config import Config
import nwae.utils.Profiling as prf
from nwae.utils.UnitTest import ResultObj, UnitTestParams, UnitTest
from nwae.lang.preprocessing.BasicPreprocessor import BasicPreprocessor
from nwae.lang.model.WordFreqDocMatrix import WordFreqDocMatrix


class UnitTestMetricSpaceModel:

    IDENTIFIER_STRING = 'demo_ut1'

    DATA_TEXTS = [
        # 0
        ['하나', '두', '두', '셋', '넷'],
        ['하나', '하나', '두', '셋', '셋', '넷'],
        ['하나', '두', '셋', '넷'],
        # 1
        ['두', '셋', '셋', '넷'],
        ['두', '두', '셋', '셋', '넷', '넷'],
        ['두', '두', '셋', '넷', '넷'],
        # 2
        ['넷', '다섯', '다섯', '여섯', '여섯', '여섯'],
        ['두', '넷', '넷', '다섯', '다섯', '여섯', '여섯'],
        ['두', '넷', '다섯', '여섯', '여섯'],
        # 3
        ['하나', '여섯'],
        ['하나', '여섯', '여섯'],
        ['하나', '하나', '여섯'],
        ['두', '셋', '넷', '다섯'],
        ['두', '셋', '셋', '넷', '다섯'],
        ['두', '셋', '넷', '넷', '다섯']
    ]
    # Must be in sync with DATA_TEXTS
    DATA_X = np.array(
        [
            # в таком порядке ['하나', '두', '셋', '넷', '다섯', '여섯']
            # 무리 0
            [1, 2, 1, 1, 0, 0],
            [2, 1, 2, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            # 무리 1
            [0, 1, 2, 1, 0, 0],
            [0, 2, 2, 2, 0, 0],
            [0, 2, 1, 2, 0, 0],
            # 무리 2
            [0, 0, 0, 1, 2, 3],
            [0, 1, 0, 2, 2, 2],
            [0, 1, 0, 1, 1, 2],
            # 무리 3 (mix)
            [1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 2],
            [2, 0, 0, 0, 0, 1],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 2, 1, 1, 0],
            [0, 1, 1, 2, 1, 0]
        ]
    )
    # Порядок изменяется по самому частому
    EXPECTED_X_FEATURE_NAMES = ['넷', '두', '셋', '여섯', '하나', '다섯', '_unk']
    REORDER_FEATURE_NAMES = [3, 1, 2, 5, 0, 4]
    REORDER_FEATURE_NAMES_WITH_UNK = np.array([3, 1, 2, 5, 0, 4, 6])
    EXPECTED_DATA_X_FREQ_NORM = np.array(
        [
            # 무리 0
            [0.37796447, 0.75592895, 0.37796447, 0.,         0.37796447, 0.,         0.       ],
            [0.31622777, 0.31622777, 0.63245553, 0.,         0.63245553, 0.,         0.       ],
            [0.5,        0.5,        0.5,        0.,         0.5,        0.,         0.       ],
            # 무리 1
            [0.40824829, 0.40824829, 0.81649658, 0.,         0.,         0.,         0.       ],
            [0.57735027, 0.57735027, 0.57735027, 0.,         0.,         0.,         0.       ],
            [0.66666667, 0.66666667, 0.33333333, 0.,         0.,         0.,         0.       ],
            # 무리 2
            [0.26726124, 0.0,        0.0,        0.80178373, 0.0,        0.53452248, 0.       ],
            [0.5547002,  0.2773501,  0.0,        0.5547002,  0.0,        0.5547002,  0.       ],
            [0.37796447, 0.37796447, 0.0,        0.75592895, 0.0,        0.37796447, 0.       ],
            # 무리 3 (mix)
            [0.,         0.,         0.,         0.70710678, 0.70710678, 0.,         0.       ],
            [0.,         0.,         0.,         0.89442719, 0.4472136,  0.,         0.       ],
            [0.,         0. ,        0.,         0.4472136,  0.89442719, 0.,         0.       ],
            [0.5,        0.5,        0.5,        0.,         0.,         0.5,        0.       ],
            [0.37796447, 0.37796447, 0.75592895, 0.,         0.,         0.37796447, 0.       ],
            [0.75592895, 0.37796447, 0.37796447, 0.,         0.,         0.37796447, 0.       ],
        ]
    )
    DATA_Y = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    )
    DATA_X_NAME = np.array(['하나', '두', '셋', '넷', '다섯', '여섯'])

    #
    # To test against trained models
    # Need to add one more dimension at the end for "_unk"
    #
    DATA_TEST_X = np.array(
        [
            # 무리 0
            [1.2, 2.0, 1.1, 1.0, 0, 0, 0],
            [2.1, 1.0, 2.4, 1.0, 0, 0, 0],
            [1.5, 1.0, 1.3, 1.0, 0, 0, 0],
            # 무리 1
            [0, 1.1, 2.5, 1.5, 0, 0, 0],
            [0, 2.2, 2.6, 2.4, 0, 0, 0],
            [0, 2.3, 1.7, 2.1, 0, 0, 0],
            # 무리 2
            [0, 0.0, 0, 1.6, 2.1, 3.5, 0],
            [0, 1.4, 0, 2.7, 1.2, 2.4, 0],
            [0, 1.1, 0, 1.3, 1.3, 2.1, 0],
            # 무리 3
            [1.1, 0.0, 0.0, 0.0, 0.0, 1.5, 0],
            [0.0, 1.4, 0.9, 1.7, 1.2, 0.0, 0]
        ]
    )
    DATA_TEST_X_NAME = np.array(['하나', '두', '셋', '넷', '다섯', '여섯', BasicPreprocessor.W_UNK])

    #
    # Layers Design
    #
    NEURAL_NETWORK_LAYERS = [
        {
            'units': 128,
            'activation': 'relu',
            'input_shape': (DATA_X.shape[1],)
        },
        {
            # 4 unique classes
            'units': 4,
            'activation': 'softmax'
        }
    ]

    def __init__(
            self,
            ut_params,
            model_name
    ):
        self.ut_params = ut_params
        if self.ut_params is None:
            # We only do this for convenience, so that we have access to the Class methods in UI
            self.ut_params = UnitTestParams()
        self.res_final = ResultObj()

        self.identifier_string = UnitTestMetricSpaceModel.IDENTIFIER_STRING
        self.model_name = model_name

        self.x_expected = UnitTestMetricSpaceModel.DATA_X
        self.texts = UnitTestMetricSpaceModel.DATA_TEXTS

        self.y = UnitTestMetricSpaceModel.DATA_Y
        self.x_name = UnitTestMetricSpaceModel.DATA_X_NAME

        return

    def unit_test_train(
            self,
            word_freq_model,
            model_params = None
    ):
        #
        # Finally we have our text data in the desired format
        #
        y_list = self.y.tolist()
        y_list = list(y_list)
        self.tdm_obj = tdm.TrainingDataModel.unify_word_features_for_text_data(
            label_id                 = y_list.copy(),
            label_name               = y_list.copy(),
            sentences_list           = self.texts,
            word_frequency_model     = word_freq_model,
            keywords_remove_quartile = 0,
            add_unknown_word_in_keywords_list = True,
        )

        trainer_obj = TextTrainer(
            identifier_string = self.identifier_string,
            model_name        = self.model_name,
            model_params      = model_params,
            dir_path_model    = self.ut_params.dirpath_model,
            training_data     = self.tdm_obj,
            word_freq_model   = word_freq_model,
        )

        trainer_obj.train(
            write_training_data_to_storage = True
        )

        # проверка имен атрибутов, порядок который изменился с самого частого слова к самому редкому
        xname = self.tdm_obj.get_x_name()
        self.res_final.update_bool(res_bool=UnitTest.assert_true(
            observed = list(xname),
            expected = list(self.EXPECTED_X_FEATURE_NAMES),
            test_comment = 'x features ' + str(xname)
        ))

        expected_data_x_norm = self.DATA_X
        # Need to reorder
        expected_data_x_norm = np.array([row[self.REORDER_FEATURE_NAMES] for row in expected_data_x_norm])
        # Add unknown word
        expected_data_x_norm = expected_data_x_norm.tolist()
        [row.append(0) for row in expected_data_x_norm]
        expected_data_x_norm = np.array(expected_data_x_norm)
        if word_freq_model in [WordFreqDocMatrix.BY_SIGMOID_FREQ, WordFreqDocMatrix.BY_SIGMOID_FREQ_NORM]:
            expected_data_x_norm = 2 * ((1 / (1 + np.exp(-expected_data_x_norm))) - 0.5)
        elif word_freq_model in [WordFreqDocMatrix.BY_LOG_FREQ, WordFreqDocMatrix.BY_LOG_FREQ_NORM]:
            expected_data_x_norm = np.log(1 + expected_data_x_norm)
        else:
            pass
        vector_lengths = np.sum(expected_data_x_norm**2, axis=1) ** 0.5
        vector_lengths = np.reshape(vector_lengths, newshape=(len(vector_lengths),1))
        expected_data_x_norm = expected_data_x_norm / vector_lengths

        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Mapped expected x norm to word freq model "' + str(word_freq_model) + '" to ' + str(expected_data_x_norm)
        )

        # проверка
        x_transformed = self.tdm_obj.get_x()
        for i in range(len(x_transformed)):
            x_line = x_transformed[i]
            x_line_expected = expected_data_x_norm[i]
            self.res_final.update_bool(res_bool=UnitTest.assert_true(
                observed = list(np.round(x_line, 8)),
                expected = list(np.round(x_line_expected, 8)),
                test_comment = 'Train test line #' + str(i) + ' test x ' + str(x_line)
            ))

    def unit_test_predict_classes(
            self,
            word_freq_model,
            include_match_details = False,
            top = 5,
    ):
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Test predict classes using model "' + str(self.model_name) + '".'
        )

        # Unit test using direct text (PredictClass.py) is in PredictClass.py itself
        model_obj = ModelHelper.get_model(
            model_name        = self.model_name,
            model_params      = None,
            identifier_string = self.identifier_string,
            dir_path_model    = self.ut_params.dirpath_model,
            training_data     = None
        )
        model_obj.start()
        model_obj.wait_for_model()
        #model_obj.load_model_parameters()

        test_x = UnitTestMetricSpaceModel.DATA_TEST_X
        test_x_name = UnitTestMetricSpaceModel.DATA_TEST_X_NAME
        model_x_name = model_obj.get_model_features()
        if model_x_name is None:
            model_x_name = UnitTestMetricSpaceModel.DATA_X_NAME

        word_freq_model_mapped = WordFreqDocMatrix.map_to_feature_vect_word_freq_measure(freq_measure=word_freq_model)
        if word_freq_model_mapped in [WordFreqDocMatrix.BY_SIGMOID_FREQ, WordFreqDocMatrix.BY_SIGMOID_FREQ_NORM]:
            test_x = 2 * ( (1 / (1 + np.exp(-test_x))) - 0.5 )
        elif word_freq_model_mapped in [WordFreqDocMatrix.BY_LOG_FREQ, WordFreqDocMatrix.BY_LOG_FREQ_NORM]:
            test_x = np.log(1 + test_x)
        else:
            pass
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Mapped to word freq model "' + str(word_freq_model_mapped) + '" to ' + str(test_x)
        )

        if model_x_name.ndim == 2:
            model_x_name = model_x_name[0]
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Model x_name: ' + str(model_x_name)
        )

        # Reorder by model x_name
        df_x_name = pd.DataFrame(data={'word': model_x_name, 'target_order': range(0, len(model_x_name), 1)})
        df_test_x_name = pd.DataFrame(data={'word': test_x_name, 'original_order': range(0, len(test_x_name), 1)})
        # Log.debug('**** Target Order: ' + str(model_x_name))
        # Log.debug('**** Original order: ' + str(test_x_name))
        # Left join to ensure the order follows target order and target symbols
        df_x_name = df_x_name.merge(df_test_x_name, how='left')
        # Log.debug('**** Merged Order: ' + str(df_x_name))
        # Then order by original order
        df_x_name = df_x_name.sort_values(by=['target_order'], ascending=True)
        # Then the order we need to reorder is the target_order column
        reorder = np.array(df_x_name['original_order'])
        self.res_final.update_bool(res_bool=UnitTest.assert_true(
            observed = reorder.tolist(),
            expected = self.REORDER_FEATURE_NAMES_WITH_UNK.tolist(),
            test_comment = 'Test reorder of feature names ' + str(reorder)
        ))

        test_x_transpose = test_x.transpose()
        Log.debugdebug(test_x_transpose)

        reordered_test_x = np.zeros(shape=test_x_transpose.shape)
        Log.debugdebug(reordered_test_x)

        for i in range(0, reordered_test_x.shape[0], 1):
            reordered_test_x[i] = test_x_transpose[reorder[i]]

        reordered_test_x = reordered_test_x.transpose()
        Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Reordered test x = ' + str(reordered_test_x)
        )

        x_classes_expected = self.y
        # Just the top predicted ones
        all_y_observed_top = []
        all_y_observed = []
        mse = 0
        count_all = reordered_test_x.shape[0]

        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Predict classes for x:\n\r' + str(reordered_test_x)
        )
        prf_start = prf.Profiling.start()

        for i in range(reordered_test_x.shape[0]):
            v = npUtil.NumpyUtil.convert_dimension(arr=reordered_test_x[i], to_dim=2)
            Log.debugdebug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Testing x: ' + str(v)
            )
            if self.model_name == ModelHelper.MODEL_NAME_HYPERSPHERE_METRICSPACE:
                predict_result = model_obj.predict_class(
                    x           = v,
                    include_match_details = include_match_details,
                    top         = top
                )
            else:
                predict_result = model_obj.predict_class(
                    x           = v
                )
            y_observed = predict_result.predicted_classes
            all_y_observed_top.append(y_observed[0])
            all_y_observed.append(y_observed)
            top_class_distance = predict_result.top_class_distance
            match_details = predict_result.match_details

            Log.debugdebug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Point v ' + str(v) + ', predicted ' + str(y_observed)
                + ', Top Class Distance: ' + str(top_class_distance)
                + ', Match Details:\n\r' + str(match_details)
            )

            if self.model_name == ModelHelper.MODEL_NAME_HYPERSPHERE_METRICSPACE:
                metric = top_class_distance
                mse += metric ** 2

        prf_dur = prf.Profiling.get_time_dif(prf_start, prf.Profiling.stop())
        Log.important(
            str(self.__class__) + str(getframeinfo(currentframe()).lineno)
            + ' PROFILING ' + str(count_all) + ' calculations: ' + str(round(1000*prf_dur,0))
            + ', or ' + str(round(1000*prf_dur/count_all,2)) + ' milliseconds per calculation'
        )

        # Compare with expected
        compare_top_x = {}

        for t in range(1, top + 1, 1):
            # True or '1' means not correct or error
            compare_top_x[t] = np.array([True] * len(all_y_observed))
            for i in range(len(all_y_observed)):
                matches_i = all_y_observed[i]
                if x_classes_expected[i] in matches_i[0:t]:
                    # False of '0' means no error
                    compare_top_x[t][i] = False
                    self.res_final.count_ok += 1*(t==1)
                else:
                    self.res_final.count_fail += 1*(t==1)
            Log.info(compare_top_x[t])
            Log.info(
                'Total Errors (compare top #' + str(t) + ') = ' + str(np.sum(compare_top_x[t] * 1))
            )

        Log.info('mse = ' + str(mse))

        if self.model_name == ModelHelper.MODEL_NAME_HYPERSPHERE_METRICSPACE:
            predict_result = model_obj.predict_classes(
                    x           = reordered_test_x,
                    include_match_details = include_match_details,
                    top = top
                )
            Log.info('Predicted Classes:\n\r' + str(predict_result.predicted_classes))
            Log.info('Top class distance:\n\r' + str(predict_result.top_class_distance))
            Log.info('Match Details:\n\r' + str(predict_result.match_details))
            Log.info('MSE = ' + str(predict_result.mse))

        model_obj.join()

        #
        # Test using PredictClass
        #
        from nwae.lang.LangFeatures import LangFeatures
        from nwae.ml.PredictClass import PredictClass
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
            word_freq_model        = word_freq_model_mapped,
            do_spelling_correction = False,
            do_profiling           = True
        )

        for i in range(len(self.DATA_TEXTS)):
            label = self.DATA_Y[i]
            text_arr = self.DATA_TEXTS[i]
            text = ' '.join(text_arr)
            # Return all results in the top 5
            res = predict.predict_class_text_features(
                inputtext                  = text,
                match_pct_within_top_score = 0,
                include_match_details      = True,
                top                        = 5,
            )
            self.res_final.update_bool(res_bool=UnitTest.assert_true(
                observed = res.predict_result.predicted_classes[0],
                expected = label,
                test_comment = 'Test "' + str(text) + '" label ' + str(label)
            ))
            Log.debug(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ': ' + str(i) + '. Match Details word freq model "' + str(predict.word_freq_model)
                + '" ' + str(res.predict_result.match_details)
            )
            predict.word_freq_model = WordFreqDocMatrix.map_to_feature_vect_word_freq_measure(freq_measure=WordFreqDocMatrix.BY_SIGMOID_FREQ)
            res = predict.predict_class_text_features(
                inputtext                  = text,
                match_pct_within_top_score = 0,
                include_match_details      = True,
                top                        = 5,
            )
            Log.debug(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ': ' + str(i) + '. Match Details word freq model "' + str(predict.word_freq_model)
                + '" ' + str(res.predict_result.match_details)
            )

        # Kill any background jobs
        predict.stop_model_thread()

        return

    def run_unit_test(self):
        for wfm in [
            WordFreqDocMatrix.BY_FREQ_NORM,
            WordFreqDocMatrix.BY_SIGMOID_FREQ_NORM,
            # WordFreqDocMatrix.BY_LOG_FREQ_NORM,
        ]:
            Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Start testing using word freq model "' + str(wfm) + '"'
            )
            self.unit_test_train(word_freq_model=wfm)
            self.unit_test_predict_classes(
                word_freq_model = wfm,
                include_match_details = True,
                top = 2
            )
        return self.res_final


if __name__ == '__main__':
    config = Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = Config,
        default_config_file = Config.CONFIG_FILE_DEFAULT
    )
    ut_params = UnitTestParams(
        dirpath_wordlist     = config.get_config(param=Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist     = config.get_config(param=Config.PARAM_NLP_POSTFIX_WORDLIST),
        dirpath_app_wordlist = config.get_config(param=Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_app_wordlist = config.get_config(param=Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        dirpath_synonymlist  = config.get_config(param=Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist  = config.get_config(param=Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        dirpath_model        = config.get_config(param=Config.PARAM_MODEL_DIR)
    )
    print('Unit Test Params: ' + str(ut_params.to_string()))

    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1

    for model_name in [
            ModelHelper.MODEL_NAME_HYPERSPHERE_METRICSPACE,
            #modelHelper.ModelHelper.MODEL_NAME_KERAS,
    ]:
        obj = UnitTestMetricSpaceModel(
            ut_params         = ut_params,
            model_name        = model_name
        )
        res = obj.run_unit_test()
        print('***** PASS ' + str(res.count_ok) + ', FAIL ' + str(res.count_fail))

