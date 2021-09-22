# -*- coding: utf-8 -*-

import pandas as pd
from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
import nwae.utils.Profiling as pf
import nwae.ml.config.Config as cf
import nwae.ml.PredictClass as predictclass
import nwae.math.NumpyUtil as nputil
import nwae.ml.ModelInterface as modelif


#
# Model Engine back test on training data
#
class ModelBackTest:

    TEST_TOP_X = 5

    KEY_STATS_START_TEST_TIME = 'start_test_time'
    KEY_STATS_RESULT_TOTAL = 'result_total'
    KEY_STATS_RESULT_CORRECT = 'result_correct'
    KEY_STATS_RESULT_TOP = 'result_top'
    # How many % in top X
    KEY_STATS_RESULT_ACCURACY = 'result_accuracy'
    KEY_STATS_RESULT_WRONG = 'result_wrong'
    # Ratio of score between top 2 and top 1, when the correct answer is at position #1
    KEY_STATS_WHEN_CORRECT_ANSWER_AT_TOP1_RATIO_SCORE_TOP_2_AND_TOP_1 = 'ratio_at_top1_pos_1_2'
    KEY_STATS_WHEN_CORRECT_ANSWER_NOT_AT_TOP1_RATIO_SCORE_TO_TOP_1 = 'ratio_not_at_top1_to_pos_1'
    KEY_STATS_DF_SCORES = 'df_scores'

    def __init__(
            self,
            config,
            word_freq_model = None,
    ):
        self.config = config
        self.word_freq_model = word_freq_model

        if self.config is None:
            Log.warning(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Not initializing predictor, since config is None. Do that on your own.'
            )
            return
        else:
            self.init_predictor()

    def init_predictor(self):
        self.include_detailed_accuracy_stats = config.get_config(param=cf.Config.PARAM_MODEL_BACKTEST_DETAILED_STATS)
        self.model_name = config.get_config(param=cf.Config.PARAM_MODEL_NAME)
        self.model_lang = config.get_config(param=cf.Config.PARAM_MODEL_LANG)
        self.model_dir = config.get_config(param=cf.Config.PARAM_MODEL_DIR)
        self.model_identifier = config.get_config(param=cf.Config.PARAM_MODEL_IDENTIFIER)
        self.do_profiling = config.get_config(param=cf.Config.PARAM_DO_PROFILING)

        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Include detailed stats = ' + str(self.include_detailed_accuracy_stats)
            + ', model name "' + str(self.model_name)
            + '", model lang "' + str(self.model_lang)
            + '", model directory "' + str(self.model_dir)
            + '", model identifier "' + str(self.model_identifier)
            + '", word freq model "' + str(self.word_freq_model)
            + '", do profiling = ' + str(self.do_profiling)
        )

        try:
            self.predictor = predictclass.PredictClass(
                model_name           = self.model_name,
                identifier_string    = self.model_identifier,
                dir_path_model       = self.model_dir,
                lang                 = self.model_lang,
                dirpath_synonymlist  = self.config.get_config(param=cf.Config.PARAM_NLP_DIR_SYNONYMLIST),
                postfix_synonymlist  = self.config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
                dir_wordlist         = self.config.get_config(param=cf.Config.PARAM_NLP_DIR_WORDLIST),
                postfix_wordlist     = self.config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_WORDLIST),
                dir_wordlist_app     = self.config.get_config(param=cf.Config.PARAM_NLP_DIR_APP_WORDLIST),
                postfix_wordlist_app = self.config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
                word_freq_model      = self.word_freq_model,
                do_profiling         = self.config.get_config(param=cf.Config.PARAM_DO_PROFILING)
            )
            self.model = self.predictor.model
            self.test_stats = None
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Could not load PredictClass: ' + str(ex)
            Log.error(errmsg)
            # Don't raise exception
            # raise Exception(errmsg)
        return

    def reset_test_stats(self):
        self.test_stats = {
            ModelBackTest.KEY_STATS_START_TEST_TIME: 0,
            ModelBackTest.KEY_STATS_RESULT_TOTAL: 0,
            ModelBackTest.KEY_STATS_RESULT_CORRECT: 0,
            ModelBackTest.KEY_STATS_RESULT_TOP: {},
            # How many % in top X
            ModelBackTest.KEY_STATS_RESULT_ACCURACY: {},
            ModelBackTest.KEY_STATS_RESULT_WRONG: 0,
            # Ratio of score between top 2 and top 1, when the correct answer is at position #1
            ModelBackTest.KEY_STATS_WHEN_CORRECT_ANSWER_AT_TOP1_RATIO_SCORE_TOP_2_AND_TOP_1: [],
            ModelBackTest.KEY_STATS_WHEN_CORRECT_ANSWER_NOT_AT_TOP1_RATIO_SCORE_TO_TOP_1: [],
            ModelBackTest.KEY_STATS_DF_SCORES: pd.DataFrame(columns=['Score', 'ConfLevel', 'Correct'])
        }
        for top_i in range(ModelBackTest.TEST_TOP_X):
            self.test_stats[ModelBackTest.KEY_STATS_RESULT_TOP][top_i] = 0
            self.test_stats[ModelBackTest.KEY_STATS_RESULT_ACCURACY][top_i] = 0

    #
    # TODO: Include data that is suppose to fail (e.g. run LeBot through our historical chats to get that data)
    # TODO: This way we can measure both what is suppose to pass and what is suppose to fail
    #
    def test_model_against_training_data(
            self,
            include_detailed_accuracy_stats = False
    ):
        start_get_td_time = pf.Profiling.start()
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '.   Start Load Training Data: ' + str(start_get_td_time)
        )

        # Get training data to improve LeBot intent/command detection
        self.model.load_training_data_from_storage()
        td = self.model.training_data
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': TD x_name, shape=' + str(td.get_x_name().shape) + ': ' +  str(td.get_x_name())
            + '\n\rTD shape=' + str(td.get_x().shape)
            + '\n\rTD[0:10] =' + str(td.get_x()[0:10])
        )

        stop_get_td_time = pf.Profiling.stop()
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '.   Stop Load Training Data: '
            + str(pf.Profiling.get_time_dif_str(start_get_td_time, stop_get_td_time)))

        start_test_time = pf.Profiling.start()
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '.   Start Testing of Training Data from DB Time : ' + str(start_get_td_time)
        )
        #
        # Read from chatbot training files to compare with LeBot performance
        #
        self.reset_test_stats()
        self.test_stats[ModelBackTest.KEY_STATS_START_TEST_TIME] = start_test_time

        x_name = td.get_x_name()
        x = td.get_x()
        y = td.get_y()
        for i in range(0, x.shape[0], 1):
            y_expected = y[i]
            v = nputil.NumpyUtil.convert_dimension(arr=x[i], to_dim=2)
            x_features = x_name[v[0]>0]

            df_match_details = self.predict_top_x(
                v = v,
                y_expected = y_expected,
                x_features = x_features
            )

            self.update_test_stats(
                df_match_details = df_match_details,
                y_expected = y_expected,
                x_features = x_features,
                include_detailed_accuracy_stats = include_detailed_accuracy_stats
            )

        stop_test_time = pf.Profiling.stop()
        Log.important('.   Stop Testing of Training Data from DB Time : '
                   + str(pf.Profiling.get_time_dif_str(start_test_time, stop_test_time)))

        Log.important(
            str(self.test_stats[ModelBackTest.KEY_STATS_RESULT_WRONG]) + ' wrong results from '
            + str(self.test_stats[ModelBackTest.KEY_STATS_RESULT_WRONG]) + ' total tests.'
        )
        for q in (0.0, 0.05, 0.1, 0.25, 0.50, 0.75, 0.9, 0.95):
            Log.important(
                'Score Quantile (' + str(q) + '): '
                + str(self.test_stats[ModelBackTest.KEY_STATS_DF_SCORES]['Score'].quantile(q))
            )

        return

    def predict_top_x(
            self,
            # 2D numpy ndarray, shape (1,n)
            v,
            y_expected,
            x_features
    ):
        predict_result = self.model.predict_class(
            x   = v,
            top = ModelBackTest.TEST_TOP_X,
            include_match_details = True
        )
        df_match_details = predict_result.match_details
        Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': y expected: ' + str(y_expected) + ', x features: ' + str(x_features)
            + '\n\rMatch Details:\n\r' + str(df_match_details)
        )
        return df_match_details

    def update_test_stats(
            self,
            df_match_details,
            y_expected,
            x_features,
            include_detailed_accuracy_stats
    ):
        com_idx = 0
        com_class = '-'
        # com_match = None
        com_score = 0
        com_conflevel = 0
        correct = False
        ratio_score_answer_at_top1 = None
        ratio_score_answer_not_at_top1 = None

        if df_match_details is not None:
            # We define correct by having the targeted intent in the top closest
            com_results_list = list(df_match_details[modelif.ModelInterface.TERM_CLASS])
            correct = (y_expected in com_results_list)

            if correct:
                com_idx = df_match_details.index[
                    df_match_details[modelif.ModelInterface.TERM_CLASS] == y_expected
                    ][0]
                com_class = df_match_details[modelif.ModelInterface.TERM_CLASS].loc[com_idx]
                com_score = df_match_details[modelif.ModelInterface.TERM_SCORE].loc[com_idx]
                com_conflevel = df_match_details[modelif.ModelInterface.TERM_CONFIDENCE].loc[com_idx]

                if df_match_details.shape[0] >= 2:
                    score_expected_answer = df_match_details[modelif.ModelInterface.TERM_SCORE].loc[com_idx]
                    # If answer is at position #1
                    if com_idx == 0:
                        score_pos_2 = df_match_details[modelif.ModelInterface.TERM_SCORE].loc[1]
                        ratio_score_answer_at_top1 = round(100*score_pos_2/score_expected_answer, 2)
                    else:
                        score_pos_1 = df_match_details[modelif.ModelInterface.TERM_SCORE].loc[0]
                        ratio_score_answer_not_at_top1 = round(100*score_expected_answer/score_pos_1, 2)

        self.test_stats[ModelBackTest.KEY_STATS_RESULT_TOTAL] += 1
        time_elapsed = pf.Profiling.get_time_dif(
            self.test_stats[ModelBackTest.KEY_STATS_START_TEST_TIME], pf.Profiling.stop()
        )
        rps = round(self.test_stats[ModelBackTest.KEY_STATS_RESULT_TOTAL] / time_elapsed, 1)
        # Time per request in milliseconds
        tpr = round(1000 / rps, 1)

        self.test_stats[ModelBackTest.KEY_STATS_DF_SCORES] =\
            self.test_stats[ModelBackTest.KEY_STATS_DF_SCORES].append(
                {
                    'Score': com_score, 'ConfLevel': com_conflevel, 'Correct': correct, 'TopIndex': com_idx
                },
            ignore_index=True)
        # Log.debugdebug(self.test_stats[ModelBackTest.KEY_STATS_DF_SCORES])
        if not correct:
            self.test_stats[ModelBackTest.KEY_STATS_RESULT_WRONG] += 1
            Log.error(
                str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Failed Test, expected y=' + str(y_expected)
                + ' (' + str(x_features) + ') === ' + str(com_class)
                + '\n\rMatch Details: ' + str(df_match_details)
                + '\n\rResult: ' + str(com_class))
        else:
            self.test_stats[ModelBackTest.KEY_STATS_RESULT_CORRECT] =\
                self.test_stats[ModelBackTest.KEY_STATS_RESULT_CORRECT] + 1
            result_accuracy_in_top_x =\
                round(100 * self.test_stats[ModelBackTest.KEY_STATS_RESULT_CORRECT]
                      / self.test_stats[ModelBackTest.KEY_STATS_RESULT_TOTAL], 2)
            str_result_accuracy = str(result_accuracy_in_top_x) + '%'

            if ratio_score_answer_at_top1 is not None:
                self.test_stats[ModelBackTest.KEY_STATS_WHEN_CORRECT_ANSWER_AT_TOP1_RATIO_SCORE_TOP_2_AND_TOP_1].append(
                    ratio_score_answer_at_top1
                )
            if ratio_score_answer_not_at_top1 is not None:
                self.test_stats[ModelBackTest.KEY_STATS_WHEN_CORRECT_ANSWER_NOT_AT_TOP1_RATIO_SCORE_TO_TOP_1].append(
                    ratio_score_answer_not_at_top1
                )

            if include_detailed_accuracy_stats:
                # Update the result at the index it appeared
                self.test_stats[ModelBackTest.KEY_STATS_RESULT_TOP][com_idx] += 1
                self.test_stats[ModelBackTest.KEY_STATS_RESULT_ACCURACY][com_idx] =\
                    round(100 * self.test_stats[ModelBackTest.KEY_STATS_RESULT_TOP][com_idx]
                          / self.test_stats[ModelBackTest.KEY_STATS_RESULT_TOTAL], 1)

                # Only show 3
                for iii in range(min(3, ModelBackTest.TEST_TOP_X)):
                    str_result_accuracy = \
                        str_result_accuracy \
                        + ', p' + str(iii + 1) + '='\
                        + str(self.test_stats[ModelBackTest.KEY_STATS_RESULT_ACCURACY][iii]) + '%'

            Log.important(
                'Passed ' + str(self.test_stats[ModelBackTest.KEY_STATS_RESULT_CORRECT])
                + ' (' + str_result_accuracy + ', ' + str(rps) + ' rps, ' + str(tpr) + 'ms per/req'
                + '): ' + str(y_expected) + ':' + str(x_features)
                + '). Score=' + str(com_score) + ', ConfLevel=' + str(com_conflevel)
                + ', Index=' + str(com_idx + 1)
                + ', Ratio=' + str(ratio_score_answer_at_top1) + '/' + str(ratio_score_answer_not_at_top1)
            )
            if com_idx != 0:
                Log.log('   Result not 1st (in position #' + str(com_idx) + ')')

    def run(
            self,
            test_training_data = False
    ):
        while True:
            user_choice = None
            if not test_training_data:
                print('Lang=' + self.model_lang + ', Model Identifier=' + self.model_identifier + ': Choices')
                print('1: Test Model Against Training Data')
                print('e: Exit')
                user_choice = input('Enter Choice: ')

            if user_choice == '1' or test_training_data:
                start = pf.Profiling.start()
                Log.log('Start Time: ' + str(start))

                self.test_model_against_training_data(
                    include_detailed_accuracy_stats = self.include_detailed_accuracy_stats
                )

                stop = pf.Profiling.stop()
                Log.log('Stop Time : ' + str(stop))
                Log.log(pf.Profiling.get_time_dif_str(start, stop))

                if test_training_data:
                    break

            elif user_choice == 'e':
                break
            else:
                print('No such choice [' + user_choice + ']')


if __name__ == '__main__':
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class       = cf.Config,
        default_config_file = cf.Config.CONFIG_FILE_DEFAULT
    )

    mt = ModelBackTest(
        config     = config
    )
    mt.run()
