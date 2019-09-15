# -*- coding: utf-8 -*-

import pandas as pd
import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo
import nwae.utils.Profiling as pf
import nwae.Config as cf
import nwae.lib.math.PredictClass as predictclass
import nwae.lib.math.NumpyUtil as nputil
import nwae.lib.math.ml.ModelInterface as modelif


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
    KEY_STATS_DF_SCORES = 'df_scores'

    def __init__(
            self,
            config
    ):
        self.config = config

        self.include_detailed_accuracy_stats = config.get_config(param=cf.Config.PARAM_MODEL_BACKTEST_DETAILED_STATS)
        self.model_name = config.get_config(param=cf.Config.PARAM_MODEL_NAME)
        self.model_lang = config.get_config(param=cf.Config.PARAM_MODEL_LANG)
        self.model_dir = config.get_config(param=cf.Config.PARAM_MODEL_DIR)
        self.model_identifier = config.get_config(param=cf.Config.PARAM_MODEL_IDENTIFIER)
        self.do_profiling = config.get_config(param=cf.Config.PARAM_DO_PROFILING)

        lg.Log.info('Include detailed stats = ' + str(self.include_detailed_accuracy_stats) + '.')
        lg.Log.info('Model Name "' + str(self.model_name) + '"')
        lg.Log.info('Model Lang "' + str(self.model_lang) + '"')
        lg.Log.info('Model Directory "' + str(self.model_dir) + '"')
        lg.Log.info('Model Identifier "' + str(self.model_identifier) + '"')
        lg.Log.info('Do profiling = ' + str(self.do_profiling) + '.')

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
                do_profiling         = self.config.get_config(param=cf.Config.PARAM_DO_PROFILING)
            )
            self.model = self.predictor.model
        except Exception as ex:
            lg.Log.warning(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Could not load PredictClass: ' + str(ex)
            )
        return

    #
    # TODO: Include data that is suppose to fail (e.g. run LeBot through our historical chats to get that data)
    # TODO: This way we can measure both what is suppose to pass and what is suppose to fail
    #
    def test_model_against_training_data(
            self,
            include_detailed_accuracy_stats = False
    ):
        start_get_td_time = pf.Profiling.start()
        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '.   Start Load Training Data: ' + str(start_get_td_time)
        )

        # Get training data to improve LeBot intent/command detection
        self.model.load_training_data_from_storage()
        td = self.model.training_data
        lg.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': TD x_name, shape=' + str(td.get_x_name().shape) + ': ' +  str(td.get_x_name())
            + '\n\rTD shape=' + str(td.get_x().shape)
            + '\n\rTD[0:10] =' + str(td.get_x()[0:10])
        )

        stop_get_td_time = pf.Profiling.stop()
        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '.   Stop Load Training Data: '
            + str(pf.Profiling.get_time_dif_str(start_get_td_time, stop_get_td_time)))

        start_test_time = pf.Profiling.start()
        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '.   Start Testing of Training Data from DB Time : ' + str(start_get_td_time)
        )
        #
        # Read from chatbot training files to compare with LeBot performance
        #
        test_stats = {
            ModelBackTest.KEY_STATS_START_TEST_TIME: start_test_time,
            ModelBackTest.KEY_STATS_RESULT_TOTAL: 0,
            ModelBackTest.KEY_STATS_RESULT_CORRECT: 0,
            ModelBackTest.KEY_STATS_RESULT_TOP: {},
            # How many % in top X
            ModelBackTest.KEY_STATS_RESULT_ACCURACY: {},
            ModelBackTest.KEY_STATS_RESULT_WRONG: 0,
            ModelBackTest.KEY_STATS_DF_SCORES: pd.DataFrame(columns=['Score', 'ConfLevel', 'Correct'])
        }
        for top_i in range(ModelBackTest.TEST_TOP_X):
            test_stats[ModelBackTest.KEY_STATS_RESULT_TOP][top_i] = 0
            test_stats[ModelBackTest.KEY_STATS_RESULT_ACCURACY][top_i] = 0

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
                test_stats = test_stats,
                include_detailed_accuracy_stats = include_detailed_accuracy_stats
            )

        stop_test_time = pf.Profiling.stop()
        lg.Log.log('.   Stop Testing of Training Data from DB Time : '
                   + str(pf.Profiling.get_time_dif_str(start_test_time, stop_test_time)))

        lg.Log.log(
            str(test_stats[ModelBackTest.KEY_STATS_RESULT_WRONG]) + ' wrong results from '
            + str(test_stats[ModelBackTest.KEY_STATS_RESULT_WRONG]) + ' total tests.'
        )
        lg.Log.log("Score Quantile (0): " + str(test_stats[ModelBackTest.KEY_STATS_DF_SCORES]['Score'].quantile(0)))
        lg.Log.log("Score Quantile (5%): " + str(test_stats[ModelBackTest.KEY_STATS_DF_SCORES]['Score'].quantile(0.05)))
        lg.Log.log("Score Quantile (25%): " + str(test_stats[ModelBackTest.KEY_STATS_DF_SCORES]['Score'].quantile(0.25)))
        lg.Log.log("Score Quantile (50%): " + str(test_stats[ModelBackTest.KEY_STATS_DF_SCORES]['Score'].quantile(0.5)))
        lg.Log.log("Score Quantile (75%): " + str(test_stats[ModelBackTest.KEY_STATS_DF_SCORES]['Score'].quantile(0.75)))
        lg.Log.log("Score Quantile (95%): " + str(test_stats[ModelBackTest.KEY_STATS_DF_SCORES]['Score'].quantile(0.95)))

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
        lg.Log.debugdebug(
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
            test_stats,
            include_detailed_accuracy_stats
    ):
        com_idx = 0
        com_class = '-'
        # com_match = None
        com_score = 0
        com_conflevel = 0
        correct = False
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

        test_stats[ModelBackTest.KEY_STATS_RESULT_TOTAL] =\
            test_stats[ModelBackTest.KEY_STATS_RESULT_TOTAL] + 1
        time_elapsed = pf.Profiling.get_time_dif(test_stats[ModelBackTest.KEY_STATS_START_TEST_TIME], pf.Profiling.stop())
        rps = round(test_stats[ModelBackTest.KEY_STATS_RESULT_TOTAL] / time_elapsed, 1)
        # Time per request in milliseconds
        tpr = round(1000 / rps, 1)

        test_stats[ModelBackTest.KEY_STATS_DF_SCORES] = test_stats[ModelBackTest.KEY_STATS_DF_SCORES].append({
            'Score': com_score, 'ConfLevel': com_conflevel, 'Correct': correct, 'TopIndex': com_idx
        },
            ignore_index=True)
        lg.Log.debugdebug(test_stats[ModelBackTest.KEY_STATS_DF_SCORES])
        if not correct:
            lg.Log.log('Failed Test y: ' + str(y_expected) + ' (' + str(x_features) + ') === ' + str(com_class))
            lg.Log.log(df_match_details)
            lg.Log.log('   Result: ' + str(com_class))
        else:
            test_stats[ModelBackTest.KEY_STATS_RESULT_CORRECT] = test_stats[ModelBackTest.KEY_STATS_RESULT_CORRECT] + 1
            result_accuracy_in_top_x =\
                round(100 * test_stats[ModelBackTest.KEY_STATS_RESULT_CORRECT]
                      / test_stats[ModelBackTest.KEY_STATS_RESULT_TOTAL], 2)
            str_result_accuracy = str(result_accuracy_in_top_x) + '%'

            if include_detailed_accuracy_stats:
                # Update the result at the index it appeared
                test_stats[ModelBackTest.KEY_STATS_RESULT_TOP][com_idx] =\
                    test_stats[ModelBackTest.KEY_STATS_RESULT_TOP][com_idx] + 1
                test_stats[ModelBackTest.KEY_STATS_RESULT_ACCURACY][com_idx] =\
                    round(100 * test_stats[ModelBackTest.KEY_STATS_RESULT_TOP][com_idx]
                          / test_stats[ModelBackTest.KEY_STATS_RESULT_TOTAL], 1)

                # Only show 3
                for iii in range(min(3, ModelBackTest.TEST_TOP_X)):
                    str_result_accuracy = \
                        str_result_accuracy \
                        + ', p' + str(iii + 1) + '='\
                        + str(test_stats[ModelBackTest.KEY_STATS_RESULT_ACCURACY][iii]) + '%'

            lg.Log.log('Passed ' + str(test_stats[ModelBackTest.KEY_STATS_RESULT_CORRECT])
                       + ' (' + str_result_accuracy
                       + ', ' + str(rps) + ' rps, ' + str(tpr) + 'ms per/req'
                       + '): ' + str(y_expected) + ':' + str(x_features)
                       + '). Score=' + str(com_score) + ', ConfLevel=' + str(com_conflevel)
                       + ', Index=' + str(com_idx + 1))
            if com_idx != 0:
                lg.Log.log('   Result not 1st (in position #' + str(com_idx) + ')')

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
                lg.Log.log('Start Time: ' + str(start))

                self.test_model_against_training_data(
                    include_detailed_accuracy_stats = self.include_detailed_accuracy_stats
                )

                stop = pf.Profiling.stop()
                lg.Log.log('Stop Time : ' + str(stop))
                lg.Log.log(pf.Profiling.get_time_dif_str(start, stop))

                if test_training_data:
                    break

            elif user_choice == 'e':
                break
            else:
                print('No such choice [' + user_choice + ']')


if __name__ == '__main__':
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config
    )

    mt = ModelBackTest(
        config     = config
    )
    mt.run()
