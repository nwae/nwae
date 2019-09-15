# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import pandas as pd
import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo
import nwae.utils.Profiling as pf
import nwae.Config as cf
import nwae.lib.math.PredictClass as predictclass
import nwae.lib.math.NumpyUtil as nputil
import nwae.lib.math.ml.ModelInterface as modelif


#
# This only tests intent engine, does not test word segmentation
#
class ModelBackTest:

    TEST_TOP_X = 5

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
            ignore_db = False,
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
        result_total = 0
        result_correct = 0
        result_accuracy_in_top_x = 0
        result_top = {}
        # How many % in top X
        result_accuracy = {}
        for top_i in range(ModelBackTest.TEST_TOP_X):
            result_top[top_i] = 0
            result_accuracy[top_i] = 0
        result_wrong = 0
        df_scores = pd.DataFrame(columns=['Score', 'ConfLevel', 'Correct'])

        x_name = td.get_x_name()
        x = td.get_x()
        y = td.get_y()
        for i in range(0, x.shape[0], 1):
            # if i<=410: continue
            y_expected = y[i]
            v = nputil.NumpyUtil.convert_dimension(arr=x[i], to_dim=2)
            x_features = x_name[v[0]>0]

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

            result_total = result_total + 1
            time_elapsed = pf.Profiling.get_time_dif(start_test_time, pf.Profiling.stop())
            rps = round(result_total / time_elapsed, 1)
            # Time per request in milliseconds
            tpr = round(1000 / rps, 1)

            df_scores = df_scores.append({
                'Score': com_score, 'ConfLevel': com_conflevel, 'Correct': correct, 'TopIndex': com_idx
            },
                ignore_index=True)
            lg.Log.debugdebug(df_scores)
            if not correct:
                result_wrong = result_wrong + 1
                lg.Log.log('Failed Test y: ' + str(y_expected) + ' (' + str(x_features) + ') === ' + str(com_class))
                lg.Log.log(df_match_details)
                lg.Log.log('   Result: ' + str(com_class))
            else:
                result_correct = result_correct + 1
                result_accuracy_in_top_x = round(100 * result_correct / result_total, 2)
                str_result_accuracy = str(result_accuracy_in_top_x) + '%'

                if include_detailed_accuracy_stats:
                    result_top[com_idx] = result_top[com_idx] + 1
                    result_accuracy[com_idx] = round(100 * result_top[com_idx] / result_total, 1)
                    for iii in range(min(3,ModelBackTest.TEST_TOP_X)):
                        str_result_accuracy =\
                            str_result_accuracy\
                            + ', p' + str(iii+1) + '=' + str(result_accuracy[iii]) + '%'

                if result_correct % 100 == 0:
                    lg.Log.log('Passed ' + str(result_correct) + '..')
                lg.Log.log('Passed ' + str(result_correct)
                           + ' (' + str_result_accuracy
                           + ', ' + str(rps) + ' rps, ' + str(tpr) + 'ms per/req'
                           + '): ' + str(y_expected) + ':' + str(x_features)
                           + '). Score=' + str(com_score) + ', ConfLevel=' + str(com_conflevel)
                           + ', Index=' + str(com_idx+1))
                if com_idx != 0:
                    lg.Log.log('   Result not 1st (in position #' + str(com_idx) + ')')

            # if i>1: break

        stop_test_time = pf.Profiling.stop()
        lg.Log.log('.   Stop Testing of Training Data from DB Time : '
                   + str(pf.Profiling.get_time_dif_str(start_test_time, stop_test_time)))

        lg.Log.log(str(result_wrong) + ' wrong results from ' + str(result_total) + ' total tests.')
        lg.Log.log("Score Quantile (0): " + str(df_scores['Score'].quantile(0)))
        lg.Log.log("Score Quantile (5%): " + str(df_scores['Score'].quantile(0.05)))
        lg.Log.log("Score Quantile (25%): " + str(df_scores['Score'].quantile(0.25)))
        lg.Log.log("Score Quantile (50%): " + str(df_scores['Score'].quantile(0.5)))
        lg.Log.log("Score Quantile (75%): " + str(df_scores['Score'].quantile(0.75)))
        lg.Log.log("Score Quantile (95%): " + str(df_scores['Score'].quantile(0.95)))

        return

    def run(
            self,
            ignore_db = False,
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
                    ignore_db = ignore_db,
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
