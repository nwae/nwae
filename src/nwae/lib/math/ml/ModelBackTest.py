# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import sys
import numpy as np
import pandas as pd
import nwae.utils.Log as lg
import nwae.utils.Log as nwaelog
from inspect import currentframe, getframeinfo
import nwae.utils.Profiling as pf
import nwae.Config as cf


#
# This only tests intent engine, does not test word segmentation
#
class ModelBackTest:

    TEST_TOP_X = 5

    def __init__(
            self,
            config,
            account_id,
            bot_id,
            lang,
            bot_key,
            do_text_segmentation,
            do_profiling,
            minimal,
            include_detailed_accuracy_stats = False
    ):
        self.config = config
        self.account_id = account_id
        self.bot_id = bot_id
        self.lang = lang
        self.bot_key = bot_key
        self.do_text_segmentation = do_text_segmentation
        self.do_profiling = do_profiling
        self.minimal = minimal
        self.include_detailed_accuracy_stats = include_detailed_accuracy_stats

        self.dir_testdata = self.config.DIR_INTENTTEST_TESTDATA

        self.bot_config = cfbot.BotConfigFile.get_botconfig_singleton(
            bot_id = self.bot_id,
            dir_bot_config = self.config.DIR_BOT_SPECIFIC_CONFIG_FILES,
            db_profile     = self.config.DB_PROFILE
        )
        self.bot_config.init_from_app_config_file()

        dir_path_model = self.config.DIR_INTENT_MODELS
        if self.bot_config.MATH_ENGINE == cfbot.BotConfigFile.CONST_MATH_ENGINE_BUILT_IN:
            dir_path_model = self.config.DIR_RFV_INTENTS

        try:
            import ie.lib.chat.bot.IntentModel as intentModel
            self.bot = intentModel.IntentModel.get_bot(
                model_name        = self.bot_config.MATH_MODEL,
                dir_path_model    = dir_path_model,
                account_id        = self.account_id,
                bot_id            = self.bot_id,
                lang              = self.lang,
                bot_key           = self.bot_key,
                minimal           = self.minimal,
                dir_synonymlist      = self.config.DIR_SYNONYMLIST,
                postfix_synonymlist  = self.config.POSTFIX_SYNONYMLIST,
                dir_wordlist         = self.config.DIR_WORDLIST,
                postfix_wordlist     = self.config.POSTFIX_WORDLIST,
                dir_wordlist_app     = self.config.DIR_APP_WORDLIST,
                postfix_wordlist_app = self.config.POSTFIX_APP_WORDLIST,
                use_db               = self.config.USE_DB,
                db_profile           = self.config.DB_PROFILE,
                do_profiling         = self.do_profiling
            )
        except Exception as ex:
            lg.Log.warning(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Could not load IntentModel: ' + str(ex)
            )
            self.bot = intwrapper.IntentWrapper(
                use_db     = self.config.USE_DB,
                db_profile = self.config.DB_PROFILE,
                account_id = self.account_id,
                bot_id     = self.bot_id,
                lang       = self.lang,
                bot_key    = self.bot_key,
                minimal    = self.minimal,
                dir_rfv_commands     = self.config.DIR_RFV_INTENTS,
                dir_synonymlist      = self.config.DIR_SYNONYMLIST,
                dir_wordlist         = self.config.DIR_WORDLIST,
                postfix_wordlist     = self.config.POSTFIX_WORDLIST,
                dir_wordlist_app     = self.config.DIR_APP_WORDLIST,
                postfix_wordlist_app = self.config.POSTFIX_APP_WORDLIST,
                do_profiling         = self.do_profiling
            )
            self.bot.init()

        return

    #
    # TODO: Include data that is suppose to fail (e.g. run LeBot through our historical chats to get that data)
    # TODO: This way we can measure both what is suppose to pass and what is suppose to fail
    #
    def test_lebot_against_training_data(
            self,
            ignore_db = False,
            include_detailed_accuracy_stats = False
    ):

        start_get_td_time = pf.Profiling.start()
        lg.Log.critical('.   Start Get Training Data from DB Time : ' + str(start_get_td_time))

        # Get training data to improve LeBot intent/command detection
        ctdata = ctd.ChatTrainingData(
            use_db                 = self.config.USE_DB,
            db_profile             = self.config.DB_PROFILE,
            account_id             = self.account_id,
            bot_id                 = self.bot_id,
            lang                   = self.lang,
            bot_key                = self.bot_key,
            dirpath_traindata      = self.config.DIR_INTENT_TRAINDATA,
            postfix_training_files = self.config.POSTFIX_INTENT_TRAINING_FILES,
            dirpath_wordlist       = self.config.DIR_WORDLIST,
            postfix_wordlist       = self.config.POSTFIX_WORDLIST,
            dirpath_app_wordlist   = self.config.DIR_APP_WORDLIST,
            postfix_app_wordlist   = self.config.POSTFIX_APP_WORDLIST,
            dirpath_synonymlist    = self.config.DIR_SYNONYMLIST,
            postfix_synonymlist    = self.config.POSTFIX_SYNONYMLIST
        )
        if not self.config.USE_DB:
            ctdata.get_training_data(max_lines=0)
        else:
            ctdata.get_training_data_from_db()

        stop_get_td_time = pf.Profiling.stop()
        lg.Log.log('.   Stop Get Training Data from DB Time : '
                   + str(pf.Profiling.get_time_dif_str(start_get_td_time, stop_get_td_time)))

        start_test_time = pf.Profiling.start()
        lg.Log.log('.   Start Testing of Training Data from DB Time : ' + str(start_get_td_time))
        #
        # Read from chatbot training files to compare with LeBot performance
        #
        result_total = 0
        result_correct = 0
        result_accuracy_in_top_x = 0
        result_top = {}
        # How many % in top X
        result_accuracy = {}
        for top_i in range(BotTest.TEST_TOP_X):
            result_top[top_i] = 0
            result_accuracy[top_i] = 0
        result_wrong = 0
        rps = 0
        # Time per request in milliseconds
        tpr = 0
        df_scores = pd.DataFrame(columns=['Score', 'ConfLevel', 'Correct'])
        cache_intent_id_name = {}

        for i in range(0, ctdata.df_training_data.shape[0], 1):
            # if i<=410: continue
            com = str(ctdata.df_training_data[ctd.ChatTrainingData.COL_TDATA_INTENT_ID].loc[i])
            text_segmented = ctdata.df_training_data[ctd.ChatTrainingData.COL_TDATA_TEXT_SEGMENTED].loc[i]
            text_not_segmented = ctdata.df_training_data[ctd.ChatTrainingData.COL_TDATA_TEXT].loc[i]

            intent_name = str(com)

            if self.config.USE_DB:
                if not ignore_db:
                    if com in cache_intent_id_name.keys():
                        intent_name = cache_intent_id_name[com]
                    else:
                        # Get the intent name from DB
                        # TODO This part slows down the profiling, put option to not do
                        db_intent = dbint.Intent(
                            db_profile = self.config.DB_PROFILE
                        )
                        row_intent = db_intent.get(intentId=int(com))
                        intent_name = row_intent[0][dbint.Intent.COL_INTENT_NAME]
                        cache_intent_id_name[com] = intent_name
                        lg.Log.log('.   Got from DB intent id ' + str(com) + ' as [' + intent_name + ']')

            inputtext = text_segmented
            if self.do_text_segmentation or (text_segmented is None):
                inputtext = text_not_segmented

            if len(su.StringUtils.trim(inputtext)) > 1:
                df_com_class = self.bot.get_text_class(
                    chatid               = None,
                    inputtext            = inputtext,
                    do_segment_inputtext = (self.do_text_segmentation or (text_segmented is None)),
                    top                  = BotTest.TEST_TOP_X,
                    # Use training data die die for test
                    not_necessary_to_use_training_data_samples = False
                )
                com_idx = 0
                com_class = '-'
                # com_match = None
                com_score = 0
                com_conflevel = 0
                correct = False
                if df_com_class is not None:
                    if self.bot_config.MATH_ENGINE != cfbot.BotConfigFile.CONST_MATH_ENGINE_BUILT_IN:
                        import ie.lib.chat.bot.IntentModel as intentModel
                        df_com_class = intentModel.IntentModel.map_text_class_to_legacy_column_names(df_com_class)
                    lg.Log.debugdebug(df_com_class.index)
                    lg.Log.debugdebug(df_com_class.columns)
                    lg.Log.debugdebug(df_com_class.values)
                    lg.Log.debugdebug(df_com_class)
                    # We define correct by having the targeted intent in the top closest
                    com_results_list = list(df_com_class[lb.IntentEngine.COL_COMMAND])
                    correct = (com in com_results_list)
                    if not correct:
                        # Because new model no longer keeps keys as string, we need to convert to integer
                        com = int(com)
                        correct = (com in com_results_list)

                    lg.Log.debugdebug('Correct=' + str(correct))
                    if correct:
                        com_idx = df_com_class.index[df_com_class[lb.IntentEngine.COL_COMMAND] == com][0]
                        com_class = df_com_class[lb.IntentEngine.COL_COMMAND].loc[com_idx]
                        # com_match = df_com_class[lb.IntentEngine.COL_MATCH].loc[com_idx]
                        com_score = df_com_class[lb.IntentEngine.COL_SCORE].loc[com_idx]
                        com_conflevel = df_com_class[lb.IntentEngine.COL_SCORE_CONFIDENCE_LEVEL].loc[com_idx]

                result_total = result_total + 1
                time_elapsed = pf.Profiling.get_time_dif(start_test_time, pf.Profiling.stop())
                rps = round(result_total / time_elapsed, 1)
                tpr = round(1000 / rps, 1)

                df_scores = df_scores.append({
                    'Score': com_score, 'ConfLevel': com_conflevel, 'Correct': correct, 'TopIndex': com_idx
                },
                    ignore_index=True)
                lg.Log.debugdebug(df_scores)
                if not correct:
                    result_wrong = result_wrong + 1
                    lg.Log.log('Failed Command: ' + str(com) + ' (' + str(text_segmented) + ') === ' + str(com_class))
                    lg.Log.log(df_com_class)
                    lg.Log.log('   Result: ' + str(com_class))
                else:
                    result_correct = result_correct + 1
                    result_accuracy_in_top_x = round(100 * result_correct / result_total, 2)
                    str_result_accuracy = str(result_accuracy_in_top_x) + '%'

                    if include_detailed_accuracy_stats:
                        result_top[com_idx] = result_top[com_idx] + 1
                        result_accuracy[com_idx] = round(100 * result_top[com_idx] / result_total, 1)
                        for iii in range(min(3,BotTest.TEST_TOP_X)):
                            str_result_accuracy =\
                                str_result_accuracy\
                                + ', p' + str(iii+1) + '=' + str(result_accuracy[iii]) + '%'

                    if result_correct % 100 == 0:
                        lg.Log.log('Passed ' + str(result_correct) + '..')
                    lg.Log.log('Passed ' + str(result_correct)
                               + ' (' + str_result_accuracy
                               + ', ' + str(rps) + ' rps, ' + str(tpr) + 'ms per/req'
                               + '):' + str(intent_name) + ':' + str(com)
                               +' (' + str(text_not_segmented) + '||' + str(text_segmented)
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

    def test_lebot_against_text_in_csv(self, filename):
        lang = 'cn'
        brand = 'betway'
        path_qatestdata = self.dir_testdata + filename

        # Get training data to improve LeBot intent/command detection
        qatestdata = None
        try:
            qatestdata = pd.read_csv(filepath_or_buffer=path_qatestdata, sep=',', header=0)
            lg.Log.log('Read QA test data [' + path_qatestdata + '], ' + qatestdata.shape[0].__str__() + ' lines.')
        except IOError as e:
            lg.Log.log('Can\'t open file [' + path_qatestdata + ']. ')
            return

        # Add result column to data
        qatestdata['Intent.1'] = [''] * qatestdata.shape[0]
        qatestdata['Intent.1.Score'] = [0] * qatestdata.shape[0]
        qatestdata['Intent.1.ConfLevel'] = [0] * qatestdata.shape[0]
        qatestdata['Intent.2'] = [''] * qatestdata.shape[0]

        for i in range(0, qatestdata.shape[0], 1):
            inputtext = qatestdata[qatestdata.columns[0]].loc[i]
            lg.Log.log(str(i) + ': ' + inputtext)

            com_class = '-'
            com_score = 0
            com_conflevel = 0
            com_class_2 = '-'
            df_com_class = self.bot.get_text_class(
                chatid    = None,
                inputtext = inputtext,
                top       = lb.IntentEngine.SEARCH_TOPX_RFV,
                # Use training data die die for test
                not_necessary_to_use_training_data_samples = False
            )

            if df_com_class is not None:
                com_class = df_com_class[lb.IntentEngine.COL_COMMAND].loc[0]
                com_score = df_com_class[lb.IntentEngine.COL_SCORE].loc[0]
                com_conflevel = df_com_class[lb.IntentEngine.COL_SCORE_CONFIDENCE_LEVEL].loc[0]
                if df_com_class.shape[0] > 1:
                    com_class_2 = df_com_class[lb.IntentEngine.COL_COMMAND].loc[1]

            lg.Log.log('   Intent: ' + com_class + ', Score=' + str(com_score) + ', Confidence Level=' + str(
                com_conflevel))
            qatestdata['Intent.1'].loc[i] = com_class
            qatestdata['Intent.1.Score'].loc[i] = com_score
            qatestdata['Intent.1.ConfLevel'].loc[i] = com_conflevel
            qatestdata['Intent.2'].loc[i] = com_class_2

        lg.Log.log(qatestdata)
        avg_score = qatestdata['Intent.1.Score'].mean()
        quantile90 = np.percentile(qatestdata['Intent.1.Score'], 90)
        lg.Log.log('Average Score = ' + str(avg_score) + ', 90% Quantile = ' + str(quantile90))

        outputfilepath = self.dir_testdata + filename + '.lebot-result.csv'
        qatestdata.to_csv(path_or_buf=outputfilepath)

        return

    def run(
            self,
            ignore_db = False,
            test_training_data = False
    ):
        while True:
            user_choice = None
            if not test_training_data:
                print('Lang=' + self.lang + ', Botkey=' + self.bot_key + ': Choices')
                print('1: Test Bot Against Training Data')
                print('2: Test Bot Against Text in CSV File')
                print('e: Exit')
                user_choice = input('Enter Choice: ')

            if user_choice == '1' or test_training_data:
                start = pf.Profiling.start()
                lg.Log.log('Start Time: ' + str(start))

                self.test_lebot_against_training_data(
                    ignore_db = ignore_db,
                    include_detailed_accuracy_stats = self.include_detailed_accuracy_stats
                )

                stop = pf.Profiling.stop()
                lg.Log.log('Stop Time : ' + str(stop))
                lg.Log.log(pf.Profiling.get_time_dif_str(start, stop))

                if test_training_data:
                    break

            elif user_choice == '2':
                filename = cmdline.CommandLine.get_user_filename()
                if filename != None:
                    self.test_lebot_against_text_in_csv(filename = filename)
            elif user_choice == 'e':
                break
            else:
                print('No such choice [' + user_choice + ']')


if __name__ == '__main__':
    cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config
    )
    detailed_stats = True

    for arg in args:
        arg_split = arg.split('=')
        if len(arg_split) == 2:
            param = arg_split[0].lower()
            value = arg_split[1]
            if param in list(pv.keys()):
                print('Command line param [' + param + '], value [' + str(value) + '].')
                pv[param] = value

    if (pv['configfile'] is None):
        errmsg = usage_msg
        raise (Exception(errmsg))

    if pv['detailedstats'] == '0':
        detailed_stats = False
        lg.Log.log('Detailed stats = ' + str(detailed_stats))

    #
    # Main config
    #
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config
    )
    if pv['debug'] == '1':
        lg.Log.DEBUG_PRINT_ALL_TO_SCREEN = True
        nwaelog.Log.DEBUG_PRINT_ALL_TO_SCREEN = True
        print('OVERWRITE CONFIG FILE SETTING of DEBUG to ' + str(lg.Log.DEBUG_PRINT_ALL_TO_SCREEN))

    test_training_data = False
    # Logs
    lg.Log.set_path(config.FILEPATH_GENERAL_LOG)
    lg.Log.log('** Bot Test startup. Using the following parameters..')
    lg.Log.log(str(pv))

    # DB Stuff initializations
    au.Auth.init_instances()

    [accountId, botId, botLang, botkey] = cmdline.CommandLine.get_parameters_to_run_bot(
        db_profile = config.DB_PROFILE
    )
    bt = BotTest(
        config     = config,
        account_id = accountId,
        bot_id     = botId,
        lang       = botLang,
        bot_key    = botkey,
        do_text_segmentation = do_text_segmentation,
        do_profiling = config.DO_PROFILING,
        minimal      = config.MINIMAL_SERVER,
        include_detailed_accuracy_stats = detailed_stats
    )
    bt.run(
        ignore_db          = ignore_db,
        test_training_data = test_training_data
    )
