# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import sys
import ie.app.ConfigFile as cf
import numpy as np
import pandas as pd
import mozg.lib.chat.classification.training.ChatTrainingData as ctd
import mozg.common.util.StringUtils as su
import mozg.common.util.Log as lg
import mozg.lib.chat.bot.IntentEngine as lb
import mozg.lib.chat.bot.IntentWrapper as gb
import mozg.common.util.CommandLine as cmdline
import mozg.common.util.Profiling as pf
import mozg.common.data.Intent as dbint
import mozg.common.data.security.Auth as au


#
# This only tests intent engine, does not test word segmentation
#
class BotTest:

    def __init__(
            self,
            account_id,
            bot_id,
            lang,
            bot_key,
            reduce_features,
            do_text_segmentation,
            do_profiling,
            minimal,
            verbose=0.0
    ):
        self.account_id = account_id
        self.bot_id = bot_id
        self.lang = lang
        self.bot_key = bot_key
        self.reduce_features = reduce_features
        self.do_text_segmentation = do_text_segmentation
        self.do_profiling = do_profiling
        self.minimal = minimal

        self.dir_testdata = cf.ConfigFile.DIR_INTENTTEST_TESTDATA
        self.verbose = verbose

        self.crmbot = gb.IntentWrapper(
            use_db      = cf.ConfigFile.USE_DB,
            db_profile  = cf.ConfigFile.DB_PROFILE,
            account_id  = self.account_id,
            bot_id      = self.bot_id,
            lang        = self.lang,
            bot_key     = self.bot_key,
            dir_rfv_commands = cf.ConfigFile.DIR_RFV_INTENTS,
            dir_synonymlist  = cf.ConfigFile.DIR_SYNONYMLIST,
            dir_wordlist     = cf.ConfigFile.DIR_WORDLIST,
            postfix_wordlist = cf.ConfigFile.POSTFIX_WORDLIST,
            dir_wordlist_app = cf.ConfigFile.DIR_APP_WORDLIST,
            postfix_wordlist_app = cf.ConfigFile.POSTFIX_APP_WORDLIST,
            do_profiling = self.do_profiling,
            minimal      = self.minimal,
            verbose      = self.verbose
        )
        self.crmbot.init()
        return

    #
    # TODO: Include data that is suppose to fail (e.g. run LeBot through our historical chats to get that data)
    # TODO: This way we can measure both what is suppose to pass and what is suppose to fail
    #
    def test_lebot_against_training_data(
        self,
        ignore_db = False
    ):

        start_get_td_time = pf.Profiling.start()
        lg.Log.critical('.   Start Get Training Data from DB Time : ' + str(start_get_td_time))

        # Get training data to improve LeBot intent/command detection
        ctdata = ctd.ChatTrainingData(
            use_db                 = cf.ConfigFile.USE_DB,
            db_profile             = cf.ConfigFile.DB_PROFILE,
            account_id             = self.account_id,
            bot_id                 = self.bot_id,
            lang                   = self.lang,
            bot_key                = self.bot_key,
            dirpath_traindata      = cf.ConfigFile.DIR_INTENT_TRAINDATA,
            postfix_training_files = cf.ConfigFile.POSTFIX_INTENT_TRAINING_FILES,
            dirpath_wordlist       = cf.ConfigFile.DIR_WORDLIST,
            dirpath_app_wordlist   = cf.ConfigFile.DIR_APP_WORDLIST,
            dirpath_synonymlist    = cf.ConfigFile.DIR_SYNONYMLIST
        )
        if not cf.ConfigFile.USE_DB:
            ctdata.get_training_data(max_lines=0, verbose=self.verbose)
        else:
            ctdata.get_training_data_from_db(max_lines=0, verbose=self.verbose)

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
        result_wrong = 0
        df_scores = pd.DataFrame(columns=['Score', 'ConfLevel', 'Correct'])
        cache_intent_id_name = {}

        for i in range(0, ctdata.df_training_data.shape[0], 1):
            # if i<=410: continue
            com = str(ctdata.df_training_data[ctd.ChatTrainingData.COL_TDATA_INTENT_ID].loc[i])
            text_segmented = ctdata.df_training_data[ctd.ChatTrainingData.COL_TDATA_TEXT_SEGMENTED].loc[i]
            text_not_segmented = ctdata.df_training_data[ctd.ChatTrainingData.COL_TDATA_TEXT].loc[i]

            intent_name = str(com)

            if cf.ConfigFile.USE_DB:
                if not ignore_db:
                    if com in cache_intent_id_name.keys():
                        intent_name = cache_intent_id_name[com]
                    else:
                        # Get the intent name from DB
                        # TODO This part slows down the profiling, put option to not do
                        db_intent = dbint.Intent(
                            db_profile = cf.ConfigFile.DB_PROFILE,
                            verbose    = self.verbose
                        )
                        row_intent = db_intent.get(intentId=int(com))
                        intent_name = row_intent[0][dbint.Intent.COL_INTENT_NAME]
                        cache_intent_id_name[com] = intent_name
                        lg.Log.log('.   Got from DB intent id ' + str(com) + ' as [' + intent_name + ']')

            inputtext = text_segmented
            if self.do_text_segmentation or (text_segmented is None):
                inputtext = text_not_segmented

            if len(su.StringUtils.trim(inputtext)) > 1:
                df_com_class = self.crmbot.get_text_class(
                    chatid               = None,
                    inputtext            = inputtext,
                    reduced_features     = self.reduce_features,
                    do_segment_inputtext = (self.do_text_segmentation or (text_segmented is None)),
                    top                  = lb.IntentEngine.SEARCH_TOPX_RFV,
                    # Use training data die die for test
                    not_necessary_to_use_training_data_samples = False
                )
                com_idx = 0
                com_class = '-'
                com_match = None
                com_score = 0
                com_conflevel = 0
                correct = False
                if df_com_class is not None:
                    lg.Log.debugdebug(df_com_class.index)
                    lg.Log.debugdebug(df_com_class.columns)
                    lg.Log.debugdebug(df_com_class.values)
                    # We define correct by having the targeted intent in the top closest
                    correct = com in list(df_com_class[lb.IntentEngine.COL_COMMAND])
                    lg.Log.debugdebug('Correct=' + str(correct))
                    if correct:
                        com_idx = df_com_class.index[df_com_class[lb.IntentEngine.COL_COMMAND] == com][0]
                        com_class = df_com_class[lb.IntentEngine.COL_COMMAND].loc[com_idx]
                        com_match = df_com_class[lb.IntentEngine.COL_MATCH].loc[com_idx]
                        com_score = df_com_class[lb.IntentEngine.COL_SCORE].loc[com_idx]
                        com_conflevel = df_com_class[lb.IntentEngine.COL_SCORE_CONFIDENCE_LEVEL].loc[com_idx]

                result_total = result_total + 1
                df_scores = df_scores.append({
                    'Score': com_score, 'ConfLevel': com_conflevel, 'Correct': correct, 'TopIndex': com_idx
                },
                    ignore_index=True)
                lg.Log.debugdebug(df_scores)
                if not correct:
                    result_wrong = result_wrong + 1
                    lg.Log.log('Failed Command: ' + str(com) + ' (' + str(text_segmented) + ') === ' + str(com_class))
                    lg.Log.log(df_com_class)
                    lg.Log.log('   Result: ' + str(com_class) + ', Match: ' + str(com_match))
                else:
                    result_correct = result_correct + 1
                    if result_correct % 100 == 0:
                        lg.Log.log('Passed ' + str(result_correct) + '..')
                    lg.Log.log('Passed ' + str(result_correct) + ':' + str(intent_name) + ':' + str(com)
                               +' (' + str(text_not_segmented) + '||' + str(text_segmented)
                               + '). Score=' + str(com_score) + ', ConfLevel=' + str(com_conflevel)
                               + ', Index=' + str(com_idx+1))
                    if com_idx != 0:
                        lg.Log.log('   Result not 1st!')

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
            df_com_class = self.crmbot.get_text_class(
                chatid    = None,
                inputtext = inputtext,
                reduced_features = self.reduce_features,
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

                self.test_lebot_against_training_data(ignore_db=ignore_db)

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
    # Default values
    pv = {
        'topdir': None,
        'bot': None,
        'ignoredb': '1',
        'dotextsegmentation': '1',
        'reducefeatures': '0',
        'doprofiling': '0',
        'minimal': '0',
        'debug': '0',
        'verbose': lg.Log.LOG_LEVEL_IMPORTANT
    }
    args = sys.argv
    usage_msg = 'Usage: ./run.bottest.sh topdir=/Users/mark.tan/git/mozg.nlp'

    ignore_db = False
    do_text_segmentation = True
    reduce_features = False
    do_profiling = False
    minimal = False

    for arg in args:
        arg_split = arg.split('=')
        if len(arg_split) == 2:
            param = arg_split[0].lower()
            value = arg_split[1]
            if param in list(pv.keys()):
                print('Command line param [' + param + '], value [' + str(value) + '].')
                pv[param] = value

    if (pv['topdir'] is None):
        errmsg = usage_msg
        raise (Exception(errmsg))

    #
    # !!!MOST IMPORTANT, top directory, otherwise all other config/NLP/training/etc. files we won't be able to find
    #
    cf.ConfigFile.TOP_DIR = pv['topdir']
    cf.ConfigFile.top_dir_changed()

    lg.Log.LOGLEVEL = float(pv['verbose'])
    if pv['debug'] == '1':
        lg.Log.log('Print all logs to screen.')
        lg.Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    if pv['dotextsegmentation'] == '0':
        do_text_segmentation = False
    if pv['ignoredb'] == '1':
        lg.Log.log('Ignoring DB, not fetching intent names, just using intent ID to speed things up.')
        ignore_db = True
    if pv['reducefeatures'] == '1':
        lg.Log.log('Reducing Features of the feature RFVs to speed up core math calculation.')
        reduce_features = True
    if pv['doprofiling'] == '1':
        lg.Log.log('Doing profiling code.')
        do_profiling = True
    if pv['minimal'] == '1':
        lg.Log.log('Minimal RAM = True')
        minimal = True

    test_training_data = False
    # Logs
    lg.Log.set_path(cf.ConfigFile.FILEPATH_GENERAL_LOG)
    lg.Log.log('** Bot Test startup. Using the following parameters..')
    lg.Log.log(str(pv))

    # DB Stuff initializations
    au.Auth.init_instances()

    [accountId, botId, botLang, botkey] = cmdline.CommandLine.get_parameters_to_run_bot(
        db_profile = cf.ConfigFile.DB_PROFILE
    )
    bt = BotTest(
        account_id = accountId,
        bot_id     = botId,
        lang       = botLang,
        bot_key    = botkey,
        reduce_features = reduce_features,
        do_text_segmentation = do_text_segmentation,
        do_profiling = do_profiling,
        minimal      = minimal,
        verbose    = float(pv['verbose'])
    )
    bt.run(
        ignore_db          = ignore_db,
        test_training_data = test_training_data
    )
