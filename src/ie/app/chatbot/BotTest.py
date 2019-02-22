# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import ie.app.ConfigFile as cf
import numpy as np
import pandas as pd
import ie.lib.chat.classification.training.ChatTrainingData as ctd
import ie.lib.util.StringUtils as su
import ie.lib.chat.bot.Intent as lb
import ie.lib.chat.bot.LeBotWrapper as gb
import ie.app.CommandLine as cmdline


class BotTest:

    def __init__(self, lang, brand, verbose=0):
        self.lang = lang
        self.brand = brand
        self.dir_testdata = cf.ConfigFile.DIR_BOTTESTING_TESTDATA
        self.verbose = verbose

        self.crmbot = gb.LeBotWrapper(lang      = self.lang,
                              brand     = self.brand,
                              dir_rfv_commands = cf.ConfigFile.DIR_RFV_INTENTS,
                              dir_synonymlist  = cf.ConfigFile.DIR_SYNONYMLIST,
                              dir_wordlist     = cf.ConfigFile.DIR_WORDLIST,
                              postfix_wordlist = cf.ConfigFile.POSTFIX_WORDLIST,
                              dir_wordlist_app = cf.ConfigFile.DIR_APP_WORDLIST,
                              postfix_wordlist_app = '.' + self.brand + cf.ConfigFile.POSTFIX_APP_WORDLIST)
        self.crmbot.init(verbose=self.verbose)
        return

    #
    # TODO: Include data that is suppose to fail (e.g. run LeBot through our historical chats to get that data)
    # TODO: This way we can measure both what is suppose to pass and what is suppose to fail
    #
    def test_lebot_against_training_data(self):

        # Get training data to improve LeBot intent/command detection
        ctdata = ctd.ChatTrainingData(lang                   = self.lang,
                                      brand                  = self.brand,
                                      dirpath_traindata      = cf.ConfigFile.DIR_CHATBOT_TRAINDATA,
                                      postfix_training_files = cf.ConfigFile.POSTFIX_CHATBOT_TRAINING_FILES,
                                      dirpath_wordlist       = cf.ConfigFile.DIR_WORDLIST,
                                      dirpath_app_wordlist   = cf.ConfigFile.DIR_APP_WORDLIST,
                                      dirpath_synonymlist    = cf.ConfigFile.DIR_SYNONYMLIST)
        ctdata.get_training_data(max_lines=0, verbose=self.verbose)

        #
        # Read from chatbot training files to compare with LeBot performance
        #
        result_total = 0
        result_correct = 0
        result_wrong = 0
        df_scores = pd.DataFrame(columns=['Score', 'ConfLevel', 'Correct'])
        for i in range(0, ctdata.df_training_data.shape[0], 1):
            # if i<=410: continue
            com = ctdata.df_training_data[ctd.ChatTrainingData.COL_TDATA_INTENT_ID].loc[i]
            text_segmented = ctdata.df_training_data[ctd.ChatTrainingData.COL_TDATA_TEXT_SEGMENTED].loc[i]

            if len(su.StringUtils.trim(text_segmented)) > 1:
                df_com_class = self.crmbot.get_text_class(
                    inputtext           = text_segmented,
                    top                 = lb.Intent.SEARCH_TOPX_RFV
                )
                com_idx = 0
                com_class = '-'
                com_match = None
                com_score = 0
                com_conflevel = 0
                correct = False
                if df_com_class is not None:
                    # We define correct by having the targeted intent in the top closest
                    correct = com in list(df_com_class[lb.Intent.COL_COMMAND])
                    if correct:
                        com_idx = df_com_class.index[df_com_class[lb.Intent.COL_COMMAND] == com][0]
                        com_class = df_com_class[lb.Intent.COL_COMMAND].loc[com_idx]
                        com_match = df_com_class[lb.Intent.COL_MATCH].loc[com_idx]
                        com_score = df_com_class[lb.Intent.COL_SCORE].loc[com_idx]
                        com_conflevel = df_com_class[lb.Intent.COL_SCORE_CONFIDENCE_LEVEL].loc[com_idx]

                result_total = result_total + 1
                df_scores = df_scores.append({'Score': com_score, 'ConfLevel': com_conflevel, 'Correct': correct, 'TopIndex': com_idx},
                                             ignore_index=True)
                if not correct:
                    result_wrong = result_wrong + 1
                    print('Failed Command: ' + str(com) + ' (' + text_segmented + ') === ' + com_class)
                    print(df_com_class)
                    print('   Result: ' + str(com_class) + ', Match: ' + str(com_match))
                else:
                    result_correct = result_correct + 1
                    if result_correct % 100 == 0:
                        print('Passed ' + str(result_correct) + '..')
                    print('Passed ' + str(result_correct) + ':' + str(com) + ' (' + text_segmented + '). Score=' + str(
                        com_score) +
                          ', ConfLevel=' + str(com_conflevel)
                          + ', Index=' + str(com_idx+1))
                    if com_idx != 0:
                        print('   Result not 1st!')

            # if i>1: break

        print(str(result_wrong) + ' wrong results from ' + str(result_total) + ' total tests.')
        print("Score Quantile (0): " + str(df_scores['Score'].quantile(0)))
        print("Score Quantile (5%): " + str(df_scores['Score'].quantile(0.05)))
        print("Score Quantile (25%): " + str(df_scores['Score'].quantile(0.25)))
        print("Score Quantile (50%): " + str(df_scores['Score'].quantile(0.5)))
        print("Score Quantile (75%): " + str(df_scores['Score'].quantile(0.75)))
        print("Score Quantile (95%): " + str(df_scores['Score'].quantile(0.95)))

        return

    def test_lebot_against_text_in_csv(self, filename):
        lang = 'cn'
        brand = 'betway'
        path_qatestdata = self.dir_testdata + filename

        # Get training data to improve LeBot intent/command detection
        qatestdata = None
        try:
            qatestdata = pd.read_csv(filepath_or_buffer=path_qatestdata, sep=',', header=0)
            print('Read QA test data [' + path_qatestdata + '], ' + qatestdata.shape[0].__str__() + ' lines.')
        except IOError as e:
            print('Can\'t open file [' + path_qatestdata + ']. ')
            return

        # Add result column to data
        qatestdata['Intent.1'] = [''] * qatestdata.shape[0]
        qatestdata['Intent.1.Score'] = [0] * qatestdata.shape[0]
        qatestdata['Intent.1.ConfLevel'] = [0] * qatestdata.shape[0]
        qatestdata['Intent.2'] = [''] * qatestdata.shape[0]

        for i in range(0, qatestdata.shape[0], 1):
            inputtext = qatestdata[qatestdata.columns[0]].loc[i]
            print(str(i) + ': ' + inputtext)

            com_class = '-'
            com_score = 0
            com_conflevel = 0
            com_class_2 = '-'
            df_com_class = self.crmbot.get_text_class(inputtext=inputtext, top=lb.Intent.SEARCH_TOPX_RFV)

            if df_com_class is not None:
                com_class = df_com_class[lb.Intent.COL_COMMAND].loc[0]
                com_score = df_com_class[lb.Intent.COL_SCORE].loc[0]
                com_conflevel = df_com_class[lb.Intent.COL_SCORE_CONFIDENCE_LEVEL].loc[0]
                if df_com_class.shape[0] > 1:
                    com_class_2 = df_com_class[lb.Intent.COL_COMMAND].loc[1]

            print('   Intent: ' + com_class + ', Score=' + str(com_score) + ', Confidence Level=' + str(
                com_conflevel))
            qatestdata['Intent.1'].loc[i] = com_class
            qatestdata['Intent.1.Score'].loc[i] = com_score
            qatestdata['Intent.1.ConfLevel'].loc[i] = com_conflevel
            qatestdata['Intent.2'].loc[i] = com_class_2

        print(qatestdata)
        avg_score = qatestdata['Intent.1.Score'].mean()
        quantile90 = np.percentile(qatestdata['Intent.1.Score'], 90)
        print('Average Score = ' + str(avg_score) + ', 90% Quantile = ' + str(quantile90))

        outputfilepath = self.dir_testdata + filename + '.lebot-result.csv'
        qatestdata.to_csv(path_or_buf=outputfilepath)

        return

    def run(self):
        while True:
            print(self.lang + '.' + self.brand + ': Choices')
            print('1: Test Bot Against Training Data')
            print('2: Test Bot Against Text in CSV File')
            print('e: Exit')
            user_choice = input('Enter Choice: ')

            if user_choice == '1':
                self.test_lebot_against_training_data()
            elif user_choice == '2':
                filename = cmdline.CommandLine.get_user_filename()
                if filename != None:
                    self.test_lebot_against_text_in_csv(filename = filename)
            elif user_choice == 'e':
                break
            else:
                print('No such choice [' + user_choice + ']')


if __name__ == '__main__':
    ui_lang = None
    ui_brand = None

    while ui_lang is None:
        ui_lang = cmdline.CommandLine.get_user_input_language()

    while ui_brand is None:
        ui_brand = cmdline.CommandLine.get_user_input_brand()

    bt = BotTest(lang=ui_lang, brand=ui_brand)
    bt.run()
