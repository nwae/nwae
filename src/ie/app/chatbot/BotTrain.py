# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import ie.app.ConfigFile as cf
import ie.lib.chat.bot.BotIntentAnswerTrData as botdata
import ie.lib.chat.classification.training.ChatTrainingData as ctd
import ie.lib.chat.classification.training.ChatTraining as ct
import ie.app.CommandLine as cmdline
import ie.lib.util.Log as lg
import sys


#
# Wrap our entire Bot Training steps in one class
#
class BotTrain:

    def __init__(self, lang, brand):
        self.lang = lang
        self.brand = brand
        self.dir_traindata = cf.ConfigFile.DIR_INTENT_TRAINDATA
        self.postfix_chatbot_training_files = cf.ConfigFile.POSTFIX_INTENT_TRAINING_FILES
        self.dir_wordlist = cf.ConfigFile.DIR_WORDLIST
        self.dir_app_wordlist = cf.ConfigFile.DIR_APP_WORDLIST
        self.dirpath_synonymlist = cf.ConfigFile.DIR_SYNONYMLIST

        # For training
        self.dir_rfv = cf.ConfigFile.DIR_RFV_INTENTS

        return

    def extract_training_data(self):
        print(str(self.__class__) + ' Extracting Training Data...')
        bd = botdata.BotIntentAnswerTrData(
            lang  = self.lang,
            brand = self.brand,
            dirpath = cf.ConfigFile.DIR_INTENT_TRAINDATA,
            postfix_intent_answer_trdata = cf.ConfigFile.POSTFIX_INTENT_ANSWER_TRDATA_FILE,
            postfix_trdata = cf.ConfigFile.POSTFIX_INTENT_TRAINING_FILES
        )
        # No need to call explicitly, constructor already calls it
        # bd.extract_training_data()

        return

    #
    # For training data preprocessing, we will split the text first as it is (for now)
    # the slowest relatively.
    # Other things include removal of empty lines, etc.
    #
    def preprocess_training_data(self):
        ctdata = ctd.ChatTrainingData(lang                   = self.lang,
                                      # Brand must be empty, so we process the entire training data
                                      brand                  = self.brand,
                                      dirpath_traindata      = self.dir_traindata,
                                      postfix_training_files = self.postfix_chatbot_training_files,
                                      dirpath_wordlist       = self.dir_wordlist,
                                      dirpath_app_wordlist   = self.dir_app_wordlist,
                                      dirpath_synonymlist    = self.dirpath_synonymlist)
        ctdata.pre_process_training_data(segment_words=True, verbose=1)
        print(ctdata.df_training_data)

        ctdata.write_split_training_data_to_file()
        return

    def train(self):
        ctdata = ctd.ChatTrainingData(lang                   = self.lang,
                                      brand                  = self.brand,
                                      dirpath_traindata      = self.dir_traindata,
                                      postfix_training_files = self.postfix_chatbot_training_files,
                                      dirpath_wordlist       = self.dir_wordlist,
                                      dirpath_app_wordlist   = self.dir_app_wordlist,
                                      dirpath_synonymlist    = self.dirpath_synonymlist)
        ctdata.get_training_data(verbose=1)

        trainer = ct.ChatTraining(lang              = self.lang,
                                 brand              = self.brand,
                                 dirpath_rfv        = self.dir_rfv,
                                 chat_training_data = ctdata
                          )

        # No stopwords (IDF will provide word weights), no removal of words
        trainer.train(stopwords=[], keywords_remove_quartile=0, weigh_idf=True, verbose=1)
        return

    def run(self):

        while True:
            print(self.brand + '.' + self.lang + ': ' + 'Yuna Bot Training (you need to run in the following sequence):')
            print('  1: Extract Raw Training Data (chat/bot/BotIntentAnswerTrData.py)')
            print('     This step will extract training data (which is in a chunk separated by newline) into a separate file.')
            print('  2: Preprocess Raw Training Data (chat/classification/ChatTrainingData.py)')
            print('     This step will split the original training text into segmented words, remove invalid lines, etc.')
            print('  3: Train the Bot (chat/classification/ChatTraining.py)')
            print('     This step converts all split training text into mathematical objects, and performs mathematical operations to represent Intents in a single/few math objects/matrices')
            print('  e: Exit')
            print(     '              ')
            ui = input('Enter Choice: ')

            if ui == '1':
                self.extract_training_data()
            elif ui == '2':
                self.preprocess_training_data()
            elif ui == '3':
                self.train()
            elif ui == 'e':
                break
            else:
                lg.Log.log('Unrecognized Command [' + ui + ']!')
                lg.Log.log('')


if __name__ == '__main__':

    #
    # Run like '/usr/local/bin/python3.6 -m ie.app.chatbot.BotTrain topdir=/home/mark/svn/yuna'
    #
    # Default values
    pv = {
        'topdir': None,
        'debug': '0',
        'verbose': '1'
    }
    args = sys.argv
    usage_msg = 'Usage: /usr/local/bin/python3.6 -m ie.app.chatbot.BotTrain topdir=/home/mark/svn/yuna'

    for arg in args:
        arg_split = arg.split('=')
        if len(arg_split) == 2:
            param = arg_split[0].lower()
            value = arg_split[1]
            if param in list(pv.keys()):
                pv[param] = value

    if ((pv['topdir'] is None)):
        errmsg = usage_msg
        raise (Exception(errmsg))

    #
    # !!!MOST IMPORTANT, top directory, otherwise all other config/NLP/training/etc. files we won't be able to find
    #
    cf.ConfigFile.TOP_DIR = pv['topdir']
    cf.ConfigFile.top_dir_changed()
    # Where to log messages
    lg.Log.set_path(cf.ConfigFile.DIR_INTENT_TRAIN_LOGS + '/trainlogs.txt')

    for p in list(pv.keys()):
        v = pv[p]
        if v is None:
            errmsg = usage_msg
            raise (Exception(errmsg))

    # Logs
    if pv['debug'] == '1':
        lg.Log.DEBUG_PRINT_ALL_TO_SCREEN = True

    ui_lang = None
    ui_brand = None

    while ui_lang is None:
        ui_lang = cmdline.CommandLine.get_user_input_language()

    while ui_brand is None:
        ui_brand = cmdline.CommandLine.get_user_input_brand()

    bt = BotTrain(lang=ui_lang, brand=ui_brand)
    bt.run()
