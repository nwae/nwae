# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import ie.app.ConfigFile as cf
import mozg.common.bot.BotIntentAnswer as botdata
import ie.lib.chat.classification.training.ChatTrainingData as ctd
import ie.lib.chat.classification.training.ChatTraining as ct
import mozg.common.data.security.Auth as au
import mozg.common.util.CommandLine as cmdline
import mozg.common.util.Log as lg
import sys
import mozg.common.util.Profiling as pf


#
# Wrap our entire Bot Training steps in one class
#
class BotTrain:

    def __init__(
            self,
            account_id,
            bot_id,
            lang,
            botkey,
            verbose = 0.0
    ):
        self.account_id = account_id
        self.bot_id = bot_id

        self.lang = lang
        self.botkey = botkey

        self.verbose = verbose

        self.dir_traindata = cf.ConfigFile.DIR_INTENT_TRAINDATA
        self.postfix_chatbot_training_files = cf.ConfigFile.POSTFIX_INTENT_TRAINING_FILES

        self.dir_wordlist = cf.ConfigFile.DIR_WORDLIST
        self.postfix_wordlist = cf.ConfigFile.POSTFIX_WORDLIST
        self.dir_app_wordlist = cf.ConfigFile.DIR_APP_WORDLIST
        self.postfix_wordlist_app = cf.ConfigFile.POSTFIX_APP_WORDLIST
        self.dirpath_synonymlist = cf.ConfigFile.DIR_SYNONYMLIST

        # For training
        self.dir_rfv = cf.ConfigFile.DIR_RFV_INTENTS
        return

    #
    # TODO When we point to DB, this is no longer needed. Throw this away.
    #
    def extract_training_data(self):
        if cf.ConfigFile.USE_DB:
            print('Nothing to do. Using DB.')
        else:
            print(str(self.__class__) + ' Extracting Training Data...')
            bd = botdata.BotIntentAnswer(
                db_profile = cf.ConfigFile.DB_PROFILE,
                account_id = self.account_id,
                bot_id     = self.bot_id,
                bot_lang   = self.lang,
                botkey     = self.botkey,
                dirpath    = cf.ConfigFile.DIR_INTENT_TRAINDATA,
                postfix_intent_answer_trdata = cf.ConfigFile.POSTFIX_INTENT_ANSWER_TRDATA_FILE,
                train_mode = True,
                postfix_trdata = cf.ConfigFile.POSTFIX_INTENT_TRAINING_FILES
            )
            # No need to call explicitly, constructor already calls it
            # bd.extract_training_data()

        return

    #
    # For training data preprocessing, we will split the text first as it is (for now)
    # the slowest relatively.
    # Other things include removal of empty lines, etc.
    # TODO When we point to DB, this is no longer needed. Throw this away.
    #
    def preprocess_training_data(self):
        if cf.ConfigFile.USE_DB:
            print('Nothing to do. Using DB.')
        else:
            ctdata = ctd.ChatTrainingData(
                use_db                 = cf.ConfigFile.USE_DB,
                db_profile             = cf.ConfigFile.DB_PROFILE,
                account_id             = self.account_id,
                bot_id                 = self.bot_id,
                lang                   = self.lang,
                # Brand must be empty, so we process the entire training data
                bot_key                = self.botkey,
                dirpath_traindata      = self.dir_traindata,
                postfix_training_files = self.postfix_chatbot_training_files,
                dirpath_wordlist       = self.dir_wordlist,
                dirpath_app_wordlist   = self.dir_app_wordlist,
                dirpath_synonymlist    = self.dirpath_synonymlist
            )

            start = pf.Profiling.start()
            print('Start Time: ' + str(start))

            ctdata.pre_process_text_file_training_data(segment_words=True, verbose=1)
            print(ctdata.df_training_data)

            stop = pf.Profiling.stop()
            print('Stop Time : ' + str(stop))
            print(pf.Profiling.get_time_dif_str(start, stop))

            ctdata.write_split_training_data_to_file()
        return

    def train(self):
        ctdata = ctd.ChatTrainingData(
            use_db                 = cf.ConfigFile.USE_DB,
            db_profile             = cf.ConfigFile.DB_PROFILE,
            account_id             = self.account_id,
            bot_id                 = self.bot_id,
            lang                   = self.lang,
            bot_key                = self.botkey,
            dirpath_traindata      = self.dir_traindata,
            postfix_training_files = self.postfix_chatbot_training_files,
            dirpath_wordlist       = self.dir_wordlist,
            dirpath_app_wordlist   = self.dir_app_wordlist,
            dirpath_synonymlist    = self.dirpath_synonymlist
        )

        if not cf.ConfigFile.USE_DB:
            # Prepare data first if reading from csv
            ctdata.get_training_data()

        trainer = ct.ChatTraining(
            botkey             = self.botkey,
            dirpath_rfv        = self.dir_rfv,
            chat_training_data = ctdata
        )

        # No stopwords (IDF will provide word weights), no removal of words
        trainer.train(stopwords=[], keywords_remove_quartile=0, weigh_idf=True)
        return

    def run(self):

        while True:
            print(self.botkey + ': ' + ' Bot Training (you need to run in the following sequence):')

            lbl = 1
            if not cf.ConfigFile.USE_DB:
                print('  ' + str(lbl) + ': Extract Raw Training Data (chat/bot/BotIntentAnswer.py)')
                print('     This step will extract training data (which is in a chunk separated by newline) into a separate file.')
                lbl = lbl + 1
                print('  ' + str(lbl) + ': Preprocess Raw Training Data (chat/classification/ChatTrainingData.py)')
                print('     This step will split the original training text into segmented words, remove invalid lines, etc.')
                lbl = lbl + 1

            print('  ' + str(lbl) + ': Train the Bot (chat/classification/ChatTraining.py)')
            lbl = lbl + 1
            print('     This step converts all split training text into mathematical objects, and performs mathematical operations to represent Intents in a single/few math objects/matrices')
            print('  e: Exit')
            print(     '              ')
            ui = input('Enter Choice: ')

            lbl = 1
            if not cf.ConfigFile.USE_DB:
                if ui == str(lbl):
                    self.extract_training_data()
                lbl = lbl + 1
                if ui == str(lbl):
                    self.preprocess_training_data()
                lbl = lbl + 1

            if ui == str(lbl):
                self.train()
            lbl = lbl + 1
            if ui == 'e':
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
        'verbose': lg.Log.LOG_LEVEL_IMPORTANT
    }
    args = sys.argv
    usage_msg = 'Usage: /usr/local/bin/python3.6 -m ie.app.chatbot.BotTrain topdir=/home/mark/git/mozg.nlp'

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
    lg.Log.LOGLEVEL = float(pv['verbose'])
    if pv['debug'] == '1':
        lg.Log.DEBUG_PRINT_ALL_TO_SCREEN = True

    # DB Stuff initializations
    au.Auth.init_instances()

    [accountId, botId, botLang, botkey] = cmdline.CommandLine.get_parameters_to_run_bot(
        db_profile=cf.ConfigFile.DB_PROFILE
    )
    bt = BotTrain(
        account_id = accountId,
        bot_id     = botId,
        lang       = botLang,
        botkey     = botkey,
        verbose    = float(pv['verbose'])
    )
    bt.run()
