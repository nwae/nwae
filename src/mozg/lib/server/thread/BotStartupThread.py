# -*- coding: utf-8 -*-

import time as t
import datetime as dt
import threading
import mozg.common.util.Log as lg
from inspect import currentframe, getframeinfo
import ie.lib.chat.bot.IntentWrapper as intwrapper
import mozg.common.util.Db as dbutil
import mozg.common.data.Account as dbacc
import mozg.common.data.Bot as dbBot
import mozg.common.util.DbCache as dbCache
import json
import ie.lib.chat.bot.IntentEngineTest as intentEngine
import ie.lib.chat.classification.training.ChatTrainingData as ctd
import ie.lib.chat.classification.training.ChatTraining as ct
import ie.app.ConfigFile as cf
import re


#
# This thread manages chat timeouts, constantly checks which chat should be cleaned up and removed.
#
class BotStartupThread(threading.Thread):

    def __init__(
            self,
            account_name_to_startup,
            minimal,
            use_db,
            db_profile,
            cf_dir_rfv_commands,
            cf_dir_synonymlist,
            cf_dir_wordlist,
            cf_postfix_wordlist,
            cf_dir_wordlist_app,
            cf_postfix_wordlist_app,
            cf_dirpath_traindata,
            cf_postfix_training_files,
            do_profiling,
            accept_training_requests = False
    ):
        super(BotStartupThread, self).__init__()
        self.is_bot_ready = False
        self.stoprequest = threading.Event()

        self.account_name_to_startup = account_name_to_startup
        self.minimal = minimal
        self.use_db = use_db
        self.db_profile = db_profile

        self.dir_rfv_commands = cf_dir_rfv_commands
        self.dir_synonymlist = cf_dir_synonymlist
        self.dir_wordlist = cf_dir_wordlist
        self.postfix_wordlist = cf_postfix_wordlist
        self.dir_wordlist_app = cf_dir_wordlist_app
        self.postfix_wordlist_app = cf_postfix_wordlist_app
        self.dirpath_traindata = cf_dirpath_traindata
        self.postfix_training_files = cf_postfix_training_files

        self.do_profiling = do_profiling
        self.accept_training_requests = accept_training_requests

        #
        # Bots
        #
        self.db_account_id_bots = {}
        self.bots_info = {}
        self.bots = {}
        self.bots_trainer = {}
        self.__bots_mutex = threading.Lock()
        self.__bots_trainer_mutex = threading.Lock()

        # Thread safe when cleaning up
        self.__mutex = threading.Lock()
        return

    def join(self, timeout=None):
        lg.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Join called..')
        self.stoprequest.set()
        super(BotStartupThread, self).join(timeout=timeout)
        lg.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Bot Thread ended..')

    def __start_bots(self):
        if self.use_db:
            all_account_ids = dbutil.Db.get_all_account_id(
                db_profile=self.db_profile
            )

            for item_acc in all_account_ids:
                #
                # Get account ID
                #
                account_id = item_acc[dbacc.Account.COL_ACCOUNT_ID]
                account_name = item_acc[dbacc.Account.COL_ACCOUNT_NAME]

                if not (self.account_name_to_startup == ''):
                    if not (self.account_name_to_startup == account_name):
                        continue

                lg.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                + ': Initializing for account ' + str(account_name)
                                + ', account ID ' + str(account_id) + '.')

                #
                # This will return all bots in a dict with botId as key, language will be standardized
                # by guessing the language from DB
                #
                self.db_account_id_bots[account_id] = dbutil.Db.get_bots_for_account_id(
                    db_profile=self.db_profile,
                    account_id=account_id
                )
                for item_bot_id in self.db_account_id_bots[account_id].keys():
                    bot_details = self.db_account_id_bots[account_id][item_bot_id]
                    bot_id = item_bot_id
                    bot_lang = bot_details[dbBot.Bot.COL_BOT_LANGUAGE]
                    bot_name = bot_details[dbBot.Bot.COL_BOT_NAME]

                    # All keys need to be string
                    bot_id = str(bot_id)

                    botkey = ''
                    try:
                        botkey = dbBot.Bot.get_bot_key(
                            db_profile=self.db_profile,
                            account_id=account_id,
                            bot_id=bot_id,
                            lang=bot_lang
                        )
                        self.bots[botkey] = self.get_bot(
                            account_id=account_id,
                            bot_id=bot_id,
                            lang=bot_lang,
                            bot_key=botkey,
                            minimal=self.minimal
                        )
                        self.bots_info[bot_id] = {
                            dbBot.Bot.COL_ACCOUNT_ID: account_id,
                            dbBot.Bot.COL_BOT_ID: bot_id,
                            dbBot.Bot.COL_BOT_NAME: bot_name,
                            dbBot.Bot.COL_BOT_LANGUAGE: bot_lang
                        }
                        # We only start db cache after everything is successful
                        dbCache.DbCache.start_singleton_job(botkey=botkey)
                    except Exception as ex:
                        errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                                 + ': Loading of bot of botkey "' + botkey + '" failed. ' + str(ex)
                        lg.Log.critical(errmsg)
                        # Remove botkeys from container
                        lg.Log.warning(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                       + ': Removing botkey "' + botkey + '" from containers..')
                        if botkey in self.bots.keys():
                            del self.bots[botkey]
                        if bot_id in self.bots_info.keys():
                            del self.bots_info[bot_id]

            lg.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': ' + str(len(self.bots_info.keys()))
                            + ' Bots started successfully: ' + str(self.bots_info))
        else:
            raise Exception(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ' Running server from text files not implented!')

        self.is_bot_ready = True

    def run(self):
        lg.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ' Thread started..')
        self.__start_bots()

    def get_bot_lang(
            self,
            bot_id
    ):
        bot_id = str(bot_id)

        # If already in started bots
        if bot_id in self.bots_info.keys():
            bot_lang = self.bots_info[bot_id][dbBot.Bot.COL_BOT_LANGUAGE]
            return bot_lang
        else:
            # Get bot from DB if not in started bots
            db_bot = dbBot.Bot(db_profile=cf.ConfigFile.DB_PROFILE)
            bot = db_bot.get(botId=bot_id)

            lg.Log.info(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Bot id from DB returned ' + str(bot) + '.')
            if type(bot) is list:
                if len(bot) == 1:
                    bot_lang = bot[0][dbBot.Bot.COL_BOT_LANGUAGE]
                    # Standardized language
                    bot_lang = dbBot.Bot.get_bot_lang(language=bot_lang)
                    return bot_lang
                else:
                    errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                             + ' ' + str(getframeinfo(currentframe()).lineno) \
                             + ': Error. Bot ID ' + str(bot_id) + ' returned ' + str(len(bot)) + ' rows. ' \
                             + str(bot) + '.'
                    lg.Log.critical(errmsg)
                    raise Exception(errmsg)
            else:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': Error. Bot ID ' + str(bot_id) + ' returned wrong type ' + str(bot) + '.'
                lg.Log.critical(errmsg)
                raise Exception(errmsg)

    def get_bot(
            self,
            account_id,
            bot_id,
            # TODO lang and bot_key to be removed
            lang,
            bot_key,
            minimal
    ):
        lg.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                   + ': Initializing Bot for bot ID ' + str(bot_id) +', language '
                   + lang + ', botkey ' + bot_key + '.')
        bot = intwrapper.IntentWrapper(
            use_db           = self.use_db,
            db_profile       = self.db_profile,
            account_id       = account_id,
            bot_id           = bot_id,
            # TODO lang and bot_key to be removed
            lang             = lang,
            bot_key          = bot_key,
            minimal          = minimal,
            dir_rfv_commands = self.dir_rfv_commands,
            dir_synonymlist  = self.dir_synonymlist,
            dir_wordlist     = self.dir_wordlist,
            postfix_wordlist = self.postfix_wordlist,
            dir_wordlist_app = self.dir_wordlist_app,
            postfix_wordlist_app = self.postfix_wordlist_app,
            do_profiling     = self.do_profiling
        )
        bot.init()
        return bot

    def handle_intent_request(
            self,
            account_id,
            bot_id,
            question,
            chatid,
            use_only_cache_data
    ):
        try:
            # We need to lock because we could be in the process of restarting bots
            self.__bots_mutex.acquire()

            bot_lang = self.get_bot_lang(bot_id=bot_id)
            botkey = dbBot.Bot.get_bot_key(
                db_profile = cf.ConfigFile.DB_PROFILE,
                account_id = account_id,
                bot_id     = bot_id,
                lang       = bot_lang
            )
            # if not self.bot_startup_thread.is_bot_ready:
            #     return 'Bot is still starting up...'

            if botkey not in self.bots.keys():
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': Fatal error, botkey [' + str(botkey) + '] not in bots!'
                lg.Log.critical(errmsg)
                raise Exception(errmsg)

            df_com_class = self.bots[botkey].get_text_class(
                inputtext = question,
                top       = intentEngine.IntentEngine.SEARCH_TOPX_RFV,
                # When we are facing load and using only cache data, we also ask intent engine to reduce features
                reduced_features = use_only_cache_data,
                chatid    = chatid
            )
            answer_json = self.bots[botkey].get_json_response(
                chatid    = chatid,
                df_intent = df_com_class,
                use_only_cache_data = use_only_cache_data
            )

            # load_back = json.loads(answer_json)
            # lg.Log.log('Loaded back as ' + str(load_back))

            if answer_json is None:
                return json.dumps({})
            else:
                return answer_json
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ' Exception occurred for chat id "' + str(chatid)\
                     + '", question "' + question + '": ' + str(ex)
            lg.Log.critical(errmsg)
            raise Exception(errmsg)
        finally:
            self.__bots_mutex.release()

    def handle_train_bot_request(
            self,
            account_id,
            bot_id
    ):
        if not self.accept_training_requests:
            return 'This server does not accept training requests.'

        ALLOWED_TO_TRAIN_AFTER_N_MINS = 300

        try:
            bot_lang = self.get_bot_lang(bot_id = bot_id)

            botkey = dbBot.Bot.get_bot_key(
                db_profile = self.db_profile,
                account_id = account_id,
                bot_id     = bot_id,
                lang       = bot_lang
            )

            bot_trainer = None
            if botkey in self.bots_trainer.keys():
                bot_trainer = self.bots_trainer[botkey]
                if bot_trainer.is_training_done:
                    last_trained_time = bot_trainer.bot_training_end_time

                    train_dur = bot_trainer.bot_training_end_time - bot_trainer.bot_training_start_time
                    train_dur_secs = round(train_dur.seconds + train_dur.microseconds / 1000000, 1)

                    since_trained_dur = dt.datetime.now() - last_trained_time
                    since_trained_dur_secs = round(
                        since_trained_dur.seconds + since_trained_dur.microseconds / 1000000, 1)

                    if since_trained_dur_secs <= ALLOWED_TO_TRAIN_AFTER_N_MINS:
                        return str(train_dur_secs) + ' SECS DONE. Bot training for account id ' + str(account_id)\
                               + ', bot id ' + str(bot_id) + ' training started '\
                               + str(bot_trainer.bot_training_start_time)\
                               + ', ended ' + str(bot_trainer.bot_training_end_time)\
                               + '. You can train again in '\
                               + str(ALLOWED_TO_TRAIN_AFTER_N_MINS - since_trained_dur_secs) + ' secs.'\
                               + '<br><br>Training log:<br><br>' \
                               + re.sub(pattern='", "', repl='<br/>', string=str(bot_trainer.log_training))
                else:
                    progress_time = dt.datetime.now() - bot_trainer.bot_training_start_time
                    progress_time_secs = round(progress_time.seconds + progress_time.microseconds / 1000000, 1)
                    return str(progress_time_secs) + ' SECS IN PROGRESS. Bot training for account id '\
                           + str(account_id) + ', bot id ' + str(bot_id) + ' training started ' \
                           + str(bot_trainer.bot_training_start_time) \
                           + '<br><br>Training log:<br><br>' \
                           + re.sub(pattern='", "', repl='<br/>', string=str(bot_trainer.log_training))

            self.__bots_trainer_mutex.acquire()
            try:
                ctdata = ctd.ChatTrainingData(
                    use_db                 = self.use_db,
                    db_profile             = self.db_profile,
                    account_id             = account_id,
                    bot_id                 = bot_id,
                    lang                   = bot_lang,
                    bot_key                = botkey,
                    dirpath_traindata      = self.dirpath_traindata,
                    postfix_training_files = self.postfix_training_files,
                    dirpath_wordlist       = self.dir_wordlist,
                    dirpath_app_wordlist   = self.dir_wordlist_app,
                    dirpath_synonymlist    = self.dir_synonymlist
                )

                if not self.use_db:
                    ctdata.get_training_data(verbose=1)

                trainer = ct.ChatTraining(
                    botkey             = botkey,
                    dirpath_rfv        = self.dir_rfv_commands,
                    chat_training_data = ctdata,
                    keywords_remove_quartile = 0,
                    stopwords          = (),
                    weigh_idf          = True
                )
                self.bots_trainer[botkey] = trainer

                # No stopwords (IDF will provide word weights), no removal of words
                trainer.start()

                return 'STARTED. Bot training for account id ' + str(account_id) \
                               + ', bot id ' + str(bot_id) + ' training in started ' \
                               + str(trainer.bot_training_start_time)
            except Exception as ex:
                raise ex
            finally:
                self.__bots_trainer_mutex.release()
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Exception [' + str(ex) + '] training bot for account id '\
                     + str(account_id) + ', bot ID ' + str(bot_id) + '.'
            lg.Log.critical(errmsg)
            raise errmsg

    def handle_segment_words_request(
            self,
            account_id,
            bot_id,
            text
    ):
        if text is None:
            return "Error: No txt field provided. Please specify a txt."
        else:
            bot_lang = self.get_bot_lang(bot_id = bot_id)

            botkey = dbBot.Bot.get_bot_key(
                db_profile = self.db_profile,
                account_id = account_id,
                bot_id     = bot_id,
                lang       = bot_lang
            )
            try:
                answer_json = self.bots[botkey].get_word_segmentation(
                    txt = text,
                    reply_format = 'json'
                )

                if answer_json is None:
                    return json.dumps({})
                else:
                    return answer_json
            except Exception as ex:
                lg.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                           + ' Exception occurred for segment words text ' + str(text)
                                + ', exception ' + str(ex) + '.')
                return 'Sorry, could not get answer'


if __name__ == '__main__':
    msgq = BotStartupThread()

    print('Starting thread...')
    msgq.start()

    t.sleep(5)
    print('Stopping job...')
    msgq.join(timeout=5)
    print('Done')
