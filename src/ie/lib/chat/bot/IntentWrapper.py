# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import ie.lib.chat.bot.IntentEngineTest as intEng
import mozg.common.util.StringUtils as su
import ie.lib.lang.nlp.WordSegmentation as ws
import mozg.common.util.Log as log
import threading
import json
import numpy as np
import time
import mozg.common.util.Profiling as prf
import mozg.common.util.DbCache as dbcache
from inspect import currentframe, getframeinfo


#
# Wrap LeBot here, and put state management
# THREAD SAFE CLASS
#   Make sure there is no class variable that is modified during function calls.
#
class IntentWrapper:

    #
    # This is to decide how many top answers to keep.
    # If this value is say 70%, and our top scores are 70, 60, 40, 20, then
    # 70% * 70 is 49, thus only scores 70, 60 will be kept as it is higher than 49
    #
    CONSTANT_PERCENT_WITHIN_TOP_SCORE = 0.6
    MAX_QUESTION_LENGTH = 100

    JSON_ENCODING = 'utf-8'

    JSON_COL_POSITION    = 'pos'
    JSON_COL_INTENT_NAME = 'intent'
    JSON_COL_INTENT_ID   = 'intentId'
    JSON_COL_SCORE       = 'score'
    JSON_COL_CONFIDENCE  = 'confidence'

    def __init__(
            self,
            use_db,
            db_profile,
            account_id,
            bot_id,
            # TODO lang and bot_key to be removed
            lang,
            # We do not specify here whether this is bot id or bot name or account-campaign-botname, etc.
            bot_key,
            dir_rfv_commands,
            dir_synonymlist,
            dir_wordlist,
            postfix_wordlist,
            dir_wordlist_app,
            postfix_wordlist_app,
            do_profiling = False,
            minimal = False,
            min_score_threshold = intEng.IntentEngine.DEFAULT_SCORE_MIN_THRESHOLD,
            verbose = 0
    ):
        #
        # All class variables are constants only to ensure thread safety
        # except for the intent cache
        #
        self.use_db = use_db
        self.db_profile = db_profile
        self.account_id = account_id
        self.bot_id = bot_id

        self.lang = su.StringUtils.trim(lang.lower())
        self.bot_key = su.StringUtils.trim(bot_key.lower())
        self.dir_rfv_commands = su.StringUtils.trim(dir_rfv_commands)
        self.dir_synonymlist = su.StringUtils.trim(dir_synonymlist)

        self.dir_wordlist = dir_wordlist
        self.postfix_wordlist = postfix_wordlist
        self.dir_wordlist_app = dir_wordlist_app
        self.postfix_wordlist_app = postfix_wordlist_app

        self.do_profiling = do_profiling
        self.minimal = minimal
        self.min_score_threshold = min_score_threshold

        self.lebot = None
        #self.lebot_reduced = None
        self.wseg = None
        self.bot_reply = None
        self.verbose = verbose

        # Thread safe mutex, not used
        self.mutex = threading.Lock()
        return

    #
    # Only during initialization we modify class variables, after this, no other writes should happen.
    #
    def init(self):

        # Initialize AI/NLP Bot Engine
        self.lebot = intEng.IntentEngine(
            lang    = self.lang,
            bot_key = self.bot_key,
            dir_rfv_commands    = self.dir_rfv_commands,
            dirpath_synonymlist = self.dir_synonymlist,
            reduce_features = False,
            do_profiling = self.do_profiling,
            minimal      = self.minimal,
            verbose      = self.verbose
        )
        self.lebot.do_background_load()

        self.wseg = ws.WordSegmentation(
            lang = self.lang,
            dirpath_wordlist = self.dir_wordlist,
            postfix_wordlist = self.postfix_wordlist,
            do_profiling = self.do_profiling,
            verbose      = self.verbose
        )
        # Add application wordlist
        self.wseg.add_wordlist(
            dirpath=self.dir_wordlist_app,
            postfix=self.postfix_wordlist_app
        )

        # Add synonym list to wordlist (just in case they are not synched)
        len_before = self.wseg.lang_wordlist.wordlist.shape[0]
        self.wseg.add_wordlist(
            dirpath     = None,
            postfix     = None,
            array_words = list(self.lebot.synonymlist_ro.synonymlist['Word'])
        )
        len_after = self.wseg.lang_wordlist.wordlist.shape[0]
        if len_after - len_before > 0:
            log.Log.log(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ": Warning. These words not in word list but in synonym list:")
            words_not_synched = self.wseg.lang_wordlist.wordlist['Word'][len_before:len_after]
            log.Log.log(words_not_synched)

        self.db_cache = dbcache.DbCache.get_singleton(
            db_profile      = self.db_profile,
            account_id      = self.account_id,
            bot_id          = self.bot_id,
            bot_lang        = self.lang
        )

        return

    #def set_intent_reply(self, bot_reply):
    #    if type(bot_reply) == botiat.BotIntentAnswerTrData:
    #        self.bot_reply = bot_reply
    #        self.regex_intents = self.bot_reply.get_regex_intent_ids()
    #    else:
    #        errmsg = 'Bot reply setting incorrect type [' + str(type(bot_reply)) + ']'
    #        log.Log.log(errmsg)
    #        raise(Exception(errmsg))

    def get_word_segmentation(
            self,
            txt,
            reply_format='text'
    ):
        txt_split = self.wseg.segment_words(text=su.StringUtils.trim(txt))

        answer = txt_split
        if reply_format == 'json':
            answer = {
                'txt': txt_split
            }

            try:
                answer = json.dumps(obj=answer, ensure_ascii=False).encode(encoding=IntentWrapper.JSON_ENCODING)
            except Exception as ex:
                raise Exception(str(self.__class__) + ' Unable to dump to JSON format for [' + str(answer) + ']' + str(ex))

        return answer

    #
    # Returns the closest top X of the matches, where X <= top
    # THREAD SAFE
    #
    def get_text_class(
            self,
            inputtext,
            top = intEng.IntentEngine.SEARCH_TOPX_RFV,
            reduced_features = False,
            do_segment_inputtext = True,
            chatid = None,
            not_necessary_to_use_training_data_samples = True
    ):
        start_func = None
        if self.do_profiling:
            start_func = prf.Profiling.start()
            if self.verbose >= 2:
                log.Log.log('.  '
                            + '[ChatID=' + str(chatid) + ', Txt=' + inputtext + ']'
                            + ' PROFILING GET TEXT CLASS Start (Reduced Features = ' + str(reduced_features)
                            + '): ' + str(start_func))

        if len(inputtext) > IntentWrapper.MAX_QUESTION_LENGTH:
            if self.verbose >= 2:
                log.Log.log(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': Warning. ChatID [' + str(chatid) + '] message exceeds ' +
                            str(IntentWrapper.MAX_QUESTION_LENGTH) +
                            ' in length. Truncating..')
            inputtext = inputtext[0:IntentWrapper.MAX_QUESTION_LENGTH]

        a = None
        if self.do_profiling:
            a = prf.Profiling.start()
            if self.verbose >= 2:
                log.Log.log('.    '
                            + '[ChatID=' + str(chatid) + ', Txt=' + inputtext + ']'
                            + ' PROFILING Word Segmentation Start: ' + str(a))
        # Segment words first
        chatstr_segmented = su.StringUtils.trim(inputtext)
        if do_segment_inputtext:
            chatstr_segmented = self.wseg.segment_words(text=su.StringUtils.trim(inputtext))
        if self.do_profiling:
            b = prf.Profiling.stop()
            if self.verbose >= 2:
                log.Log.log('.    '
                            + '[ChatID=' + str(chatid) + ', Txt=' + inputtext + ']'
                            + ' PROFILING Word Segmentation End: ' + prf.Profiling.get_time_dif_str(start=a, stop=b))

        if self.verbose > 3:
            log.Log.log(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Segmented Text: [' + chatstr_segmented + ']')

        intent_engine = self.lebot

        df_intent = intent_engine.get_text_class(
            text_segmented            = chatstr_segmented,
            chatid                    = chatid,
            weigh_idf                 = True,
            top                       = top,
            return_match_results_only = True,
            score_min_threshold       = self.min_score_threshold,
            not_necessary_to_use_training_data_samples = not_necessary_to_use_training_data_samples
        )

        if df_intent is None:
            return None

        # Only keep scores > 0
        df_intent = df_intent[df_intent[intEng.IntentEngine.COL_SCORE]>0]

        if df_intent.shape[0] == 0:
            return None

        #
        # Choose which scores to keep.
        #
        top_score = float(df_intent[intEng.IntentEngine.COL_SCORE].loc[df_intent.index[0]])
        df_intent_keep = df_intent[df_intent[intEng.IntentEngine.COL_SCORE] >=
                                   top_score*IntentWrapper.CONSTANT_PERCENT_WITHIN_TOP_SCORE]
        df_intent_keep = df_intent_keep.reset_index(drop=True)

        if self.do_profiling:
            end_func = prf.Profiling.stop()
            log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                             + '.  ' + '[ChatID=' + str(chatid) + ', Txt=' + chatstr_segmented + ']'
                             + ' PROFILING GET TEXT CLASS End (Reduced Features = ' + str(reduced_features)
                             + '): ' + prf.Profiling.get_time_dif_str(start=start_func, stop=end_func))

        return df_intent_keep

    #
    # Forms a JSON response from get_text_class() result
    # THREAD SAFE
    # This only changes the original data frame into a desired format to return,
    # and also looking up the intent name from intent ID, that is all
    #
    def get_json_response(
            self,
            # Only for debugging, tracking purpose
            chatid,
            df_intent,
            # This is set to true by the caller when our RPS (request per second) hits too high
            use_only_cache_data = False
    ):
        start_func = None
        if self.do_profiling:
            start_func = prf.Profiling.start()
            log.Log.debug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                          + '.  ' + '[ChatID=' + str(chatid) + ']'
                          + 'PROFILING Intent JSON Start: ' + str(start_func))

        answer_json = {'matches':{}}

        if df_intent is None:
            return json.dumps(obj=answer_json, ensure_ascii=False).encode(encoding=IntentWrapper.JSON_ENCODING)

        log.Log.debug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                      + ': DF Intent for conversion to json: ')
        log.Log.debug(df_intent)

        a = None
        if self.do_profiling:
            a = prf.Profiling.start()
            log.Log.debug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        +'.    '
                        + '[ChatID=' + str(chatid) + ']'
                        + ' PROFILING Intent JSON (Loop all '
                        + str(df_intent.shape[0]) + ' intents) Start: ' + str(a))

        indexes_intent = df_intent.index.tolist()
        for i in range(0, len(indexes_intent), 1):
            idx = indexes_intent[i]
            intent_class = df_intent[intEng.IntentEngine.COL_COMMAND].loc[idx]
            intent_score = df_intent[intEng.IntentEngine.COL_SCORE].loc[idx]
            intent_confidence = df_intent[intEng.IntentEngine.COL_SCORE_CONFIDENCE_LEVEL].loc[idx]

            intent_name = str(intent_class)

            #
            # This part is slow getting the intent name from intentId, which is why we do caching
            #
            if self.use_db:
                aa = None
                if self.do_profiling:
                    aa = prf.Profiling.start()
                    log.Log.debug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                  + '.      ' + '[ChatID=' + str(chatid) + ']'
                                  + 'PROFILING Intent JSON (Get DB/Cache Intent Name for ' + str(intent_name)
                                  + ') Start: ' + str(aa))

                # Only go to DB if not in cache or already expired
                fetch_from_db = False
                # Key is the Intent ID
                intent_name = self.db_cache.get_intent_name(
                    intent_id           = intent_class,
                    use_only_cache_data = use_only_cache_data
                )

                if self.do_profiling:
                    bb = prf.Profiling.stop()
                    log_msg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                              + '.    ' + '[ChatID=' + str(chatid) + ']' \
                              + ' END PROFILING Intent JSON '
                    if fetch_from_db:
                        log_msg = log_msg + 'DB Intent Name for ' + str(intent_name) + ' (' + str(intent_class) + '): '
                    else:
                        log_msg = log_msg + 'Cache Intent Name for ' + str(intent_name) + ' (' + str(intent_class) + '): '
                    log.Log.critical(log_msg + prf.Profiling.get_time_dif_str(start=aa, stop=bb))

            # JSON can't serialize the returned int64 type
            if type(intent_class) is np.int64:
                intent_class = int(intent_class)

            answer_json['matches'][i+1] = {
                IntentWrapper.JSON_COL_POSITION    : i+1,
                IntentWrapper.JSON_COL_INTENT_NAME : intent_name,
                IntentWrapper.JSON_COL_INTENT_ID   : intent_class,
                IntentWrapper.JSON_COL_SCORE       : float(intent_score),
                IntentWrapper.JSON_COL_CONFIDENCE  : float(intent_confidence)
            }

        if self.do_profiling:
            b = prf.Profiling.stop()
            log.Log.log(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + '.    ' + '[ChatID=' + str(chatid) + ']'
                        + ' PROFILING Intent JSON (loop all ' + str(df_intent.shape[0]) +
                        ' intents) End: ' + prf.Profiling.get_time_dif_str(start=a, stop=b))

        json_reply = None
        try:
            json_reply = json.dumps(obj=answer_json, ensure_ascii=False).encode(encoding=IntentWrapper.JSON_ENCODING)
        except Exception as ex:
            raise Exception(str(self.__class__) + ' Unable to dump to JSON format for [' + str(answer_json) + ']' + str(ex))

        if self.do_profiling:
            end_func = prf.Profiling.stop()
            log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + '.  ' + '[ChatID=' + str(chatid) + ']'
                        + ' PROFILING Intent JSON End: '
                        + prf.Profiling.get_time_dif_str(start=start_func, stop=end_func))

        return json_reply

    #
    # Run this bot!
    #
    def test_run_on_command_line(
            self,
            top     = intEng.IntentEngine.SEARCH_TOPX_RFV,
            verbose = 0
    ):
        if self.lebot is None or self.wseg == None:
            raise Exception('CRMBot not initialized!!')

        while (1):
            chatstr = input("Enter question: ")
            if chatstr == 'quit':
                break

            start_intent = prf.Profiling.start()
            df_com_class = self.get_text_class(
                chatid              = 'xx',
                inputtext           = chatstr,
                top                 = top
            )
            end_intent = prf.Profiling.stop()
            print('PROFILING Intent Time: ' + prf.Profiling.get_time_dif_str(start=start_intent, stop=end_intent))

            start_intent_txt = prf.Profiling.start()
            answer = self.get_json_response(
                chatid    = 'xx',
                df_intent = df_com_class
            )
            end_intent_txt = prf.Profiling.stop()
            print('PROFILING Intent Format Text Time: ' + prf.Profiling.get_time_dif_str(
                start=start_intent_txt, stop=end_intent_txt))

            print(json.loads(answer, encoding=IntentWrapper.JSON_ENCODING))

        return

