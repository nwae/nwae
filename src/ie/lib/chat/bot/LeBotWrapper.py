# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import ie.lib.chat.bot.Intent as lb
import ie.lib.chat.bot.BotIntentAnswerTrData as botiat
import ie.lib.util.StringUtils as su
import ie.lib.lang.nlp.WordSegmentation as ws
import ie.lib.util.Log as log
import re
import time
import datetime as dt
import threading
import json
import ie.lib.chat.bot.BotIntentAnswerTrData as br
import ie.lib.chat.chatsession.Chat as chat


#
# Wrap LeBot here, and put state management
# THREAD SAFE CLASS
#   Make sure there is no class variable that is modified during function calls.
#
class LeBotWrapper:

    CONSTANT_PERCENT_WITHIN_TOP_SCORE = 0.5
    MAX_QUESTION_LENGTH = 100

    JSON_ENCODING = 'utf-8'

    def __init__(self,
                 lang,
                 brand,
                 dir_rfv_commands,
                 dir_synonymlist,
                 dir_wordlist,
                 postfix_wordlist,
                 dir_wordlist_app,
                 postfix_wordlist_app,
                 min_score_threshold = lb.Intent.CONFIDENCE_LEVEL_1_SCORE):
        #
        # All class variables are constants only to ensure thread safety
        #
        self.lang = su.StringUtils.trim(lang.lower())
        self.brand = su.StringUtils.trim(brand.lower())
        self.dir_rfv_commands = su.StringUtils.trim(dir_rfv_commands)
        self.dir_synonymlist = su.StringUtils.trim(dir_synonymlist)

        self.dir_wordlist = dir_wordlist
        self.postfix_wordlist = postfix_wordlist
        self.dir_wordlist_app = dir_wordlist_app
        self.postfix_wordlist_app = postfix_wordlist_app

        self.min_score_threshold = min_score_threshold

        self.lebot = None
        self.wseg = None
        self.bot_reply = None
        self.verbose = 0

        # Thread safe mutex, not used
        self.mutex = threading.Lock()
        return

    #
    # Only during initialization we modify class variables, after this, no other writes should happen.
    #
    def init(self, verbose=0):
        self.verbose = verbose

        # Initialize AI/NLP Bot Engine
        self.lebot = lb.Intent(lang=self.lang,
                              brand=self.brand,
                              dir_rfv_commands=self.dir_rfv_commands,
                              dirpath_synonymlist=self.dir_synonymlist)

        self.lebot.load_rfv_commands_from_file()

        self.wseg = ws.WordSegmentation(lang=self.lang,
                                        dirpath_wordlist=self.dir_wordlist,
                                        postfix_wordlist=self.postfix_wordlist)
        # Add application wordlist
        self.wseg.add_wordlist(dirpath=self.dir_wordlist_app,
                               postfix=self.postfix_wordlist_app)

        # Add synonym list to wordlist (just in case they are not synched)
        len_before = self.wseg.lang_wordlist.wordlist.shape[0]
        self.wseg.add_wordlist(dirpath=None, postfix=None, array_words=list(self.lebot.synonymlist.synonymlist['Word']))
        len_after = self.wseg.lang_wordlist.wordlist.shape[0]
        if len_after - len_before > 0:
            log.Log.log("Warning. These words not in word list but in synonym list:")
            words_not_synched = self.wseg.lang_wordlist.wordlist['Word'][len_before:len_after]
            log.Log.log(words_not_synched)

        return

    def set_intent_reply(self, bot_reply):
        if type(bot_reply) == botiat.BotIntentAnswerTrData:
            self.bot_reply = bot_reply
            self.regex_intents = self.bot_reply.get_regex_intent_ids()
        else:
            errmsg = 'Bot reply setting incorrect type [' + str(type(bot_reply)) + ']'
            log.Log.log(errmsg)
            raise(Exception(errmsg))

    #
    # Returns the closest top X of the matches, where X <= top
    # THREAD SAFE
    #
    def get_text_class(
            self,
            inputtext,
            top = lb.Intent.SEARCH_TOPX_RFV
    ):

        if len(inputtext) > LeBotWrapper.MAX_QUESTION_LENGTH:
            log.Log.log(str(self.__class__) + ' Warning. Message exceeds ' +
                        str(LeBotWrapper.MAX_QUESTION_LENGTH) +
                        ' in length. Truncating..')
            inputtext = inputtext[0:LeBotWrapper.MAX_QUESTION_LENGTH]

        chatstr_segmented = self.wseg.segment_words(text=su.StringUtils.trim(inputtext))
        if self.verbose > 1:
            log.Log.log('Segmented Text: [' + chatstr_segmented + ']')

        df_intent = self.lebot.get_text_class(
            text_segmented            = chatstr_segmented,
            weigh_idf                 = True,
            top                       = top,
            return_match_results_only = True,
            score_min_threshold       = self.min_score_threshold,
            verbose                   = self.verbose
        )

        if df_intent is None:
            return None

        # Only keep scores > 0
        df_intent = df_intent[df_intent[lb.Intent.COL_SCORE]>0]

        if df_intent.shape[0] == 0:
            return None

        #
        # Choose which scores to keep.
        #
        top_score = float(df_intent['Score'].loc[0])
        df_intent_keep = df_intent[df_intent[lb.Intent.COL_SCORE] >= top_score*LeBotWrapper.CONSTANT_PERCENT_WITHIN_TOP_SCORE]
        df_intent_keep = df_intent_keep.reset_index(drop=True)
        return df_intent_keep

    #
    # Forms a formatted text answer from get_text_class() result
    # THREAD SAFE
    #
    def get_text_answer(self, df_intent, reply_format='text'):
        answer = ''
        answer_json = {'matches':{}}

        if df_intent is None:
            if reply_format == 'json':
                return json.dumps(obj=answer_json, ensure_ascii=False).encode(encoding=LeBotWrapper.JSON_ENCODING)
            else:
                return None

        if self.verbose >= 2:
            log.Log.log(df_intent)

        for i in range(0, df_intent.shape[0], 1):
            intent_class = df_intent[lb.Intent.COL_COMMAND].loc[i]
            intent_score = df_intent[lb.Intent.COL_SCORE].loc[i]
            intent_confidence = df_intent[lb.Intent.COL_SCORE_CONFIDENCE_LEVEL].loc[i]

            if reply_format == 'text':
                if i > 0:
                    answer = answer + '\n'

                answer = answer +\
                         str(i+1) + '. ' +\
                         'Intent=' + intent_class +\
                         ', Score=' + str(intent_score) +\
                         ', Confidence=' + str(intent_confidence)
            elif reply_format == 'json':
                answer_json['matches'][i+1] = {
                    'intent'    : intent_class,
                    'score'     : float(intent_score),
                    'confidence': float(intent_confidence)
                }
            else:
                raise Exception(str(self.__class__) + ' Unrecognized reply format requested [' + str(reply_format) + ']')

        if reply_format == 'json':
            try:
                answer = json.dumps(obj=answer_json, ensure_ascii=False).encode(encoding=LeBotWrapper.JSON_ENCODING)
            except Exception as ex:
                raise Exception(str(self.__class__) + ' Unable to dump to JSON format for [' + str(answer_json) + ']' + str(ex))

        return answer

    #
    # The difference with this function is that it maintains a conversation, remembering state
    # THREAD SAFE
    #
    def new_msg(self, msg, botname, chat_hist, remove_internal_intents = True):
        # TODO Do we need to lock here?
        lock = False
        if lock:
            self.mutex.acquire()

        if self.verbose > 1:
            timestamp_start = dt.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            log.Log.log(str(timestamp_start) + ' Locking function call for bot [' + botname + ']..')

        if (msg is None) and (chat_hist is None):
            bot_welcome_msg = self.bot_reply.get_random_reply(intent_id=br.BotIntentAnswerTrData.INTENT_WELCOME)
            # TODO Add chat id into string, to identify which chat for client
            if lock:
                self.mutex.release()
            return bot_welcome_msg

        last_server_reply = None
        if not chat_hist is None:
            try:
                df_bot_lines = chat_hist[chat_hist[chat.Chat.CHAT_HISTORY_COL_SPEAKER_NAME]==botname]
                df_bot_lines = df_bot_lines.reset_index(drop=True)
                l = df_bot_lines.shape[0]
                last_server_reply = str(df_bot_lines[chat.Chat.CHAT_HISTORY_COL_MESSAGE].loc[l-1])
            except Exception as ex:
                log.Log.log(str(self.__class__) + ' Error occurred looking into chat log history')
                log.Log.log(ex)

        answer_msg = ''
        intent_answer = ''

        #
        # User entered a number (0-99), means he is choosing an item from menu of last message sent
        #
        is_menu_choice = False

        if re.search(pattern='^[0-9]{1,2}$', string=msg) and (last_server_reply is not None):
            menu_choice_number = int(msg)

            # First check is the intent choices due to bot unsure
            # The choices links need to be separated by newlines if we don't put flags=re.DOTALL
            fa = re.findall(pattern='[\t\s]*[0-9]+[.].*Intent=.*, Score=.*[\n\r]*', string=last_server_reply)
            if fa:
                if len(fa) >= menu_choice_number:
                    intent_line = fa[menu_choice_number - 1]
                    intent_line = su.StringUtils.trim(intent_line)
                    sr = re.search(pattern='Intent=(.*), Score=', string=intent_line)
                    if sr:
                        # Only if more than one intent means he is choosing from list of possible intents
                        if len(fa) > 1:
                            intent_id = sr.group(1)
                            intent_answer = self.bot_reply.get_random_reply(intent_id=intent_id)
                            if intent_answer is None:
                                intent_answer = 'Cannot find answer for intent [' + intent_id + '] in bot config.'
                            is_menu_choice = True
                else:
                    log.Log.log('Menu number ' + str(menu_choice_number) + ' not valid!')

            # Now check for intentlinks
            if not is_menu_choice:
                # The intent links need to be separated by newlines if we don't put flags=re.DOTALL
                fa = re.findall(pattern='<intentlink>.*</intentlink>', string=last_server_reply)

                if len(fa) >= menu_choice_number:
                    intent = fa[menu_choice_number - 1]
                    # Remove the markup
                    intent = intent.replace('<intentlink>', '')
                    intent = intent.replace('</intentlink>', '')
                    answer_msg = '[' + intent + ']\n'
                    intent_id = self.bot_reply.get_intent_id(intent=intent)
                    if intent_id is None:
                        intent_answer = 'Cannot find answer for intent [' + intent + '] in bot config'
                    else:
                        intent_answer = self.bot_reply.get_random_reply(intent_id=intent_id)
                    is_menu_choice = True

        if not is_menu_choice:
            #
            # We try to match regex Intents first
            #
            is_regex_intent = False
            for int_id in self.regex_intents:
                try:
                    regex = self.bot_reply.get_regex(intent_id=int_id)
                    m = re.search(pattern=regex, string=msg)
                    if m:
                        # TODO: Reply with the intent answer
                        is_regex_intent = True
                        log.Log.log('Match regex for intent ID [' + int_id + '], ' + m.group(1) + ']')
                        intent_answer = self.bot_reply.get_random_reply(intent_id=int_id)
                except Exception as ex:
                    log.Log.log('Exception occurred trying to match regex intents.')
                    log.Log.log(ex)
                    is_regex_intent = False

            if not is_regex_intent:
                # Get bot answer
                df_com_class = self.get_text_class(inputtext=msg)
                answer_msg_lebot = self.get_text_answer(df_intent=df_com_class)

                if answer_msg_lebot is None:
                    answer_msg = self.bot_reply.get_random_reply(
                        intent_id=br.BotIntentAnswerTrData.INTENT_NOANSWER) + '\n'
                else:
                    answer_msg = answer_msg_lebot
                    try:
                        # Get intent ID (command in our legacy naming) and answer
                        intent_id = df_com_class[lb.Intent.COL_COMMAND].loc[0]
                        # Only show answer if there is only 1 intent detected
                        if df_com_class.shape[0] == 1:
                            intent_answer = self.bot_reply.get_random_reply(intent_id=intent_id)
                    except Exception as ex:
                        log.Log.log('Error occurred when trying to get intent answer..')
                        log.Log.log(ex)
                        self.mutex.release()
                        return None

        bot_answer = answer_msg
        if intent_answer != '':
            if bot_answer != '':
                bot_answer = bot_answer + '\n\n' + str(intent_answer)
            else:
                bot_answer = str(intent_answer)

        if self.verbose > 1:
            timestamp_end = dt.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            log.Log.log(str(timestamp_end) + ' Releasing lock for function call for bot [' + botname + ']..')

        if lock:
            self.mutex.release()
        return bot_answer

    #
    # Run this bot!
    #
    def run(self, top=lb.Intent.SEARCH_TOPX_RFV, verbose=0):
        if self.lebot is None or self.wseg == None:
            raise Exception('CRMBot not initialized!!')

        while (1):
            chatstr = input("Enter question: ")
            if chatstr == 'quit':
                break
            df_com_class = self.get_text_class(inputtext=chatstr, top=top)

            answer = self.get_text_answer(df_intent=df_com_class)

            print(answer)

        return

