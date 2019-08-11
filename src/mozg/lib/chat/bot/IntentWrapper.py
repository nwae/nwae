# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import mozg.lib.math.ml.ModelHelper as modelHelper
import mozg.lib.lang.model.FeatureVector as fv
import mozg.lib.chat.bot.IntentEngine as intEng
import mozg.utils.StringUtils as su
import mozg.lib.lang.nlp.WordSegmentation as ws
import mozg.lib.lang.nlp.SynonymList as sl
import mozg.utils.Log as log
import threading
import json
import numpy as np
import mozg.utils.Profiling as prf
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
            model_name,
            identifier_string,
            dir_path_model,
            lang,
            dir_synonymlist,
            dir_wordlist,
            postfix_wordlist,
            dir_wordlist_app,
            postfix_wordlist_app,
            do_profiling = False,
            min_score_threshold = intEng.IntentEngine.DEFAULT_SCORE_MIN_THRESHOLD
    ):
        #
        # All class variables are constants only to ensure thread safety
        # except for the intent cache
        #
        self.model_name = model_name
        self.identifier_string = identifier_string
        self.dir_path_model = dir_path_model

        self.lang = su.StringUtils.trim(lang.lower())
        self.dir_synonymlist = su.StringUtils.trim(dir_synonymlist)
        self.dir_wordlist = dir_wordlist
        self.postfix_wordlist = postfix_wordlist
        self.dir_wordlist_app = dir_wordlist_app
        self.postfix_wordlist_app = postfix_wordlist_app

        self.do_profiling = do_profiling
        self.min_score_threshold = min_score_threshold

        self.math_engine = None
        self.wseg = None
        self.bot_reply = None

        # Thread safe mutex, not used
        self.mutex = threading.Lock()
        return

    #
    # Only during initialization we modify class variables, after this, no other writes should happen.
    #
    def init(self):
        self.model = modelHelper.ModelHelper.get_model(
            model_name=self.model_name,
            identifier_string=self.identifier_string,
            dir_path_model=self.dir_path_model,
            training_data=None
        )
        self.lebot.do_background_load()

        self.wseg = ws.WordSegmentation(
            lang = self.lang,
            dirpath_wordlist = self.dir_wordlist,
            postfix_wordlist = self.postfix_wordlist,
            do_profiling = self.do_profiling
        )
        # Add application wordlist
        self.wseg.add_wordlist(
            dirpath=self.dir_wordlist_app,
            postfix=self.postfix_wordlist_app
        )

        self.synonymlist = sl.SynonymList(
            lang                = self.lang,
            dirpath_synonymlist = self.dir_synonymlist,
            postfix_synonymlist = '.synonymlist.txt'
        )
        self.synonymlist.load_synonymlist()

        # Add synonym list to wordlist (just in case they are not synched)
        len_before = self.wseg.lang_wordlist.wordlist.shape[0]
        self.wseg.add_wordlist(
            dirpath     = None,
            postfix     = None,
            array_words = list(self.synonymlist.synonymlist['Word'])
        )
        len_after = self.wseg.lang_wordlist.wordlist.shape[0]
        if len_after - len_before > 0:
            log.Log.warning(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ": These words not in word list but in synonym list:"
            )
            words_not_synched = self.wseg.lang_wordlist.wordlist['Word'][len_before:len_after]
            log.Log.log(words_not_synched)

        self.db_cache = dbcache.DbIntentCache.get_singleton(
            db_profile      = self.db_profile,
            account_id      = self.account_id,
            bot_id          = self.bot_id,
            bot_lang        = self.lang,
            cache_intent_name    = True,
            cache_intent_answers = False,
            cache_intent_regex   = False,
            cache_expiry_time_secs = 5*60
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
                raise Exception(
                    str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                    + ': Unable to dump to JSON format for [' + str(answer) + ']' + str(ex)
                )

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
        intent_engine = self.model

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
            log.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + '[Botkey=' + str(self.bot_key) + ', ChatID=' + str(chatid) + ', Txt=' + str(inputtext) + ']'
                + ' PROFILING GET TEXT CLASS End (Reduced Features = ' + str(reduced_features)
                + '): ' + str(prf.Profiling.get_time_dif_str(start=start_func, stop=end_func))
            )

        return df_intent_keep

    def convert_text_to_math_object(
            self,
            inputtext,
            chatid = None
    ):
        start_func = None
        if self.do_profiling:
            start_func = prf.Profiling.start()

        if len(inputtext) > IntentWrapper.MAX_QUESTION_LENGTH:
            log.Log.warning(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Warning. ChatID [' + str(chatid) + '] message exceeds '
                + str(IntentWrapper.MAX_QUESTION_LENGTH)
                + ' in length. Truncating..'
            )
            inputtext = inputtext[0:IntentWrapper.MAX_QUESTION_LENGTH]

        a = None
        if self.do_profiling:
            a = prf.Profiling.start()

        # Segment words first
        text_segmented = self.wseg.segment_words(text=su.StringUtils.trim(inputtext))
        if self.do_profiling:
            b = prf.Profiling.stop()
            log.Log.info(
                '.    '
                + '[Identifier=' + str(self.identifier_string)
                + ', ChatID=' + str(chatid) + ', Txt=' + str(inputtext) + ']'
                + ' PROFILING Word Segmentation End: '
                + str(prf.Profiling.get_time_dif_str(start=a, stop=b))
            )

        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Segmented Text: [' + text_segmented + ']'
        )

        #
        # This routine is thread safe, no writes to class variables, just read.
        #
        if self.do_profiling:
            start_func = prf.Profiling.start()

        space_profiling = '      '

        #
        # Replace words with root words
        # This step uses synonyms and replaces say 存钱, 入钱, 入款, all with the standard 存款
        # This will reduce training data without needing to put all versions of the same thing.
        #
        if self.do_profiling:
            a = prf.Profiling.start()
            log.Log.debug(
                '.' + space_profiling
                + '[Identifier "' + str(self.identifier_string)
                + '", ChatID "' + str(chatid) + '", Txt "' + str(text_segmented) + '"]'
                + ' PROFILING Intent (replace root words) Start: ' + str(a)
            )

        text_normalized = self.synonymlist.normalize_text(text_segmented=text_segmented)
        text_normalized = text_normalized.lower()
        log.Log.debugdebug('#')
        log.Log.debugdebug('# TEXT NORMALIZATION')
        log.Log.debugdebug('#')
        log.Log.debugdebug('Text [' + text_segmented + '] normalized to [' + text_normalized + ']')
        if self.do_profiling:
            b = prf.Profiling.stop()
            log.Log.info(
                '.' + space_profiling
                + '[Identifier "' + str(self.identifier_string)
                + '", ChatID "' + str(chatid) + '", Txt "' + str(text_segmented) + '"]'
                + ' PROFILING Intent (replace root words): ' + str(prf.Profiling.get_time_dif_str(a, b))
            )

        keywords_all = list(self.model.get_model_features())
        log.Log.debugdebug('Keywords all: ' + str(keywords_all))

        #
        # Convert sentence to a mathematical object (feature vector)
        #
        log.Log.debugdebug('#')
        log.Log.debugdebug('# FEATURE VECTOR & NORMALIZATION')
        log.Log.debugdebug('#')

        if self.do_profiling:
            a = prf.Profiling.start()

        model_fv = fv.FeatureVector()
        model_fv.set_freq_feature_vector_template(list_symbols=keywords_all)

        # Get feature vector of text
        try:
            df_fv = model_fv.get_freq_feature_vector(text=text_normalized)
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Exception occurred calculating FV for "' + str(text_normalized)\
                     + '": Exception "' + str(ex)\
                     + '. Using FV Template ' + str(model_fv.get_fv_template())\
                     + ', FV Weights ' + str(model_fv.get_fv_weights()) \
                     + ', identifier "' + str(self.identifier_string) + '".'
            log.Log.critical(errmsg)
            raise Exception(ex)

        # This creates a single row matrix that needs to be transposed before matrix multiplications
        # ndmin=2 will force numpy to create a 2D matrix instead of a 1D vector
        # For now we make it 1D first
        fv_text_1d = np.array(df_fv['Frequency'].values, ndmin=1)
        if fv_text_1d.ndim != 1:
            raise Exception(str(self.__class__) + ': Expected a 1D vector, got ' + str(fv_text_1d.ndim) + 'D!')
        fv_text_normalized_1d = np.array(df_fv['FrequencyNormalized'].values, ndmin=1)
        if fv_text_normalized_1d.ndim != 1:
            raise Exception(str(self.__class__) + ': Expected a 1D vector, got ' + str(fv_text_normalized_1d.ndim) + 'D!')
        log.Log.debug(fv_text_1d)
        log.Log.debug(fv_text_normalized_1d)

        if self.do_profiling:
            b = prf.Profiling.stop()
            log.Log.info(
                '.' + space_profiling
                + '[Identifier "' + str(self.identifier_string)
                + '", ChatID "' + str(chatid) + '", Txt "' + str(text_segmented) + '"]'
                + ' PROFILING Intent (FV & Normalization): ' + str(prf.Profiling.get_time_dif_str(a, b))
            )

        return fv_text_1d

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
            log.Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + '[Botkey=' + str(self.bot_key) + ', ChatID=' + str(chatid) + ']'
                + 'PROFILING Intent JSON Start: ' + str(start_func)
            )

        answer_json = {'matches':{}}

        if df_intent is None:
            return json.dumps(obj=answer_json, ensure_ascii=False).encode(encoding=IntentWrapper.JSON_ENCODING)

        log.Log.debug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                      + ': DF Intent for conversion to json: ')
        log.Log.debug(df_intent)

        a = None
        if self.do_profiling:
            a = prf.Profiling.start()
            log.Log.debug(
                '.    '
                + '[Botkey=' + str(self.bot_key) + ', ChatID=' + str(chatid) + ']'
                + ' PROFILING Intent JSON (Loop all '
                + str(df_intent.shape[0]) + ' intents) Start: ' + str(a)
            )

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
                    log.Log.debug(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + '[Botkey=' + str(self.bot_key) + ', ChatID=' + str(chatid) + ']'
                        + 'PROFILING Intent JSON (Get DB/Cache Intent Name for ' + str(intent_name)
                        + ') Start: ' + str(aa)
                    )

                # Only go to DB if not in cache or already expired
                fetch_from_db = False
                # Key is the Intent ID
                try:
                    intent_name = self.db_cache.get_intent_name(
                        intent_id           = intent_class,
                        use_only_cache_data = use_only_cache_data
                    )
                except Exception as ex:
                    errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                             + ': Could not get intent name for intent ID ' + str(intent_class)\
                             + '. Got exception ' + str(ex)
                    log.Log.error(errmsg)
                    intent_name = str(intent_class)

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
            log.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + '[Botkey=' + str(self.bot_key) + ', ChatID=' + str(chatid) + ']'
                + ' PROFILING Intent JSON (loop all ' + str(df_intent.shape[0])
                + ' intents) End: ' + str(prf.Profiling.get_time_dif_str(start=a, stop=b))
            )

        json_reply = None
        try:
            json_reply = json.dumps(obj=answer_json, ensure_ascii=False).encode(encoding=IntentWrapper.JSON_ENCODING)
        except Exception as ex:
            raise Exception(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' Unable to dump to JSON format for [' + str(answer_json) + ']' + str(ex)
            )

        if self.do_profiling:
            end_func = prf.Profiling.stop()
            log.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': [Botkey=' + str(self.bot_key) + ', ChatID=' + str(chatid) + ']'
                + ' PROFILING Intent JSON End: '
                + str(prf.Profiling.get_time_dif_str(start=start_func, stop=end_func))
            )

        return json_reply

    #
    # Run this bot!
    #
    def test_run_on_command_line(
            self,
            top     = intEng.IntentEngine.SEARCH_TOPX_RFV,
            verbose = 0
    ):
        if self.model is None or self.wseg == None:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Model not initialized!! Model ' + str(self.model)
                + ', Word Segmentation ' + str(self.wseg)
            )

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


if __name__ == '__main__':

