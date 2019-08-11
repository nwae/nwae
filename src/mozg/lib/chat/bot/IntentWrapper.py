# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import mozg.lib.math.ml.ModelHelper as modelHelper
import mozg.lib.lang.model.FeatureVector as fv
import mozg.utils.StringUtils as su
import mozg.lib.lang.nlp.WordSegmentation as ws
import mozg.lib.lang.nlp.SynonymList as sl
import mozg.utils.Log as log
import threading
import json
import numpy as np
import mozg.utils.Profiling as prf
from inspect import currentframe, getframeinfo
import mozg.lib.math.NumpyUtil as npUtil


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

    SEARCH_TOPX_RFV = 5
    DEFAULT_SCORE_MIN_THRESHOLD = 5

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
            min_score_threshold = DEFAULT_SCORE_MIN_THRESHOLD
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

        self.__init()
        return

    #
    # Only during initialization we modify class variables, after this, no other writes should happen.
    #
    def __init(self):
        self.model = modelHelper.ModelHelper.get_model(
            model_name=self.model_name,
            identifier_string=self.identifier_string,
            dir_path_model=self.dir_path_model,
            training_data=None
        )
        self.model.load_model_parameters()
        #self.lebot.do_background_load()

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
        return

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
            top = SEARCH_TOPX_RFV,
            chatid = None
    ):
        fv_text_2d = self.convert_text_to_math_object(
            inputtext = inputtext,
            chatid    = chatid
        )

        pred = self.model.predict_class(
            x = fv_text_2d,
            top = top,
            include_match_details = True
        )
        match_details = pred.match_details
        return match_details

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
        fv_text_2d = npUtil.NumpyUtil.convert_dimension(
            arr    = fv_text_1d,
            to_dim = 2
        )
        if fv_text_2d.ndim != 2:
            raise Exception(str(self.__class__) + ': Expected a 2D vector, got ' + str(fv_text_2d.ndim) + 'D!')
        log.Log.debug(fv_text_2d)

        if self.do_profiling:
            b = prf.Profiling.stop()
            log.Log.info(
                '.' + space_profiling
                + '[Identifier "' + str(self.identifier_string)
                + '", ChatID "' + str(chatid) + '", Txt "' + str(text_segmented) + '"]'
                + ' PROFILING Intent (FV & Normalization): ' + str(prf.Profiling.get_time_dif_str(a, b))
            )

        return fv_text_2d


if __name__ == '__main__':
    import mozg.ConfigFile as cf
    cf.ConfigFile.get_cmdline_params_and_init_config()
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_INFO

    obj = IntentWrapper(
        model_name = modelHelper.ModelHelper.MODEL_NAME_HYPERSPHERE_METRICSPACE,
        identifier_string = 'botkey_db_mario.production.accountid_4.botid_22.lang_cn',
        dir_path_model    = '/Users/mark.tan/git/mozg.nlp/app.data/intent/models',
        lang              = 'cn',
        dir_synonymlist   = cf.ConfigFile.DIR_SYNONYMLIST,
        dir_wordlist      = cf.ConfigFile.DIR_WORDLIST,
        postfix_wordlist  = cf.ConfigFile.POSTFIX_WORDLIST,
        dir_wordlist_app  = cf.ConfigFile.DIR_APP_WORDLIST,
        postfix_wordlist_app = cf.ConfigFile.POSTFIX_APP_WORDLIST
    )

    c = obj.get_text_class(
        inputtext = '存款提款扫码'
    )
    print(c)
