# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
import nwae.utils.Profiling as prf
from nwae.lang.nlp.sajun.SpellCheckWord import SpellCheckWord, SpellCheckWordUnitTest
from nwae.lang.nlp.WordList import WordList
from nwae.lang.LangHelper import LangHelper
from nwae.lang.LangFeatures import LangFeatures
import nwae.utils.UnitTest as ut
from nwae.lang.config.Config import Config


#
# Описание Алгоритма
#   1. Правописание Отдельных Слов Без Контекста.
#      Если слово найдется в массиве self.words_list, то проверка слова не продолжится.
#      А если нашло, то поиск через Левенштейн с максимальным расстоянием = 1 или 2 пройдет.
#      Возможно подходящее слово не найдет, и слово не изменится.
#   2. Правописание Предложения с Контекстом
#      Нам нужны сбор соответствующий с приложением предложений.
#      С Этими предложениями, мы обучаем последовательность слов в предложениях используя
#      LSTM и т.п.
#      TODO
#
class SpellCheckSentence:

    def __init__(
            self,
            lang,
            # This words list can be a full dictionary (for languages with natural space
            # as word separator) or just a common words list in our usage application context
            # for languages without a natural space as word separator.
            # This is because for languages without space, the word splitting itself might
            # be wrong, and the spelling correction algorithm might need to look at previous
            # or subsequent words.
            words_list,
            # Directory and identifier string for looking up EIDF files
            dir_path_model     = None,
            identifier_string  = None,
            # Option to pass in EIDF DataFrame instead of using directory and identifier string
            eidf_dataframe     = None,
            do_profiling       = False
    ):
        self.lang = LangFeatures.map_to_lang_code_iso639_1(
            lang_code = lang
        )
        self.words_list = words_list
        self.dir_path_model = dir_path_model
        self.identifier_string = identifier_string
        self.eidf_dataframe = eidf_dataframe
        self.do_profiling = do_profiling

        self.sep_type = LangFeatures().get_word_separator_type(
            lang = lang
        )
        self.spell_check_word = SpellCheckWord(
            lang              = self.lang,
            words_list        = self.words_list,
            dir_path_model    = self.dir_path_model,
            identifier_string = self.identifier_string,
            eidf_dataframe    = self.eidf_dataframe,
            do_profiling      = self.do_profiling
        )
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Initialize Spelling Correction for "' + str(lang)
            + '", separator type "' + str(self.sep_type) + '"'
        )
        return

    def check(
            self,
            # Text array e.g. ['это', 'мое', 'предложение']
            text_segmented_arr,
            max_cost = 1
    ):
        start_prf = prf.Profiling.start()

        #
        # 1. Правописание Отдельных Слов Без Контекста.
        #

        len_text = len(text_segmented_arr)
        corrected_text_arr = []
        # Get the list of words in the model
        for i in range(len_text):
            w = text_segmented_arr[i]
            if (w is None) or (len(w) == 0):
                continue

            w_corrected = w

            if w not in self.words_list:
                df_correction_matches = self.spell_check_word.search_close_words(
                    word     = w,
                    max_cost = max_cost
                )
                # Забрать первое слово с максимальным весом
                if df_correction_matches is not None:
                    # В случае индексы не в порядке
                    top_loc = df_correction_matches.index[0]
                    w_corrected = df_correction_matches[SpellCheckWord.COL_CORRECTED_WORD].loc[top_loc]

            corrected_text_arr.append(w_corrected)

        #
        #   2. Правописание Предложения с Контекстом
        #
        # TODO

        if self.do_profiling:
            ms = 1000 * prf.Profiling.get_time_dif_secs(start=start_prf, stop=prf.Profiling.stop())
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Spelling correction for ' + str(text_segmented_arr)
                + ' to ' + str(corrected_text_arr) + ' took '
                + str(round(ms,2)) + 'ms'
            )
        return corrected_text_arr


class SpellCheckSentenceUnitTest:

    def __init__(
            self,
            ut_params
    ):
        self.ut_params = ut_params
        if self.ut_params is None:
            # We only do this for convenience, so that we have access to the Class methods in UI
            self.ut_params = ut.UnitTestParams()

        self.word_segmenter = {}
        self.synonymlist = {}
        self.wordlist = {}
        self.spell_corr = {}

        for lang in [LangFeatures.LANG_TH]:
            ret_obj = LangHelper.get_word_segmenter(
                lang                 = lang,
                dirpath_wordlist     = ut_params.dirpath_wordlist,
                postfix_wordlist     = ut_params.postfix_wordlist,
                dirpath_app_wordlist = ut_params.dirpath_app_wordlist,
                postfix_app_wordlist = ut_params.postfix_app_wordlist,
                dirpath_synonymlist  = ut_params.dirpath_synonymlist,
                postfix_synonymlist  = ut_params.postfix_synonymlist,
                # We can only allow root words to be words from the model features
                allowed_root_words   = None,
                do_profiling         = False
            )
            self.word_segmenter[lang] = ret_obj.wseg
            self.synonymlist[lang] = ret_obj.snnlist

            self.wordlist[lang] = WordList(
                lang             = lang,
                dirpath_wordlist = ut_params.dirpath_wordlist,
                postfix_wordlist = ut_params.postfix_wordlist
            )
            words = self.wordlist[lang].wordlist[WordList.COL_WORD].tolist()

            self.spell_corr[lang] = SpellCheckSentence(
                lang              = lang,
                words_list        = words,
                eidf_dataframe    = SpellCheckWordUnitTest.SAMPLE_EIDF[lang],
                do_profiling      = True
            )

        return

    def run_unit_test(
            self
    ):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)

        lang = LangFeatures.LANG_TH
        test_sent = [
            # Case words segmented correctly to ['มี', 'เงน', 'ที่', 'ไหน'] and 'เงน' corrected to 'เงิน'
            ['มีเงนที่ไหน',
             ['มี', 'เงิน', 'ที่', 'ไหน']],
            # ['การแพร่ระบาดของเชื้อไวรัสโควิด-19',
            #  ['การ', 'แพร่', 'ระบาด', 'ของ', 'เชื้อ', 'ไวรัส', 'โค', 'วิด', '-19']],
            # ['ในทั่วโลกยังเพิ่มขึ้นไม่หยุด',
            #  ['ใน', 'ทั่ว', 'โลก', 'ยัง', 'เพิ่ม', 'ขึ้น', 'ไม่', 'หยุด']]
        ]

        for obj in test_sent:
            s = obj[0]
            arr_expected = obj[1]

            seg = self.word_segmenter[lang].segment_words(
                text = s,
                return_array_of_split_words = True
            )
            Log.debug(
                '"' + s + '" segmented to ' + str(seg)
            )

            arr_cor = self.spell_corr[lang].check(
                text_segmented_arr=seg
            )
            Log.debug('Corrections array: ' + str(arr_cor))
            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = arr_cor,
                expected = arr_expected,
                test_comment = 'Test "' + str(s) + '" to ' + str(arr_cor)
            ))

        return res_final


if __name__ == '__main__':
    config = Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class       = Config,
        default_config_file = Config.CONFIG_FILE_PATH_DEFAULT
    )
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1

    lang = LangFeatures.LANG_TH

    ut_params = ut.UnitTestParams(
        dirpath_wordlist     = config.get_config(param=Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist     = config.get_config(param=Config.PARAM_NLP_POSTFIX_WORDLIST),
        dirpath_app_wordlist = config.get_config(param=Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_app_wordlist = config.get_config(param=Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        dirpath_synonymlist  = config.get_config(param=Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist  = config.get_config(param=Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        dirpath_model        = None,
    )

    res = SpellCheckSentenceUnitTest(ut_params=ut_params).run_unit_test()
    exit(res.count_fail)
