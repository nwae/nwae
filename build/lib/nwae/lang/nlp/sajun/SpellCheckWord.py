# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
import nwae.lang.nlp.WordList as wl
from nwae.lang.LangFeatures import LangFeatures
import nwae.utils.UnitTest as ut
from nwae.lang.config.Config import Config
import numpy as np
import pandas as pd
from nwae.lang.nlp.sajun.EditDistance import EditDistance
from nwae.lang.nlp.sajun.TrieNode import TrieNode
import nwae.math.optimization.Eidf as eidf
import os


#
# Проверка правописания (орфографии) единственного слова без никакого контекста
#
class SpellCheckWord:

    COL_CORRECTED_WORD = 'corrected_word'
    COL_EDIT_DISTANCE  = 'edit_distance'
    COL_EIDF_VALUE     = 'eidf_value'

    def __init__(
            self,
            lang,
            # Список слов из словаря или любых
            words_list,
            # Directory and identifier string for looking up EIDF files
            dir_path_model     = None,
            identifier_string  = None,
            # Option to pass in EIDF DataFrame instead of using directory and identifier string
            eidf_dataframe     = None,
            use_word_weighting = True,
            do_profiling       = False
    ):
        self.lang = LangFeatures.map_to_lang_code_iso639_1(
            lang_code = lang
        )
        self.words_list = words_list
        self.dir_path_model = dir_path_model
        self.identifier_string = identifier_string
        self.use_word_weighting = use_word_weighting
        self.eidf_dataframe = eidf_dataframe
        self.do_profiling = do_profiling

        self.trie = TrieNode.build_trie_node(
            words = self.words_list
        )
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Read ' + str(TrieNode.WORD_COUNT) + ' words, '
            + str(TrieNode.NODE_COUNT) + ' trie nodes from wordlist '
            + str(self.words_list[0:50]) + ' (first 50 of ' + str(len(self.words_list)) + ')'
        )

        if not self.use_word_weighting:
            self.eidf_words = None
            self.eidf_value = None
        else:
            try:
                Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Initializing EIDF object.. try to read from file..'
                )
                # Try to read from file
                df_eidf_file = eidf.Eidf.read_eidf_from_storage(
                    data_pd_dataframe = self.eidf_dataframe,
                    dir_path_model    = self.dir_path_model,
                    identifier_string = self.identifier_string,
                    # No need to reorder the words in EIDF file
                    x_name            = None
                )
                Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Successfully Read EIDF from file from directory "' + str(self.dir_path_model)
                    + '" for model "' + str(self.identifier_string) + '".'
                )
                Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': EIDF initialized as:' + str(df_eidf_file)
                )
                self.eidf_words = np.array(df_eidf_file[eidf.Eidf.STORAGE_COL_X_NAME], dtype=str)
                self.eidf_value = np.array(df_eidf_file[eidf.Eidf.STORAGE_COL_EIDF], dtype=float)
            except Exception as ex_eidf:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': No EIDF from file available. Exception ' + str(ex_eidf) + '.'
                Log.error(errmsg)
                raise Exception(errmsg)
        return

    #
    # Описание алгоритма
    #   1. Поиск близких слов через структуру "Префиксное Дерево" (Trie)
    #   2. Ранжирование слов поиска через любую "весовую систему" слов
    #
    # Разница между TrieNode и этого алгоритма находится в филтрировании ресултатов поиска
    # TrieNode через поиск списка слов с весами
    #
    def search_close_words(
            self,
            word,
            # Cost can be any measure of edit distance, e.g. Levenshtein, Damerau-Levenshtein, etc.
            max_cost = 2,
            edit_distance_algo = EditDistance.EDIT_DIST_ALGO_DAMLEV
    ):
        # Returns tuples of (word, edit-distance)
        # E.g. from word bg to [('be',1), ('big',1), ('bag',1), ('brag',2)]
        results = TrieNode.search_close_words(
            trie     = self.trie,
            word     = word,
            max_cost = max_cost,
            edit_distance_algo = edit_distance_algo
        )
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': For word "' + str(word) + '", found trie node matches ' + str(results)
        )
        if (results is None) or (len(results) == 0):
            return None

        #
        # Можно использовать любую весовую систему слов
        #
        corrected_words = []
        edit_distances = []
        eidf_values = []
        for obj in results:
            # The corrected word returned in tuple
            cor_word = obj[0]
            # The edit distance returned in tuple
            edit_dist = obj[1]
            if self.use_word_weighting:
                eidf_val = self.eidf_value[self.eidf_words == cor_word]
                if len(eidf_val) != 1:
                    Log.debugdebug(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': No EIDF value found for corrected word "' + str(cor_word) + '"'
                    )
                    continue
                else:
                    eidf_values.append(round(eidf_val[0], 2))
            else:
                eidf_values.append(None)
            corrected_words.append(cor_word)
            edit_distances.append(edit_dist)

        df = pd.DataFrame({
            SpellCheckWord.COL_CORRECTED_WORD: corrected_words,
            SpellCheckWord.COL_EDIT_DISTANCE:  edit_distances,
            SpellCheckWord.COL_EIDF_VALUE:     eidf_values
        })


        df = df.sort_values(
            by = [SpellCheckWord.COL_EDIT_DISTANCE, SpellCheckWord.COL_EIDF_VALUE],
            ascending = True
        )
        df = df.reset_index(drop=True)
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Corrected words and eidf values: ' + str(df)
        )
        return df
        # return (list(df['corrected_word'][0:3]), list(df['eidf_value'][0:3]))


class SpellCheckWordUnitTest:

    SAMPLE_EIDF = {
        LangFeatures.LANG_TH: pd.DataFrame({
            'x_name': ['เงิน', 'ถอน', 'ถือ', 'ถนน', 'เงา', 'สหรัฐ', 'เชื้อ'],
            'eidf':   [1.55,  1.76,  1.84, 2.11,  2.33,  2.50,   2.55]
        }),
        LangFeatures.LANG_VI: pd.DataFrame({
            'x_name': ['giao diện', 'ăn diện'],
            'eidf':   [1.42,        1.89]
        }),
    }

    def __init__(
            self,
            ut_params
    ):
        self.ut_params = ut_params
        if self.ut_params is None:
            # We only do this for convenience, so that we have access to the Class methods in UI
            self.ut_params = ut.UnitTestParams()

        self.wordlist = {}
        self.spell_corr = {}

        for lang in [LangFeatures.LANG_TH, LangFeatures.LANG_VI]:
            try:
                self.wordlist[lang] = wl.WordList(
                    lang             = lang,
                    dirpath_wordlist = ut_params.dirpath_wordlist,
                    postfix_wordlist = ut_params.postfix_wordlist
                )
                words = self.wordlist[lang].wordlist[wl.WordList.COL_WORD].tolist()
            except Exception as ex:
                # For some languages we might not have the words list
                words = None

            self.spell_corr[lang] = SpellCheckWord(
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

        test_sets = [
            {
                'lang': LangFeatures.LANG_TH,
                'tests': [
                    ['เงน', EditDistance.EDIT_DIST_ALGO_DAMLEV, 1, True, ['เงา', 'เงิน']],
                    ['เงน', EditDistance.EDIT_DIST_ALGO_LEV, 1, True, ['เงา', 'เงิน']],
                    ['ถนอ', EditDistance.EDIT_DIST_ALGO_DAMLEV, 1, True, ['ถอน', 'ถนน', 'ถือ']],
                    ['ถนอ', EditDistance.EDIT_DIST_ALGO_LEV, 1, True, ['ถนน', 'ถือ']],
                    ['สหรฐฯ', EditDistance.EDIT_DIST_ALGO_DAMLEV, 1, True, None],
                    ['สหรฐ', EditDistance.EDIT_DIST_ALGO_DAMLEV, 2, True, ['สหรัฐ']],
                    ['เชื่อ', EditDistance.EDIT_DIST_ALGO_DAMLEV, 1, True, ['เชื้อ']],
                    ['เชื่อ', EditDistance.EDIT_DIST_ALGO_LEV, 1, True, ['เชื้อ']],
                    #
                    # Don't use any EIDF weights
                    #
                    ['เชื่อ', EditDistance.EDIT_DIST_ALGO_DAMLEV, 1, False,
                     ['ชื่อ', 'เชื่อ', 'เชื่อง', 'เชื่อม', 'เชื้อ', 'เดื่อ', 'เทื่อ', 'เบื่อ', 'เผื่อ', 'เพื่อ', 'เฟื่อ', 'เมื่อ', 'เยื่อ', 'เรื่อ', 'เสื่อ', 'เหื่อ']],
                ]
            },
            {
                'lang': LangFeatures.LANG_VI,
                'tests': [
                    ['gaio diện', EditDistance.EDIT_DIST_ALGO_LEV, 1, True, None],
                    ['gaio diện', EditDistance.EDIT_DIST_ALGO_DAMLEV, 1, True, ['giao diện']],
                    ['go diện', EditDistance.EDIT_DIST_ALGO_DAMLEV, 2, True, ['giao diện', 'ăn diện']],
                    #
                    # Don't use any EIDF weights
                    #
                    ['go diện', EditDistance.EDIT_DIST_ALGO_DAMLEV, 2, False,
                     ['giao diện', 'lộ diện', 'sĩ diện', 'tứ diện', 'ăn diện', 'đa diện']],
                ]
            }
        ]

        for x in test_sets:
            lang = x['lang']
            for xx in x['tests']:
                w = xx[0]
                algo = xx[1]
                max_dist = xx[2]
                use_eidf_weights = xx[3]
                expected = xx[4]
                if expected:
                    expected = sorted(expected)

                self.spell_corr[lang].use_word_weighting = use_eidf_weights

                df_search = self.spell_corr[lang].search_close_words(
                    word = w,
                    max_cost = max_dist,
                    edit_distance_algo = algo
                )
                if df_search is not None:
                    close_words = sorted( list( df_search[SpellCheckWord.COL_CORRECTED_WORD] ) )
                    # arr_dist  = list( df_search[SpellCheckWord.COL_EDIT_DISTANCE] )
                    Log.debug('Corrections: ' + str(df_search.values))
                else:
                    close_words = None

                res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                    observed = close_words,
                    expected = expected,
                    test_comment = 'Test "' + str(w) + '", algo "' + str(algo)
                                   + '", close words: ' + str(close_words)
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

    res = SpellCheckWordUnitTest(ut_params=ut_params).run_unit_test()
    exit(res.count_fail)
