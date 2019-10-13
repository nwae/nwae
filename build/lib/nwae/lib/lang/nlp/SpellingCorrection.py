# -*- coding: utf-8 -*-

import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo
import nwae.utils.Profiling as prf
import nwae.lib.lang.nlp.sajun.TrieNode as trienod
import nwae.lib.lang.nlp.WordList as wl
import nwae.lib.lang.LangFeatures as langfeatures
import nwae.lib.math.optimization.Eidf as eidf
import numpy as np


#
# Проверка написания зависит на 2 алгоритма, разбиение слов и расстояние Левенштейна
#
class SpellingCorrection:

    def __init__(
            self,
            words_list,
            dir_path_model,
            identifier_string,
            do_profiling = False
    ):
        self.words_list = words_list
        self.dir_path_model = dir_path_model
        self.identifier_string = identifier_string
        self.do_profiling = do_profiling

        self.trie = trienod.TrieNode.build_trie_node(
            words = self.words_list
        )
        lg.Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Read ' + str(trienod.TrieNode.WORD_COUNT) + ' words, '
            + str(trienod.TrieNode.NODE_COUNT) + ' trie nodes from wordlist '
            + str(self.words_list[0:50]) + ' (first 50 of ' + str(len(self.words_list)) + ')'
        )

        try:
            lg.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Initializing EIDF object.. try to read from file..'
            )
            # Try to read from file
            df_eidf_file = eidf.Eidf.read_eidf_from_storage(
                dir_path_model    = self.dir_path_model,
                identifier_string = self.identifier_string,
                # No need to reorder the words in EIDF file
                x_name            = None
            )
            lg.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Successfully Read EIDF from file. ' + str(df_eidf_file)
            )
            self.eidf_words = np.array(df_eidf_file[eidf.Eidf.STORAGE_COL_X_NAME], dtype=str)
            self.eidf_value = np.array(df_eidf_file[eidf.Eidf.STORAGE_COL_EIDF], dtype=float)
        except Exception as ex_eidf:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': No EIDF from file available. Exception ' + str(ex_eidf) + '.'
            lg.Log.error(errmsg)
            raise Exception(errmsg)

        return

    def do_spelling_correction(
            self,
            text_segmented_arr,
            max_cost = 1
    ):
        start_prf = prf.Profiling.start()

        corrected_text_arr = text_segmented_arr.copy()
        # Get the list of words in the model
        for i in range(len(text_segmented_arr)):
            w = text_segmented_arr[i]
            if w not in self.words_list:
                lg.Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Word "' + str(w) + '" not found in features model. Searching trie node...'
                )
                results = trienod.TrieNode.search(
                    trie = self.trie,
                    word = w,
                    max_cost = max_cost
                )
                lg.Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': For word "' + str(w) + '", found trie node matches ' + str(results)
                )
                best_eidf = np.infty
                best_word = None
                for word_dist in results:
                    eidf_val = self.eidf_value[self.eidf_words==word_dist[0]]
                    if len(eidf_val) != 1:
                        continue
                    if eidf_val[0] < best_eidf:
                        best_eidf = eidf_val[0]
                        best_word = word_dist[0]
                if best_word is not None:
                    corrected_text_arr[i] = best_word
                    lg.Log.info(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Corrected word "' + str(w) + '" to "' + str(best_word)
                        + '" in sentence ' + str(text_segmented_arr) + '.'
                    )

        if self.do_profiling:
            lg.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Spelling correction for ' + str(text_segmented_arr)
                + ' to ' + str(corrected_text_arr) + ' took '
                + str(prf.Profiling.get_time_dif_str(start=start_prf, stop=prf.Profiling.stop()))
            )
        return corrected_text_arr


if __name__ == '__main__':
    import nwae.config.Config as cf
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config
    )
    lg.Log.LOGLEVEL = lg.Log.LOG_LEVEL_INFO

    wl_obj = wl.WordList(
        lang             = langfeatures.LangFeatures.LANG_TH,
        dirpath_wordlist = config.get_config(cf.Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist = config.get_config(cf.Config.PARAM_NLP_POSTFIX_WORDLIST)
    )
    words = wl_obj.wordlist[wl.WordList.COL_WORD].tolist()
    s = ['ฝาก','เงน','ที่','ไหน']

    obj = SpellingCorrection(
        words_list = words,
        dir_path_model = '/usr/local/git/mozig/mozg.nlp/app.data/intent/models',
        identifier_string = config.get_config(param=cf.Config.PARAM_MODEL_IDENTIFIER),
        do_profiling = True
    )

    correction = obj.do_spelling_correction(
        text_segmented_arr = s
    )
    print(correction)

    exit(0)
