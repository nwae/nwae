# -*- coding: utf-8 -*-

import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo
import nwae.utils.Profiling as prf
import nwae.lib.lang.nlp.sajun.TrieNode as trienod
import nwae.lib.lang.nlp.WordList as wl
import nwae.lib.lang.LangFeatures as langfeatures
import nwae.lib.lang.LangHelper as langhelper
import nwae.lib.math.optimization.Eidf as eidf
import numpy as np


#
# Проверка написания зависит на 2 алгоритма, разбиение слов и расстояние Левенштейна
#
class SpellingCorrection:

    NO_SPACE_DELIMITER_LANGUAGES = (
        langfeatures.LangFeatures.LANG_TH
    )

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
            dir_path_model,
            identifier_string,
            do_profiling = False
    ):
        self.lang = lang
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
                + ': Successfully Read EIDF from file for model "' + str(self.identifier_string) + '".'
            )
            lg.Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': EIDF:' + str(df_eidf_file)
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

        #
        # For languages with no space as word/syllable delimiter like Thai, if we detect
        # single character words, no point to look for matches, instead we need to join
        # them to either the previous word or next word.
        #

        corrected_text_arr = []
        # Get the list of words in the model
        for i in range(len(text_segmented_arr)):
            w = text_segmented_arr[i]
            if (w is None) or (len(w) == 0):
                continue

            #
            # We join words only for alphabet type languages without space as word separator
            # The concept works as follows, take for example this Thai sentence
            #  ['ฝาก', 'เงน', 'ที่', 'ไหน']
            # The word 'เงน' has been split means this word is either in the full dictionary or
            # they are separate single alphabets joined together by the word tokenizer.
            # We then compare this to our common words list to get the best match.
            #
            possible_words = [w]
            is_concatenate_single_alphabet_to_neighboring_words = False
            if self.lang in SpellingCorrection.NO_SPACE_DELIMITER_LANGUAGES:
                if len(w) == 1:
                    possible_words = []
                    if i>0 is not None:
                        possible_words.append(text_segmented_arr[i-1]+w)
                    if i < len(text_segmented_arr)-1:
                        possible_words.append(w+text_segmented_arr[i+1])
                    lg.Log.debug(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Checking appended words "' + str(possible_words) + '".'
                    )
                    is_concatenate_single_alphabet_to_neighboring_words = True

            best_word_final = w
            best_eidf_final = 99999
            #
            # There can only be multiple possibilities if we are concatenating single alphabets in
            # languages without space delimiter to previous & next word
            #
            for w_possible in possible_words:
                if w_possible not in self.words_list:
                    lg.Log.debug(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Word "' + str(w_possible) + '" not found in features model. Searching trie node...'
                    )
                    (best_word, best_eidf) = self.find_best_match(
                        w = w_possible,
                        max_cost = max_cost
                    )
                    if best_word is not None:
                        if best_eidf_final > best_eidf:
                            if is_concatenate_single_alphabet_to_neighboring_words:
                                # If same with previous word due to single alphabet concatenation, ignore
                                if (i>0) and (best_word == text_segmented_arr[i-1]):
                                    continue
                                # If same with next word due to concatenating single alphabet, ignore
                                if (i<len(text_segmented_arr)-1) and (best_word == text_segmented_arr[i+1]):
                                    continue
                            best_eidf_final = best_eidf
                            best_word_final = best_word
                            lg.Log.info(
                                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                + ': Corrected word "' + str(w_possible) + '" using best EIDF value '
                                + str(best_eidf) + ' to "' + str(best_word)
                                + '" in sentence ' + str(text_segmented_arr) + '.'
                            )
                else:
                    best_word_final = w_possible
            corrected_text_arr.append(best_word_final)

        if self.do_profiling:
            lg.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Spelling correction for ' + str(text_segmented_arr)
                + ' to ' + str(corrected_text_arr) + ' took '
                + str(prf.Profiling.get_time_dif_str(start=start_prf, stop=prf.Profiling.stop()))
            )
        return corrected_text_arr

    def find_best_match(
            self,
            w,
            max_cost
    ):
        results = trienod.TrieNode.search(
            trie     = self.trie,
            word     = w,
            max_cost = max_cost
        )
        if (results is None) or (len(results) == 0):
            return None

        lg.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': For word "' + str(w) + '", found trie node matches ' + str(results)
        )
        best_eidf = np.infty
        best_word = None
        for word_dist in results:
            eidf_val = self.eidf_value[self.eidf_words == word_dist[0]]
            if len(eidf_val) != 1:
                continue
            if eidf_val[0] < best_eidf:
                best_eidf = eidf_val[0]
                best_word = word_dist[0]
        return (best_word, round(best_eidf,2))


if __name__ == '__main__':
    import nwae.config.Config as cf
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config
    )
    lg.Log.LOGLEVEL = lg.Log.LOG_LEVEL_INFO

    lang = langfeatures.LangFeatures.LANG_TH

    ret_obj = langhelper.LangHelper.get_word_segmenter(
        lang             = lang,
        dirpath_wordlist = config.get_config(param=cf.Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_WORDLIST),
        dirpath_app_wordlist = config.get_config(param=cf.Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_app_wordlist = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        dirpath_synonymlist  = config.get_config(param=cf.Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist  = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        # We can only allow root words to be words from the model features
        allowed_root_words   = None,
        do_profiling         = False
    )
    wseg = ret_obj.wseg
    synonymlist = ret_obj.snnlist

    test_sent = [
        'มีโปรไหนใช้ได้กัลยุสนี้',
        'ฝากเงนที่ไหน'
    ]

    wl_obj = wl.WordList(
        lang             = lang,
        dirpath_wordlist = config.get_config(cf.Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist = config.get_config(cf.Config.PARAM_NLP_POSTFIX_WORDLIST)
    )
    words = wl_obj.wordlist[wl.WordList.COL_WORD].tolist()

    obj = SpellingCorrection(
        lang              = lang,
        words_list        = words,
        dir_path_model    = '/usr/local/git/mozig/mozg.nlp/app.data/intent/models',
        identifier_string = config.get_config(param=cf.Config.PARAM_MODEL_IDENTIFIER),
        do_profiling      = True
    )

    for s in test_sent:
        seg = wseg.segment_words(
            text = s,
            return_array_of_split_words = True
        )
        print('"' + s + '" segmented to ' + str(seg))

        correction = obj.do_spelling_correction(
            text_segmented_arr = seg
        )
        print(correction)

    exit(0)
