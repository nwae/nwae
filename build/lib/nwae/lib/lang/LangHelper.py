# --*-- coding: utf-8 --*--

import nwae.utils.Log as lg
from inspect import getframeinfo, currentframe
import nwae.lib.lang.nlp.WordList as wl
import nwae.lib.lang.nlp.SynonymList as sl
import nwae.lib.lang.nlp.WordSegmentation as ws


class LangHelper:

    @staticmethod
    def get_word_segmenter(
            lang,
            dirpath_wordlist,
            postfix_wordlist,
            dirpath_app_wordlist,
            postfix_app_wordlist,
            dirpath_synonymlist,
            postfix_synonymlist,
            # If this is not None, then the synonym list will only choose root words from here
            allowed_root_words = None,
            do_profiling = False
    ):
        wseg_obj = ws.WordSegmentation(
            lang             = lang,
            dirpath_wordlist = dirpath_wordlist,
            postfix_wordlist = postfix_wordlist,
            do_profiling     = do_profiling
        )
        # Add application wordlist
        wseg_obj.add_wordlist(
            dirpath = dirpath_app_wordlist,
            # This is a general application wordlist file, shared between all
            postfix = postfix_app_wordlist,
        )

        # We need synonyms to normalize all text with "rootwords"
        sl_obj = sl.SynonymList(
            lang                = lang,
            dirpath_synonymlist = dirpath_synonymlist,
            postfix_synonymlist = postfix_synonymlist
        )
        sl_obj.load_synonymlist(
            allowed_root_words = allowed_root_words
        )

        len_before = wseg_obj.lang_wordlist.wordlist.shape[0]
        wseg_obj.add_wordlist(
            dirpath     = None,
            postfix     = None,
            array_words = list(sl_obj.synonymlist[sl.SynonymList.COL_WORD])
        )
        len_after = wseg_obj.lang_wordlist.wordlist.shape[0]
        if len_after - len_before > 0:
            words_not_synched = wseg_obj.lang_wordlist.wordlist[sl.SynonymList.COL_WORD][len_before:len_after]
            lg.Log.warning(
                str(LangHelper.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': These words not in word list but in synonym list:\n\r' + str(words_not_synched)
            )

        class retclass:
            def __init__(self, wseg, snnlist):
                self.wseg = wseg
                self.snnlist = snnlist

        return retclass(
            wseg = wseg_obj,
            snnlist = sl_obj
        )

