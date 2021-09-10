# --*-- coding: utf-8 --*--

import nwae.utils.Log as lg
from inspect import getframeinfo, currentframe
import nwae.lang.LangFeatures as lf
import nwae.lang.nlp.WordList as wl
import nwae.lang.nlp.SynonymList as sl
import nwae.lang.nlp.WordSegmentation as ws


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
            do_profiling       = False,
    ):
        lang_std = lf.LangFeatures.map_to_lang_code_iso639_1(
            lang_code = lang
        )
        wseg_obj = ws.WordSegmentation(
            lang             = lang_std,
            dirpath_wordlist = dirpath_wordlist,
            postfix_wordlist = postfix_wordlist,
            do_profiling     = do_profiling,
        )

        # We need synonyms to normalize all text with "rootwords"
        sl_obj = sl.SynonymList(
            lang                = lang_std,
            dirpath_synonymlist = dirpath_synonymlist,
            postfix_synonymlist = postfix_synonymlist
        )
        sl_obj.load_synonymlist(
            allowed_root_words = allowed_root_words
        )

        if not wseg_obj.have_simple_word_separator:
            # Add application wordlist
            wseg_obj.add_wordlist(
                dirpath = dirpath_app_wordlist,
                # This is a general application wordlist file, shared between all
                postfix = postfix_app_wordlist,
            )

            len_before = wseg_obj.get_wordlist_length()

            # We assume words from model features are the same with allowed root words
            words_from_model_features = allowed_root_words
            if words_from_model_features is not None:
                wseg_obj.add_wordlist(
                    dirpath     = None,
                    postfix     = None,
                    array_words = words_from_model_features
                )
                lg.Log.important(
                    str(LangHelper.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Lang "' + str(lang_std) + '". Added ' + str(len(words_from_model_features))
                    + ' words from model features to wordlist.'
                )

            wseg_obj.add_wordlist(
                dirpath     = None,
                postfix     = None,
                array_words = sl_obj.get_synonym_list_words()
            )
            len_after = wseg_obj.get_wordlist_length()
            if len_after - len_before > 0:
                words_not_synched = wseg_obj.lang_wordlist.wordlist[wl.WordList.COL_WORD][len_before:len_after]
                lg.Log.warning(
                    str(LangHelper.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Lang "' + str(lang_std) + '". These words not in word list but in model features & synonym list: '
                    + str(words_not_synched.values)
                )

        class retclass:
            def __init__(self, wseg, snnlist):
                self.wseg = wseg
                self.snnlist = snnlist

        return retclass(
            wseg = wseg_obj,
            snnlist = sl_obj
        )


if __name__ == '__main__':
    import nwae.config.Config as cf
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config,
        default_config_file='/usr/local/git/nwae/nwae.lang/app.data/config/default.cf'
    )
    retobj = LangHelper.get_word_segmenter(
        lang                 = lf.LangFeatures.LANG_ZH,
        dirpath_wordlist     = config.get_config(param=cf.Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist     = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_WORDLIST),
        dirpath_app_wordlist = config.get_config(param=cf.Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_app_wordlist = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        dirpath_synonymlist  = config.get_config(param=cf.Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist  = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        # During training, we don't care about allowed root words
        # We just take the first word in the synonym list as root
        # word. Only during detection, we need to do this to make
        # sure that whatever word we replace is in the feature list.
        allowed_root_words = None,
        do_profiling = False
    )
    ws = retobj.wseg
    synonymlist = retobj.snnlist

    text = '香港抗议 盘点本周最新出现的五个重大情况'
    print(ws.segment_words(text=text, return_array_of_split_words=True, look_from_longest=True))

