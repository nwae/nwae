#!/use/bin/python
# --*-- coding: utf-8 --*--

# !!! Will work only on Python 3 and above

import re
import pandas as pd
import numpy as np
import nwae.utils.FileUtils as futil
import nwae.utils.StringUtils as su
import nwae.lib.lang.nlp.LatinEquivalentForm as lef
import nwae.lib.lang.characters.LangCharacters as langchar
import nwae.utils.Log as log
from inspect import getframeinfo, currentframe


class SynonymList:

    COL_ROOTWORD      = 'RootWord'
    COL_WORD          = 'Word'
    COL_WORD_NO       = 'WordNumber'
    COL_WORD_LATIN    = 'WordLatin'
    COL_WORD_LATIN_NO = 'WordLatinNumber'

    def __init__(
            self,
            lang,
            dirpath_synonymlist,
            postfix_synonymlist
    ):
        self.lang = lang

        self.dirpath_synonymlist = dirpath_synonymlist
        self.postfix_synonymlist = postfix_synonymlist

        self.synonymlist = None
        self.synonymlist_words = None
        self.synonymlist_rootwords = None
        return

    def load_synonymlist(
            self,
            # List of words that should be in the first position (index 0)
            allowed_root_words = None
    ):
        if self.synonymlist is None:
            self.synonymlist = self.__load_list(
                dirpath = self.dirpath_synonymlist,
                postfix = self.postfix_synonymlist,
                allowed_root_words = allowed_root_words
            )
            self.synonymlist_words = np.array(self.synonymlist[SynonymList.COL_WORD], dtype=str)
            self.synonymlist_rootwords = np.array(self.synonymlist[SynonymList.COL_ROOTWORD], dtype=str)

        return

    # General function to load wordlist or stopwords
    def __load_list(
            self,
            dirpath,
            postfix,
            allowed_root_words = None
    ):
        lc = langchar.LangCharacters()

        filepath = dirpath + '/' + self.lang + postfix
        log.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Loading list for lang "' + self.lang + '" from file path "' + filepath + '"'
        )

        fu = futil.FileUtils()
        content = fu.read_text_file(filepath)

        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Synonym Read ' + str(len(content)) + ' lines.'
            + ' List of main words (' + str(type(allowed_root_words)) + '):\n\r' + str(allowed_root_words)
        )

        words = []
        rootwords = []
        # Convert words to some number
        measures = []
        # In Latin form
        words_latin = []
        measures_latin = []
        for line in content:
            line = str(line)
            line = su.StringUtils.trim(line)
            # Remove empty lines
            if len(line)<=0: continue
            # Remove comment lines starting with '#'
            if re.match(pattern='^#', string=line): continue

            linewords = line.split(sep=',')
            if type(linewords) not in (list, tuple):
                log.Log.warning(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Line "' + str(linewords) + '" not split into list or tuple but type '
                    + str(type(linewords)) + '.'
                )
                continue

            log.Log.debugdebug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Processing line "' + str(linewords) + '".'
            )
            rootword = None
            for i in range(len(linewords)):
                rootword_test = su.StringUtils.trim(linewords[i]).lower()
                if len(rootword_test) <= 0:
                    continue

                if allowed_root_words is not None:
                    if rootword_test not in allowed_root_words:
                        log.Log.warning(
                            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': Word "' + str(rootword_test) + '" is not in allowed root words list! Trying next word in line.'
                        )
                        continue

                # root word is in list of main words, or list main words is None
                rootword = rootword_test
                break

            # If 1st word is empty, ignore entire line
            if rootword is None:
                log.Log.warning(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Line "' + str(linewords) + '" ignored, no root words found!'
                )
                continue

            log.Log.debugdebug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Line "' + str(linewords) + '" root word "' + str(rootword) + '"'
            )

            for j in range(0, len(linewords), 1):
                word = su.StringUtils.trim(linewords[j]).lower()
                # Make sure to convert all to Unicode
                # word = unicode(word, encoding='utf-8')
                # Remove empty words
                if len(word)<=0: continue

                rootwords.append(rootword)
                words.append(word)
                measures.append(lc.convert_string_to_number(word))

                wordlatin = lef.LatinEquivalentForm.get_latin_equivalent_form(lang=self.lang, word=word)
                words_latin.append(wordlatin)

                measures_latin.append(lc.convert_string_to_number(wordlatin))

        # Convert to pandas data frame
        df_synonyms = pd.DataFrame({
            SynonymList.COL_ROOTWORD:      rootwords,
            SynonymList.COL_WORD:          words,
            SynonymList.COL_WORD_NO:       measures,
            SynonymList.COL_WORD_LATIN:    words_latin,
            SynonymList.COL_WORD_LATIN_NO: measures_latin
        })
        log.Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Successfully loaded synonym list:\n\r' + str(df_synonyms)
        )
        df_synonyms = df_synonyms.drop_duplicates(subset=[SynonymList.COL_WORD])
        # Need to reset indexes, otherwise some index will be missing
        df_synonyms = df_synonyms.reset_index(drop=True)

        return df_synonyms

    # Replace with root words, thus normalizing the text
    def normalize_text_array(
            self,
            # A word array. e.g. ['this','is','a','sentence','or','just','any','word','array','.']
            text_segmented_array
    ):
        words_normalized = []
        #
        # Replace words with root words
        #
        for i in range(0, len(text_segmented_array), 1):
            word = text_segmented_array[i]
            if len(word)==0:
                continue

            rootword = []
            if self.synonymlist_words is not None:
                rootword = self.synonymlist_rootwords[self.synonymlist_words == word].tolist()
                # Causes the noisy warning from numpy if dataframe empty,
                # 'FutureWarning: elementwise comparison failed; returning scalar instead,
                # but in the future will perform elementwise comparison result = method(y)'
                # rootword = self.synonymlist[self.synonymlist[SynonymList.COL_WORD]==word][SynonymList.COL_ROOTWORD].values
            if len(rootword)==1:
                log.Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Rootword of "' + str(word) + '" is "' + str(rootword[0]) + '"'
                )
                words_normalized.append(rootword[0])
            else:
                words_normalized.append(word)

        return words_normalized


if __name__ == '__main__':
    import nwae.config.Config as cf
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config
    )

    for lang in ['cn', 'th']:
        sl = SynonymList(
            lang                = lang,
            dirpath_synonymlist = config.get_config(param=cf.Config.PARAM_NLP_DIR_SYNONYMLIST),
            postfix_synonymlist = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_SYNONYMLIST)
        )
        sl.load_synonymlist()
        print(sl.synonymlist)

