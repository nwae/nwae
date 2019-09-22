#!/use/bin/python
# --*-- coding: utf-8 --*--

# !!! Will work only on Python 3 and above

import re
import pandas as pd
import nwae.utils.FileUtils as futil
import nwae.utils.StringUtils as sutil
import nwae.lib.lang.LangFeatures as lf
import nwae.lib.lang.nlp.LatinEquivalentForm as lef
import nwae.lib.lang.characters.LangCharacters as langchar
import nwae.utils.Log as log
from inspect import currentframe, getframeinfo


#
# Any simple word list, stopwords, etc. that can be read line by line as a single word, with no other properties
#
# Example Word Lists
#   - Chinese: i)  https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.big
#              ii) https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.small
#              Both above taken from the project jieba - https://github.com/fxsjy/jieba
#   - Thai:    https://github.com/pureexe/thai-wordlist
#
class WordList:

    COL_WORD = 'Word'
    COL_WORD_NUMBER = 'WordNumber'
    COL_LATIN = 'WordLatin'
    COL_LATIN_NUMBER = 'WordLatinNumber'
    # In the case of languages with syllable separator (e.g. Vietnamese, Korean),
    # the word length is actually the ngram length. In other cases, it is just the word length
    COL_NGRAM_LEN = 'WordLen'

    MAX_NGRAMS = 20

    def __init__(
            self,
            lang,
            dirpath_wordlist,
            postfix_wordlist
    ):
        self.lang = lang

        self.dirpath_wordlist = dirpath_wordlist
        self.postfix_wordlist = postfix_wordlist

        self.wordlist = None
        # Break the wordlist into ngrams for faster word segmentation
        self.ngrams = {}

        self.lang_feature = lf.LangFeatures()
        self.syl_split_token = self.lang_feature.get_split_token(
            lang  = self.lang,
            level = 'syllable'
        )
        if self.syl_split_token is None:
            self.syl_split_token = ''
        log.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Lang "' + str(lang) + '" syllable split token is "' + self.syl_split_token + '"')

        self.__load_wordlist()
        return

    def __load_wordlist(
            self
    ):
        if self.wordlist is None:
            self.wordlist = self.__load_list(
                dirpath = self.dirpath_wordlist,
                postfix = self.postfix_wordlist
            )
            self.update_ngrams()
        return

    def update_ngrams(self):
        try:
            # Get the unique length unigrams
            max_length = int( max( set(self.wordlist[WordList.COL_NGRAM_LEN]) ) )
            max_length = int( min(max_length, WordList.MAX_NGRAMS ) )

            if max_length < 1:
                errmsg =\
                    ': Lang "' + str(self.lang) + '" have no ngrams!! Max ngram length = ' + str(max_length) + '!!'
                log.Log.error(errmsg)
                raise Exception(errmsg)

            for i in range(1, max_length+1, 1):
                condition = self.wordlist[WordList.COL_NGRAM_LEN] == i
                self.ngrams[i] = self.wordlist[WordList.COL_WORD][condition].tolist()
                log.Log.debugdebug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Lang "' + str(self.lang)
                    + '" ngrams [' + str(i) + '] (list len = ' + str(len(self.ngrams[i])) + '):\n\r'
                    + str(self.ngrams[i])
                )
        except Exception as ex:
            errmsg = \
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                + ': Exception updating ngrams list for lang "' + str(self.lang)\
                + '", exception message: ' + str(ex) + '.'
            log.Log.error(errmsg)
            raise Exception(errmsg)

        return

    def append_wordlist(
            self,
            dirpath     = None,
            postfix     = None,
            array_words = None,
    ):
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Lang ' + str(self.lang) + '" Initial wordlist length = ' + str(self.wordlist.shape[0]) + '.'
            + ', appending wordlist:\n\r' + str(array_words)
        )
        wordlist_additional = None
        if array_words is not None:
            wordlist_additional = self.__load_list(
                dirpath     = None,
                postfix     = None,
                array_words = array_words
            )
        else:
            wordlist_additional = self.__load_list(
                dirpath = dirpath,
                postfix = postfix
            )
        # Join general and application wordlist
        self.wordlist = self.wordlist.append(wordlist_additional)
        # Remove duplicates
        self.wordlist = self.wordlist.drop_duplicates(subset=[WordList.COL_WORD])
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Lang "' + str(self.lang) + '" final wordlist length = ' + str(self.wordlist.shape[0]) + '.'
        )

        self.update_ngrams()

        return

    # General function to load wordlist or stopwords
    def __load_list(
            self,
            dirpath,
            postfix,
            array_words = None
    ):

        lc = langchar.LangCharacters()

        content = None
        if array_words is not None:
            content = array_words
        else:
            filepath = dirpath + '/' + self.lang + postfix
            log.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Lang "' + str(self.lang) + '" loading list for [' + self.lang + ']' + '[' + filepath + ']'
            )

            fu = futil.FileUtils()
            content = fu.read_text_file(filepath)

            #
            # We will not tolerate missing file. This is because when adding user wordlists,
            # it will significantly effect Bot quality.
            # For example if file is missing, we will miss out on user keywords like "必威" or
            # "云闪付" or "彩金", etc, which will severely reduce Bot efficiency.
            #
            if len(content) == 0:
                warning_msg =\
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                    + ': Lang "' + str(self.lang) + '" file [' + filepath + '] is empty or non-existent!!'
                log.Log.warning(warning_msg)

            log.Log.info(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                         + ': Lang "' + str(self.lang) + '" read ' + str(len(content)) + ' lines.')

        words = []
        # Convert words to some number
        measures = []
        # In Latin form
        words_latin = []
        measures_latin = []
        #
        # TODO Don't loop
        #
        for line in content:
            line = sutil.StringUtils.trim(line)
            # Remove empty lines
            if len(line)<=0: continue
            # Remove comment lines starting with '#'
            if re.match(u'^#', line): continue

            word = line

            # Make sure to convert all to Unicode
            # word = unicode(word, encoding='utf-8')
            # Remove empty words
            if len(word)<=0: continue

            words.append(word)
            measures.append(lc.convert_string_to_number(word))

            wordlatin = lef.LatinEquivalentForm.get_latin_equivalent_form(lang=self.lang, word=word)
            words_latin.append(wordlatin)

            measures_latin.append(lc.convert_string_to_number(wordlatin))

        try:
            # Convert to pandas data frame
            df_wordlist = pd.DataFrame({
                WordList.COL_WORD        : words,
                WordList.COL_WORD_NUMBER : measures,
                WordList.COL_LATIN       : words_latin,
                WordList.COL_LATIN_NUMBER: measures_latin,
                WordList.COL_NGRAM_LEN   : [0] * len(words)
            })
            if self.syl_split_token == '':
                log.Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Lang "' + str(self.lang) + '" ngram length for ' + self.lang + ' is just the WORD length.'
                )
                if df_wordlist.shape[0] > 0:
                    df_wordlist[WordList.COL_NGRAM_LEN] = pd.Series(data=words).str.len()
            else:
                log.Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Lang "' + str(self.lang) + '" ngram length for ' + self.lang + ' is just the SYLLABLE length.'
                )
                if df_wordlist.shape[0] > 0:
                    df_wordlist[WordList.COL_NGRAM_LEN] = pd.Series(data=words).str.replace('[^ ]','').str.len() + 1

            df_wordlist = df_wordlist.drop_duplicates(subset=[WordList.COL_WORD])
            # Need to reset indexes, otherwise some index will be missing
            df_wordlist = df_wordlist.reset_index(drop=True)

            return df_wordlist
        except Exception as ex:
            errmsg =\
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                + ': Lang ' + str(self.lang) + ' wordlist loading exception: ' + str(ex) + '.'
            log.Log.error(errmsg)
            raise Exception(errmsg)



if __name__ == '__main__':
    import nwae.config.Config as cf
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config
    )

    for lang in ['cn', 'th']:
        wl = WordList(
            lang             = lang,
            dirpath_wordlist = config.get_config(cf.Config.PARAM_NLP_DIR_WORDLIST),
            postfix_wordlist = config.get_config(cf.Config.PARAM_NLP_POSTFIX_WORDLIST)
        )
        log.Log.log('')
        log.Log.log( lang + ': Read Word List ' + str(wl.wordlist.shape[0]) + " lines" )
        s = ''
        sm = ''
        s_latin = ''
        sm_latin = ''
        df = wl.wordlist
        for i in range(0, min(100, df.shape[0]), 1):
            s = s + str(df[WordList.COL_WORD].loc[i]) + ','
            sm = sm + str(df[WordList.COL_WORD_NUMBER].loc[i]) + ','
            s_latin = s_latin + str(df[WordList.COL_LATIN].loc[i]) + ','
            sm_latin = sm_latin + str(df[WordList.COL_LATIN_NUMBER].loc[i]) + ','

        log.Log.log ( s )
        log.Log.log ( sm )
        log.Log.log ( s_latin )
        log.Log.log ( sm_latin )

        # Stopwords
        sw = WordList(
            lang             = lang,
            dirpath_wordlist = config.get_config(cf.Config.PARAM_NLP_DIR_WORDLIST),
            postfix_wordlist = config.get_config(cf.Config.PARAM_NLP_POSTFIX_STOPWORDS)
        )
        log.Log.log('')
        log.Log.log ( lang + ': Read Stopword List ' + str(sw.wordlist.shape[0]) + " lines" )
        s = ''
        sm = ''
        s_latin = ''
        sm_latin = ''
        df = sw.wordlist
        for i in range(0, min(100, df.shape[0]), 1):
            s = s + str(df[WordList.COL_WORD].loc[i]) + ','
            sm = sm + str(df[WordList.COL_WORD_NUMBER].loc[i]) + ','
            s_latin = s_latin + str(df[WordList.COL_LATIN].loc[i]) + ','
            sm_latin = sm_latin + str(df[WordList.COL_LATIN_NUMBER].loc[i]) + ','

        log.Log.log ( s )
        log.Log.log ( sm )
        log.Log.log ( s_latin )
        log.Log.log ( sm_latin )

