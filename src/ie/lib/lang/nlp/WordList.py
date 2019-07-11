#!/use/bin/python
# --*-- coding: utf-8 --*--

# !!! Will work only on Python 3 and above

import re
import pandas as pd
import mozg.common.util.FileUtils as futil
import mozg.common.util.StringUtils as sutil
import ie.lib.lang.LangFeatures as lf
import ie.lib.lang.nlp.LatinEquivalentForm as lef
import ie.lib.lang.characters.LangCharacters as langchar
import mozg.common.util.Log as log

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
            postfix_wordlist = '-wordlist.txt',
            verbose = 0
    ):
        self.lang = lang

        self.dirpath_wordlist = dirpath_wordlist
        self.postfix_wordlist = postfix_wordlist

        self.verbose = verbose

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
        log.Log.log('Syllable split token is [' + self.syl_split_token + ']')

        self.load_wordlist()
        return

    def load_wordlist(
            self
    ):
        if self.wordlist is None:
            self.wordlist = self.load_list(
                dirpath = self.dirpath_wordlist,
                postfix = self.postfix_wordlist
            )
            self.update_ngrams()
        return

    def update_ngrams(self):
        # Get the unique length unigrams
        max_length = max( set(self.wordlist[WordList.COL_NGRAM_LEN]) )
        max_length = min(max_length, WordList.MAX_NGRAMS)

        for i in range(1, max_length+1, 1):
            condition = self.wordlist[WordList.COL_NGRAM_LEN] == i
            self.ngrams[i] = self.wordlist[WordList.COL_WORD][condition].tolist()
            if self.verbose >= 3:
                log.Log.log('Ngrams[' + str(i) + '] (list len = ' + str(len(self.ngrams[i])) + '):')
                log.Log.log(self.ngrams[i])

        return

    def append_wordlist(
            self,
            dirpath     = None,
            postfix     = None,
            array_words = None,
    ):
        if self.verbose >= 3:
            log.Log.log('Initial wordlist length = ' + str(self.wordlist.shape[0]) + '.')
        wordlist_additional = None
        if array_words is not None:
            wordlist_additional = self.load_list(
                dirpath     = None,
                postfix     = None,
                array_words = array_words
            )
        else:
            wordlist_additional = self.load_list(
                dirpath = dirpath,
                postfix = postfix
            )
        # Join general and application wordlist
        self.wordlist = self.wordlist.append(wordlist_additional)
        # Remove duplicates
        self.wordlist = self.wordlist.drop_duplicates(subset=[WordList.COL_WORD])
        if self.verbose >= 3:
            log.Log.log('Final wordlist length = ' + str(self.wordlist.shape[0]) + '.')

        self.update_ngrams()

        return

    # General function to load wordlist or stopwords
    def load_list(
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
            if self.verbose >= 1:
                log.Log.log('Loading list for [' + self.lang + ']' + '[' + filepath + ']')

            fu = futil.FileUtils()
            content = fu.read_text_file(filepath)

            #
            # We will not tolerate missing file. This is because when adding user wordlists,
            # it will significantly effect Bot quality.
            # For example if file is missing, we will miss out on user keywords like "必威" or
            # "云闪付" or "彩金", etc, which will severely reduce Bot efficiency.
            #
            if len(content) == 0:
                raise Exception('File [' + filepath + '] is empty or non-existent!!')

            if self.verbose >= 1:
                log.Log.log('   Read ' + str(len(content)) + ' lines.')

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

        # Convert to pandas data frame
        df_wordlist = pd.DataFrame({
            WordList.COL_WORD        : words,
            WordList.COL_WORD_NUMBER : measures,
            WordList.COL_LATIN       : words_latin,
            WordList.COL_LATIN_NUMBER: measures_latin
        })
        if self.syl_split_token == '':
            log.Log.log('Ngram length for ' + self.lang + ' is just the WORD length.')
            df_wordlist[WordList.COL_NGRAM_LEN] = pd.Series(data=words).str.len()
        else:
            log.Log.log('Ngram length for ' + self.lang + ' is just the SYLLABLE length.')
            df_wordlist[WordList.COL_NGRAM_LEN] = pd.Series(data=words).str.replace('[^ ]','').str.len() + 1

        df_wordlist = df_wordlist.drop_duplicates(subset=[WordList.COL_WORD])
        # Need to reset indexes, otherwise some index will be missing
        df_wordlist = df_wordlist.reset_index(drop=True)

        #if self.verbose >= 3:
        #    log.Log.log(df_wordlist)

        return df_wordlist



if __name__ == '__main__':

    dirpath_wordlist = '/Users/mark.tan/git/mozg.nlp/nlp.data/wordlist'

    for lang in ['cn', 'th']:
        wl = WordList(
            lang             = lang,
            dirpath_wordlist = dirpath_wordlist,
            postfix_wordlist = '-wordlist.txt',
            verbose          = 3,
        )
        wl.load_wordlist()
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
            dirpath_wordlist = dirpath_wordlist,
            postfix_wordlist = '-stopwords.txt',
            verbose          = 3
        )
        sw.load_wordlist()
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

