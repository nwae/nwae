#!/use/bin/python
# --*-- coding: utf-8 --*--

# !!! Will work only on Python 3 and above

import re
import pandas as pd
import ie.lib.util.FileUtils
import ie.lib.util.StringUtils
import ie.lib.lang.characters.LangCharacters
import ie.lib.lang.LangFeatures
import ie.lib.lang.nlp.LatinEquivalentForm as lef
import ie.lib.lang.characters.LangCharacters as langchar
import ie.lib.util.Log as log

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

    def __init__(self, lang, dirpath_wordlist, postfix_wordlist='-wordlist.txt'):
        self.lang = lang

        self.dirpath_wordlist = dirpath_wordlist
        self.postfix_wordlist = postfix_wordlist

        self.wordlist = None
        return

    def load_wordlist(self, verbose=0):
        if self.wordlist is None:
            self.wordlist = self.load_list(dirpath=self.dirpath_wordlist, postfix=self.postfix_wordlist, verbose=verbose)
        return

    def append_wordlist(self, dirpath=None, postfix=None, array_words=None, verbose=0):
        if verbose >= 1:
            log.Log.log('Initial wordlist length = ' + str(self.wordlist.shape[0]) + '.')
        wordlist_additional = None
        if array_words is not None:
            wordlist_additional = self.load_list(dirpath=None, postfix=None, array_words=array_words, verbose=verbose)
        else:
            wordlist_additional = self.load_list(dirpath=dirpath, postfix=postfix, verbose=verbose)
        # Join general and application wordlist
        self.wordlist = self.wordlist.append(wordlist_additional)
        # Remove duplicates
        self.wordlist = self.wordlist.drop_duplicates(subset=['Word'])
        if verbose>= 1:
            log.Log.log('Final wordlist length = ' + str(self.wordlist.shape[0]) + '.')
        return

    # General function to load wordlist or stopwords
    def load_list(self, dirpath, postfix, array_words=None, verbose=0):

        lc = langchar.LangCharacters()

        content = None
        if array_words is not None:
            content = array_words
        else:
            filepath = dirpath + '/' + self.lang + postfix
            if verbose >= 1:
                log.Log.log('Loading list for [' + self.lang + ']' + '[' + filepath + ']')

            fu = ie.lib.util.FileUtils.FileUtils()
            content = fu.read_text_file(filepath)

            #
            # We will not tolerate missing file. This is because when adding user wordlists,
            # it will significantly effect Bot quality.
            # For example if file is missing, we will miss out on user keywords like "必威" or
            # "云闪付" or "彩金", etc, which will severely reduce Bot efficiency.
            #
            if len(content) == 0:
                raise Exception('File [' + filepath + '] is empty or non-existent!!')

            if verbose >= 1:
                log.Log.log('   Read ' + str(len(content)) + ' lines.')

        words = []
        # Convert words to some number
        measures = []
        # In Latin form
        words_latin = []
        measures_latin = []
        for line in content:
            line = ie.lib.util.StringUtils.StringUtils.trim(line)
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
        df_wordlist = pd.DataFrame({'Word':words, 'WordNumber':measures,
                                         'WordLatin':words_latin, 'WordLatinNumber':measures_latin})
        df_wordlist = df_wordlist.drop_duplicates(subset=['Word'])
        # Need to reset indexes, otherwise some index will be missing
        df_wordlist = df_wordlist.reset_index(drop=True)

        return df_wordlist



def demo_1():

    dirpath_wordlist = '/Users/mark.tan/Documents/dev/ie/nlp.data/wordlist'

    for lang in ['cn', 'th', 'vn']:
        wl = WordList(lang=lang,
                      dirpath_wordlist=dirpath_wordlist,
                      postfix_wordlist='-wordlist.txt')
        wl.load_wordlist(verbose=0)
        print('')
        print ( lang + ': Read Word List ' + str(wl.wordlist.shape[0]) + " lines" )
        s = ''
        sm = ''
        s_latin = ''
        sm_latin = ''
        df = wl.wordlist
        for i in range(0, min(100, df.shape[0]), 1):
            s = s + str(df['Word'].loc[i]) + ','
            sm = sm + str(df['WordNumber'].loc[i]) + ','
            s_latin = s_latin + str(df['WordLatin'].loc[i]) + ','
            sm_latin = sm_latin + str(df['WordLatinNumber'].loc[i]) + ','

        print ( s )
        print ( sm )
        print ( s_latin )
        print ( sm_latin )

        # Stopwords
        sw = WordList(lang=lang,
                      dirpath_wordlist=dirpath_wordlist,
                      postfix_wordlist='-stopwords.txt')
        sw.load_wordlist(verbose=0)
        print('')
        print ( lang + ': Read Stopword List ' + str(sw.wordlist.shape[0]) + " lines" )
        s = ''
        sm = ''
        s_latin = ''
        sm_latin = ''
        df = sw.wordlist
        for i in range(0, min(100, df.shape[0]), 1):
            s = s + str(df['Word'].loc[i]) + ','
            sm = sm + str(df['WordNumber'].loc[i]) + ','
            s_latin = s_latin + str(df['WordLatin'].loc[i]) + ','
            sm_latin = sm_latin + str(df['WordLatinNumber'].loc[i]) + ','

        print ( s )
        print ( sm )
        print ( s_latin )
        print ( sm_latin )

    return


#demo_1()
