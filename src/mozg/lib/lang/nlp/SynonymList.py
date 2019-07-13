#!/use/bin/python
# --*-- coding: utf-8 --*--

# !!! Will work only on Python 3 and above

import re
import pandas as pd
import mozg.common.util.FileUtils as futil
import mozg.common.util.StringUtils as su
import ie.lib.lang.characters.LangCharacters
import ie.lib.lang.LangFeatures
import ie.lib.lang.nlp.LatinEquivalentForm as lef
import ie.lib.lang.characters.LangCharacters as langchar
import mozg.common.util.Log as log


class SynonymList:

    def __init__(self, lang, dirpath_synonymlist, postfix_synonymlist='.synonymlist.txt'):
        self.lang = lang

        self.dirpath_synonymlist = dirpath_synonymlist
        self.postfix_synonymlist = postfix_synonymlist

        self.synonymlist = None
        return

    def load_synonymlist(self, verbose=0):
        if self.synonymlist is None:
            self.synonymlist = self.load_list(dirpath=self.dirpath_synonymlist, postfix=self.postfix_synonymlist, verbose=verbose)
        return

    # General function to load wordlist or stopwords
    def load_list(self, dirpath, postfix, verbose=0):

        lc = langchar.LangCharacters()

        filepath = dirpath + '/' + self.lang + postfix
        if verbose >= 1:
            log.Log.log('Loading list for [' + self.lang + ']' + '[' + filepath + ']')

        fu = futil.FileUtils()
        content = fu.read_text_file(filepath)

        if verbose >= 1:
            log.Log.log('   Read ' + str(len(content)) + ' lines.')

        words = []
        rootwords = []
        # Convert words to some number
        measures = []
        # In Latin form
        words_latin = []
        measures_latin = []
        for line in content:
            line = su.StringUtils.trim(line)
            # Remove empty lines
            if len(line)<=0: continue
            # Remove comment lines starting with '#'
            if re.match(u'^#', line): continue

            linewords = line.split(sep=',')

            rootword = su.StringUtils.trim(linewords[0])
            # If 1st word is empty, ignore entire line
            if len(rootword)==0:
                continue

            for j in range(0, len(linewords), 1):

                word = su.StringUtils.trim(linewords[j])
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
        df_synonyms = pd.DataFrame({'RootWord':rootwords,
                                         'Word':words,
                                         'WordNumber':measures,
                                         'WordLatin':words_latin,
                                         'WordLatinNumber':measures_latin})
        df_synonyms = df_synonyms.drop_duplicates(subset=['Word'])
        # Need to reset indexes, otherwise some index will be missing
        df_synonyms = df_synonyms.reset_index(drop=True)

        return df_synonyms

    # Replace with root words, thus normalizing the text
    def normalize_text(self, text_segmented, verbose):
        #
        # Replace words with root words
        #
        words = text_segmented.split(sep=' ')
        for i in range(0, len(words), 1):
            word = words[i]
            if len(word)==0:
                continue
            rootword = self.synonymlist[self.synonymlist['Word']==word]['RootWord'].values
            if len(rootword)==1:
                # log.Log.log('Rootword of [' + word + '] is [' + rootword + ']')
                words[i] = rootword[0]
        text_normalized = ' '.join(words)
        #if verbose >= 2:
        #    log.Log.log('Normalized Text:')
        #    log.Log.log(text_normalized)

        return text_normalized


def demo_1():
    dirpath_synonymlist = '/Users/mark.tan/Documents/dev/ie/nlp.data/app/chats'

    for lang in ['cn', 'th']:
        sl = SynonymList(lang=lang,
                         dirpath_synonymlist=dirpath_synonymlist,
                         postfix_synonymlist='.synonymlist.txt')
        sl.load_synonymlist(verbose=0)
        print(sl.synonymlist)


#demo_1()
