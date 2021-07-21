#!/use/bin/python
# --*-- coding: utf-8 --*--

# !!! Will work only on Python 3 and above

import os
import sys
import re
import numpy as np
import pandas as pd
import collections
import nwae.lang.LangFeatures as langfeatures
import nwae.lang.characters.LangCharacters as langcharacters
import nwae.utils.FileUtils as futil
import nwae.utils.StringUtils as sutil
import nwae.utils.Log as log
from inspect import currentframe, getframeinfo


#
# Study unigram stats & unigram boundary conditional probabilities
# Unigram stats are used for better language detection improving
# the results of the alphabet/syllable language detection.
#
# Unigram can only be done for space separated languages grouped
# by either alphabets or syllables.
# (en,ko,vn,in,my,ru,es) Not for (cn,th,jp)
#
# Unigram boundary conditional probabilities are used to determine
# word/phrase boundaries, to solve the problem of word/phrase
# segmentation.
#
class LangStats:

    def __init__(
            self,
            dirpath_traindata,
            dirpath_collocation
    ):
        self.dirpath_traindata = dirpath_traindata
        self.dirpath_collocation = dirpath_collocation
        self.lang_features = langfeatures.LangFeatures()
        self.encoding = langcharacters.LangCharacters.encoding
        self.collocation_stats = {}
        return

    def get_collocation_probability(
            self,
            lang,
            pretoken,
            posttoken,
            conditional
    ):
        prob = 0
        cs = self.collocation_stats[lang]
        log.Log.debugdebug(cs)
        if cs is None:
            return None

        # Make sure token exists
        is_exist_pretoken = pretoken in cs.index
        is_exist_posttoken = posttoken in cs.columns
        log.Log.debugdebug('Pre token [' + pretoken + '] exists: ' + str(is_exist_pretoken))
        log.Log.debugdebug('Post token [' + posttoken + '] exists: ' + str(is_exist_posttoken))

        if not is_exist_pretoken:
            return None
        if not is_exist_posttoken:
            return 0

        total_count = 0
        if conditional == 'post':
            # Get column
            col = cs[posttoken]
            log.Log.debugdebug(col)
            total_count = col.sum()
            log.Log.debugdebug('Total column sum = ' + str(total_count))
        else:
            # Get row
            row = cs.loc[pretoken]
            log.Log.debugdebug(row)
            total_count = row.sum()
            log.Log.debugdebug('Total row sum = ' + str(total_count))

        cellcount = cs[posttoken][pretoken]
        log.Log.debugdebug('Cell count [' + pretoken + ',' + posttoken + '] = ' + str(cellcount))

        return float( cellcount / float(total_count) )

    def get_training_files(self, lang, dirpath):
        traindata_langfiles_all = os.listdir(dirpath + '/' + lang + "/")

        # Only .txt and .csv files
        traindata_langfiles = []
        for file in traindata_langfiles_all:
            if re.match('[a-zA-Z0-9._-]*.csv$', file) or re.match('[a-zA-Z0-9._-]*.txt$', file):
                traindata_langfiles.append(dirpath + '/' + lang + '/' + file)

        return traindata_langfiles

    def train_unigram_distribution_stats(self, dirpath):
        for lang in self.lang_features.langfeatures['Language']:
            print ('Processing language [' + lang + ']...')
            # Get training data file names
            traindata_langfiles = self.get_training_files(lang, dirpath)

            # Get unigram distribution stats
            df = self.get_lang_unigram_distribution_stats(lang, traindata_langfiles)

        return

    def get_character_sticky_distribution_stats(self, lang):

        traindata_files = self.get_training_files(lang, self.dirpath_traindata)
        log.Log.debugdebug('Training data files: ' + str(traindata_files))

        df = self.get_lang_unigram_distribution_stats(lang, verbose)
        token_list = df['TokenList']
        df_freqtable = df['FreqTable']

        log.Log.debugdebug('Read ' + str(len(df['Lines'])) + ' lines from training files.')
        log.Log.debugdebug('Read ' + str(len(df['TokenList'])) + ' characters from training files.')
        log.Log.debugdebug(df['TokenList'][0:1000])
        log.Log.debugdebug(df['FreqTable'])

        token_index = list(range(0, len(token_list), 1))

        # Create a dictionary as matrix of conditional probabilities
        tokens = list(df_freqtable['Token'])
        # Create a matrix of dimension (no.rows, no.columns)
        m = np.zeros((len(tokens), len(tokens)))
        # Convert to a Data Frame
        df_pm_preceding = pd.DataFrame(m, columns=tokens, index=tokens)

        print (df_pm_preceding)

        total_characters = len(tokens)
        count = 0
        # Loop through all unique tokens
        for e in tokens:
            count += 1
            log.Log.debugdebug( 'Processing character [' + e + ']..' + str(count) + ' of ' + str(total_characters))

            # Get all positions of this token
            allpos = [ x for x in token_index if token_list[x]==e ]

            log.Log.debugdebug( 'Positions of [' + e + ': ' )
            log.Log.debugdebug( allpos )

            ones = [1]*len(allpos)
            # All valid positions preceding this token
            allpos_preceding = [ (x-y) for x,y in zip(allpos, ones) if (x-y)>=0 ]
            log.Log.debugdebug( 'Positions of preceding characters:' )
            log.Log.debugdebug( allpos_preceding )

            all_tokens_preceding = [ token_list[x] for x in allpos_preceding ]
            log.Log.debugdebug( all_tokens_preceding )

            counter = collections.Counter(all_tokens_preceding)
            counter_sorted = counter.most_common()
            log.Log.debugdebug( counter_sorted )

            for pair in counter_sorted:
                char_precede = pair[0]
                f = pair[1]
                if char_precede not in tokens: continue
                df_pm_preceding[e][char_precede] = f

        return df_pm_preceding

    def load_collocation_stats(self):
        for lang in self.lang_features.langfeatures['Language']:
            log.Log.debugdebug('Loading collocation stats for language [' + lang + ']...')

            # Get collocation stats
            try:
                df = pd.read_csv(self.dirpath_collocation + '/collocation.stats.' + lang + '.csv' , sep=',', index_col=0)
                self.collocation_stats[lang] = df
                log.Log.debugdebug('   Loaded [' + lang + '] successfully.')
            except:
                log.Log.debugdebug('   Unexpected error: ', sys.exc_info()[0] )
                self.collocation_stats[lang] = None
                continue

        return

    #
    # Mainly for language detection.
    # A unigram is not necessarily a word, it can also be a character or syllable, e.g. Chinese character,
    # Korean Hangul syllable, or Thai alphabet (because Thai don't have syllable nor word separator)
    #
    def get_lang_unigram_distribution_stats(self, lang):

        traindata_files = self.get_training_files(lang, self.dirpath_traindata)
        log.Log.debugdebug('Training data files: ' + str(traindata_files))

        langcharset = langcharacters.LangCharacters.get_language_charset(lang)
        wordseparators = langcharacters.LangCharacters.UNICODE_BLOCK_WORD_SEPARATORS
        punctuations = langcharacters.LangCharacters.UNICODE_BLOCK_PUNCTUATIONS

        # Languages that have the tokens as the character set (those without syllable/word boundary as space)
        is_lang_token_same_with_charset = self.lang_features.is_lang_token_same_with_charset(lang)

        df = self.preprocess_text_into_tokens(lang, traindata_files, True)
        token_list = df['TokenList']
        contents = df['Lines']

        # TODO: Handle cases like "I'm", "it's", etc. In these cases 'm' and 's' will be separated even though they are not words.
        # By right, there should be some grammatical intelligence to split them into 2 words.

        # Get unique elements
        unique_tokens = list( set( token_list) )
        counter = collections.Counter(token_list)
        # Order the counter
        counter_sorted = counter.most_common()
        print ( 'Found ' + str(len(unique_tokens)) + ' unique tokens.' )

        table_tokens = {}
        for i in range(0, len(counter_sorted), 1):
            token = counter_sorted[i][0]
            # We don't want spaces, punctuations, empty '', etc.
            accept_token = token not in (wordseparators + punctuations + [''])
            if is_lang_token_same_with_charset:
                accept_token = accept_token and token in langcharset
            if accept_token:
                table_tokens[token] = counter_sorted[i][1]

        log.Log.debugdebug( 'Found ' + str(len(table_tokens)) + ' unique allowed tokens.' )

        keys_tokens = list(table_tokens.keys())
        count_tokens = list(table_tokens.values())
        df_table_tokens = pd.DataFrame({'Token':keys_tokens, 'Frequency':count_tokens})

        # token_list contains all the tokens, whereas df_table_tokens are already filtered (punctuations, etc.)
        return { 'Lines':contents, 'TokenList':token_list, 'FreqTable':df_table_tokens }

    #
    # Returns the contents of text files split by lines and by tokens in a list
    # A "token" can be a character or unigram, depending if the language has word/syllable
    # separators (if none, token=character, if yes, token=unigram)
    # A "unigram" is either a syllable or a word. In most languages a unigram is a word,
    # however in Vietnamese, syllables are standalone, so the unigram is a syllable or a word.
    #
    def preprocess_text_into_tokens(self, lang, filenames, string_tolower):

        contents = futil.FileUtils.read_text_files(filenames, self.encoding)

        for i in range(0, len(contents), 1):
            s = contents[i]
            # Remove newlines and trim line
            s = sutil.StringUtils.remove_newline(s)
            s = sutil.StringUtils.trim(s)
            if string_tolower:
                s = s.lower()
            # Remove start/end '"' (don't know why this appears after extraction of chats into lines csv)
            s = re.sub('^["]', '', re.sub('["]$', '', s))
            # Add a space to separate lines
            s = s + ' '
            contents[i] = s

        text = u''.join(contents)
        # Convert to list
        token_list = []
        sylsep = self.lang_features.get_split_token(lang, langfeatures.LangFeatures.LEVEL_SYLLABLE)
        wordsep = self.lang_features.get_split_token(lang, langfeatures.LangFeatures.LEVEL_UNIGRAM)
        # If either syllable or unigram is split by space
        if  sylsep == ' ' or wordsep == ' ':
            token_list = text.split(' ')
        else:
            token_list = list(text)

        return { 'Lines':contents, 'TokenList':token_list }

