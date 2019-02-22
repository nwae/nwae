# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import pandas as pd
import ie.lib.lang.nlp.WordSegmentation as ws
import ie.lib.lang.nlp.SynonymList as sl
import ie.lib.util.StringUtils as su
import ie.lib.util.Log as log


#
# Preprocessing of raw training data into desired form, including slow text segmentation done first.
#
class ChatTrainingData:

    COL_TDATA_BRAND = 'Brand'
    COL_TDATA_LANGUAGE = 'Lang'
    COL_TDATA_CATEGORY = 'Category'
    COL_TDATA_INTENT = 'Intent'
    COL_TDATA_INTENT_ID = 'Intent ID'
    COL_TDATA_INTENT_INDEX = 'IntentIndex'      # Derived Column
    COL_TDATA_TEXT = 'Text'
    COL_TDATA_TEXT_LENGTH = 'TextLength'        # Derived Column
    COL_TDATA_TEXT_SEGMENTED = 'TextSegmented'  # Derived Column

    def __init__(self,
                 lang,
                 brand,
                 dirpath_traindata,
                 postfix_training_files,
                 dirpath_wordlist,
                 dirpath_app_wordlist,
                 dirpath_synonymlist):
        self.lang = lang
        self.brand = brand
        self.dirpath_traindata = dirpath_traindata
        self.postfix_training_files = postfix_training_files
        self.dirpath_wordlist = dirpath_wordlist
        self.dirpath_app_wordlist = dirpath_app_wordlist
        self.dirpath_synonymlist = dirpath_synonymlist
        self.df_training_data = None

        postfix_wordlist = '-wordlist.txt'
        self.wseg = ws.WordSegmentation(lang=self.lang,
                                        dirpath_wordlist=self.dirpath_wordlist,
                                        postfix_wordlist=postfix_wordlist)
        # Load application wordlist
        postfix_wordlist_app = '.wordlist.app.txt'
        # Add application wordlist
        self.wseg.add_wordlist(dirpath=self.dirpath_app_wordlist,
                               postfix='.'+self.brand+postfix_wordlist_app,
                               verbose=0)

        # We need synonyms to normalize all text with "rootwords"
        self.synonymlist = sl.SynonymList(lang=self.lang,
                                          dirpath_synonymlist=self.dirpath_synonymlist,
                                          postfix_synonymlist='.synonymlist.txt')
        self.synonymlist.load_synonymlist(verbose=0)

        return

    #
    # Pre-process training data, remove empty lines, segment words, etc.
    #
    def pre_process_training_data(self, segment_words=True, verbose=0):
        # File name of training data depends on whether to split or not to split
        fpath_training_data = self.dirpath_traindata + '/' + self.lang + '.' + self.brand + '.' + self.postfix_training_files + '.csv'

        df = None
        try:
            # Row of header is row 0, no index column
            if verbose >= 1:
                log.Log.log('Reading training data from file [' + fpath_training_data + '], Split = ' + str(segment_words))
            df = pd.read_csv(filepath_or_buffer=fpath_training_data, sep=',', header=0, index_col=None)
            df = df.fillna(value=0)
            if verbose >= 1:
                log.Log.log('Read ' + str(df.shape[0]) + ' lines.')
        except IOError as e:
            log.Log.log(str(self.__class__) + ' Cannot open file [' + fpath_training_data + ']')
            return None

        # Remove empty lines
        df[ChatTrainingData.COL_TDATA_TEXT_LENGTH] = df[ChatTrainingData.COL_TDATA_TEXT].str.len()
        is_text_not_empty = df[ChatTrainingData.COL_TDATA_TEXT_LENGTH] > 0
        df = df[is_text_not_empty]

        # Sort the raw training data by Intent ID
        log.Log.log('Sorting raw training data by Category, Intent...')
        df = df.sort_values(by=[ChatTrainingData.COL_TDATA_CATEGORY, ChatTrainingData.COL_TDATA_INTENT], ascending=True)
        # Reset indexes to get normal indexes
        df = df.reset_index(drop=True)
        # Add intent index
        df[ChatTrainingData.COL_TDATA_INTENT_INDEX] = [0]*df.shape[0]
        prev_cat_int = ''
        prev_cat_int_index = 0

        log.Log.log('Assigning numbers to training data based on intent...')
        for i in range(0, df.shape[0], 1):
            cur_cat_int = df[ChatTrainingData.COL_TDATA_INTENT_ID].loc[i]
            if cur_cat_int != prev_cat_int:
                prev_cat_int = cur_cat_int
                prev_cat_int_index = 0
            prev_cat_int_index = prev_cat_int_index + 1
            df[ChatTrainingData.COL_TDATA_INTENT_INDEX].at[i] = prev_cat_int_index

        # Finally reset indexes
        df = df.reset_index(drop=True)
        log.Log.log('After brand (' + self.brand + ') filtering, remain ' + str(df.shape[0]) + ' lines.')

        # Segment words
        if segment_words:
            log.Log.log('Doing word segmentation on training data...')
            nlines = df.shape[0]
            df[ChatTrainingData.COL_TDATA_TEXT_SEGMENTED] = ['']*nlines
            for line in range(0, nlines, 1):
                text = str( df[ChatTrainingData.COL_TDATA_TEXT].loc[line] )
                text = su.StringUtils.trim(text)
                # For some reason, if I don't remove '\n' or '\r' before segmenting text, the segmented text when
                # written to csv afterwards, cannot be read back.
                text = su.StringUtils.remove_newline(str=text, replacement=' ')
                df[ChatTrainingData.COL_TDATA_TEXT].at[line] = text

                log.Log.log('Line ' + str(line) + ' (of ' + str(nlines) + ' lines): + "' + text + '"')
                text_segmented = self.wseg.segment_words(text=text, verbose=0)
                log.Log.log('          Segmented Text: ' + '"' + text_segmented + '"')

                #
                # Replace words with root words
                #
                words = text_segmented.split(sep=' ')
                for i in range(0, len(words), 1):
                    word = words[i]
                    if len(word) == 0:
                        continue
                    rootword = self.synonymlist.synonymlist[self.synonymlist.synonymlist['Word'] == word]['RootWord'].values
                    if len(rootword) == 1:
                        # log.Log.log('Rootword of [' + word + '] is [' + rootword + ']')
                        words[i] = rootword[0]
                text_segmented_normalized = ' '.join(words)
                #if verbose >= 1:
                #    log.Log.log('Normalized Text:')
                #    log.Log.log(text_segmented_normalized)

                df[ChatTrainingData.COL_TDATA_TEXT_SEGMENTED].at[line] = text_segmented_normalized

        log.Log.log(str(self.__class__) + ' DONE!')
        self.df_training_data = df[[ChatTrainingData.COL_TDATA_INTENT_INDEX,
                                    ChatTrainingData.COL_TDATA_CATEGORY,
                                    ChatTrainingData.COL_TDATA_INTENT,
                                    ChatTrainingData.COL_TDATA_INTENT_ID,
                                    ChatTrainingData.COL_TDATA_BRAND,
                                    ChatTrainingData.COL_TDATA_LANGUAGE,
                                    ChatTrainingData.COL_TDATA_TEXT_LENGTH,
                                    ChatTrainingData.COL_TDATA_TEXT,
                                    ChatTrainingData.COL_TDATA_TEXT_SEGMENTED]]
        return self.df_training_data

    #
    # Returns the already pre-processed training data
    #
    def get_training_data(self, max_lines=0, verbose=0):
        # File name of training data depends on whether to split or not to split
        fpath_training_data = self.dirpath_traindata + '/' + self.lang + '.' + self.brand +\
                                  '.' + self.postfix_training_files + '.split.csv'

        df = None
        try:
            # Row of header is row 0, no index column
            if verbose >= 1:
                log.Log.log('Reading training data from file [' + fpath_training_data + ']')
            df = pd.read_csv(filepath_or_buffer=fpath_training_data, sep=',', header=0, index_col=None)
            df = df.fillna(value=0)
            if verbose >= 1:
                log.Log.log('Read ' + str(df.shape[0]) + ' lines.')
        except IOError as e:
            log.Log.log('Cannot open file [' + fpath_training_data + ']')
            return None

        # Extract only given brand or 'all' brand. If brand=='', then we keep everything
        #if self.brand != '':
        #    is_all_brand = df[ChatTrainingData.COL_TDATA_BRAND] == 'all'
        #    is_this_brand = df[ChatTrainingData.COL_TDATA_BRAND] == self.brand
        #    df = df[is_all_brand | is_this_brand]

        # Finally reset indexes
        df = df.reset_index(drop=True)
        if verbose >= 1:
            log.Log.log('Training data has ' + str(df.shape[0]) + ' lines.')

        self.df_training_data = df[[ChatTrainingData.COL_TDATA_INTENT_INDEX,
                                    ChatTrainingData.COL_TDATA_CATEGORY,
                                    ChatTrainingData.COL_TDATA_INTENT,
                                    ChatTrainingData.COL_TDATA_INTENT_ID,
                                    ChatTrainingData.COL_TDATA_BRAND,
                                    ChatTrainingData.COL_TDATA_LANGUAGE,
                                    ChatTrainingData.COL_TDATA_TEXT_LENGTH,
                                    ChatTrainingData.COL_TDATA_TEXT,
                                    ChatTrainingData.COL_TDATA_TEXT_SEGMENTED]]
        return self.df_training_data

    def write_split_training_data_to_file(self):
        fpath_training_data_splitted = self.dirpath_traindata + '/' + self.lang + '.' + self.brand +\
                                       '.' + self.postfix_training_files + '.split.csv'
        log.Log.log(str(self.__class__) + ' Writing split training data to file [' + fpath_training_data_splitted + ']...')
        # Do not write index to file, so that format remains consistent with non-split file
        try:
            self.df_training_data.to_csv(path_or_buf=fpath_training_data_splitted, index=0)
            log.Log.log(str(self.__class__) + ' Successfully written to file [' + fpath_training_data_splitted + ']')
        except Exception as ex:
            log.Log.log(str(self.__class__) + ' Failed to write split training data to file.')
            log.Log.log(ex)
        return

    #
    # We cluster the Intents into a few clusters, for recognition purposes later
    #
    def cluster_training_data(self):
        return
