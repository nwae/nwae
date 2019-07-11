# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import pandas as pd
import numpy as np
import ie.lib.lang.nlp.WordSegmentation as ws
import ie.lib.lang.nlp.SynonymList as sl
import mozg.common.util.StringUtils as su
import mozg.common.util.Log as log
import mozg.common.data.BasicData as bd
import mozg.common.data.Bot as dbbot
import mozg.common.data.IntentCategory as dbintcat
import mozg.common.data.Intent as dbint
import mozg.common.data.IntentTraining as dbinttrn
import mozg.common.data.ViewBotTrainingData as viewBotTd
import ie.app.ConfigFile as cf
import mozg.common.data.security.Auth as au
from inspect import currentframe, getframeinfo


#
# Preprocessing of raw training data into desired form, including slow text segmentation done first.
#
class ChatTrainingData:

    COL_TDATA_TRAINING_DATA_ID = dbinttrn.IntentTraining.COL_INTENT_TRAINING_ID
    COL_TDATA_CATEGORY = 'Category'
    COL_TDATA_INTENT = 'Intent'
    COL_TDATA_INTENT_ID = 'Intent ID'
    COL_TDATA_INTENT_TYPE = viewBotTd.ViewBotTrainingData.COL_INTENT_TYPE
    COL_TDATA_INTENT_INDEX = 'IntentIndex'      # Derived Column for indexing training data sentences
    COL_TDATA_TEXT = 'Text'
    COL_TDATA_TEXT_LENGTH = 'TextLength'        # Derived Column
    COL_TDATA_TEXT_SEGMENTED = 'TextSegmented'  # Derived Column
    COL_TDATA_TEXT_INTENT_FULL_PATH = 'intentPath'  # Derived Column

    DB_COLNAME_MAP = {
        dbintcat.IntentCategory.COL_INTENT_CATEGORY_NAME: COL_TDATA_CATEGORY,
        dbint.Intent.COL_INTENT_NAME: COL_TDATA_INTENT,
        dbint.Intent.COL_INTENT_ID: COL_TDATA_INTENT_ID,
        dbinttrn.IntentTraining.COL_SENTENCE: COL_TDATA_TEXT,
        dbinttrn.IntentTraining.COL_SPLIT_SENTENCE: COL_TDATA_TEXT_SEGMENTED
    }

    def __init__(
            self,
            use_db,
            db_profile,
            account_id,
            bot_id,
            lang,
            # TODO Remove this when fully switched to DB, this is only identifier for csv files
            bot_key,
            dirpath_traindata,
            postfix_training_files,
            dirpath_wordlist,
            dirpath_app_wordlist,
            dirpath_synonymlist,
            verbose = 0
    ):
        self.use_db = use_db
        self.db_profile = db_profile
        self.account_id = account_id
        self.bot_id = bot_id

        self.lang = lang
        self.bot_key = bot_key
        self.dirpath_traindata = dirpath_traindata
        self.postfix_training_files = postfix_training_files
        self.dirpath_wordlist = dirpath_wordlist
        self.dirpath_app_wordlist = dirpath_app_wordlist
        self.dirpath_synonymlist = dirpath_synonymlist
        self.df_training_data = None

        self.verbose = verbose

        postfix_wordlist = '-wordlist.txt'
        self.wseg = ws.WordSegmentation(
            lang             = self.lang,
            dirpath_wordlist = self.dirpath_wordlist,
            postfix_wordlist = postfix_wordlist,
            verbose          = self.verbose
        )
        # Load application wordlist
        postfix_wordlist_app = '.wordlist.app.txt'
        # Add application wordlist
        self.wseg.add_wordlist(
            dirpath = self.dirpath_app_wordlist,
            # This is a general application wordlist file, shared between all
            postfix = postfix_wordlist_app,
        )

        # We need synonyms to normalize all text with "rootwords"
        self.synonymlist = sl.SynonymList(
            lang                = self.lang,
            dirpath_synonymlist = self.dirpath_synonymlist,
            postfix_synonymlist = '.synonymlist.txt')

        self.synonymlist.load_synonymlist(verbose=self.verbose)

        self.viewBotTrainingData = viewBotTd.ViewBotTrainingData(
            db_profile = cf.ConfigFile.DB_PROFILE
        )

        return

    #
    # Pre-process training data, remove empty lines, segment words, etc.
    # TODO function Remove when using DB
    #
    def pre_process_text_file_training_data(
            self,
            segment_words = True,
            verbose = 0
    ):
        if self.use_db:
            raise Exception(str(self.__class__) + ': Using DB should not call this function!')

        # File name of training data depends on whether to split or not to split
        fpath_training_data = self.dirpath_traindata + '/' + self.bot_key + '.' + self.postfix_training_files + '.csv'

        df = None
        try:
            # Row of header is row 0, no index column
            log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Reading training data from file [' + fpath_training_data + '], Split = ' + str(segment_words))
            df = pd.read_csv(filepath_or_buffer=fpath_training_data, sep=',', header=0, index_col=None)
            df = df.fillna(value=0)
            log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                              + ': Read ' + str(df.shape[0]) + ' lines.')
        except IOError as e:
            log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                             + ': Cannot open file [' + fpath_training_data + ']')
            return None

        # Remove empty lines
        df[ChatTrainingData.COL_TDATA_TEXT_LENGTH] = df[ChatTrainingData.COL_TDATA_TEXT].str.len()
        is_text_not_empty = df[ChatTrainingData.COL_TDATA_TEXT_LENGTH] > 0
        df = df[is_text_not_empty]

        # Sort the raw training data by Intent ID
        log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                          + ': Sorting raw training data by Category, Intent...')
        df = df.sort_values(
            by        = [ChatTrainingData.COL_TDATA_CATEGORY, ChatTrainingData.COL_TDATA_INTENT],
            ascending = True
        )
        # Reset indexes to get normal indexes
        df = df.reset_index(drop=True)
        # Add intent index
        df[ChatTrainingData.COL_TDATA_INTENT_INDEX] = [0]*df.shape[0]
        prev_cat_int = ''
        prev_cat_int_index = 0

        log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                          + ': Assigning numbers to training data based on intent...')
        for i in range(0, df.shape[0], 1):
            cur_cat_int = df[ChatTrainingData.COL_TDATA_INTENT_ID].loc[i]
            if cur_cat_int != prev_cat_int:
                prev_cat_int = cur_cat_int
                prev_cat_int_index = 0
            prev_cat_int_index = prev_cat_int_index + 1
            df[ChatTrainingData.COL_TDATA_INTENT_INDEX].at[i] = prev_cat_int_index

        # Finally reset indexes
        df = df.reset_index(drop=True)
        #log.Log.log('After brand (' + self.brand + ') filtering, remain ' + str(df.shape[0]) + ' lines.')

        # Segment words
        if segment_words:
            log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                              + ': Doing word segmentation on training data...')
            nlines = df.shape[0]
            df[ChatTrainingData.COL_TDATA_TEXT_SEGMENTED] = ['']*nlines
            for line in range(0, nlines, 1):
                text = str( df[ChatTrainingData.COL_TDATA_TEXT].loc[line] )
                text = su.StringUtils.trim(text)
                # For some reason, if I don't remove '\n' or '\r' before segmenting text, the segmented text when
                # written to csv afterwards, cannot be read back.
                text = su.StringUtils.remove_newline(str=text, replacement=' ')
                df[ChatTrainingData.COL_TDATA_TEXT].at[line] = text

                log.Log.info(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                             + ': Line ' + str(line) + ' (of ' + str(nlines) + ' lines): + "' + text + '"')
                text_segmented = self.wseg.segment_words(text=text)
                log.Log.info(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                             + '          Segmented Text: ' + '"' + text_segmented + '"')

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

        log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                          + ' DONE!')
        self.df_training_data = df[[
            ChatTrainingData.COL_TDATA_INTENT_INDEX,
            ChatTrainingData.COL_TDATA_CATEGORY,
            ChatTrainingData.COL_TDATA_INTENT,
            ChatTrainingData.COL_TDATA_INTENT_ID,
            ChatTrainingData.COL_TDATA_TEXT_LENGTH,
            ChatTrainingData.COL_TDATA_TEXT,
            ChatTrainingData.COL_TDATA_TEXT_SEGMENTED
        ]]
        return self.df_training_data

    def get_training_data_from_db(
            self,
            max_lines=0,
            verbose=0
    ):
        #
        # Needed columns:
        #  Intent Category ID, Intent Category, Intent ID, Intent, Text, Text Segmented
        #
        td_intents = self.viewBotTrainingData.get(
            botId = self.bot_id
        )

        self.df_training_data_db = pd.DataFrame(td_intents)
        log.Log.info(self.df_training_data_db[1:10])
        log.Log.info(self.df_training_data_db.columns)
        log.Log.info(self.df_training_data_db.shape)

        # Rename the columns to be compatible with csv column names
        self.df_training_data_db.rename(columns=ChatTrainingData.DB_COLNAME_MAP, inplace=True)
        log.Log.info(self.df_training_data_db[1:10])
        log.Log.info(self.df_training_data_db.columns)
        log.Log.info(self.df_training_data_db.shape)

        # Remove empty lines
        self.df_training_data_db[ChatTrainingData.COL_TDATA_TEXT_LENGTH] =\
            self.df_training_data_db[ChatTrainingData.COL_TDATA_TEXT].str.len()
        is_text_not_empty = self.df_training_data_db[ChatTrainingData.COL_TDATA_TEXT_LENGTH] > 0
        self.df_training_data_db = self.df_training_data_db[is_text_not_empty]

        # Filter out certain intent types
        condition = np.invert(
            np.isin(
                element=self.df_training_data_db[ChatTrainingData.COL_TDATA_INTENT_TYPE],
                test_elements = dbint.Intent.EXCLUDED_INTENT_TYPES_FOR_TRAINING
            )
        )
        self.df_training_data_db = self.df_training_data_db[condition]
        log.Log.info('After filtering out intent types ' + str(dbint.Intent.EXCLUDED_INTENT_TYPES_FOR_TRAINING))
        log.Log.info(self.df_training_data_db[1:10])
        log.Log.info(self.df_training_data_db.shape)

        # Sort by 'intentPath' and reset index
        self.df_training_data_db = self.df_training_data_db.sort_values(
            [ChatTrainingData.COL_TDATA_INTENT_ID],
            ascending=True
        )
        self.df_training_data_db = self.df_training_data_db.reset_index(drop=True)

        # Now derive the training data index
        log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                          + ': Assigning numbers to training data based on intent...')
        # Add intent index
        self.df_training_data_db[ChatTrainingData.COL_TDATA_INTENT_INDEX] = [0]*self.df_training_data_db.shape[0]
        prev_cat_int = ''
        prev_cat_int_index = 0
        for i in range(0, self.df_training_data_db.shape[0], 1):
            cur_cat_int = self.df_training_data_db[ChatTrainingData.COL_TDATA_INTENT_ID].loc[i]
            if cur_cat_int != prev_cat_int:
                prev_cat_int = cur_cat_int
                prev_cat_int_index = 0
            prev_cat_int_index = prev_cat_int_index + 1
            self.df_training_data_db[ChatTrainingData.COL_TDATA_INTENT_INDEX].at[i] = prev_cat_int_index

        #self.df_training_data_db.to_csv(path_or_buf='/Users/mark.tan/Downloads/td.csv', header=True, index=False)

        # Point csv data to DB data
        self.df_training_data = self.df_training_data_db

        return self.df_training_data_db

    #
    # Segment text if necessary
    # This can happen when there is new training text in the DB, and we just write NULL
    # to the segmented text column.
    #
    def segment_db_training_data(
            self
    ):
        # Training Data DB
        db_int_td = dbinttrn.IntentTraining(
            db_profile = self.db_profile,
            verbose    = self.verbose
        )

        # Shorten variable
        td = self.df_training_data_db

        for i in list(td.index):
            text_segmented = td[ChatTrainingData.COL_TDATA_TEXT_SEGMENTED].loc[i]
            if text_segmented is None:
                intent_td_id = td[ChatTrainingData.COL_TDATA_TRAINING_DATA_ID].loc[i]
                intent_id = td[ChatTrainingData.COL_TDATA_INTENT_ID].loc[i]

                text = td[ChatTrainingData.COL_TDATA_TEXT].loc[i]
                text = str(text)

                text_segmented = self.wseg.segment_words(text=su.StringUtils.trim(text))

                log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                  + ': Training Data ID ' + str(intent_td_id)
                                  + '. No segmented text for training data "'
                                  + text + '". Segmented to "' + str(text_segmented) + '"')

                td[ChatTrainingData.COL_TDATA_TEXT_SEGMENTED].at[i] = text_segmented

                try:
                    # Write this segmented text back to DB
                    res = db_int_td.insert_or_update(
                        operation = bd.BasicData.OPERATION_UPDATE,
                        sentence  = text,
                        splitSentence = text_segmented,
                        intentId      = int(intent_id),
                        intentTrainingId = int(intent_td_id)
                    )
                    if res:
                        log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                          + ': Successfully updated training data id '
                                          + str(intent_td_id) + '.')
                    else:
                        log.Log.error(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                    + ': Failed to update training data id '
                                    + str(intent_td_id) + '.')
                except Exception as ex:
                    log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                + ': Exception updating training data id '
                                + str(intent_td_id) + 'with intent id=' + str(intent_id)
                                + ', sentence=' + text + ', segmented text=' + str(text_segmented)
                                + '. Exception message: ' + str(ex) + '.')

    #
    # Returns the already pre-processed training data
    #
    def get_training_data(self, max_lines=0, verbose=0):
        if self.use_db:
            raise Exception(str(self.__class__) + ': Using DB should not call this function!')

        # File name of training data depends on whether to split or not to split
        fpath_training_data = self.dirpath_traindata + '/' + self.bot_key +\
                                  '.' + self.postfix_training_files + '.split.csv'

        df = None
        try:
            # Row of header is row 0, no index column
            log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                              + ': Reading training data from file [' + fpath_training_data + ']')
            df = pd.read_csv(filepath_or_buffer=fpath_training_data, sep=',', header=0, index_col=None)
            df = df.fillna(value=0)
            log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                              + ': Read ' + str(df.shape[0]) + ' lines.')
        except IOError as e:
            log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                             + ': Cannot open file [' + fpath_training_data + ']')
            return None

        # Extract only given brand or 'all' brand. If brand=='', then we keep everything
        #if self.brand != '':
        #    is_all_brand = df[ChatTrainingData.COL_TDATA_BRAND] == 'all'
        #    is_this_brand = df[ChatTrainingData.COL_TDATA_BRAND] == self.brand
        #    df = df[is_all_brand | is_this_brand]

        # Finally reset indexes
        df = df.reset_index(drop=True)
        log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                          + ': Training data has ' + str(df.shape[0]) + ' lines.')

        self.df_training_data = df[[
            ChatTrainingData.COL_TDATA_INTENT_INDEX,
            ChatTrainingData.COL_TDATA_CATEGORY,
            ChatTrainingData.COL_TDATA_INTENT,
            ChatTrainingData.COL_TDATA_INTENT_ID,
            #ChatTrainingData.COL_TDATA_BRAND,
            #ChatTrainingData.COL_TDATA_LANGUAGE,
            ChatTrainingData.COL_TDATA_TEXT_LENGTH,
            ChatTrainingData.COL_TDATA_TEXT,
            ChatTrainingData.COL_TDATA_TEXT_SEGMENTED
        ]]
        return self.df_training_data

    def write_split_training_data_to_file(self):
        fpath_training_data_splitted = self.dirpath_traindata + '/' + self.bot_key +\
                                       '.' + self.postfix_training_files + '.split.csv'
        log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                          + ': Writing split training data to file [' + fpath_training_data_splitted + ']...')
        # Do not write index to file, so that format remains consistent with non-split file
        try:
            self.df_training_data.to_csv(path_or_buf=fpath_training_data_splitted, index=0)
            log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                              + ': Successfully written to file [' + fpath_training_data_splitted + ']')
        except Exception as ex:
            log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                             + ': Failed to write split training data to file.')
            log.Log.critical(ex)
        return

    def get_split_training_data_from_file(self):
        fpath_training_data_splitted = self.dirpath_traindata + '/' + self.bot_key +\
                                       '.' + self.postfix_training_files + '.split.csv'
        df = None
        try:
            # Row of header is row 0, no index column
            log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                              + ': Reading training data from file [' + fpath_training_data_splitted + ']')
            df = pd.read_csv(filepath_or_buffer=fpath_training_data_splitted, sep=',', header=0, index_col=None)
            df = df.fillna(value=0)
            log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                              + ': Read ' + str(df.shape[0]) + ' lines.')
        except IOError as e:
            log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                             + ': Cannot open file [' + fpath_training_data_splitted + ']')
            return None

        return df

    #
    # We cluster the Intents into a few clusters, for recognition purposes later
    #
    def cluster_training_data(self):
        return


if __name__ == '__main__':

    accountId = 4
    botId = 22
    lang = 'cn'

    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_INFO

    # DB Stuff initializations
    au.Auth.init_instances()

    botkey = dbbot.Bot.get_bot_key(
        db_profile = cf.ConfigFile.DB_PROFILE,
        account_id = accountId,
        bot_id     = botId,
        lang       = lang
    )
    ctdata = ChatTrainingData(
        use_db     = cf.ConfigFile.USE_DB,
        db_profile = 'mario2',
        account_id = accountId,
        bot_id     = botId,
        lang       = lang,
        bot_key    = botkey,
        dirpath_traindata      = cf.ConfigFile.DIR_INTENT_TRAINDATA,
        postfix_training_files = cf.ConfigFile.POSTFIX_INTENT_TRAINING_FILES,
        dirpath_wordlist       = cf.ConfigFile.DIR_WORDLIST,
        dirpath_app_wordlist   = cf.ConfigFile.DIR_APP_WORDLIST,
        dirpath_synonymlist    = cf.ConfigFile.DIR_SYNONYMLIST
    )

    td = ctdata.get_training_data_from_db()
    td.to_csv(path_or_buf='/Users/mark.tan/Downloads/td.csv', index=True, index_label='INDEX')

    ctdata.segment_db_training_data()
