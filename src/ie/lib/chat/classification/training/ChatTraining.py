# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import numpy as np
import pandas as pd
import os
import re
import ie.lib.lang.classification.TextClusterBasic as tcb
import ie.lib.chat.classification.training.RefFeatureVec as reffv
import ie.lib.chat.classification.training.ChatTrainingData as ctd
import mozg.common.util.Log as log
from inspect import currentframe, getframeinfo


#
# Supervised training of chats, grouping into command/intent types.
# We provide a few approaches to recognize intents:
#  1. Closest Distance using IDF weights
#  2. TODO: Naive Bayesian Classification
#  3. TODO: Logistic Classification by Every Intent Pair (I suspect
#     TODO: this method will not work well thus not a priority for now)
#
class ChatTraining:

    MINIMUM_THRESHOLD_DIST_TO_RFV = 0.5

    def __init__(
            self,
            botkey,
            dirpath_rfv,
            chat_training_data,
            # TODO Not needed anymore?
            lang = None
    ):

        self.botkey = botkey
        self.dirpath_rfv = dirpath_rfv
        self.chat_training_data = chat_training_data

        self.df_idf = None
        # Just a single RFV for each Intent
        self.df_rfv = None
        # The furthest point distance of training data to each Intent RFV
        self.df_rfv_distance_furthest = None
        # FV of all training data
        # TODO: Replace with clusters instead of all, only when our auto optimal cluster count algo works better.
        self.df_fv_all = None
        self.df_intent_tf = None
        # Closest distance of a non-class point to a class RFV
        self.df_dist_closest_non_class = None
        # Average distance of all non-class points to a class RFV
        self.df_dist_avg_non_class = None

        self.textcluster = None
        self.textcluster_bycategory = None
        # Commands or categories or classes of the classification
        self.commands = None

        return

    #
    # When we index FV of all training data, we keep in the format 'intentId-1', 'intentId-2,...
    # So we just filter off the '-\d+'
    #
    @staticmethod
    def retrieve_intent_from_training_data_index(
            index_list,
            convert_to_type = 'str'
    ):
        for i in range(0,len(index_list),1):
            v = str(index_list[i])
            v = re.sub(
                pattern = '[-]\d+.*',
                repl    = '',
                string  = v
            )
            if convert_to_type == 'int':
                index_list[i] = int(v)
            else:
                index_list[i] = v
        return index_list

    #
    # TODO: Include training/optimization of vector weights to best define the category and differentiate with other categories.
    # TODO: Currently uses static IDF weights.
    #
    def train(
            self,
            keywords_remove_quartile = 50,
            stopwords = (),
            weigh_idf = False
    ):
        td = None

        log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                         + ': Training for botkey=' + self.botkey
                         + '. Using keywords remove quartile = ' + str(keywords_remove_quartile)
                         + ', stopwords = [' + str(stopwords) + ']'
                         + ', weigh by IDF = ' + str(weigh_idf))

        if self.chat_training_data.use_db:
            td = self.chat_training_data.get_training_data_from_db()
            # Segment sentence and write back to DB if necessary
            self.chat_training_data.segment_db_training_data()
        else:
            if self.chat_training_data.df_training_data is None:
                raise Exception(str(self.__class__) + ' No training data from file!!')

            td = self.chat_training_data.df_training_data

        log.Log.debug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                      + ': Training data: ' + str(td))

        #
        # Extract all keywords
        # Our training now doesn't remove any word, uses no stopwords, but uses an IDF weightage to measure
        # keyword value.
        #
        log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                         + ': Starting text cluster, calculate top keywords...')
        self.textcluster = tcb.TextClusterBasic(
            text      = list(td[ctd.ChatTrainingData.COL_TDATA_TEXT_SEGMENTED]),
            stopwords = stopwords
        )
        self.textcluster.calculate_top_keywords(
            remove_quartile = keywords_remove_quartile
        )
        log.Log.info(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                         + ': Keywords extracted as follows:' + str(self.textcluster.keywords_for_fv))

        # Extract unique Commands/Intents
        log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                         + ': Extracting unique commands/intents..')
        self.commands = set( list( td[ctd.ChatTrainingData.COL_TDATA_INTENT_ID] ) )
        # Change back to list, this list may change due to deletion of invalid commands.
        self.commands = list(self.commands)
        log.Log.critical(self.commands)

        # Prepare data frames to hold RFV, etc. These also may change due to deletion of invalid commands.
        log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                         + ': Preparing RFV data frames...')
        m = np.zeros((len(self.commands), len(self.textcluster.keywords_for_fv)))
        self.df_rfv = pd.DataFrame(m, columns=self.textcluster.keywords_for_fv, index=self.commands)
        self.df_rfv_distance_furthest = pd.DataFrame({
            reffv.RefFeatureVector.COL_COMMAND:list(self.commands),
            reffv.RefFeatureVector.COL_DISTANCE_TO_RFV_FURTHEST:[ChatTraining.MINIMUM_THRESHOLD_DIST_TO_RFV]*len(self.commands),
            reffv.RefFeatureVector.COL_TEXT: ['']*len(self.commands)
        })

        #
        # Get IDF first
        #   We join all text from the same intent, to get IDF
        # TODO: IDF may not be the ideal weights, design an optimal one.
        #
        log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                         + ': Joining all training data in the same command/intent to get IDF...')
        i = 0
        text_bycategory = [''] * len(self.commands)
        for com in self.commands:
            # Join all text of the same command/intent together and treat them as one
            is_same_command = td[ctd.ChatTrainingData.COL_TDATA_INTENT_ID]==com
            text_samples = list(td[ctd.ChatTrainingData.COL_TDATA_TEXT_SEGMENTED].loc[is_same_command])
            text_com = ' '.join(text_samples)
            text_bycategory[i] = text_com
            i = i + 1
        log.Log.info(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                          + ': Joined intents: ' + str(text_bycategory))
        # Create a new TextCluster object
        self.textcluster_bycategory = tcb.TextClusterBasic(text=text_bycategory, stopwords=stopwords)
        # Always use the same keywords FV!!
        self.textcluster_bycategory.set_keywords(df_keywords=self.textcluster.df_keywords_for_fv.copy())

        log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                         + ': Calculating sentence matrix of combined Intents to get IDF...')
        self.textcluster_bycategory.calculate_sentence_matrix(
            freq_measure='normalized',
            feature_presence_only=False,
            idf_matrix=None
        )

        log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                         + ': Calculating IDF...')
        self.textcluster_bycategory.calculate_idf()
        # Create a column matrix for IDF
        idf_matrix = self.textcluster_bycategory.idf_matrix.copy()
        log.Log.info(idf_matrix)
        if not weigh_idf:
            log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                             + ': Not using IDF')
            idf_matrix = None

        #
        # Get RFV for every command/intent, representative feature vectors by command type
        #
        # Get sentence matrix for all sentences first
        log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                         + ': Calculating sentence matrix for all training data...')
        self.textcluster.calculate_sentence_matrix(
            freq_measure          = 'normalized',
            feature_presence_only = False,
            idf_matrix            = idf_matrix
        )

        #
        # Data frame for all training data
        # Get index for training data df
        #
        td_indexes = ['']*td.shape[0]
        for i in range(0, td.shape[0], 1):
            intent = td[ctd.ChatTrainingData.COL_TDATA_INTENT_ID].loc[i]
            intent_index = td[ctd.ChatTrainingData.COL_TDATA_INTENT_INDEX].loc[i]
            #
            # IMPORTANT!!
            # Don't use '.' here, as the index will become a number such that 1234.1 becomes
            # equal to 1234.10 when read back from CSV !!!
            #
            td_indexes[i] = str(intent) + '-' + str(intent_index)

        self.df_fv_all = pd.DataFrame(
            data    = self.textcluster.sentence_matrix.copy(),
            columns = self.textcluster.keywords_for_fv,
            index   = td_indexes
        )

        # Sanity check for training data with incorrect FV, e.g. no common features at all
        for idx in list(self.df_fv_all.index):
            # Convert to array
            fv = np.array(self.df_fv_all.loc[idx].values)
            check_dist = np.sum(np.multiply(fv,fv))**0.5
            if abs(check_dist-1) > 0.000001:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': Warning! FV for intent [' + str(idx) + '] not 1, but [' + str(check_dist) + '].'
                log.Log.critical(errmsg)
                log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                 + ': Dropping this training data [' + str(idx) + ']...')
                len_before = self.df_fv_all.shape[0]
                self.df_fv_all = self.df_fv_all.drop(labels=idx, axis=0)
                len_after = self.df_fv_all.shape[0]
                log.Log.debug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                             + ': Length before = ' + str(len_before) + ', length now = ' + str(len_after))
            else:
                log.Log.debug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                             + ': Normalization check for training data array ' + str(idx) + ' ok')


        #
        # 1. Cluster training data of the same intent.
        #    Instead of a single RFV to represent a single intent, we should have multiple.
        # 2. Get word importance or equivalent term frequency (TF) within an intent
        #    This can only be used to confirm if a detected intent is indeed the intent,
        #    can't be used as a general method to detect intent because it is intent specific.
        #
        m_tmp = np.zeros(self.df_rfv.shape)
        self.df_intent_tf = pd.DataFrame(m_tmp,
                                    columns=self.textcluster.keywords_for_fv,
                                    index=self.df_rfv.index.tolist())
        tmp_fv_td_matrix = self.textcluster.sentence_matrix.copy()
        row_indexes = None
        for com in self.commands:
            try:
                if type(com) is str:
                    # Use training data FVs (if text contains '?', '+',... need to replace with '[?]',... otherwise regex will fail
                    for specchar in ['?', '+', '*', '(', ')']:
                        com = com.replace(specchar, '[' + specchar + ']')

                # Extract only rows of this intent
                row_indexes = self.df_fv_all.index.str.match(str(com)+'.[0-9]+')
                tmp_intent_matrix = self.df_fv_all[row_indexes]
                if tmp_intent_matrix.shape[0] == 0:
                    errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                             + ': For intent ID [' + str(com) + '], no rows matched! ' + 'Possibly data has no common features in FV...'
                    log.Log.critical(errmsg)
                    continue
                    #raise Exception(errmsg)

                #
                # TODO FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
                #
                #log.Log.log(tmp_fv_td_intent_matrix)
                # tmp_intent_matrix = tmp_intent_matrix.as_matrix()
                tmp_intent_array = tmp_intent_matrix.values

                #
                # TODO: Cluster intent
                #

                #
                # Get some measure equivalent to "TF"
                #
                word_presence_array = (tmp_intent_array>0)*1
                # Sum columns
                keyword_presence_tf = np.sum(word_presence_array, axis=0) / word_presence_array.shape[0]
                #log.Log.log(keyword_presence_tf)
                self.df_intent_tf.loc[com] = keyword_presence_tf
                #log.Log.log(df_intent_tf)
                #raise Exception('Debug End')
            except Exception as ex:
                log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                 + ': Error for intent ID [' + com + '], at row indexes ' + str(row_indexes))
                raise(ex)

        #
        # Weigh by IDF above and normalize back
        #
        # Because self.commands may change, due to deletion of invalid commands
        all_commands = self.commands.copy()
        for com in all_commands:
            log.Log.info(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                             + ': Doing intent ID [' + str(com) + ']')
            is_same_command = td[ctd.ChatTrainingData.COL_TDATA_INTENT_ID]==com
            text_samples = list(td[ctd.ChatTrainingData.COL_TDATA_TEXT_SEGMENTED].loc[is_same_command])
            text_samples_indexes = list(td[ctd.ChatTrainingData.COL_TDATA_TEXT_SEGMENTED].loc[is_same_command].index)
            # Convert to numpy ndarray
            sample_matrix = self.textcluster.sentence_matrix[text_samples_indexes] * 1

            #
            # Sanity check to make sure FV is normalized 1
            #
            ok = False
            while not ok:
                ok = True
                for mrow in range(0, sample_matrix.shape[0], 1):
                    x = sample_matrix[mrow]
                    textsam = text_samples[mrow]

                    check_magnitude = np.sum( np.multiply(x,x) ) ** 0.5
                    if abs(check_magnitude-1) > 0.000001:
                        # Remove this row
                        sample_matrix = np.delete(arr=sample_matrix, obj=mrow, axis=0)
                        del text_samples[mrow]
                        del text_samples_indexes[mrow]

                        errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                                 + ': Removing row [' + str(mrow) + '] of training data.' +\
                                 'Error in training data ['  + textsam + ']! FV for sample command [' + str(com) +\
                                 '] not 1, but [' + str(check_magnitude) + '].'
                        log.Log.critical(errmsg)
                        # We need to break because the matrix dimensions changed in the loop
                        ok = False
                        break

            #
            # This case of empty sample matrix means all training data have been deleted due to bad data
            #
            if sample_matrix.shape[0] == 0:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': Empty matrix for command [' + str(com) + '] Removing this command from list..'
                log.Log.critical(errmsg)
                len_before = len(self.commands)
                self.commands.remove(com)
                len_after = len(self.commands)
                log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                 + ': ' + str(len_after) + ' commands left from previous ' + str(len_before)
                                 + ' commands. Not calculating RFV for [' + str(com) + ']')

                log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                 + ': Now dropping command [' + str(com) + '] from df_rfv..')
                len_before = self.df_rfv.shape[0]
                self.df_rfv = self.df_rfv.drop(labels=com, axis=0)
                len_after = self.df_rfv.shape[0]
                log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                 + ': ' + str(len_after) + ' df_rfv left from previous ' + str(len_before)
                                 + ' commands.')

                log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                 + ': Now dropping command [' + str(com) + '] from df_rfv_distance_furthest..')
                len_before = self.df_rfv_distance_furthest.shape[0]
                com_index = \
                    self.df_rfv_distance_furthest.loc[
                        self.df_rfv_distance_furthest[reffv.RefFeatureVector.COL_COMMAND] == com].index
                self.df_rfv_distance_furthest = self.df_rfv_distance_furthest.drop(labels=com_index, axis=0)
                len_after = self.df_rfv_distance_furthest.shape[0]
                log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                 + ':' + str(len_after) + ' df_rfv_distance_furthest left from previous '
                                 + str(len_before) + ' commands.')
                continue

            #
            # Reference feature vector for the command is the average of all feature vectors
            # TODO: Important!!! Cluster a training intent into several clusters for even more accuracy
            #
            rfv = np.sum(sample_matrix, axis=0) / sample_matrix.shape[0]
            # Renormalize it again
            normalize_factor = np.sum(np.multiply(rfv, rfv)) ** 0.5
            rfv = rfv / normalize_factor
            # A single array will be created as a column dataframe, thus we have to name the index and not columns
            self.df_rfv.loc[com] = rfv

            check_dist = np.sum(np.multiply(rfv,rfv))**0.5
            if abs(check_dist-1) > 0.000001:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': Warning! RFV for command [' + str(com) + '] not 1, but [' + str(check_dist) + '].'
                raise(errmsg)
            else:
                log.Log.info(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                             + ': Check RFV normalized ok [' + str(check_dist) + '].')

            #
            # Get furthest point of classification to rfv
            # This will be used to accept or reject a classified point to a particular class,
            # once the nearest class is found (in which no class is found then).
            #
            # Minimum value of threshold, don't allow 0's
            dist_furthest = ChatTraining.MINIMUM_THRESHOLD_DIST_TO_RFV
            com_index =\
                self.df_rfv_distance_furthest.loc[self.df_rfv_distance_furthest[reffv.RefFeatureVector.COL_COMMAND] == com].index
            for i in range(0, sample_matrix.shape[0], 1):
                log.Log.info(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                 + '   Checking ' + str(i) + ': [' + text_samples[i] + ']')
                fv_text = sample_matrix[i]
                dist_vec = rfv - fv_text
                dist = np.sum(np.multiply(dist_vec, dist_vec)) ** 0.5
                if dist > dist_furthest:
                    dist_furthest = dist
                    self.df_rfv_distance_furthest.at[com_index, reffv.RefFeatureVector.COL_DISTANCE_TO_RFV_FURTHEST] = dist
                    self.df_rfv_distance_furthest.at[com_index, reffv.RefFeatureVector.COL_TEXT] = text_samples[i]

            log.Log.info('Command [' + str(com) + ']..')
            log.Log.info('Furthest distance = ' +
                  str(float(self.df_rfv_distance_furthest[reffv.RefFeatureVector.COL_DISTANCE_TO_RFV_FURTHEST][com_index])))
            s_tmp = self.df_rfv_distance_furthest[reffv.RefFeatureVector.COL_TEXT].loc[com_index].values
            log.Log.info('  [' + s_tmp + ']')

            #if re.match('choice.[0-9]+', com):
            #    log.Log.log(com)
            #    log.Log.log(text_samples)
            #    log.Log.log(rfv)
            #    raise Exception('Done debug')

        #
        # TODO: Optimization
        # TODO: One idea is to find the biggest difference between features and magnify this difference
        # TODO: (means higher weight on the feature), or some form of optimization algorithm
        # TODO: A simpler way to start is to use IDF where document in this case is the categories. This
        # TODO: IDF measure can then be used as weight to the features. Means we use the above average of the
        # TODO: RFV to start the iteration with the IDF of the feature.
        #
        #word_presence_matrix = np.zeros((len(self.commands), len(self.keywords_for_fv)))
        #i = 0
        # Create a copy to ensure we don't overwrite the original
        #df_rfv_copy = self.df_rfv.copy(deep=True)
        #for cmd in df_rfv_copy.index:
        #    rfv_cmd = df_rfv_copy.loc[cmd].as_matrix()
        #    # Set to 1 if word is present
        #    rfv_cmd[rfv_cmd>0] = 1
        #    word_presence_matrix[i] = rfv_cmd
        #    i = i + 1

        # Sort
        self.df_rfv = self.df_rfv.sort_index()
        self.df_rfv_distance_furthest = self.df_rfv_distance_furthest.sort_index()
        self.df_fv_all = self.df_fv_all.sort_index()
        self.df_intent_tf = self.df_intent_tf.sort_index()

        #
        # Check RFV is normalized
        #
        count = 1
        for com in list(self.df_rfv.index):
            rfv = np.array(self.df_rfv.loc[com].values)
            dist = np.sum(np.multiply(rfv,rfv))**0.5
            if abs(dist-1) > 0.000001:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': Warning! RFV for command [' + str(com) + '] not 1, but [' + str(dist) + '].'
                raise(Exception(errmsg))
            else:
                log.Log.info(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                             + ': ' + str(count) + '. Check RFV for [' + str(com)
                             + '] normalized ok [' + str(dist) + '].')
                count = count + 1

        #
        # Save to file
        # TODO: This needs to be saved to DB, not file
        #
        fpath_idf = self.dirpath_rfv + '/' + self.botkey + '.' + 'chatbot.words.idf.csv'
        self.textcluster_bycategory.df_idf.to_csv(path_or_buf=fpath_idf, index=True, index_label='INDEX')

        fpath_rfv = self.dirpath_rfv + '/' + self.botkey + '.' + 'chatbot.commands.rfv.csv'
        self.df_rfv.to_csv(path_or_buf=fpath_rfv, index=True, index_label='INDEX')
        log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                         + ': Saved RFV dimensions [' + str(self.df_rfv.shape) + '] filepath [' + fpath_rfv + ']')

        fpath_dist_furthest = self.dirpath_rfv + '/' + self.botkey + '.' + 'chatbot.commands.rfv.distance.csv'
        self.df_rfv_distance_furthest.to_csv(path_or_buf=fpath_dist_furthest, index=True, index_label='INDEX')

        fpath_fv_all = self.dirpath_rfv + '/' + self.botkey + '.' + 'chatbot.fv.all.csv'
        self.df_fv_all.to_csv(path_or_buf=fpath_fv_all, index=True, index_label='INDEX')

        fpath_intent_tf = self.dirpath_rfv + '/' + self.botkey + '.' + 'chatbot.commands.words.tf.csv'
        self.df_intent_tf.to_csv(path_or_buf=fpath_intent_tf, index=True, index_label='INDEX')

        # We write a dummy file to change the directory timestamp, so that our intent server can detect change
        fpath_dummy_file = self.dirpath_rfv + '/.dummyfile'
        try:
            f = open(file=fpath_dummy_file, mode='w')
            f.close()
            os.remove(fpath_dummy_file)
        except Exception as ex:
            log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                             + ': Could not create/delete dummy file "' + fpath_dummy_file
                             + '". ' + str(ex))

        return

    #
    # Get the measure of "separation" between different classes to the reference feature vector
    #
    def get_separation(self):

        self.df_dist_closest_non_class = pd.DataFrame({'Command':list(self.commands),
                                                       'DistanceClosestNonClass':[0]*len(self.commands)})
        self.df_dist_avg_non_class = pd.DataFrame({'Command':list(self.commands),
                                                   'DistanceAvgNonClass':[0]*len(self.commands)})

        td = self.chat_training_data.df_training_data

        for com in self.commands:
            # RFV for this command
            rfv_com = self.df_rfv.loc[com]

            # Get average distance to all other non-class points
            not_same_command = td[ctd.ChatTrainingData.COL_TDATA_INTENT_ID] != com
            #text_non_class = list(td[ChatTraining.COL_TDATA_TEXT_SEGMENTED].loc[not_same_command])
            text_non_class_indexes = list(td[ctd.ChatTrainingData.COL_TDATA_TEXT_SEGMENTED].loc[not_same_command].index)

            com_index = self.df_dist_closest_non_class.loc[self.df_dist_closest_non_class['Command'] == com].index

            # Get sentence matrix of those not in this class
            text_non_class_matrix = self.textcluster.sentence_matrix[text_non_class_indexes] * 1

            # Now get the distance of all points to the RFV and sum them up
            total_distance = 0
            total_points = 0
            distance_closest = 99999999
            for i in range(0, text_non_class_matrix.shape[0], 1):
                fv_text = text_non_class_matrix[i]
                dist_vec = rfv_com - fv_text
                dist = np.sum(dist_vec * dist_vec) ** 0.5
                if distance_closest > dist:
                    distance_closest = dist

                total_points = total_points + 1
                total_distance = total_distance + dist

            avg_non_class = (float)(total_distance / total_points)
            self.df_dist_closest_non_class.at[com_index, 'DistanceClosestNonClass'] = round(distance_closest,3)
            self.df_dist_avg_non_class.at[com_index, 'DistanceAvgNonClass'] = round(avg_non_class,3)

            same_class_furthest =\
                self.df_rfv_distance_furthest[reffv.RefFeatureVector.COL_DISTANCE_TO_RFV_FURTHEST].loc[com_index].values
            overlap = same_class_furthest - distance_closest > 0
            if overlap:
                log.Log.warning('Warning: Intent '+ com +
                      ', same class furthest = ' + str(same_class_furthest) +
                      ', closest non-class point = ' + str(distance_closest) +
                      ', overlap = ' + str(overlap) +
                      ', avg = ' + str(avg_non_class))

            log.Log.info('Command = ' + com +
                  ', same class furthest = ' + str(same_class_furthest) +
                  ', closest non-class point = ' + str(distance_closest) +
                  ', overlap = ' + str(overlap) +
                  ', avg = ' + str(avg_non_class))

        return

    def read_fv_training_data_into_text(self):
        fpath_fv_all = self.dirpath_rfv + '/' + self.botkey + '.' + 'chatbot.fv.all.csv'
        df = pd.read_csv(filepath_or_buffer=fpath_fv_all, sep=',', index_col=0)

        df_text = pd.DataFrame(data={'Text':['']*df.shape[0]}, index=df.index)
        # Only numpy array type can work like a data frame
        kwlist = np.array(df.columns)
        df_word_presence = df>0
        for idx in df.index:
            presence_i = (df_word_presence.loc[idx]).tolist()
            text_nonzero = kwlist[presence_i]
            text_i = ' '.join(text_nonzero)
            df_text['Text'].at[idx] = text_i

        return df_text


if __name__ == '__main__':
    index_list = ['1111-1', '1111-2', '1111-3', '2222-1', '2222-2', '3333-10']
    ChatTraining.retrieve_intent_from_training_data_index(index_list = index_list)
    print(index_list)