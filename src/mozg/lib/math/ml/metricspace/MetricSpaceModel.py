# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import numpy as np
import pandas as pd
import threading
import re
import datetime as dt
import mozg.lib.chat.classification.training.RefFeatureVec as reffv
import mozg.lib.math.ml.TrainingDataModel as tdm
import mozg.lib.chat.classification.training.ChatTrainingData as ctd
import mozg.common.util.Log as log
from inspect import currentframe, getframeinfo
import mozg.common.data.security.Auth as au
import mozg.lib.math.Cluster as clstr


#
# MetricSpace Machine Learning Model
#
class MetricSpaceModel(threading.Thread):

    MINIMUM_THRESHOLD_DIST_TO_RFV = 0.5
    SMALL_VALUE = 0.0000001

    def __init__(
            self,
            # Unique identifier to identify this set of trained data+other files after training
            identifier_string,
            # Directory to keep all our model files
            dir_path_model,
            # Training data in TrainingDataModel class type
            training_data,
            # From all the initial features, how many we should remove by quartile. If 0 means remove nothing.
            key_features_remove_quartile = 50,
            # Initial features to remove, should be an array of numbers (0 index) indicating column to delete in training data
            stop_features = (),
            # If we will create an "IDF" based on the initial features
            weigh_idf = False
    ):
        super(MetricSpaceModel, self).__init__()

        self.identifier_string = identifier_string
        self.dir_path_model = dir_path_model
        self.training_data = training_data
        if type(self.training_data) is not tdm.TrainingDataModel:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Training data must be of type "' + str(tdm.TrainingDataModel.__class__)
                + '", got type "' + str(type(self.training_data)) + '" instead from object ' + str(self.training_data) + '.'
            )

        self.key_features_remove_quartile = key_features_remove_quartile
        self.stop_features = stop_features
        self.weigh_idf = weigh_idf

        self.df_feature_idf = None
        # Can be multiple RFVs for each class
        self.df_class_rfv = None
        # The furthest point distance of training data (in the same class) to the (one or more) class RFV
        self.df_sample_rfv_distance_furthest = None
        # FV of all training data
        self.df_trainingdata_fv_all = None
        # FV of clustered training data
        self.df_trainingdata_fv_clustered = None
        # Feature term frequency for curiosity?
        self.df_feature_tf = None

        # Closest distance of a non-class point to a class RFV
        self.df_sample_dist_closest_non_class = None
        # Average distance of all non-class points to a class RFV
        self.df_sample_dist_avg_non_class = None

        self.cluster = None
        self.cluster_bycategory = None
        #  Classes of the classification
        self.classes = None

        self.bot_training_start_time = None
        self.bot_training_end_time = None
        self.logs_training = None
        self.is_training_done = False
        self.__mutex_training = threading.Lock()

        return

    def run(self):
        self.__mutex_training.acquire()
        try:
            self.bot_training_start_time = dt.datetime.now()
            self.log_training = []
            self.train(
                self.key_features_remove_quartile,
                self.stop_features,
                self.weigh_idf
            )
            self.bot_training_end_time = dt.datetime.now()
        except Exception as ex:
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Identifier ' + str(self.identifier_string) + '" training exception: ' + str(ex) + '.'
            )
        finally:
            self.is_training_done = True
            self.__mutex_training.release()

        log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Identifier ' + str(self.identifier_string) + '" trained successfully.')

    def get_feature_weight_idf(
            self,
            x,
            y,
            x_name
    ):
        df_tmp = pd.DataFrame(data=x, index=y)
        df_agg_sum = df_tmp.groupby(df_tmp.index).sum()
        np_agg_sum = df_agg_sum.values
        # Get presence only by cell, then sum up by columns to get total presence by document
        np_feature_presence = (np_agg_sum>0)*1
        # Sum by column axis=0
        np_feature_presence_sum = np.sum(np_feature_presence, axis=0)
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tAggregated sum by labels:\n\r' + str(np_agg_sum)
            + '\n\r\tPresence array:\n\r' + str(np_feature_presence)
            + '\n\r\tPresence sum:\n\r' + str(np_feature_presence_sum)
            + '\n\r\tx_names: ' + str(x_name) + '.'
        )

        # Total document count
        n_documents = np_feature_presence.shape[0]
        log.Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Total unique documents/intents to calculate IDF = ' + str(n_documents)
        )

        # If using outdated np.matrix, this IDF will be a (1,n) array, but if using np.array, this will be 1-dimensional vector
        idf = np.log(n_documents / np_feature_presence_sum)
        # Replace infinity with 1 count or log(n_documents)
        idf[idf==np.inf] = np.log(n_documents)
        # If only 1 document, all IDF will be zero, we will handle below
        if n_documents <= 1:
            log.Log.warning(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Only ' + str(n_documents) + ' document in IDF calculation. Setting IDF to 1.'
            )
            idf = np.array([1]*len(x.shape[1]))
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tWeight IDF:\n\r' + str(idf)
        )
        return idf

    #
    # TODO: Include training/optimization of vector weights to best define the category and differentiate with other categories.
    # TODO: Currently uses static IDF weights.
    #
    def train(
            self,
            key_features_remove_quartile = 50,
            stop_features = (),
            weigh_idf = False
    ):
        self.log_training = []

        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Training for identifier=' + self.identifier_string
            + '. Using key features remove quartile = ' + str(key_features_remove_quartile)
            + ', stop features = [' + str(stop_features) + ']'
            + ', weigh by IDF = ' + str(weigh_idf)
            , log_list = self.log_training
        )

        x = self.training_data.get_x()
        y = self.training_data.get_y()
        x_name = self.training_data.get_x_name()
        # Unique y or classes
        # We have to list() the set(), to make it into a proper 1D vector
        self.classes = np.array(list(set(y)))

        #
        # Here training data must be prepared in the correct format already
        # Значит что множество свойств уже объединено как одно (unified features)
        #
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tTraining data:\n\r' + str(self.training_data.get_x())
            + '\n\r\tx names: ' + str(self.training_data.get_x_name())
            + '\n\r\ty labels: ' + str(self.training_data.get_y())
        )

        #
        # Get IDF first
        #   We join all text from the same intent, to get IDF
        # TODO: IDF may not be the ideal weights, design an optimal one.
        #
        if weigh_idf:
            # Sum x by class
            idf = self.get_feature_weight_idf(x=x, y=y, x_name=x_name)
            self.df_idf = pd.DataFrame({'Word': x_name, 'IDF': idf})
            log.Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': IDF Dataframe:\n\r' + str(self.df_idf)
                , log_list = self.log_training
            )

            self.training_data.weigh_x(w=idf)

            # Refetch again after weigh
            x = self.training_data.get_x()
            y = self.training_data.get_y()
            x_name = self.training_data.get_x_name()

            log.Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + '\n\r\tx weighted by idf and renormalized:\n\r' + str(x)
                + '\n\r\ty\n\r' + str(y)
                + '\n\r\tx_name\n\r' + str(x_name)
                , log_list=self.log_training
            )

        #
        # Get RFV for every command/intent, representative feature vectors by command type
        #

        #
        # 1. Cluster training data of the same intent.
        #    Instead of a single RFV to represent a single intent, we should have multiple.
        # 2. Get word importance or equivalent term frequency (TF) within an intent
        #    This can only be used to confirm if a detected intent is indeed the intent,
        #    can't be used as a general method to detect intent because it is intent specific.
        #
        x_clustered = None
        y_clustered = None
        for cs in self.classes:
            try:
                # Extract only rows of this class
                rows_of_class = x[y==cs]
                if rows_of_class.shape[0] == 0:
                    continue

                log.Log.debugdebug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + '\n\r\tRows of class "' + str(cs) + ':'
                    + '\n\r' + str(rows_of_class)
                )

                #
                # Cluster intent
                #
                # If there is only 1 row, then the cluster is the row
                np_class_cluster = rows_of_class
                # Otherwise we do proper clustering
                if rows_of_class.shape[0] > 1:
                    class_cluster = clstr.Cluster.cluster(
                        matx          = rows_of_class,
                        feature_names = x_name,
                        # Not more than 5 clusters per label
                        ncenters      = min(5, round(rows_of_class.shape[0] * 2/3)),
                        iterations    = 20
                    )
                    np_class_cluster = class_cluster[clstr.Cluster.COL_CLUSTER_NDARRY]
                if x_clustered is None:
                    x_clustered = np_class_cluster
                    y_clustered = np.array([cs]*x_clustered.shape[0])
                else:
                    # Append rows (thus 1st dimension at axis index 0)
                    x_clustered = np.append(x_clustered, np_class_cluster, axis=0)
                    # Appending to a 1D array always at axis=0
                    y_clustered = np.append(y_clustered, [cs]*np_class_cluster.shape[0], axis=0)
            except Exception as ex:
                log.Log.error(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Error for class "' + str(cs) + '", Exception msg ' + str(ex) + '.'
                    , log_list = self.log_training
                )
                raise(ex)

        self.x_clustered = x_clustered
        self.y_clustered = y_clustered
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tCluster of x\n\r' + str(self.x_clustered)
            + '\n\r\ty labels for cluster: ' + str(self.y_clustered)
        )

        raise Exception('DEBUGGING HERE')

        #
        # RFV Derivation
        #
        all_classes = self.classes.copy()
        for cs in all_classes:
            log.Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Doing class [' + str(cs) + ']'
                , log_list = self.log_training
            )
            # Extract class points
            x_tmp_class = x[x==cs]
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
                log.Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Check RFV normalized ok [' + str(check_dist) + '].'
                )

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
                log.Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + '   Checking ' + str(i) + ': [' + text_samples[i] + ']'
                )
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
                log.Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': ' + str(count) + '. Check RFV for [' + str(com)
                    + '] normalized ok [' + str(dist) + '].'
                    , log_list = self.log_training
                )
                count = count + 1

        #
        # Save to file
        # TODO: This needs to be saved to DB, not file
        #
        fpath_idf = self.dirpath_rfv + '/' + self.botkey + '.' + 'chatbot.words.idf.csv'
        self.textcluster_bycategory.df_idf.to_csv(path_or_buf=fpath_idf, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved IDF dimensions [' + str(self.textcluster_bycategory.df_idf.shape) + '] filepath [' + fpath_idf + ']'
            , log_list = self.log_training
        )

        fpath_rfv = self.dirpath_rfv + '/' + self.botkey + '.' + 'chatbot.commands.rfv.csv'
        self.df_rfv.to_csv(path_or_buf=fpath_rfv, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved RFV dimensions [' + str(self.df_rfv.shape) + '] filepath [' + fpath_rfv + ']'
            , log_list = self.log_training
        )

        fpath_dist_furthest = self.dirpath_rfv + '/' + self.botkey + '.' + 'chatbot.commands.rfv.distance.csv'
        self.df_rfv_distance_furthest.to_csv(path_or_buf=fpath_dist_furthest, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved RFV (furthest) dimensions [' + str(self.df_rfv_distance_furthest.shape)
            + '] filepath [' + fpath_dist_furthest + ']'
            , log_list = self.log_training
        )

        fpath_intent_tf = self.dirpath_rfv + '/' + self.botkey + '.' + 'chatbot.commands.words.tf.csv'
        self.df_intent_tf.to_csv(path_or_buf=fpath_intent_tf, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved TF dimensions [' + str(self.df_intent_tf.shape)
            + '] filepath [' + fpath_intent_tf + ']'
            , log_list = self.log_training
        )

        fpath_fv_all = self.dirpath_rfv + '/' + self.botkey + '.' + 'chatbot.fv.all.csv'
        self.df_fv_all.to_csv(path_or_buf=fpath_fv_all, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved Training Data [' + str(self.df_fv_all.shape) + '] filepath [' + fpath_fv_all + ']'
            , log_list=self.log_training
        )

        # Our servers look to this file to see if RFV has changed
        # It is important to do it last (and fast), after everything is done
        fpath_updated_file = self.dirpath_rfv + '/' + self.botkey + '.lastupdated.txt'
        try:
            f = open(file=fpath_updated_file, mode='w')
            f.write(str(dt.datetime.now()))
            f.close()
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Saved update time file "' + fpath_updated_file
                + '" for other processes to detect and restart.'
                , log_list=self.log_training
            )
        except Exception as ex:
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Could not create last updated file "' + fpath_updated_file
                + '". ' + str(ex)
            , log_list = self.log_training
            )

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


def demo_chat_training():
    au.Auth.init_instances()
    topdir = '/Users/mark.tan/git/mozg.nlp'
    chat_td = ctd.ChatTrainingData(
        use_db     = True,
        db_profile = 'mario2',
        account_id = 4,
        bot_id     = 22,
        lang       = 'cn',
        bot_key    = 'db_mario2.accid4.botid22',
        dirpath_traindata      = None,
        postfix_training_files = None,
        dirpath_wordlist       = topdir + '/nlp.data/wordlist',
        dirpath_app_wordlist   = topdir + '/nlp.data/app/chats',
        dirpath_synonymlist    = topdir + '/nlp.data/app/chats'
    )

    td = chat_td.get_training_data_from_db()
    # Take just ten labels
    unique_classes = td[ctd.ChatTrainingData.COL_TDATA_INTENT_ID]
    text_segmented = td[ctd.ChatTrainingData.COL_TDATA_TEXT_SEGMENTED]

    keep = 10
    unique_classes_trimmed = list(set(unique_classes))[0:keep]
    np_unique_classes_trimmed = np.array(unique_classes_trimmed)
    np_indexes = np.isin(element=unique_classes, test_elements=np_unique_classes_trimmed)

    # By creating a new np array, we ensure the indexes are back to the normal 0,1,2...
    np_label_id = np.array(list(unique_classes[np_indexes]))
    np_text_segmented = np.array(list(text_segmented[np_indexes]))

    print(np_label_id[0:20])
    print(np_text_segmented[0:20])
    print(np_text_segmented[0])

    #
    # Finally we have our text data in the desired format
    #
    tdm_obj = tdm.TrainingDataModel.unify_word_features_for_text_data(
        label_id       = np_label_id.tolist(),
        text_segmented = np_text_segmented.tolist(),
        keywords_remove_quartile = 0
    )

    print(tdm_obj.get_x())
    print(tdm_obj.get_x_name())
    print(tdm_obj.get_y())

    ms_model = MetricSpaceModel(
        identifier_string = 'demo_msmodel_accid4_botid22',
        # Directory to keep all our model files
        dir_path_model    = topdir + '/app.data/intent/rfv',
        # Training data in TrainingDataModel class type
        training_data     = tdm_obj,
        # From all the initial features, how many we should remove by quartile. If 0 means remove nothing.
        key_features_remove_quartile = 0,
        # Initial features to remove, should be an array of numbers (0 index) indicating column to delete in training data
        stop_features                = (),
        # If we will create an "IDF" based on the initial features
        weigh_idf                    = True
    )
    ms_model.train(
        key_features_remove_quartile = 0,
        stop_features = (),
        weigh_idf     = True
    )
    return


if __name__ == '__main__':
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_DEBUG_1
    #demo_chat_training()

    x_expected = np.array(
        [
            # 무리 A
            [1, 2, 1, 1, 0, 0],
            [2, 1, 2, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            # 무리 B
            [0, 1, 2, 1, 0, 0],
            [0, 2, 2, 2, 0, 0],
            [0, 2, 1, 2, 0, 0],
            # 무리 C
            [0, 0, 0, 1, 2, 3],
            [0, 1, 0, 2, 1, 2],
            [0, 1, 0, 1, 1, 2]
        ]
    )
    texts = [
        # 'A'
        '하나 두 두 셋 넷',
        '하나 하나 두 셋 셋 넷',
        '하나 두 셋 넷',
        # 'B'
        '두 셋 셋 넷',
        '두 두 셋 셋 넷 넷',
        '두 두 셋 넷 넷',
        # 'C'
        '넷 다섯 다섯 여섯 여섯 여섯',
        '두 넷 넷 다섯 다섯 여섯 여섯',
        '두 넷 다섯 여섯 여섯',
    ]

    y = np.array(
        ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
    )
    x_name = np.array(['하나', '두', '셋', '넷', '다섯', '여섯'])
    #
    # Finally we have our text data in the desired format
    #
    tdm_obj = tdm.TrainingDataModel.unify_word_features_for_text_data(
        label_id       = y.tolist(),
        text_segmented = texts,
        keywords_remove_quartile = 0
    )

    x_friendly = tdm_obj.get_print_friendly_x()

    print(tdm_obj.get_x())
    for k in x_friendly.keys():
        print(x_friendly[k])
    print(tdm_obj.get_x_name())
    print(tdm_obj.get_y())

    topdir = '/Users/mark.tan/git/mozg.nlp'
    ms_model = MetricSpaceModel(
        identifier_string = 'demo_msmodel_testdata',
        # Directory to keep all our model files
        dir_path_model    = topdir + '/app.data/intent/rfv',
        # Training data in TrainingDataModel class type
        training_data     = tdm_obj,
        # From all the initial features, how many we should remove by quartile. If 0 means remove nothing.
        key_features_remove_quartile = 0,
        # Initial features to remove, should be an array of numbers (0 index) indicating column to delete in training data
        stop_features                = (),
        # If we will create an "IDF" based on the initial features
        weigh_idf                    = True
    )
    ms_model.train(
        key_features_remove_quartile = 0,
        stop_features = (),
        weigh_idf     = True
    )

    # How to make sure order is the same output from TextCluster in unit tests?
    x_name_expected = ['넷' '두' '셋' '여섯' '다섯' '하나']

    sentence_matrix_expected = np.array([
            [0.37796447 ,0.75592895 ,0.37796447 ,0.         ,0.         ,0.37796447],
            [0.31622777 ,0.31622777 ,0.63245553 ,0.         ,0.         ,0.63245553],
            [0.5        ,0.5        ,0.5        ,0.         ,0.         ,0.5       ],
            [0.40824829 ,0.40824829 ,0.81649658 ,0.         ,0.         ,0.        ],
            [0.57735027 ,0.57735027 ,0.57735027 ,0.         ,0.         ,0.        ],
            [0.66666667 ,0.66666667 ,0.33333333 ,0.         ,0.         ,0.        ],
            [0.26726124 ,0.         ,0.         ,0.80178373 ,0.53452248 ,0.        ],
            [0.5547002  ,0.2773501  ,0.         ,0.5547002  ,0.5547002  ,0.        ],
            [0.37796447 ,0.37796447 ,0.         ,0.75592895 ,0.37796447 ,0.        ]
        ])
    for i in range(0,sentence_matrix_expected.shape[0],1):
        v = sentence_matrix_expected[i]
        ss = np.sum(np.multiply(v,v))**0.5
        print(v)
        print(ss)

    agg_by_labels_expected = np.array([
        [1.19419224 ,1.57215671 ,1.51042001 ,0.         ,0.         ,1.51042001],
        [1.65226523 ,1.65226523 ,1.72718018 ,0.         ,0.         ,0.        ],
        [1.19992591 ,0.65531457 ,0.         ,2.11241287 ,1.46718715 ,0.        ]
    ])

    idf_expected = [0.         ,0.         ,0.40546511 ,1.09861229 ,1.09861229 ,1.09861229]

    x_w_expected = [
        [0.         ,0.         ,0.34624155 ,0.         ,0.         ,0.9381454 ],
        [0.         ,0.         ,0.34624155 ,0.         ,0.         ,0.9381454 ],
        [0.         ,0.         ,0.34624155 ,0.         ,0.         ,0.9381454 ],
        [0.         ,0.         ,1.         ,0.         ,0.         ,0.        ],
        [0.         ,0.         ,1.         ,0.         ,0.         ,0.        ],
        [0.         ,0.         ,1.         ,0.         ,0.         ,0.        ],
        [0.         ,0.         ,0.         ,0.83205029 ,0.5547002  ,0.        ],
        [0.         ,0.         ,0.         ,0.70710678 ,0.70710678 ,0.        ],
        [0.         ,0.         ,0.         ,0.89442719 ,0.4472136  ,0.        ]
    ]
    y_w_expected = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
    x_name_w_expected = ['넷', '두', '셋', '여섯', '다섯', '하나']

    # Cluster value will change everytime! So don't rely on this
    x_clustered_expected = [
        [0.         ,0.         ,0.34624155 ,0.         ,0.         ,0.9381454 ],
        [0.         ,0.         ,1.         ,0.         ,0.         ,0.        ],
        [0.         ,0.         ,0.         ,0.86323874 ,0.5009569  ,0.        ],
        [0.         ,0.         ,0.         ,0.70710678 ,0.70710678 ,0.        ]
    ]
    y_clustered_expected = ['A', 'B', 'C', 'C']


