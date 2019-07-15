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
import mozg.lib.math.Constants as const


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
        idf = None
        if weigh_idf:
            # Sum x by class
            idf = self.get_feature_weight_idf(x=x, y=y, x_name=x_name)
            log.Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + '\n\r\tIDF values:\n\r' + str(idf)
            )

            self.training_data.weigh_x(w=idf)

            # Refetch again after weigh
            x = self.training_data.get_x()
            y = self.training_data.get_y()
            x_name = self.training_data.get_x_name()
            # Unique y or classes
            # We have to list() the set(), to make it into a proper 1D vector
            self.classes = np.array(list(set(y)))

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

        #
        # RFV Derivation
        #
        m = np.zeros((len(self.classes), len(x_name)))
        # self.classes
        self.df_rfv = pd.DataFrame(
            m,
            columns = x_name,
            index   = self.classes
        )
        self.df_rfv_distance_furthest = pd.DataFrame(
            {
                reffv.RefFeatureVector.COL_COMMAND:list(self.classes),
                reffv.RefFeatureVector.COL_DISTANCE_TO_RFV_FURTHEST:[MetricSpaceModel.MINIMUM_THRESHOLD_DIST_TO_RFV]*len(self.classes),
            },
            index = self.classes
        )

        for cs in self.classes:
            log.Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Doing class [' + str(cs) + ']'
                , log_list = self.log_training
            )
            # Extract class points
            class_points = x[y==cs]
            #
            # Reference feature vector for the command is the average of all feature vectors
            #
            rfv = np.sum(class_points, axis=0) / class_points.shape[0]
            # Renormalize it again
            # At this point we don't have to check if it is a 0 vector, etc. as it was already done in TrainingDataModel
            # after weighing process
            normalize_factor = np.sum(np.multiply(rfv, rfv)) ** 0.5
            if normalize_factor < const.Constants.SMALL_VALUE:
                raise Exception(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Normalize factor for rfv in class "' + str(cs) + '" is 0.'
                )
            rfv = rfv / normalize_factor
            # A single array will be created as a column dataframe, thus we have to name the index and not columns
            self.df_rfv.at[cs] = rfv

            check_normalized = np.sum(np.multiply(rfv,rfv))**0.5
            if abs(check_normalized-1) > const.Constants.SMALL_VALUE:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': Warning! RFV for class [' + str(cs) + '] not 1, but [' + str(check_normalized) + '].'
                raise Exception(errmsg)
            else:
                log.Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Check RFV class "' + str(cs) + '" normalized ok [' + str(check_normalized) + '].'
                )

            #
            # Get furthest point of classification to rfv
            # This will be used to accept or reject a classified point to a particular class,
            # once the nearest class is found (in which no class is found then).
            #
            # Minimum value of threshold, don't allow 0's
            dist_furthest = MetricSpaceModel.MINIMUM_THRESHOLD_DIST_TO_RFV
            for i in range(0, class_points.shape[0], 1):
                log.Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + '   Checking ' + str(i) + ':\n\r' + str(class_points[i])
                )
                fv_text = class_points[i]
                dist_vec = rfv - fv_text
                dist = np.sum(np.multiply(dist_vec, dist_vec)) ** 0.5
                if dist > dist_furthest:
                    dist_furthest = dist
                    self.df_rfv_distance_furthest.at[cs, reffv.RefFeatureVector.COL_DISTANCE_TO_RFV_FURTHEST] = dist

            log.Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Class "' + str(cs) + '". Furthest distance = '
                + str(self.df_rfv_distance_furthest[reffv.RefFeatureVector.COL_DISTANCE_TO_RFV_FURTHEST].loc[cs])
            )

        #
        # TODO: Optimization
        # TODO: One idea is to find the biggest difference between features and magnify this difference
        # TODO: (means higher weight on the feature), or some form of optimization algorithm
        # TODO: A simpler way to start is to use IDF where document in this case is the categories. This
        # TODO: IDF measure can then be used as weight to the features. Means we use the above average of the
        # TODO: RFV to start the iteration with the IDF of the feature.
        #
        # Sort
        self.df_x_name = pd.DataFrame(data=x_name)
        self.df_idf = pd.DataFrame(data=idf, index=x_name)
        xy = tdm.TrainingDataModel(
            x = np.array(self.df_rfv.values),
            y = np.array(self.df_rfv.index),
            x_name = np.array(self.df_rfv.columns)
        )
        rfv_friendly = xy.get_print_friendly_x()
        self.df_rfv = self.df_rfv.sort_index()
        self.df_rfv_distance_furthest = self.df_rfv_distance_furthest.sort_index()
        self.df_x_clustered = pd.DataFrame(
            data    = self.x_clustered,
            index   = self.y_clustered,
            columns = x_name
        ).sort_index()

        log.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tx_name:\n\r' + str(self.df_x_name)
            + '\n\r\tIDF:\n\r' + str(self.df_idf)
            + '\n\r\tRFV:\n\r' + str(self.df_rfv)
            + '\n\r\tFurthest Distance:\n\r' + str(self.df_rfv_distance_furthest)
            + '\n\r\tx clustered:\n\r' + str(self.df_x_clustered)
        )

        #
        # Save to file
        # TODO: This needs to be saved to DB, not file
        #
        fpath_x_name = self.dir_path_model + '/' + self.identifier_string + '.x_name.csv'
        self.df_x_name.to_csv(path_or_buf=fpath_x_name, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved x_name shape ' + str(self.df_x_name.shape) + ', filepath "' + fpath_x_name + ']'
            , log_list = self.log_training
        )

        fpath_idf = self.dir_path_model + '/' + self.identifier_string + '.idf.csv'
        self.df_idf.to_csv(path_or_buf=fpath_idf, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved IDF dimensions ' + str(self.df_idf.shape) + ' filepath "' + fpath_idf + '"'
            , log_list = self.log_training
        )

        fpath_rfv = self.dir_path_model + '/' + self.identifier_string + '.rfv.csv'
        self.df_rfv.to_csv(path_or_buf=fpath_rfv, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved RFV dimensions ' + str(self.df_rfv.shape) + ' filepath "' + fpath_rfv + '"'
            , log_list = self.log_training
        )

        fpath_rfv_friendly = self.dir_path_model + '/' + self.identifier_string + '.rfv_friendly.csv'
        self.df_rfv.to_csv(path_or_buf=fpath_rfv_friendly, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved RFV (friendly format) dimensions ' + str(rfv_friendly) + ' filepath "' + fpath_rfv_friendly + '"'
            , log_list = self.log_training
        )

        fpath_dist_furthest = self.dir_path_model + '/' + self.identifier_string + '.rfv.distance.csv'
        self.df_rfv_distance_furthest.to_csv(path_or_buf=fpath_dist_furthest, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved RFV (furthest) dimensions ' + str(self.df_rfv_distance_furthest.shape)
            + ' filepath "' + fpath_dist_furthest + '"'
            , log_list = self.log_training
        )

        fpath_x_clustered = self.dir_path_model + '/' + self.identifier_string + '.x_clustered.csv'
        self.df_x_clustered.to_csv(path_or_buf=fpath_x_clustered, index=True, index_label='INDEX')
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Saved Clustered x with shape ' + str(self.df_x_clustered.shape) + ' filepath "' + fpath_x_clustered + '"'
            , log_list=self.log_training
        )

        # Our servers look to this file to see if RFV has changed
        # It is important to do it last (and fast), after everything is done
        fpath_updated_file = self.dir_path_model + '/' + self.identifier_string + '.lastupdated.txt'
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


def demo_chat_training():
    au.Auth.init_instances()
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_DEBUG_1
    topdir = '/Users/mark.tan/git/mozg'
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
        dir_path_model    = topdir + '/app.data/models',
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
    #exit(0)

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

    topdir = '/Users/mark.tan/git/mozg'
    ms_model = MetricSpaceModel(
        identifier_string = 'demo_msmodel_testdata',
        # Directory to keep all our model files
        dir_path_model    = topdir + '/app.data/models',
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


