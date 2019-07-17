# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import numpy as np
import pandas as pd
import threading
import json
import datetime as dt
import os
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
# The model treat all points as lying on the hypersphere (normalized measure),
# thus the maximum Euclidean Distance in the positive section of the hypersphere is 2^0.5=1.4142
# The formal problem statement is:
#
#    If given positive real numbers x_a, x_b, x_c, ... and y_a, y_b, y_c, ...
#    and the constraints (x_a^2 + x_b^2 + x_c^2 + ...) = (y_a^2 + y_b^2 + y_c^2 + ...) = 1
#    then
#         (x_a - y_a)^2 + (x_b - y_b)^2 + (x_c - y_c)^2 + ...
#         = 2 - 2(x_a*y_a + x_b_*y_b + x_c*y_c)
#         <= 2
#
class MetricSpaceModel(threading.Thread):

    MINIMUM_THRESHOLD_DIST_TO_RFV = 0.5
    SMALL_VALUE = 0.0000001

    # Hypersphere max/min Euclidean Distance
    HPS_MAX_EUCL_DIST = 2**0.5
    HPS_MIN_EUCL_DIST = 0

    CONVERT_DATAFRAME_INDEX_TO_STR = True

    def __init__(
            self,
            # Unique identifier to identify this set of trained data+other files after training
            identifier_string,
            # Directory to keep all our model files
            dir_path_model,
            # Training data in TrainingDataModel class type
            training_data = None,
            # From all the initial features, how many we should remove by quartile. If 0 means remove nothing.
            key_features_remove_quartile = 0,
            # Initial features to remove, should be an array of numbers (0 index) indicating column to delete in training data
            stop_features = (),
            # If we will create an "IDF" based on the initial features
            weigh_idf = False
    ):
        super(MetricSpaceModel, self).__init__()

        self.identifier_string = identifier_string
        self.dir_path_model = dir_path_model
        self.training_data = training_data
        if self.training_data is not None:
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

        #
        # RFVs
        # Original x, y, x_name in self.training_data
        # All np array type unless stated
        #
        # Order follows x_name
        self.idf = None
        self.rfv_x = None
        self.rfv_y = None
        self.df_rfv_distance_furthest = None
        self.x_clustered = None
        self.y_clustered = None
        self.x_name = None

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

    def calc_distance_of_point_to_x_ref(
            self,
            # Point
            v,
            x_ref
    ):
        log.Log.debugdebug('v: ' + str(v))

        # Create an array with the same number of rows with rfv
        v_ok = v
        if v.ndim == 1:
            v_ok = np.array([v])
            print(v_ok)
        vv = np.repeat(a=v_ok, repeats=x_ref.shape[0], axis=0)
        log.Log.debugdebug('vv repeat: ' + str(vv))

        dif = vv - x_ref
        log.Log.debugdebug('dif with x_ref: ' + str(dif))

        # Square every element in the matrix
        dif2 = np.power(dif, 2)
        log.Log.debugdebug('dif squared: ' + str(dif2))

        # Sum every row to create a single column matrix
        dif2_sum = dif2.sum(axis=1)
        log.Log.debugdebug('dif aggregated sum: ' + str(dif2_sum))

        # Take the square root of every element in the single column matrix as distance
        distance_x_ref = np.power(dif2_sum, 0.5)
        log.Log.debugdebug('distance to x_ref: ' + str(distance_x_ref))

        # Convert to a single row matrix
        distance_x_ref = distance_x_ref.transpose()
        log.Log.debugdebug('distance transposed: ' + str(distance_x_ref))

        # Theoretical Inequality check
        check_less_than_max = np.sum(1 * (distance_x_ref > MetricSpaceModel.HPS_MAX_EUCL_DIST))
        check_greater_than_min = np.sum(1 * (distance_x_ref < MetricSpaceModel.HPS_MIN_EUCL_DIST))

        if (check_less_than_max > 0) or (check_greater_than_min > 0):
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Point ' + str(v) + ' distance to x_ref:\n\r' + str(x_ref) + ' fail theoretical inequality test.'
                + ' Distance tensor:\n\r' + str(distance_x_ref)
            )

        return distance_x_ref

    #
    # Steps to predict classes
    #
    #  1. Weight by IDF and normalize input x
    #  2. Calculate Euclidean Distance of x to each of the x_ref (or rfv)
    #  3. Calculate Euclidean Distance of x to each of the x_clustered (or rfv)
    #  4. Normalize Euclidean Distance so that it is in the range [0,1]
    #
    def predict_classes(
            self,
            # ndarray type
            x
    ):
        x_classes = None

        log.Log.debug('x:\n\r' + str(x))
        # Weigh x with idf
        x_weighted = x * self.idf
        log.Log.debug('x_weighted:\n\r' + str(x_weighted))

        x_weighted_normalized = x_weighted.copy()
        # Normalize x_weighted
        for i in range(0,x_weighted_normalized.shape[0]):
            v = x_weighted_normalized[i]
            mag = np.sum(np.multiply(v,v)**0.5)
            #
            # In the case of 0 magnitude, we put the point right in the center of the hypersphere at 0
            #
            if mag < const.Constants.SMALL_VALUE:
                x_weighted_normalized[i] = np.multiply(v, 0)
            else:
                x_weighted_normalized[i] = v / mag
        log.Log.debug('x_weighted_normalized:\n\r' + str(x_weighted_normalized))

        x_distance_to_x_ref = None
        x_distance_to_x_clustered = None

        # Calculate distance to RFV
        for i in range(0,x_weighted_normalized.shape[0]):
            v = x_weighted_normalized[i]

            distance_x_ref = self.calc_distance_of_point_to_x_ref(v=v, x_ref=self.rfv_x)
            distance_x_clustered = self.calc_distance_of_point_to_x_ref(v=v, x_ref=self.x_clustered)

            if i == 0:
                x_distance_to_x_ref = np.array([distance_x_ref])
                x_distance_to_x_clustered = np.array([distance_x_clustered])
            else:
                x_distance_to_x_ref = np.append(x_distance_to_x_ref, np.array([distance_x_ref]), axis=0)
                x_distance_to_x_clustered = np.append(x_distance_to_x_clustered, np.array([distance_x_clustered]), axis=0)

        x_distance_to_x_ref = x_distance_to_x_ref / MetricSpaceModel.HPS_MAX_EUCL_DIST
        x_distance_to_x_clustered = x_distance_to_x_clustered / MetricSpaceModel.HPS_MAX_EUCL_DIST

        log.Log.debug('distance to rfv:\n\r' + str(x_distance_to_x_ref))
        log.Log.debug('distance to x_clustered:\n\r' + str(x_distance_to_x_clustered))

        # Get weighted score or something
        return x_classes

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
        if self.training_data is None:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Cannot train without training data for identifier "' + self.identifier_string + '"'
            )

        self.__mutex_training.acquire()
        try:
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
            self.x_name = self.training_data.get_x_name()
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
            self.idf = None
            if weigh_idf:
                # Sum x by class
                self.idf = self.get_feature_weight_idf(x=x, y=y, x_name=self.x_name)
                log.Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + '\n\r\tIDF values:\n\r' + str(self.idf)
                )

                # This will change the x in self.training data
                self.training_data.weigh_x(w=self.idf)

                # Refetch again after weigh
                x = self.training_data.get_x()
                y = self.training_data.get_y()
                # Unique y or classes
                # We have to list() the set(), to make it into a proper 1D vector
                self.classes = np.array(list(set(y)))

                log.Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + '\n\r\tx weighted by idf and renormalized:\n\r' + str(x)
                    + '\n\r\ty\n\r' + str(y)
                    + '\n\r\tx_name\n\r' + str(self.x_name)
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
                            feature_names = self.x_name,
                            # Not more than 5 clusters per label
                            ncenters      = min(5, round(rows_of_class.shape[0] * 2/3)),
                            iterations    = 20
                        )
                        np_class_cluster = class_cluster[clstr.Cluster.COL_CLUSTER_NDARRY]

                    # Renormalize x_clustered
                    for ii in range(0,np_class_cluster.shape[0],1):
                        v = np_class_cluster[ii]
                        mag = np.sum(np.multiply(v, v))**0.5
                        print('Before normalize ' + str(np_class_cluster[ii]))
                        v = v / mag
                        np_class_cluster[ii] = v
                        print('After normalize ' + str(np_class_cluster[ii]))

                    if self.x_clustered is None:
                        self.x_clustered = np_class_cluster
                        self.y_clustered = np.array([cs]*self.x_clustered.shape[0])
                    else:
                        # Append rows (thus 1st dimension at axis index 0)
                        self.x_clustered = np.append(self.x_clustered, np_class_cluster, axis=0)
                        # Appending to a 1D array always at axis=0
                        self.y_clustered = np.append(self.y_clustered, [cs]*np_class_cluster.shape[0], axis=0)
                except Exception as ex:
                    log.Log.error(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Error for class "' + str(cs) + '", Exception msg ' + str(ex) + '.'
                        , log_list = self.log_training
                    )
                    raise(ex)

            log.Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + '\n\r\tCluster of x\n\r' + str(self.x_clustered)
                + '\n\r\ty labels for cluster: ' + str(self.y_clustered)
            )

            #
            # RFV Derivation
            #
            m = np.zeros((len(self.classes), len(self.x_name)))
            # self.classes
            self.df_rfv = pd.DataFrame(
                m,
                columns = self.x_name,
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
            self.rfv_y = np.array(self.df_rfv.index)
            self.rfv_x = np.array(self.df_rfv.values)
            log.Log.debug('**************** ' + str(self.rfv_y))

            self.persist_model_to_storage()
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Training exception for identifier "' + str(self.identifier_string) + '".'\
                     + ' Exception message ' + str(ex) + '.'
            log.Log.error(errmsg)
            raise ex
        finally:
            self.__mutex_training.release()
        return

    def persist_model_to_storage(self):
        # Sort
        self.df_x_name = pd.DataFrame(data=self.x_name)
        self.df_idf = pd.DataFrame(data=self.idf, index=self.x_name)
        # We use this training data model class to get the friendly representation of the RFV
        xy = tdm.TrainingDataModel(
            x = np.array(self.df_rfv.values),
            y = np.array(self.df_rfv.index),
            x_name = np.array(self.df_rfv.columns)
        )
        rfv_friendly = xy.get_print_friendly_x()
        #json_rfv_friendly = json.dumps(obj=xy.get_print_friendly_x(), ensure_ascii=False)
        self.df_rfv = self.df_rfv.sort_index()
        self.df_rfv_distance_furthest = self.df_rfv_distance_furthest.sort_index()
        self.df_x_clustered = pd.DataFrame(
            data    = self.x_clustered,
            index   = self.y_clustered,
            columns = self.x_name
        ).sort_index()
        # We use this training data model class to get the friendly representation of the x_clustered
        xy_x_clustered = tdm.TrainingDataModel(
            x = np.array(self.x_clustered),
            y = np.array(self.y_clustered),
            x_name = self.x_name
        )
        x_clustered_friendly = xy_x_clustered.get_print_friendly_x()

        log.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tx_name:\n\r' + str(self.df_x_name)
            + '\n\r\tIDF:\n\r' + str(self.df_idf)
            + '\n\r\tRFV:\n\r' + str(self.df_rfv)
            + '\n\r\tRFV friendly:\n\r' + str(rfv_friendly)
            + '\n\r\tFurthest Distance:\n\r' + str(self.df_rfv_distance_furthest)
            + '\n\r\tx clustered:\n\r' + str(self.df_x_clustered)
            + '\n\r\tx clustered friendly:\n\r' + str(x_clustered_friendly)
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

        fpath_rfv_friendly_json = self.dir_path_model + '/' + self.identifier_string + '.rfv_friendly.json'
        # This file only for debugging
        fpath_rfv_friendly_txt = self.dir_path_model + '/' + self.identifier_string + '.rfv_friendly.txt'
        try:
            # This file only for debugging
            f = open(file=fpath_rfv_friendly_txt, mode='w', encoding='utf-8')
            for i in rfv_friendly.keys():
                line = str(rfv_friendly[i])
                f.write(str(line) + '\n\r')
            f.close()

            with open(fpath_rfv_friendly_json, 'w', encoding='utf-8') as f:
                json.dump(rfv_friendly, f, indent=2)
            f.close()
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Saved rfv friendly ' + str(rfv_friendly) +  ' to file "' + fpath_rfv_friendly_json + '".'
                , log_list=self.log_training
            )
        except Exception as ex:
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Could not create rfv friendly file "' + fpath_rfv_friendly_json
                + '". ' + str(ex)
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

        # This file only for debugging
        fpath_x_clustered_friendly_txt = self.dir_path_model + '/' + self.identifier_string + '.x_clustered_friendly.txt'
        try:
            # This file only for debugging
            f = open(file=fpath_x_clustered_friendly_txt, mode='w', encoding='utf-8')
            for i in x_clustered_friendly.keys():
                line = str(x_clustered_friendly[i])
                f.write(str(line) + '\n\r')
            f.close()
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Saved x_clustered friendly with keys ' + str(x_clustered_friendly.keys())
                + ' filepath "' + fpath_x_clustered_friendly_txt + '"'
                , log_list=self.log_training
            )
        except Exception as ex:
            log.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Could not create x_clustered friendly file "' + fpath_x_clustered_friendly_txt
                + '". ' + str(ex)
            , log_list = self.log_training
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

    def load_model_parameters_from_storage(
            self,
            dir_model
    ):
        # First check the existence of the files
        self.fpath_updated_file = dir_model + '/' + self.identifier_string + '.lastupdated.txt'
        if not os.path.isfile(self.fpath_updated_file):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Last update file "' + self.fpath_updated_file + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)
        # Keep checking time stamp of this file for changes
        self.last_updated_time_rfv = 0

        #
        # We explicitly put a '_ro' postfix to indicate read only, and should never be changed during the program
        #
        self.fpath_x_name = dir_model + '/' + self.identifier_string + '.x_name.csv'
        if not os.path.isfile(self.fpath_x_name):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': x_name file "' + self.fpath_x_name + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)

        #
        # We explicitly put a '_ro' postfix to indicate read only, and should never be changed during the program
        #
        self.fpath_idf = dir_model + '/' + self.identifier_string + '.idf.csv'
        if not os.path.isfile(self.fpath_idf):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': IDF file "' + self.fpath_idf + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)

        self.fpath_rfv = dir_model + '/' + self.identifier_string + '.rfv.csv'
        if not os.path.isfile(self.fpath_rfv):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': RFV file "' + self.fpath_rfv + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)

        self.fpath_rfv_friendly_json = dir_model + '/' + self.identifier_string + '.rfv_friendly.json'
        if not os.path.isfile(self.fpath_rfv_friendly_json):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': RFV friendly file "' + self.fpath_rfv_friendly_json + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)

        self.fpath_rfv_dist = dir_model + '/' + self.identifier_string + '.rfv.distance.csv'
        if not os.path.isfile(self.fpath_rfv):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': RFV furthest distance file "' + self.fpath_rfv_dist + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)

        self.fpath_x_clustered = dir_model + '/' + self.identifier_string + '.x_clustered.csv'
        if not os.path.isfile(self.fpath_rfv):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': x clustered file "' + self.fpath_x_clustered + '" not found!'
            log.Log.error(errmsg)
            raise Exception(errmsg)

        self.__mutex_training.acquire()
        try:
            df_x_name = pd.read_csv(
                filepath_or_buffer = self.fpath_x_name,
                sep       =',',
                index_col = 'INDEX'
            )
            if MetricSpaceModel.CONVERT_DATAFRAME_INDEX_TO_STR:
                # Convert Index column to string
                df_x_name.index = df_x_name.index.astype(str)
            self.x_name = np.array(df_x_name[df_x_name.columns[0]])
            if self.x_name.ndim == 1:
                self.x_name = np.array([self.x_name])

            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': x_name Data: Read ' + str(df_x_name.shape[0]) + ' lines'
                + '\n\r' + str(self.x_name)
            )

            df_idf = pd.read_csv(
                filepath_or_buffer = self.fpath_idf,
                sep       =',',
                index_col = 'INDEX'
            )
            if MetricSpaceModel.CONVERT_DATAFRAME_INDEX_TO_STR:
                # Convert Index column to string
                df_idf.index = df_idf.index.astype(str)
            self.idf = np.array(df_idf[df_idf.columns[0]])
            if self.idf.ndim == 1:
                self.idf = np.array([self.idf])
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': IDF Data: Read ' + str(df_idf.shape[0]) + ' lines'
                + '\n\r' + str(self.idf)
            )

            df_rfv = pd.read_csv(
                filepath_or_buffer = self.fpath_rfv,
                sep       = ',',
                index_col = 'INDEX'
            )
            if MetricSpaceModel.CONVERT_DATAFRAME_INDEX_TO_STR:
                # Convert Index column to string
                df_rfv.index = df_rfv.index.astype(str)
            # Cached the numpy array
            self.rfv_y = np.array(df_rfv.index)
            self.rfv_x = np.array(df_rfv.values)
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': RFV x read ' + str(df_rfv.shape[0]) + ' lines: '
                + '\n\r' + str(self.rfv_x)
                + '\n\rRFV y' + str(self.rfv_y)
            )

            self.df_rfv_distance_furthest = pd.read_csv(
                filepath_or_buffer = self.fpath_rfv_dist,
                sep       = ',',
                index_col = 'INDEX'
            )
            if MetricSpaceModel.CONVERT_DATAFRAME_INDEX_TO_STR:
                # Convert Index column to string
                self.df_rfv_distance_furthest.index = self.df_rfv_distance_furthest.index.astype(str)
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': RFV Furthest Distance Data: Read ' + str(self.df_rfv_distance_furthest.shape[0]) + ' lines'
                + '\n\r' + str(self.df_rfv_distance_furthest)
            )

            df_x_clustered = pd.read_csv(
                filepath_or_buffer = self.fpath_x_clustered,
                sep       = ',',
                index_col = 'INDEX'
            )
            if MetricSpaceModel.CONVERT_DATAFRAME_INDEX_TO_STR:
                # Convert Index column to string
                df_x_clustered.index = df_x_clustered.index.astype(str)
            self.y_clustered = np.array(df_x_clustered.index)
            self.x_clustered = np.array(df_x_clustered.values)
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': x clustered data: Read ' + str(df_x_clustered.shape[0]) + ' lines\n\r'
                + '\n\r' + str(self.x_clustered)
                + '\n\ry_clustered:\n\r' + str(self.y_clustered)
            )
            self.sanity_check()
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Load RFV from file failed for identifier "' + self.identifier_string\
                     + '". Error msg "' + str(ex) + '".'
            log.Log.critical(errmsg)
            raise Exception(errmsg)
        finally:
            self.__mutex_training.release()

    def sanity_check(self):
        # Check RFV is normalized
        for i in range(0,self.rfv_x.shape[0],1):
            cs = self.rfv_y[i]
            rfv = self.rfv_x[i]
            if len(rfv.shape) != 1:
                raise Exception(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': RFV vector must be 1-dimensional, got ' + str(len(rfv.shape))
                    + '-dimensional for class ' + str(cs)
                )
            dist = np.sum(np.multiply(rfv,rfv))**0.5
            if abs(dist-1) > const.Constants.SMALL_VALUE:
                log.Log.critical(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Warning: RFV for command [' + str(cs) + '] not 1, ' + str(dist)
                )
                raise Exception('RFV error')

        for i in range(0,self.x_clustered.shape[0],1):
            cs = self.y_clustered[i]
            fv = self.x_clustered[i]
            dist = np.sum(np.multiply(fv,fv))**0.5
            if abs(dist-1) > const.Constants.SMALL_VALUE:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': Warning: x fv error for class "' + str(cs)\
                         + '" at index ' + str(i) + ' not 1, ' + str(dist)
                log.Log.critical(errmsg)
                raise Exception(errmsg)
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


def unit_test():
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

if __name__ == '__main__':
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_DEBUG_1

    topdir = '/Users/mark.tan/git/mozg'
    #unit_test()
    #demo_chat_training()
    #exit(0)
    ms = MetricSpaceModel(
        identifier_string = 'demo_msmodel_testdata',
        # Directory to keep all our model files
        dir_path_model    = topdir + '/app.data/models',
    )
    ms.load_model_parameters_from_storage(
        dir_model = topdir + '/app.data/models'
    )

    test_x = np.array(
        [
            # 무리 A
            [1.2, 2.0, 1.1, 1.0, 0, 0],
            [2.1, 1.0, 2.4, 1.0, 0, 0],
            [1.5, 1.0, 1.3, 1.0, 0, 0],
            # 무리 B
            [0, 1.1, 2.5, 1.5, 0, 0],
            [0, 2.2, 2.6, 2.4, 0, 0],
            [0, 2.3, 1.7, 2.1, 0, 0],
            # 무리 C
            [0, 0.0, 0, 1.6, 2.1, 3.5],
            [0, 1.4, 0, 2.7, 1.2, 2.4],
            [0, 1.1, 0, 1.3, 1.3, 2.1]
        ]
    )
    test_x_name = np.array(['하나', '두', '셋', '넷', '다섯', '여섯'])
    model_x_name = ms.x_name
    if model_x_name.ndim == 2:
        model_x_name = model_x_name[0]
    print(model_x_name)

    # Reorder by model x_name
    df_x_name = pd.DataFrame(data={'word':model_x_name, 'target_order':range(0,len(model_x_name),1)})
    df_test_x_name = pd.DataFrame(data={'word':test_x_name, 'original_order':range(0,len(test_x_name),1)})
    df_x_name = df_x_name.merge(df_test_x_name)
    # Then order by original order
    df_x_name = df_x_name.sort_values(by=['target_order'], ascending=True)
    # Then the order we need to reorder is the target_order column
    reorder = np.array(df_x_name['original_order'])
    print(df_x_name)
    print(reorder)
    print(test_x)

    test_x_transpose = test_x.transpose()
    print(test_x_transpose)

    reordered_test_x = np.zeros(shape=test_x_transpose.shape)
    print(reordered_test_x)

    for i in range(0,reordered_test_x.shape[0],1):
        reordered_test_x[i] = test_x_transpose[reorder[i]]

    reordered_test_x = reordered_test_x.transpose()
    print(reordered_test_x)

    x_classes = ms.predict_classes(x=reordered_test_x)
    print(x_classes)


