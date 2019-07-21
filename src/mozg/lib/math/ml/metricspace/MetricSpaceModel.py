# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import numpy as np
import pandas as pd
import threading
import datetime as dt
import mozg.lib.chat.classification.training.RefFeatureVec as reffv
import mozg.lib.math.ml.TrainingDataModel as tdm
import mozg.common.util.Log as log
from inspect import currentframe, getframeinfo
import mozg.lib.math.Cluster as clstr
import mozg.lib.math.Constants as const
import mozg.lib.math.ml.metricspace.ModelData as modelData
import mozg.lib.math.NumpyUtil as npUtil
import mozg.common.util.Profiling as prf


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

    # Hypersphere max/min Euclidean Distance
    HPS_MAX_EUCL_DIST = 2**0.5
    HPS_MIN_EUCL_DIST = 0

    # Terms for dataframe, etc.
    TERM_CLASS    = 'class'
    TERM_SCORE    = 'score'
    TERM_DIST     = 'dist'
    TERM_DISTNORM = 'distnorm'

    # Matching
    MATCH_TOP = 10

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
            weigh_idf = False,
            do_profiling = True
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
        self.do_profiling = do_profiling

        #
        # All parameter for model is encapsulated in this class
        #
        self.model_data = modelData.ModelData(
            identifier_string = self.identifier_string,
            dir_path_model    = self.dir_path_model
        )

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
            self.train()
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

    #
    # Given our training data x, we get the IDF of the columns x_name
    #
    @staticmethod
    def get_feature_weight_idf(
            x,
            y,
            x_name,
            feature_presence_only_in_label_training_data = True
    ):
        df_tmp = pd.DataFrame(data=x, index=y)

        # Group by the labels y, as they are not unique
        df_agg_sum = df_tmp.groupby(df_tmp.index).sum()
        np_agg_sum = df_agg_sum.values

        # Get presence only by cell, then sum up by columns to get total presence by document
        np_feature_presence = np_agg_sum
        if feature_presence_only_in_label_training_data:
            np_feature_presence = (np_agg_sum>0)*1

        # Sum by column axis=0
        np_feature_presence_sum = np.sum(np_feature_presence, axis=0)
        log.Log.debug(
            str(MetricSpaceModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tAggregated sum by labels:\n\r' + str(np_agg_sum)
            + '\n\r\tPresence array:\n\r' + str(np_feature_presence)
            + '\n\r\tPresence sum:\n\r' + str(np_feature_presence_sum)
            + '\n\r\tx_names: ' + str(x_name) + '.'
        )

        # Total document count
        n_documents = np_feature_presence.shape[0]
        log.Log.important(
            str(MetricSpaceModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Total unique documents/intents to calculate IDF = ' + str(n_documents)
        )

        # If using outdated np.matrix, this IDF will be a (1,n) array, but if using np.array, this will be 1-dimensional vector
        idf = np.log(n_documents / np_feature_presence_sum)
        # Replace infinity with 1 count or log(n_documents)
        idf[idf==np.inf] = np.log(n_documents)
        # If only 1 document, all IDF will be zero, we will handle below
        if n_documents <= 1:
            log.Log.warning(
                str(MetricSpaceModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Only ' + str(n_documents) + ' document in IDF calculation. Setting IDF to 1.'
            )
            idf = np.array([1]*len(x.shape[1]))
        log.Log.debug(
            str(MetricSpaceModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tWeight IDF:\n\r' + str(idf)
        )
        return idf

    #
    # Calculates the normalized distance (0 to 1 magnitude range) of a point v (n dimension)
    # to a set of references (n+1 dimensions or k rows of n dimensional points) by knowing
    # the theoretical max/min of our hypersphere
    #
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
            # Convert to 2 dimensions
            v_ok = np.array([v])

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

        return distance_x_ref

    #
    # Get all class proximity scores to a point
    #
    def calc_proximity_class_score_to_point(
            self,
            # ndarray type of >= 2 dimensions, with 1 row (or 1st dimension length == 1)
            x_distance,
            y_label,
            top = MATCH_TOP
    ):
        if ( type(x_distance) is not np.ndarray ):
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Wrong type "' + type(x_distance) + '" to predict classes. Not ndarray.'
            )

        if x_distance.ndim > 1:
            if x_distance.shape[0] != 1:
                raise Exception(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Expected x has only 1 row got c shape ' + str(x_distance.shape)
                    + '". x = ' + str(x_distance)
                )
            else:
                x_distance = x_distance[0]

        log.Log.debugdebug('x_distance: ' + str(x_distance) + ', y_label ' + str(y_label))

        #
        # Normalize distance to between 0 and 1
        #
        x_distance_norm = x_distance / MetricSpaceModel.HPS_MAX_EUCL_DIST
        log.Log.debugdebug('distance normalized: ' + str(x_distance_norm))

        # Theoretical Inequality check
        check_less_than_max = np.sum(1 * (x_distance_norm > 1))
        check_greater_than_min = np.sum(1 * (x_distance_norm < 0))

        if (check_less_than_max > 0) or (check_greater_than_min > 0):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Distance ' + str(x_distance) + ' fail theoretical inequality test.' \
                     + ' Distance normalized:\n\r' + str(x_distance_norm)
            log.Log.critical(errmsg)
            raise Exception(errmsg)

        # x_score = np.round(100 - x_distance_norm*100, 1)

        df_score = pd.DataFrame({
            MetricSpaceModel.TERM_CLASS: y_label,
            # MetricSpaceModel.TERM_SCORE: x_score,
            MetricSpaceModel.TERM_DIST:  x_distance,
            MetricSpaceModel.TERM_DISTNORM: x_distance_norm
        })
        # Sort distances
        #df_score.sort_values(by=[MetricSpaceModel.TERM_DIST], ascending=True, inplace=True)
        #df_score = df_score[0:top]
        #df_score.reset_index(drop=True, inplace=True)
        #log.Log.debugdebug('DF SCORE 1:\n\r' + str(df_score))

        # Aggregate class by max score, don't make class index
        df_score = df_score.groupby(by=[MetricSpaceModel.TERM_CLASS], as_index=False, axis=0).min()
        log.Log.debugdebug('DF SCORE 2:\n\r' + str(df_score))

        # Put score last (because we need to do groupby().min() above, which will screw up the values
        # as score is in the reverse order with distances) and sort scores
        np_distnorm = np.array(df_score[MetricSpaceModel.TERM_DISTNORM])
        df_score[MetricSpaceModel.TERM_SCORE] = np.round(100 - np_distnorm*100, 1)
        df_score.sort_values(by=[MetricSpaceModel.TERM_SCORE], ascending=False, inplace=True)
        # Make sure indexes are conventional 0,1,2,...
        df_score.reset_index(drop=True, inplace=True)
        log.Log.debugdebug('DF SCORE 3:\n\r' + str(df_score))

        log.Log.debugdebug('x_score:\n\r' + str(df_score))

        return df_score

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
            # ndarray type of >= 2 dimensions
            x
    ):
        prf_start = prf.Profiling.start()

        if type(x) is not np.ndarray:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Wrong type "' + type(x) + '" to predict classes. Not ndarray.'
            )

        if x.ndim < 2:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Expected x dimension >= 2, got ' + str(x.ndim) + '".'
            )

        if x.shape[0] < 1:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Expected x has at least 1 row got c shape ' + str(x.shape) + '".'
            )

        log.Log.debug('x:\n\r' + str(x))

        #
        # Weigh x with idf
        #
        x_weighted = x * self.model_data.idf
        log.Log.debugdebug('x_weighted:\n\r' + str(x_weighted))

        x_weighted_normalized = x_weighted.copy()

        #
        # Normalize x_weighted
        #
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
        log.Log.debugdebug('x_weighted_normalized:\n\r' + str(x_weighted_normalized))

        match_details = {}
        x_classes = []
        top_class_distance = []
        # np array types
        x_distance_to_x_ref = None
        x_distance_to_x_clustered = None

        #
        # Calculate distance to x_ref & x_clustered
        #
        for i in range(0,x_weighted_normalized.shape[0]):
            v = x_weighted_normalized[i]

            # Returns absolute distance
            distance_x_ref = self.calc_distance_of_point_to_x_ref(v=v, x_ref=self.model_data.x_ref)
            distance_x_clustered = self.calc_distance_of_point_to_x_ref(v=v, x_ref=self.model_data.x_clustered)

            # We combine all the reference points, or sub-classes of the classes. Thus each class
            # is represented by more than one point, reference sub_classes.
            x_distance = np.append(distance_x_ref, distance_x_clustered)
            y_distance = np.append(self.model_data.y_ref, self.model_data.y_clustered)
            log.Log.debugdebug('x_distance combined:\n\r' + str(x_distance))
            log.Log.debugdebug('y_distance combined:\n\r' + str(y_distance))

            # Get the score of point relative to all classes.
            df_class_score = self.calc_proximity_class_score_to_point(
                x_distance = x_distance,
                y_label    = y_distance
            )
            log.Log.debugdebug('df_class_score:\n\r' + str(df_class_score))

            # For np array types, we have no choice to do this messy if/else
            if i == 0:
                x_distance_to_x_ref = np.array([distance_x_ref])
                x_distance_to_x_clustered = np.array([distance_x_clustered])
            else:
                x_distance_to_x_ref = np.append(x_distance_to_x_ref, np.array([distance_x_ref]), axis=0)
                x_distance_to_x_clustered = np.append(x_distance_to_x_clustered, np.array([distance_x_clustered]), axis=0)

            x_classes.append( df_class_score[MetricSpaceModel.TERM_CLASS].loc[df_class_score.index[0]] )
            top_class_distance.append( df_class_score[MetricSpaceModel.TERM_DIST].loc[df_class_score.index[0]] )

            match_details[i] = df_class_score

            # Get the top class
            log.Log.debugdebug('x_classes:\n\r' + str(x_classes))

        log.Log.debugdebug('distance to rfv:\n\r' + str(x_distance_to_x_ref))
        log.Log.debugdebug('distance to x_clustered:\n\r' + str(x_distance_to_x_clustered))
        log.Log.debugdebug('top class distance:\n\r' + str(top_class_distance))

        # Mean square error MSE and MSE normalized
        top_class_distance = np.array(top_class_distance)
        mse = np.sum(np.multiply(top_class_distance, top_class_distance))
        mse_norm = mse / (MetricSpaceModel.HPS_MAX_EUCL_DIST ** 2)

        class retclass:
            def __init__(
                    self,
                    predicted_classes,
                    top_class_distance,
                    match_details,
                    mse,
                    mse_norm
            ):
                self.predicted_classes = predicted_classes
                # The top class and shortest distances (so that we can calculate sum of squared error
                self.top_class_distance = top_class_distance
                self.match_details = match_details
                self.mse = mse
                self.mse_norm = mse_norm
                return

        retval = retclass(
            predicted_classes  = np.array(x_classes),
            top_class_distance = top_class_distance,
            match_details      = match_details,
            mse                = mse,
            mse_norm           = mse_norm
        )

        if self.do_profiling:
            prf_dur = prf.Profiling.get_time_dif(prf_start, prf.Profiling.stop())
            # Duration per prediction
            dpp = round(1000 * prf_dur / x.shape[0], 0)
            log.Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING predict_classes(): ' + str(prf_dur)
                + ', time per prediction = ' + str(dpp) + ' milliseconds.'
            )

        return retval

    @staticmethod
    def get_clusters(
            x,
            y,
            x_name
    ):
        class retclass:
            def __init__(self, x_cluster, y_cluster):
                self.x_cluster = x_cluster
                self.y_cluster = y_cluster

        #
        # 1. Cluster training data of the same class.
        #    Instead of a single reference class to represent a single class, we have multiple.
        #

        # Our return values, in the same dimensions with x, y respectively
        x_clustered = None
        y_clustered = None

        for cs in list(set(y)):
            try:
                # Extract only rows of this class
                rows_of_class = x[y == cs]
                if rows_of_class.shape[0] == 0:
                    continue

                log.Log.debugdebug(
                    str(MetricSpaceModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
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
                        matx=rows_of_class,
                        feature_names=x_name,
                        # Not more than 5 clusters per label
                        ncenters=min(5, round(rows_of_class.shape[0] * 2 / 3)),
                        iterations=20
                    )
                    np_class_cluster = class_cluster[clstr.Cluster.COL_CLUSTER_NDARRY]

                # Renormalize x_clustered
                for ii in range(0, np_class_cluster.shape[0], 1):
                    v = np_class_cluster[ii]
                    mag = np.sum(np.multiply(v, v)) ** 0.5
                    # print('Before normalize ' + str(np_class_cluster[ii]))
                    v = v / mag
                    np_class_cluster[ii] = v
                    # print('After normalize ' + str(np_class_cluster[ii]))

                if x_clustered is None:
                    x_clustered = np_class_cluster
                    y_clustered = np.array([cs] * x_clustered.shape[0])
                else:
                    # Append rows (thus 1st dimension at axis index 0)
                    x_clustered = np.append(
                        x_clustered,
                        np_class_cluster,
                        axis=0)
                    # Appending to a 1D array always at axis=0
                    y_clustered = np.append(
                        y_clustered,
                        [cs] * np_class_cluster.shape[0],
                        axis=0)
            except Exception as ex:
                errmsg = str(MetricSpaceModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                         + ': Error for class "' + str(cs) + '", Exception msg ' + str(ex) + '.'
                log.Log.error(errmsg)
                raise Exception(errmsg)

        retobj = retclass(x_cluster=x_clustered, y_cluster=y_clustered)

        log.Log.debug(
            str(MetricSpaceModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '\n\r\tCluster of x\n\r' + str(retobj.x_cluster)
            + '\n\r\ty labels for cluster: ' + str(retobj.y_cluster)
        )
        return retobj

    #
    # TODO: Include training/optimization of vector weights to best define the category and differentiate with other categories.
    # TODO: Currently uses static IDF weights.
    #
    def train(
            self
    ):
        prf_start = prf.Profiling.start()

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
                + '. Using key features remove quartile = ' + str(self.key_features_remove_quartile)
                + ', stop features = [' + str(self.stop_features) + ']'
                + ', weigh by IDF = ' + str(self.weigh_idf)
                , log_list = self.log_training
            )

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
            # The function of these weights are nothing more than dimension reduction
            # TODO: IDF may not be the ideal weights, design an optimal one.
            #
            if self.weigh_idf:
                # Sum x by class
                self.model_data.idf = MetricSpaceModel.get_feature_weight_idf(
                    x      = self.training_data.get_x(),
                    y      = self.training_data.get_y(),
                    x_name = self.training_data.get_x_name()
                )
                # Standardize to at least 2-dimensional, easier when weighting x
                self.model_data.idf = npUtil.NumpyUtil.convert_dimension(
                    arr    = self.model_data.idf,
                    to_dim = 2
                )

                log.Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + '\n\r\tIDF values:\n\r' + str(self.model_data.idf)
                )

                # This will change the x in self.training data
                self.training_data.weigh_x(w=self.model_data.idf[0])
            else:
                self.model_data.idf = np.array([1]*self.training_data.get_x_name().shape[0])
                # Standardize to at least 2-dimensional, easier when weighting x
                self.model_data.idf = npUtil.NumpyUtil.convert_dimension(
                    arr    = self.model_data.idf,
                    to_dim = 2
                )

            #
            # Initizalize model data
            #
            # Refetch again after weigh
            x = self.training_data.get_x()
            y = self.training_data.get_y()
            self.model_data.x_name = self.training_data.get_x_name()
            self.model_data.idf = self.training_data.get_w()

            # Unique y or classes
            # We do this again because after weighing, it will remove bad rows, which might cause some y
            # to disappear
            self.model_data.y_unique = np.array(list(set(y)))

            log.Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + '\n\r\tx weighted by idf and renormalized:\n\r' + str(x)
                + '\n\r\ty\n\r' + str(y)
                + '\n\r\tx_name\n\r' + str(self.model_data.x_name)
                , log_list=self.log_training
            )

            #
            # Get RFV for every command/intent, representative feature vectors by command type
            #

            # 1. Cluster training data of the same intent.
            #    Instead of a single RFV to represent a single intent, we should have multiple.
            xy_clstr = MetricSpaceModel.get_clusters(
                x      = x,
                y      = y,
                x_name = self.model_data.x_name
            )
            self.model_data.x_clustered = xy_clstr.x_cluster
            self.model_data.y_clustered = xy_clstr.y_cluster

            #
            # RFV Derivation
            #
            m = np.zeros((len(self.model_data.y_unique), len(self.model_data.x_name)))
            # Temporary only this data frame
            df_x_ref = pd.DataFrame(
                m,
                columns = self.model_data.x_name,
                index   = self.model_data.y_unique
            )
            self.model_data.df_y_ref_radius = pd.DataFrame(
                {
                    reffv.RefFeatureVector.COL_COMMAND:
                        list(self.model_data.y_unique),
                    reffv.RefFeatureVector.COL_DISTANCE_TO_RFV_FURTHEST:
                        [MetricSpaceModel.HPS_MAX_EUCL_DIST]*len(self.model_data.y_unique),
                },
                index = self.model_data.y_unique
            )

            #
            # Derive x_ref and y_ref
            #
            for cs in self.model_data.y_unique:
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
                df_x_ref.at[cs] = rfv

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
                radius_max = -1
                for i in range(0, class_points.shape[0], 1):
                    p = class_points[i]
                    dist_vec = rfv - p
                    dist = np.sum(np.multiply(dist_vec, dist_vec)) ** 0.5
                    log.Log.debugdebug(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + '   Class ' + str(cs) + ' check point ' + str(i)
                        + ', distance= ' + str(dist) + '. Point ' + str(class_points[i])
                        + ' with RFV ' + str(rfv)
                    )
                    if dist > radius_max:
                        radius_max = dist
                        self.model_data.df_y_ref_radius.at[
                            cs, reffv.RefFeatureVector.COL_DISTANCE_TO_RFV_FURTHEST] = dist

                log.Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Class "' + str(cs) + '". Max Radius = '
                    + str(self.model_data.df_y_ref_radius[
                              reffv.RefFeatureVector.COL_DISTANCE_TO_RFV_FURTHEST].loc[cs])
                )
            df_x_ref.sort_index(inplace=True)
            self.model_data.y_ref = np.array(df_x_ref.index)
            self.model_data.x_ref = np.array(df_x_ref.values)
            log.Log.debug('**************** ' + str(self.model_data.y_ref))

            if self.do_profiling:
                log.Log.important(
                    str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                    + ' PROFILING train(): ' + prf.Profiling.get_time_dif_str(prf_start, prf.Profiling.stop())
                )

            prf_start = prf.Profiling.start()
            self.model_data.persist_model_to_storage()
            if self.do_profiling:
                log.Log.important(
                    str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                    + ' PROFILING persist_model_to_storage(): '
                    + prf.Profiling.get_time_dif_str(prf_start, prf.Profiling.stop())
                )

            prf_start = prf.Profiling.start()
            # For debugging only, not required by model
            self.model_data.persist_training_data_to_storage(
                td = self.training_data
            )
            if self.do_profiling:
                log.Log.important(
                    str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                    + ' PROFILING persist_training_data_to_storage(): '
                    + prf.Profiling.get_time_dif_str(prf_start, prf.Profiling.stop())
                )
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Training exception for identifier "' + str(self.identifier_string) + '".'\
                     + ' Exception message ' + str(ex) + '.'
            log.Log.error(errmsg)
            raise ex
        finally:
            self.__mutex_training.release()

        return

    def load_model_parameters_from_storage(
            self
    ):
        prf_start = prf.Profiling.start()

        try:
            self.__mutex_training.acquire()
            self.model_data.load_model_parameters_from_storage()
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Failed to load model data for identifier "' + self.identifier_string\
                     + '". Exception message: ' + str(ex) + '.'
            log.Log.critical(errmsg)
            raise Exception(errmsg)
        finally:
            self.__mutex_training.release()

        if self.do_profiling:
            log.Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING load_model_parameters_from_storage(): '
                + prf.Profiling.get_time_dif_str(prf_start, prf.Profiling.stop())
            )
        return

    def load_training_data_from_storage(self):
        prf_start = prf.Profiling.start()

        try:
            self.__mutex_training.acquire()
            self.training_data = self.model_data.load_training_data_from_storage()
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Failed to load training data for identifier "' + self.identifier_string\
                     + '". Exception message: ' + str(ex) + '.'
            log.Log.critical(errmsg)
            raise Exception(errmsg)
        finally:
            self.__mutex_training.release()

        if self.do_profiling:
            log.Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING load_training_data_from_storage(): '
                + prf.Profiling.get_time_dif_str(prf_start, prf.Profiling.stop())
            )
        return
