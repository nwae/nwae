# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import threading
from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
import nwae.math.NumpyUtil as npUtil
import nwae.math.Cluster as clstr
import nwae.math.Constants as const
import nwae.math.optimization.Eidf as eidf
import nwae.ml.metricspace.ModelData as modelData
import nwae.ml.ModelInterface as modelIf
import nwae.utils.Profiling as prf
from nwae.lang.model.FeatureVect import FeatureVector


#
# MetricSpace Machine Learning Model
#
# TODO Convert to a set of linear layers to speed things up
# TODO Training must be incremental (or if not incremental, must be fast < 20s)
# TODO Model must also work for other data types like image/video
#
# Model Description:
#
# Points can be on any dimensional space, however in this model, we project it all on a hypersphere.
# Thus this model is only suitable for directionally sensitive data, and not too dependent on magnitude.
#
# The maximum distance (if euclidean) in the positive section of the hypersphere is 2^0.5=1.4142
# The formal problem statement is:
#
#    If given positive real numbers x_a, x_b, x_c, ... and y_a, y_b, y_c, ...
#    and the constraints (x_a^2 + x_b^2 + x_c^2 + ...) = (y_a^2 + y_b^2 + y_c^2 + ...) = 1
#    then
#         (x_a - y_a)^2 + (x_b - y_b)^2 + (x_c - y_c)^2 + ...
#         = 2 - 2(x_a*y_a + x_b_*y_b + x_c*y_c)
#         <= 2
#
#    Using the above formula we may also derive the expected distance between 2 random points,
#    just as a double integral. Thus
#
#         E(distance)^2 = int{int{2 - 2xy}}dxdy
#                       = 2 - int{int{2xy}}dxdy
#
# Thus on a whole hypersphere the average distance is sqrt(2) because the double integral of 2xy
# is just 0 due to symmetry, but on the positive hyper-hemisphere there is an additional term of
# the double integral of 2xy from 0 to 1.
# This double integral for positive x, y from 0 to 1 theoretically is 2-(1/sqrt(2)).
#
# However we may choose a different metric to speed up calculation, perhaps a linear one.
#
# For all classes, or cluster the radius of the class/cluster is defined as the distance of the
# "center" of the class/cluster to the furthest point. For a class the "center" may be defined
# as the center of mass.
#
# Mean Radius:
#  TODO Given 2 random points on a hypersphere, what is the expected Euclidean distance between them?
#
class MetricSpaceModel(modelIf.ModelInterface):

    MODEL_NAME = 'hypersphere_metricspace'

    # From rescoring training data (using SEARCH_TOPX_RFV=5), we find that
    #    5% quartile score  = 10
    #    25% quartile score = 22
    #    50% quartile score = 32
    #    75% quartile score = 42
    #    95% quartile score = 58
    # Using the above information, we set
    CONFIDENCE_LEVEL_SCORES_DEFAULT = {1: 10, 2: 15, 3: 20, 4:30, 5:40}

    # Our modified IDF that is much better than the IDF in literature
    # TODO For now it is unusable in production because it is too slow!
    USE_OPIMIZED_IDF = True

    # Hypersphere max/min Euclidean Distance
    HPS_MAX_EUCL_DIST = 2**0.5
    HPS_MIN_EUCL_DIST = 0

    #
    # Radius min/max
    # TODO
    #  Derive theoretical value for max cluster radius, such that it is say in the 0.1% low
    #  quantile of distances between 2 points chosen at random on a hypersphere of unit radius.
    # Math Result
    #  - The mean of distance between 2 random points on the positive hemisphere of the hypersphere
    #    is (2^0.5)/2 or 1/(2^0.5) or about 0.7071068
    #  - The 0.1% quartile for 1,000 dimension is about 0.663, and this value gets bigger with higher
    #    dimension. At 2000 dimension the 0.1% quartile is about 0.676
    #
    CLUSTER_RADIUS_MAX = 0.663
    #CLUSTER_RADIUS_MAX = HPS_MAX_EUCL_DIST / 1.618
    # TODO
    #  Max cluster should depend on theoretical value of number of points in a cluster
    N_CLUSTER_MAX = 10
    # TODO
    #  Same thing, should depend on some factor
    IDEAL_MIN_POINTS_PER_CLUSTER = 5

    def __init__(
            self,
            # NN layer configurations, etc.
            model_params,
            # Unique identifier to identify this set of trained data+other files after training
            identifier_string,
            # Directory to keep all our model files
            dir_path_model,
            # Training data in TrainingDataModel class type
            training_data,
            # Confidence scores for class detection
            confidence_level_scores      = None,
            # From all the initial features, how many we should remove by quartile. If 0 means remove nothing.
            key_features_remove_quartile = 0,
            # Initial features to remove, should be an array of numbers (0 index) indicating column to delete in training data
            stop_features                = (),
            # If we will create an "EIDF" based on the initial features
            weigh_idf                    = False,
            do_profiling                 = False,
            # Train only by y/labels and store model files in separate y_id directories
            is_partial_training          = False,
    ):
        super().__init__(
            model_name          = MetricSpaceModel.MODEL_NAME,
            model_params        = model_params,
            identifier_string   = identifier_string,
            dir_path_model      = dir_path_model,
            training_data       = training_data,
            is_partial_training = is_partial_training,
            do_profiling        = do_profiling
        )

        self.confidence_level_scores = confidence_level_scores
        if self.confidence_level_scores is None:
            self.confidence_level_scores = MetricSpaceModel.CONFIDENCE_LEVEL_SCORES_DEFAULT

        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': For identifier "' + str(self.identifier_string)
            + '", using confidence level scores ' + str(self.confidence_level_scores)
        )
        self.y_id = None

        if self.is_partial_training:
            # In this case training data must exist
            unique_y = list(set(list(self.training_data.get_y())))
            if len(unique_y) != 1:
                raise Exception(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': [' + str(self.identifier_string)
                    + '] In partial training mode, must only have 1 unique label, but found '
                    + str(unique_y) + '.'
                )
            self.y_id = int(unique_y[0])

        self.key_features_remove_quartile = key_features_remove_quartile
        self.stop_features = stop_features
        self.weigh_idf = weigh_idf
        # Only train some y/labels and store model files in separate directories by y_id
        self.is_partial_training = is_partial_training

        #
        # All parameter for model is encapsulated in this class
        #
        self.model_data = modelData.ModelData(
            model_name          = MetricSpaceModel.MODEL_NAME,
            identifier_string   = self.identifier_string,
            dir_path_model      = self.dir_path_model,
            is_partial_training = self.is_partial_training,
            y_id                = self.y_id
        )

        return

    #
    # Model interface override
    #
    def is_model_ready(
            self
    ):
        return self.model_data.is_model_ready()

    #
    # Model interface override
    #
    def check_if_model_updated(
            self
    ):
        return self.model_data.check_if_model_updated()

    #
    # Model interface override
    #
    def get_model_features(
            self
    ):
        return npUtil.NumpyUtil.convert_dimension(arr=self.model_data.x_name, to_dim=1)

    def transform_input_for_model(
            self,
            # This should be a list of words as a sentence
            x_input,
            word_freq_model = FeatureVector.COL_FREQUENCY,
    ):
        #
        # This could be numbers, words, etc.
        #
        features_model = list(self.get_model_features())
        # Log.debug(
        #    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
        #    + ': Predicting v = ' + str(v_feature_segmented)
        #    + ' using model features:\n\r' + str(features_model)
        # )

        #
        # Convert sentence to a mathematical object (feature vector)
        #
        model_fv = FeatureVector()
        model_fv.set_freq_feature_vector_template(list_symbols=features_model)

        # Get feature vector of text
        try:
            df_fv = model_fv.get_freq_feature_vector(
                text_list = x_input
            )
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Exception occurred calculating FV for "' + str(x_input) \
                     + '": Exception "' + str(ex) \
                     + '\n\rUsing FV Template:\n\r' + str(model_fv.get_fv_template()) \
                     + ', FV Weights:\n\r' + str(model_fv.get_fv_weights())
            Log.critical(errmsg)
            raise Exception(errmsg)

        # This creates a single row matrix that needs to be transposed before matrix multiplications
        # ndmin=2 will force numpy to create a 2D matrix instead of a 1D vector
        # For now we make it 1D first
        assert word_freq_model in df_fv.columns, '"' + str(word_freq_model) + '" must be in ' + str(df_fv.columns)
        fv_text_1d = np.array(df_fv[word_freq_model].values, ndmin=1)
        if fv_text_1d.ndim != 1:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Expected a 1D vector, got ' + str(fv_text_1d.ndim) + 'D!'
            )
        Log.debugdebug(fv_text_1d)

        x_transformed = npUtil.NumpyUtil.convert_dimension(arr=fv_text_1d, to_dim=2)
        return x_transformed

    #
    # Get all class proximity scores to a point
    #
    def calc_proximity_class_score_to_point(
            self,
            # ndarray type of >= 2 dimensions, with 1 row (or 1st dimension length == 1)
            # This distance metric must be normalized to [0,1] already
            x_distance,
            y_label,
            top = modelIf.ModelInterface.MATCH_TOP
    ):
        prf_start = prf.Profiling.start()

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

        # Log.debugdebug('x_distance: ' + str(x_distance) + ', y_label ' + str(y_label))

        # Theoretical Inequality check
        check_less_than_max = np.sum(1 * (x_distance > 1+const.Constants.SMALL_VALUE))
        check_greater_than_min = np.sum(1 * (x_distance < 0-const.Constants.SMALL_VALUE))

        if (check_less_than_max > 0) or (check_greater_than_min > 0):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Distance ' + str(x_distance) + ' fail theoretical inequality test.'
            Log.critical(errmsg)
            raise Exception(errmsg)

        # x_score = np.round(100 - x_distance_norm*100, 1)

        df_score = pd.DataFrame({
            MetricSpaceModel.TERM_CLASS: y_label,
            # MetricSpaceModel.TERM_SCORE: x_score,
            MetricSpaceModel.TERM_DIST:  x_distance,
        })
        # Sort distances
        # df_score.sort_values(by=[MetricSpaceModel.TERM_DIST], ascending=True, inplace=True)
        # df_score = df_score[0:top]
        # df_score.reset_index(drop=True, inplace=True)
        # Log.debugdebug('DF SCORE 1:\n\r' + str(df_score))

        # Aggregate class by min distance, don't make class index.
        df_score = df_score.groupby(by=[MetricSpaceModel.TERM_CLASS], as_index=False, axis=0).min()
        # Warning! Uncomment only when debugging, this statement printing numpy array takes up to 10ms on Mac Air
        # Log.debugdebug('DF SCORE 2:\n\r' + str(df_score))

        # Put score last (because we need to do groupby().min() above, which will screw up the values
        # as score is in the reverse order with distances) and sort scores
        np_distnorm = np.array(df_score[MetricSpaceModel.TERM_DIST])
        score_vec = np.round(100 - np_distnorm*100, 1)
        df_score[MetricSpaceModel.TERM_SCORE] = score_vec
        # Maximum confidence level is 5, minimum 0
        score_confidence_level_vec = \
            (score_vec >= self.confidence_level_scores[1]) * 1 + \
            (score_vec >= self.confidence_level_scores[2]) * 1 + \
            (score_vec >= self.confidence_level_scores[3]) * 1 + \
            (score_vec >= self.confidence_level_scores[4]) * 1 + \
            (score_vec >= self.confidence_level_scores[5]) * 1
        df_score[MetricSpaceModel.TERM_CONFIDENCE] = score_confidence_level_vec

        # Finally sort by Score
        df_score.sort_values(by=[MetricSpaceModel.TERM_SCORE], ascending=False, inplace=True)

        # Make sure indexes are conventional 0,1,2,...
        df_score = df_score[0:min(top,df_score.shape[0])]
        df_score.reset_index(drop=True, inplace=True)

        # Warning! Uncomment only when debugging, this statement printing numpy array takes up to 10ms on Mac Air
        #Log.debugdebug('x_score:\n\r' + str(df_score))

        if self.do_profiling:
            prf_dur = prf.Profiling.get_time_dif(prf_start, prf.Profiling.stop())
            Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING calc_proximity_class_score_to_point(): ' + str(round(1000*prf_dur,0))
                + ' milliseconds.'
            )

        return df_score

    #
    # Model interface override
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
            x,
            # This will slow down by a whopping 20ms!!
            include_match_details = False,
            top = modelIf.ModelInterface.MATCH_TOP
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

        # Warning! Uncomment only when debugging, this statement printing numpy array takes up to 10ms on Mac Air
        #Log.debugdebug('x:\n\r' + str(x))

        match_details = {}
        x_classes = []
        top_class_distance = []
        mse = 0

        #
        # Calculate distance to x_ref & x_clustered for all the points in the array passed in
        #
        for i in range(x.shape[0]):
            v = npUtil.NumpyUtil.convert_dimension(arr=x[i], to_dim=2)
            predict_result = self.predict_class(
                x           = v,
                include_match_details = include_match_details
            )
            x_classes.append(list(predict_result.predicted_classes))
            top_class_distance.append(predict_result.top_class_distance)
            if include_match_details:
                match_details[i] = predict_result.match_details
            metric = predict_result.top_class_distance
            mse += metric ** 2

        # Mean square error MSE and MSE normalized
        top_class_distance = np.array(top_class_distance)

        class retclass:
            def __init__(
                    self,
                    predicted_classes,
                    top_class_distance,
                    match_details,
                    mse
            ):
                self.predicted_classes = predicted_classes
                # The top class and shortest distances (so that we can calculate sum of squared error
                self.top_class_distance = top_class_distance
                self.match_details = match_details
                self.mse = mse
                return

        retval = MetricSpaceModel.PredictReturnClass(
            predicted_classes       = x_classes,
            predicted_classes_coded = x_classes,
            top_class_distance      = top_class_distance,
            match_details           = match_details,
            mse                     = mse
        )

        if self.do_profiling:
            prf_dur = prf.Profiling.get_time_dif(prf_start, prf.Profiling.stop())
            # Duration per prediction
            dpp = round(1000 * prf_dur / x.shape[0], 0)
            Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING predict_classes(): ' + str(prf_dur)
                + ', time per prediction = ' + str(dpp) + ' milliseconds.'
            )

        return retval

    #
    # Model interface override
    #
    # Required model data for prediction:
    #  1. EIDF
    #  2. x_clustered, y_clustered
    #  3. x_ref, y_ref (if include_rfv=True)
    #
    def predict_class(
            self,
            # ndarray type of >= 2 dimensions, single point/row array
            x,
            include_match_details = False,
            top = modelIf.ModelInterface.MATCH_TOP
    ):
        self.wait_for_model()
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

        if x.shape[0] != 1:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Expected x has 1 row got c shape ' + str(x.shape) + '".'
            )

        # Warning! Uncomment only when debugging, this statement printing numpy array takes up to 10ms on Mac Air
        # Log.debugdebug('predict_class() x:\n\r' + str(x))

        if self.do_profiling:
            prf_dur = prf.Profiling.get_time_dif(prf_start, prf.Profiling.stop())
            # Duration per prediction
            dpp = round(1000 * prf_dur, 0)
            Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING predict_class(): ' + str(prf_dur)
                + ', sanity check = ' + str(dpp) + ' milliseconds.'
                + ' x shape ' + str(x.shape)
            )

        #
        # Weigh x with idf
        # TODO
        #   Для Китайского языка это вычисление занимает 1ms, но в Тайском языком 25ms. Почему?
        #
        prf_start_x_weigh = prf.Profiling.start()
        x_weighted = x * self.model_data.idf
        # Warning! Uncomment only when debugging, this statement printing numpy array takes up to 10ms on Mac Air
        #Log.debugdebug(
        #    'x_weighted shape: ' + str(x_weighted.shape) + ', x shape: ' + str(x.shape)
        #    + ', eidf shape: ' + str(self.model_data.idf.shape)
        #)

        v = x_weighted

        if self.do_profiling:
            prf_dur = prf.Profiling.get_time_dif(prf_start_x_weigh, prf.Profiling.stop())
            # Duration per prediction
            dpp = round(1000 * prf_dur, 0)
            Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING predict_class(): ' + str(prf_dur)
                + ', multiply eidf weights on v = ' + str(dpp) + ' milliseconds.'
                + ' v shape ' + str(v.shape) + ', v dtype ' + str(v.dtype)
                + ' eidf shape ' + str(self.model_data.idf.shape) + ', eidf dtype ' + str(self.model_data.idf.dtype)
            )

        #
        # Normalize x_weighted
        #
        prf_start_normalize = prf.Profiling.start()
        mag = np.sum(np.multiply(v,v)**0.5)
        #
        # In the case of 0 magnitude, we put the point right in the center of the hypersphere at 0
        #
        if mag < const.Constants.SMALL_VALUE:
            v = np.multiply(v, 0)
        else:
            v = v / mag
        # Warning! Uncomment only when debugging, this statement printing numpy array takes up to 10ms on Mac Air
        # Log.debugdebug('v normalized:\n\r' + str(v))
        if self.do_profiling:
            prf_dur = prf.Profiling.get_time_dif(prf_start_normalize, prf.Profiling.stop())
            # Duration per prediction
            dpp = round(1000 * prf_dur, 0)
            Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING predict_class(): ' + str(prf_dur)
                + ', normalize v = ' + str(dpp) + ' milliseconds.'
            )

        #
        # Our distance metric
        #
        prf_start_layer_1_neural_network = prf.Profiling.start()
        metric_distance = npUtil.NumpyUtil.calc_metric_angle_distance(
            v = v,
            x = self.model_data.x_clustered
        )
        # Normalize to [0,1]
        metric_distance = metric_distance / (np.pi / 2)
        if self.do_profiling:
            prf_dur = prf.Profiling.get_time_dif(prf_start_layer_1_neural_network, prf.Profiling.stop())
            # Duration per prediction
            dpp = round(1000 * prf_dur, 0)
            Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING predict_class(): ' + str(prf_dur)
                + ', calculation of layer 1 neural network = ' + str(dpp) + ' milliseconds.'
            )

        # Get the score of point relative to all classes.
        df_class_score = self.calc_proximity_class_score_to_point(
            x_distance = metric_distance,
            y_label    = self.model_data.y_clustered,
            top        = top
        )
        # Warning! Uncomment only when debugging, this statement printing numpy array takes up to 10ms on Mac Air
        # Log.debugdebug('df_class_score:\n\r' + str(df_class_score))

        top_classes_label = list(df_class_score[MetricSpaceModel.TERM_CLASS])
        top_class_distance = df_class_score[MetricSpaceModel.TERM_DIST].loc[df_class_score.index[0]]

        # Get the top class
        # Log.debugdebug('x_classes:\n\r' + str(top_classes_label))
        # Log.debugdebug('Class for point:\n\r' + str(top_classes_label))
        # Log.debugdebug('distance metric to x_clustered:\n\r' + str(metric_distance))
        # Log.debugdebug('top class distance:\n\r' + str(top_class_distance))

        # Mean square error MSE and MSE normalized
        top_class_distance = np.array(top_class_distance)

        match_details = None
        if include_match_details:
            match_details = df_class_score
        retval = MetricSpaceModel.PredictReturnClass(
            predicted_classes       = np.array(top_classes_label),
            predicted_classes_coded = np.array(top_classes_label),
            top_class_distance      = top_class_distance,
            match_details           = match_details,
        )

        if self.do_profiling:
            prf_dur = prf.Profiling.get_time_dif(prf_start, prf.Profiling.stop())
            # Duration per prediction
            dpp = round(1000 * prf_dur / x.shape[0], 0)
            Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING predict_class(): ' + str(prf_dur)
                + ', prediction time = ' + str(dpp) + ' milliseconds.'
            )

        return retval

    @staticmethod
    def get_clusters(
            x,
            y,
            x_name,
            log_training = None
    ):
        class retclass:
            def __init__(self, x_cluster, y_cluster, y_cluster_radius):
                self.x_cluster = x_cluster
                self.y_cluster = y_cluster
                self.y_cluster_radius = y_cluster_radius

        #
        # 1. Cluster training data of the same class.
        #    Instead of a single reference class to represent a single class, we have multiple.
        #

        # Our return values, in the same dimensions with x, y respectively
        x_clustered = None
        y_clustered = None
        y_clustered_radius = None

        #
        # Loop by unique class labels
        #
        for cs in list(set(y)):
            try:
                # Extract only rows of this class
                rows_of_class = x[y == cs]
                if rows_of_class.shape[0] == 0:
                    continue

                # Log.debugdebug(
                #     str(MetricSpaceModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                #     + '\n\r\tRows of class "' + str(cs) + ':'
                #     + '\n\r' + str(rows_of_class)
                # )

                #
                # Cluster intent
                #
                # We start with 1 cluster, until the radius of the clusters satisfy our max radius condition
                #
                max_cluster_radius_condition_met = False

                # Start with 1 cluster
                n_clusters = 0
                while not max_cluster_radius_condition_met:
                    n_clusters += 1
                    np_cluster_centers = None
                    np_cluster_radius = None
                    Log.info(
                        str(MetricSpaceModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Cls label ' + str(cs) + ' Loop #' + str(n_clusters)
                        , log_list = log_training
                    )

                    # Do clustering to n_clusters only if it is less than the number of points
                    if rows_of_class.shape[0] > n_clusters:
                        cluster_result = clstr.Cluster.cluster(
                            matx          = rows_of_class,
                            feature_names = x_name,
                            ncenters      = n_clusters,
                            iterations    = 20
                        )
                        np_cluster_centers = cluster_result.np_cluster_centers
                        np_cluster_labels = cluster_result.np_cluster_labels
                        np_cluster_radius = cluster_result.np_cluster_radius
                        # Remember this distance is calculated without a normalized cluster center, but we ignore for now
                        val_max_cl_radius = max(np_cluster_radius)
                    else:
                        #
                        # If number of clusters == number of points, then the cluster centers
                        # are just the cluster points, and the cluster radiuses are all 0
                        #
                        np_cluster_centers = np.array(rows_of_class)
                        np_cluster_radius = np.array([cs] * rows_of_class.shape[0])
                        val_max_cl_radius = max(np_cluster_radius)

                    # If
                    #  1. Number of clusters already equal to points
                    #  2. (Max cluster radius < CLUSTER_RADIUS_MAX) AND (Points Per Cluster is below or equal 5)
                    #  3. Number of clusters already exceed our limit N_CLUSTER_MAX
                    # then our condition is met
                    points_per_cluster = rows_of_class.shape[0] / n_clusters
                    max_cluster_radius_condition_met = \
                        (rows_of_class.shape[0] <= n_clusters+1) \
                        or (
                                (val_max_cl_radius <= MetricSpaceModel.CLUSTER_RADIUS_MAX)
                                and (points_per_cluster <= MetricSpaceModel.IDEAL_MIN_POINTS_PER_CLUSTER)
                        ) \
                        or (n_clusters >= MetricSpaceModel.N_CLUSTER_MAX)

                    #
                    # If not the best cluster points, we keep trying with 1 more center
                    #
                    if not max_cluster_radius_condition_met:
                        continue

                    #
                    # Put the cluster center back on the hypersphere surface, renormalize cluster centers
                    #
                    for ii in range(0, np_cluster_centers.shape[0], 1):
                        cluster_label = ii
                        cc = np_cluster_centers[ii]
                        mag = np.sum(np.multiply(cc, cc)) ** 0.5
                        cc = cc / mag
                        np_cluster_centers[ii] = cc

                    if x_clustered is None:
                        # First time initializing x_clustered
                        x_clustered = np_cluster_centers
                        y_clustered = np.array([cs] * x_clustered.shape[0])
                        y_clustered_radius = np_cluster_radius
                    else:
                        # Log.debugdebug(
                        #     str(MetricSpaceModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        #     + ': Concatenating x_cls row:\n\r' + str(np_cluster_centers)
                        #     + '\n\nto x_cls array:\n\r' + str(x_clustered)
                        #     + '\n\ry_cls row:\n\r' + str([cs] * np_cluster_centers.shape[0])
                        #     + '\n\ry_radius row:\n\r' + str(np_cluster_radius) + ' to y_cls_radius ' + str(y_clustered_radius)
                        # )
                        # Append rows (thus 1st dimension at axis index 0)
                        x_clustered = np.append(
                            x_clustered,
                            np_cluster_centers,
                            axis=0)
                        # Appending to a 1D array always at axis=0
                        y_clustered = np.append(
                            y_clustered,
                            [cs] * np_cluster_centers.shape[0],
                            axis=0)
                        y_clustered_radius = np.append(
                            y_clustered_radius,
                            np_cluster_radius,
                            axis=0)
            except Exception as ex:
                errmsg = str(MetricSpaceModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                         + ': Error for class "' + str(cs) + '", Exception msg ' + str(ex) + '.'
                Log.error(errmsg, log_list=log_training)
                raise Exception(errmsg)

        retobj = retclass(x_cluster=x_clustered, y_cluster=y_clustered, y_cluster_radius=y_clustered_radius)

        # Log.debugdebug(
        #     str(MetricSpaceModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
        #     + '\n\r\tCluster of x\n\r' + str(retobj.x_cluster)
        #     + '\n\r\ty labels for cluster: ' + str(retobj.y_cluster)
        # )
        return retobj

    #
    # Train from partial model files
    #
    def train_from_partial_models(
            self,
            write_model_to_storage = True,
            write_training_data_to_storage = False,
            # Log training events
            logs = None
    ):
        #
        # Load EIDF first
        # TODO How to ensure there are no missing words?
        #
        x_name = self.training_data.get_x_name()
        try:
            if type(logs) is list:
                self.logs_training = logs
            else:
                self.logs_training = []

            Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Initializing IDF object.. try to read from file first'
                , log_list = self.logs_training
            )
            # Try to read from file
            df_eidf_file = eidf.Eidf.read_eidf_from_storage(
                dir_path_model    = self.dir_path_model,
                identifier_string = self.identifier_string,
                x_name            = x_name,
                log_training      = self.logs_training
            )
            Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Successfully Read EIDF from file'
                , log_list = self.logs_training
            )
            self.model_data.idf = np.array(df_eidf_file[eidf.Eidf.STORAGE_COL_EIDF])
        except Exception as ex_eidf:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': No EIDF from file available. Exception ' + str(ex_eidf)
            Log.critical(errmsg, log_list=self.logs_training)
            raise Exception(errmsg)

        # Standardize to at least 2-dimensional, easier when weighting x
        self.model_data.idf = npUtil.NumpyUtil.convert_dimension(
            arr    = self.model_data.idf,
            to_dim = 2
        )

        #
        # Combines
        #
        self.model_data.load_model_from_partial_trainings_data(
            td_latest = self.training_data,
            log_training = self.logs_training
        )
        return self.logs_training

    #
    # Model interface override
    #
    # TODO: Include training/optimization of vector weights to best define the category and differentiate with other categories.
    # TODO: Currently uses static IDF weights.
    #
    def train(
            self,
            write_model_to_storage = True,
            write_training_data_to_storage = False,
            # Option to train a single y ID/label
            y_id = None,
            # To keep training logs here for caller's reference
            log_list_to_populate = None
    ):
        prf_start = prf.Profiling.start()

        if self.training_data is None:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Cannot train without training data for identifier "' + self.identifier_string + '"'
            )

        self.mutex_training.acquire()
        try:
            if type(log_list_to_populate) is list:
                self.logs_training = log_list_to_populate
            else:
                self.logs_training = []

            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Training for identifier=' + self.identifier_string
                + ', y_id ' + str(y_id)
                + '. Using key features remove quartile = ' + str(self.key_features_remove_quartile)
                + ', stop features = [' + str(self.stop_features) + ']'
                + ', weigh by EIDF = ' + str(self.weigh_idf)
                , log_list = self.logs_training
            )

            #
            # Here training data must be prepared in the correct format already
            # Значит что множество свойств уже объединено как одно (unified features)
            #
            # Log.debugdebug(
            #     str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            #     + '\n\r\tTraining data:\n\r' + str(self.training_data.get_x().tolist())
            #     + '\n\r\tx names: ' + str(self.training_data.get_x_name())
            #     + '\n\r\ty labels: ' + str(self.training_data.get_y())
            # )

            #
            # Get IDF first
            # The function of these weights are nothing more than dimension reduction
            # TODO: IDF may not be the ideal weights, design an optimal one.
            #
            if self.weigh_idf:
                if MetricSpaceModel.USE_OPIMIZED_IDF:
                    try:
                        Log.info(
                            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': Initializing EIDF object.. try to read from file first',
                            log_list = self.logs_training
                        )
                        # Try to read from file
                        df_eidf_file = eidf.Eidf.read_eidf_from_storage(
                            dir_path_model = self.dir_path_model,
                            identifier_string = self.identifier_string,
                            x_name = self.training_data.get_x_name()
                        )
                        Log.info(
                            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': Successfully Read EIDF from file.',
                            log_list = self.logs_training
                        )
                        self.model_data.idf = np.array(df_eidf_file[eidf.Eidf.STORAGE_COL_EIDF])
                    except Exception as ex_eidf:
                        Log.critical(
                            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': No EIDF from file available. Recalculating EIDF..',
                            log_list = self.logs_training
                        )
                        idf_opt_obj = eidf.Eidf(
                            x      = self.training_data.get_x(),
                            y      = self.training_data.get_y(),
                            x_name = self.training_data.get_x_name()
                        )
                        idf_opt_obj.optimize(
                            initial_w_as_standard_idf = True
                        )
                        self.model_data.idf = idf_opt_obj.get_w()
                else:
                    # Sum x by class
                    self.model_data.idf = eidf.Eidf.get_feature_weight_idf_default(
                        x      = self.training_data.get_x(),
                        y      = self.training_data.get_y(),
                        x_name = self.training_data.get_x_name()
                    )
            else:
                self.model_data.idf = np.array([1.0]*self.training_data.get_x_name().shape[0], dtype=float)

            # Standardize to at least 2-dimensional, easier when weighting x
            self.model_data.idf = npUtil.NumpyUtil.convert_dimension(
                arr    = self.model_data.idf,
                to_dim = 2
            )

            Log.debugdebug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + '\n\r\tEIDF values:\n\r' + str(self.model_data.idf),
                log_list = self.logs_training
            )

            #
            # Re-weigh again. This will change the x in self.training data
            #
            self.training_data.weigh_x(
                w = self.model_data.idf[0]
            )

            #
            # Initizalize model data
            #
            # Refetch again after weigh
            x = self.training_data.get_x()
            y = self.training_data.get_y()
            self.model_data.x_name = self.training_data.get_x_name()

            # Unique y or classes
            # We do this again because after weighing, it will remove bad rows, which might cause some y
            # to disappear
            self.model_data.y_unique = np.array(list(set(y)))

            Log.debugdebug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + '\n\r\tx weighted by idf and renormalized:\n\r' + str(x.tolist())
                + '\n\r\ty\n\r' + str(y)
                + '\n\r\tx_name\n\r' + str(self.model_data.x_name)
                , log_list = self.logs_training
            )

            #
            # Get RFV for every command/intent, representative feature vectors by command type
            #

            # 1. Cluster training data of the same intent.
            #    Instead of a single RFV to represent a single intent, we should have multiple.
            xy_clstr = MetricSpaceModel.get_clusters(
                x      = x,
                y      = y,
                x_name = self.model_data.x_name,
                log_training = self.logs_training
            )
            self.model_data.x_clustered = xy_clstr.x_cluster
            self.model_data.y_clustered = xy_clstr.y_cluster
            self.model_data.y_clustered_radius = xy_clstr.y_cluster_radius

            #
            # RFV Derivation
            #
            m = np.zeros((len(self.model_data.y_unique), len(self.model_data.x_name)))
            # Temporary only this data frame
            df_x_ref = pd.DataFrame(
                m,
                columns = self.model_data.x_name,
                index   = list(self.model_data.y_unique)
            )
            #print('***** y unique type: ' + str(type(self.model_data.y_unique)) + ', df_x_ref: '  + str(df_x_ref))
            self.model_data.df_y_ref_radius = pd.DataFrame(
                {
                    MetricSpaceModel.TERM_CLASS: list(self.model_data.y_unique),
                    MetricSpaceModel.TERM_RADIUS: [MetricSpaceModel.HPS_MAX_EUCL_DIST]*len(self.model_data.y_unique),
                },
                index = list(self.model_data.y_unique)
            )
            #print('***** df_x_ref: '  + str(self.model_data.df_y_ref_radius))


            #
            # Derive x_ref and y_ref
            #
            for cs in self.model_data.y_unique:
                Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Doing class [' + str(cs) + ']'
                    , log_list = self.logs_training
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
                    Log.warning(errmsg, log_list=self.training_data)
                    raise Exception(errmsg)
                else:
                    Log.debug(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Check RFV class "' + str(cs) + '" normalized ok [' + str(check_normalized) + '].'
                        , log_list = self.logs_training
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
                    Log.debugdebug(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + '   Class ' + str(cs) + ' check point ' + str(i)
                        + ', distance= ' + str(dist) + '. Point ' + str(class_points[i])
                        + ' with RFV ' + str(rfv)
                        , log_list = self.logs_training
                    )
                    if dist > radius_max:
                        radius_max = dist
                        self.model_data.df_y_ref_radius[MetricSpaceModel.TERM_RADIUS].at[cs] = dist

                Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Class "' + str(cs) + '". Max Radius = '
                    + str(self.model_data.df_y_ref_radius[MetricSpaceModel.TERM_RADIUS].loc[cs])
                    , log_list = self.logs_training
                )
            df_x_ref.sort_index(inplace=True)
            self.model_data.y_ref = np.array(df_x_ref.index)
            self.model_data.x_ref = np.array(df_x_ref.values)
            Log.debug('**************** ' + str(self.model_data.y_ref))

            if self.do_profiling:
                Log.important(
                    str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                    + ' PROFILING train(): ' + prf.Profiling.get_time_dif_str(prf_start, prf.Profiling.stop()),
                    log_list = self.logs_training
                )

            if write_model_to_storage:
                self.persist_model_to_storage()
            if write_training_data_to_storage or (self.is_partial_training):
                self.persist_training_data_to_storage(td=self.training_data)
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Training exception for identifier "' + str(self.identifier_string) + '".'\
                     + ' Exception message ' + str(ex) + '.'
            Log.error(errmsg)
            raise ex
        finally:
            self.mutex_training.release()

        return self.logs_training

    def persist_model_to_storage(
            self
    ):
        prf_start = prf.Profiling.start()
        self.model_data.persist_model_to_storage(
            log_training = self.logs_training
        )
        if self.do_profiling:
            Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING persist_model_to_storage(): '
                + prf.Profiling.get_time_dif_str(prf_start, prf.Profiling.stop())
                , log_list = self.logs_training
            )
        return

    #
    # Model interface override
    #
    def load_model_parameters(
            self
    ):
        prf_start = prf.Profiling.start()

        try:
            self.mutex_training.acquire()
            self.model_data.load_model_parameters_from_storage()
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Failed to load model data for identifier "' + self.identifier_string\
                     + '". Exception message: ' + str(ex) + '.'
            Log.critical(errmsg)
            raise Exception(errmsg)
        finally:
            self.mutex_training.release()

        if self.do_profiling:
            Log.important(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING load_model_parameters_from_storage(): '
                + prf.Profiling.get_time_dif_str(prf_start, prf.Profiling.stop())
            )
        return
