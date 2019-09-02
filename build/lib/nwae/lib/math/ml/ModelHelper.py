# -*- coding: utf-8 -*-

import nwae.lib.math.ml.metricspace.MetricSpaceModel as msModel
import nwae.lib.math.ml.deeplearning.Keras as krModel
import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo


class ModelHelper:

    MODEL_NAME_KERAS = krModel.Keras.MODEL_NAME
    MODEL_NAME_HYPERSPHERE_METRICSPACE = msModel.MetricSpaceModel.MODEL_NAME

    @staticmethod
    def get_model(
            model_name,
            identifier_string,
            dir_path_model,
            training_data,
            is_partial_training = False
    ):
        if model_name == ModelHelper.MODEL_NAME_KERAS:
            kr_model = krModel.Keras(
                identifier_string = identifier_string,
                dir_path_model    = dir_path_model,
                training_data     = training_data
            )
            return kr_model
        else:
            ms_model = msModel.MetricSpaceModel(
                identifier_string   = identifier_string,
                dir_path_model      = dir_path_model,
                training_data       = training_data,
                is_partial_training = is_partial_training
            )
            return ms_model
