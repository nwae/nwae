# -*- coding: utf-8 -*-

from nwae.ml.metricspace.MetricSpaceModel import MetricSpaceModel
from nwae.ml.nndense.NnDenseModel import NnDenseModel
from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo


#
# For prediction purposes, use this model helper without the need of creating
# model parameters, neural networks, etc. Those will be read from model files.
# For training purposes, it is best to use the specific type model helper
# classes.
#
class ModelHelper:

    MODEL_NAME_HYPERSPHERE_METRICSPACE = MetricSpaceModel.MODEL_NAME
    MODEL_NAME_NN_DENSE = NnDenseModel.MODEL_NAME
    MODEL_NAME_SEQUENCE_MODEL = 'sequence_model'

    @staticmethod
    def get_model(
            model_name,
            model_params,
            identifier_string,
            dir_path_model,
            training_data,
            train_epochs        = 10,
            train_batch_size    = 128,
            train_optimizer     = 'rmsprop',
            train_loss          = 'categorical_crossentropy',
            evaluate_metrics    = ('accuracy'),
            confidence_level_scores = None,
            do_profiling = False,
            is_partial_training = False
    ):
        if model_name == ModelHelper.MODEL_NAME_NN_DENSE:
            model = NnDenseModel(
                identifier_string = identifier_string,
                model_params      = model_params,
                dir_path_model    = dir_path_model,
                training_data     = training_data,
                train_epochs      = train_epochs,
                train_batch_size  = train_batch_size,
                train_optimizer   = train_optimizer,
                train_loss        = train_loss,
                evaluate_metrics  = evaluate_metrics,
                do_profiling      = do_profiling
            )
        else:
            model = MetricSpaceModel(
                identifier_string       = identifier_string,
                model_params            = model_params,
                dir_path_model          = dir_path_model,
                training_data           = training_data,
                is_partial_training     = is_partial_training,
                confidence_level_scores = confidence_level_scores,
                do_profiling            = do_profiling
            )

        return model
