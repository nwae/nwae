# -*- coding: utf-8 -*-

from nwae.ml.metricspace.MetricSpaceModel import MetricSpaceModel
from nwae.ml.networkdesign.NetworkDesign import NetworkDesign
from nwae.ml.nndense.NnDenseModel import NnDenseModel
from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo


#
# Mainly used when training text related data, otherwise the general ModelHelper
# will work for most cases.
#
class TextModelHelper:

    MODEL_NAME_HYPERSPHERE_METRICSPACE = MetricSpaceModel.MODEL_NAME
    MODEL_NAME_NN_DENSE = NnDenseModel.MODEL_NAME
    MODEL_NAME_SEQUENCE_MODEL = 'sequence_model'

    @staticmethod
    def get_model(
            model_name,
            identifier_string,
            dir_path_model,
            training_data,
            # Text specific data
            embed_max_label_val,
            embed_max_sentence_len,
            embed_vocabulary_size,
            model_params            = None,
            train_epochs            = 100,
            train_batch_size        = 128,
            train_optimizer         = 'rmsprop',
            train_loss              = 'sparse_categorical_crossentropy',
            evaluate_metrics        = ('accuracy'),
            confidence_level_scores = None,
            do_profiling            = False,
            is_partial_training     = False
    ):
        if model_name == TextModelHelper.MODEL_NAME_NN_DENSE:
            #
            # Layers Design for Training (may be replaced when old model loads later)
            #
            if model_params is None:
                model_params = NetworkDesign(
                    model_type              = NetworkDesign.MODEL_TEXT_EMBEDDING,
                    max_label_value         = embed_max_label_val,
                    txtemb_max_sentence_len = embed_max_sentence_len,
                    txtemb_vocabulary_size  = embed_vocabulary_size
                )
                model_params.create_network()

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
