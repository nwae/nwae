# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
from nwae.ml.ModelInterface import ModelInterface
from nwae.math.NumpyUtil import NumpyUtil
import nwae.utils.UnitTest as ut
import nwae.ml.networkdesign.NetworkDesign as nwdesign
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# TODO Don't rely on buggy TF/Keras, write our own
try:
    from keras.utils import to_categorical
    from tensorflow.keras.models import load_model
    # This one will not work in a multi-threaded environment
    #from keras.models import load_model
except Exception as ex_keras:
    Log.warning(
        str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
        + ': Exception importing Keras modules: ' + str(ex_keras)
    )


class NnDenseModel(ModelInterface):

    MODEL_NAME = 'nn_dense'

    CONFIDENCE_LEVEL_SCORES_DEFAULT = {1: 10, 2: 15, 3: 20, 4:30, 5:40}

    def __init__(
            self,
            # NN layer configurations, etc.
            model_params,
            # Unique identifier to identify this set of trained data+other files after training
            identifier_string,
            # Directory to keep all our model files
            dir_path_model,
            # If for text data, we map words to the x integers for usage of Model after training
            train_epochs        = 10,
            train_batch_size    = None,
            train_optimizer     = 'rmsprop',
            # 'sparse_categorical_crossentropy', etc.
            train_loss          = 'categorical_crossentropy',
            evaluate_metrics    = ('accuracy'),
            # Confidence scores for class detection
            confidence_level_scores = None,
            # Training data in TrainingDataModel class type
            training_data       = None,
            do_profiling        = False,
            # Train only by y/labels and store model files in separate y_id directories
            is_partial_training = False
    ):
        super().__init__(
            model_name          = NnDenseModel.MODEL_NAME,
            model_params        = model_params,
            identifier_string   = identifier_string,
            dir_path_model      = dir_path_model,
            training_data       = training_data,
            is_partial_training = is_partial_training,
            do_profiling        = do_profiling
        )

        self.train_epochs = train_epochs
        self.train_batch_size = train_batch_size
        self.train_optimizer = train_optimizer
        self.train_loss = train_loss
        self.evaluate_metrics = evaluate_metrics
        # Keras accepts only list/tuple type
        if type(self.evaluate_metrics) not in (list, tuple):
            self.evaluate_metrics = [self.evaluate_metrics]
        self.confidence_level_scores = confidence_level_scores
        if self.confidence_level_scores is None:
            self.confidence_level_scores = NnDenseModel.CONFIDENCE_LEVEL_SCORES_DEFAULT

        self.fpath_model           = self.model_prefix + '.tf.network'
        self.fpath_model_x_one_hot = self.model_prefix + '.tf.x_one_hot_dict.csv'
        self.fpath_model_y_one_hot = self.model_prefix + '.tf.y_one_hot_dict.csv'

        self.network = None
        self.network_layer_config = None
        # In code:word format
        self.x_one_hot_dict = None
        # In word:code format usable for transforming user inputs
        self.x_one_hot_dict_inverse = None
        # In code:label format
        self.y_one_hot_dict = None
        return

    #
    # Model interface override
    #
    def get_model_features(
            self
    ):
        if self.x_one_hot_dict is not None:
            return np.array(self.x_one_hot_dict.values())
        else:
            return None

    def transform_input_for_model(
            self,
            # For the model to interpret and transform in to x usable for model input
            # (e.g. map using one-hot dictionaries)
            x_input,
            word_freq_model = None,
    ):
        try:
            Log.debugdebug('***** x input: ' + str(x_input))
            # We expect x_input to be an np array of words
            if type(x_input) is np.ndarray:
                x_input = x_input.tolist()
            if type(x_input) not in (list, tuple):
                raise Exception(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Model "' + str(self.identifier_string)
                    + '". Expect list/tuple type, got type "' + str(type(x_input))
                    + '" for x input: ' + str(x_input)
                )
            if self.x_one_hot_dict_inverse is None:
                raise Exception(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Model "' + str(self.identifier_string) + '" x one hot not yet initialized!'
                )
            x = []

            for i in range(len(x_input)):
                word = x_input[i]
                if word in self.x_one_hot_dict_inverse.keys():
                    x.append(self.x_one_hot_dict_inverse[word])
                else:
                    Log.warning(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Model "' + str(self.identifier_string) + '", could not map input value "' + str(word)
                        + '" to code x. Not in x one hot dictionary.'
                    )

            # TODO Pad with 0's to satisfy neural network in put length
            input_shape = self.network.layers[0].input_shape
            input_len = input_shape[1]
            Log.debugdebug('***** INPUT SHAPE ' + str(input_shape) + ', len ' + str(input_len) + ', x = ' + str(x))
            while len(x) < input_len:
                x = [0] + x
            Log.debugdebug('  ***** padded x: ' + str(x))

            x = np.array(x)
            x_transformed = NumpyUtil.convert_dimension(arr=x, to_dim=2)
            Log.debugdebug('  ***** transformed x: ' + str(x_transformed))

            return x_transformed
        except Exception as ex:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Model "' + str(self.identifier_string) + '", exception tranforming ' + str(x_input)
                + '. Exception: ' + str(ex)
            )

    #
    # Maps y output to original labels
    #
    def __map_top_matches_to_one_hot_labels(
            self,
            top_matches
    ):
        if (self.y_one_hot_dict is not None) and (top_matches is not None):
            top_x_matches_original = []
            for i in range(top_matches.shape[0]):
                match = top_matches[i]
                if match in self.y_one_hot_dict.keys():
                    top_x_matches_original.append(self.y_one_hot_dict[match])
                else:
                    top_x_matches_original.append(None)
            top_x_matches_original = np.array(top_x_matches_original)
        else:
            # Без one hot dict
            top_x_matches_original = top_matches
        return top_x_matches_original

    def __calculate_match_details(
            self,
            # Both as numpy ndarray
            top_matches,
            top_probs
    ):
        distances = 1 - top_probs
        scores = top_probs*100

        df_details = pd.DataFrame({
            ModelInterface.TERM_CLASS: top_matches,
            ModelInterface.TERM_DIST:  distances,
            ModelInterface.TERM_SCORE: scores
        })

        # Maximum confidence level is 5, minimum 0
        score_confidence_level_vec = \
            (scores >= self.confidence_level_scores[1]) * 1 + \
            (scores >= self.confidence_level_scores[2]) * 1 + \
            (scores >= self.confidence_level_scores[3]) * 1 + \
            (scores >= self.confidence_level_scores[4]) * 1 + \
            (scores >= self.confidence_level_scores[5]) * 1
        df_details[ModelInterface.TERM_CONFIDENCE] = score_confidence_level_vec

        return df_details

    def predict_class(
            self,
            # ndarray type of >= 2 dimensions, single point/row array
            x,
            include_match_details = False,
            top                   = ModelInterface.MATCH_TOP
    ):
        self.wait_for_model()
        # print('***** Input x: ' + str(x) + ', type ' + str(type(x)))
        if top == 1:
            # This will only return the top match
            top_match = self.network.predict_classes(x=x)
            return NnDenseModel.PredictReturnClass(
                predicted_classes = top_match
            )
        else:
            # This will return the last layer NN output, in numpy ndarray >= 2 dim, thus we can rank them
            # The return is the probabilities of classes [0,1,2,3...] in that order
            prob_distribution = self.network.predict(x=x)
            prob_dist_1d = prob_distribution[0]
            # print('***** Prob: ' + str(prob_distribution) + ', type "' + str(type(prob_distribution)) + '"')

            # Returns a 1-dim numpy ndarray
            top_x_matches = NumpyUtil.get_top_indexes(
                data      = prob_dist_1d,
                ascending = False,
                top_x     = top
            )
            # print('  ***** Top X Matches: ' + str(top_x_matches))
            top_x_probabilities = prob_dist_1d[top_x_matches]
            # print('  ***** Top X Probs:   ' + str(top_x_probabilities))
            top_x_matches_original = self.__map_top_matches_to_one_hot_labels(
                top_matches = top_x_matches
            )
            if include_match_details:
                match_details = self.__calculate_match_details(
                    top_matches = top_x_matches_original,
                    top_probs   = top_x_probabilities
                )
            else:
                match_details = None

            return ModelInterface.PredictReturnClass(
                predicted_classes       = top_x_matches_original,
                predicted_classes_coded = top_x_matches,
                match_details           = match_details
            )

    def predict_classes(
            self,
            # ndarray type of >= 2 dimensions
            x,
            include_match_details = False,
            top                   = ModelInterface.MATCH_TOP
    ):
        self.wait_for_model()
        # Returns a 1-dim
        p = self.network.predict_classes(x=x)
        top_x_matches_original = self.__map_top_matches_to_one_hot_labels(
            top_matches = p
        )
        return ModelInterface.PredictReturnClass(
            predicted_classes       = top_x_matches_original,
            predicted_classes_coded = p
        )

    def train(
            self,
            write_model_to_storage = True,
            write_training_data_to_storage = False,
            # Option to train a single y ID/label
            y_id = None,
            # To keep training logs here for caller's reference
            log_list_to_populate = None,
            # # Transform train labels to categorical or not
            # convert_train_labels_to_categorical = True
    ):
        if self.training_data is None:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Cannot train without training data for identifier "' + self.identifier_string + '"'
            )

        if type(self.model_params) is not nwdesign.NetworkDesign:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Cannot train without network for identifier "'
                + self.identifier_string + '". Got wrong type "' + str(type(self.model_params))
            )

        self.mutex_training.acquire()
        try:
            self.model_loaded = False
            Log.info(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ': Training for data, x shape ' + str(self.training_data.get_x().shape)
                + ', train labels with shape ' + str(self.training_data.get_y().shape)
            )

            if type(log_list_to_populate) is list:
                self.logs_training = log_list_to_populate
            else:
                self.logs_training = []

            x = self.training_data.get_x().copy()
            y = self.training_data.get_y().copy()
            self.x_one_hot_dict = self.training_data.get_x_one_hot_dict()
            # Form the inverse for convenience of transforming user input
            if type(self.x_one_hot_dict) is dict:
                self.x_one_hot_dict_inverse = {word:code for code,word in self.x_one_hot_dict.items()}
            self.y_one_hot_dict = self.training_data.get_y_one_hot_dict()
            # Convert labels to categorical one-hot encoding
            train_labels_categorical = to_categorical(y)

            n_labels = len(list(set(y.tolist())))
            Log.info(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ': Total unique labels = ' + str(n_labels) + '.',
                log_list = self.logs_training
            )

            try:
                self.network_layer_config = self.model_params.get_network_config()
                Log.info(
                    str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                    + ': Start creating network layers from config: ' + str(self.network_layer_config)
                )
                network = self.model_params.get_network()
                Log.info(
                    str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                    + ': Successfully created network layers from config: ' + str(self.network_layer_config)
                )
            except Exception as ex_layers:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                         + ': Error creating network layers for config: ' + str(self.network_layer_config) \
                         +'. Exception: ' + str(ex_layers)
                Log.error(
                    s = errmsg,
                    log_list = self.logs_training
                )
                raise Exception(errmsg)

            try:
                Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                    + ': Start compiling network "' + str(self.identifier_string) + '"..'
                )
                network.compile(
                    optimizer = self.train_optimizer,
                    loss      = self.train_loss,
                    metrics   = self.evaluate_metrics
                )
            except Exception as ex_compile:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                         + ': Error compiling network for config: ' + str(self.network_layer_config) \
                         +'. Exception: ' + str(ex_compile)
                Log.error(errmsg)
                raise Exception(errmsg)

            # Log model summary
            network.summary(print_fn=Log.info)

            Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + 'Categorical Train label shape "' + str(train_labels_categorical.shape)
                + '":\n\r' + str(train_labels_categorical)
            )

            try:
                Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                    + ': Start fitting network "' + str(self.identifier_string) + '"..'
                )

                # print('***** x: ' + str(x))
                # print('***** y: ' + str(train_labels_categorical))
                train_labels = y
                if self.model_params.require_label_to_categorical:
                    train_labels = train_labels_categorical
                if self.train_batch_size is not None:
                    network.fit(
                        x,
                        train_labels,
                        epochs     = self.train_epochs,
                        batch_size = self.train_batch_size
                    )
                else:
                    network.fit(
                        x,
                        train_labels,
                        epochs    = self.train_epochs,
                    )
                Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                    + ': Successfully fitted network "' + str(self.identifier_string) + '"..'
                )
            except Exception as ex_fit:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                         + ': Error training/fitting network for config: ' + str(self.network_layer_config) \
                         +'. Exception: ' + str(ex_fit)
                Log.error(errmsg)
                raise Exception(errmsg)

            self.network = network

            if write_model_to_storage:
                self.persist_model_to_storage(network=network)
            if write_training_data_to_storage:
                self.persist_training_data_to_storage(td=self.training_data)

            self.model_loaded = True
        except Exception as ex_train:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Train error for identifier "' + str(self.identifier_string)\
                     + '". Exception: ' + str(ex_train)
            Log.error(
                s = errmsg,
                log_list = self.logs_training
            )
            raise Exception(errmsg)
        finally:
            self.mutex_training.release()
        return

    def evaluate(
            self,
            data,
            # If required, labels must be converted to_categorical() by caller,
            # as certain NN architecture that uses flatten layers won't require it
            # while others will need it
            labels
    ):
        return self.network.evaluate(data, labels)

    def load_model_parameters(
            self
    ):
        try:
            self.mutex_training.acquire()

            # First check the existence of the files
            if not os.path.isfile(self.fpath_updated_file):
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                         + ': Last update file "' + self.fpath_updated_file \
                         + 'for model "' + str(self.identifier_string) + '" not found!'
                Log.error(errmsg)
                raise Exception(errmsg)

            self.network = load_model(self.fpath_model)

            try:
                df_x_one_hot_dict = pd.read_csv(
                    filepath_or_buffer = self.fpath_model_x_one_hot,
                    sep       = ',',
                    index_col = 'INDEX'
                )
                self.x_one_hot_dict = {code:word for code,word in df_x_one_hot_dict.values}
                # Form the inverse for convenience of transforming user input
                self.x_one_hot_dict_inverse = {word: code for code, word in self.x_one_hot_dict.items()}
                Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                    + ': Model "' + str(self.identifier_string)
                    + '" x one hot dict loaded: ' + str(self.x_one_hot_dict)
                )
            except Exception as ex_x_one_hot_dict:
                Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                    + ': Model "' + str(self.identifier_string) + '" no x_one_hot_dict. ' + str(ex_x_one_hot_dict)
                )
                self.x_one_hot_dict = None

            try:
                df_y_one_hot_dict = pd.read_csv(
                    filepath_or_buffer = self.fpath_model_y_one_hot,
                    sep       = ',',
                    index_col = 'INDEX'
                )
                self.y_one_hot_dict = {code:lbl for code,lbl in df_y_one_hot_dict.values}
                Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                    + ': Model "' + str(self.identifier_string)
                    + '" y one hot dict loaded: ' + str(self.y_one_hot_dict)
                )
            except Exception as ex_y_one_hot_dict:
                Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                    + ': Model "' + str(self.identifier_string) + '" no y_one_hot_dict. ' + str(ex_y_one_hot_dict)
                )
                self.y_one_hot_dict = None

            self.model_loaded = True
            self.model_updated_time = os.path.getmtime(self.fpath_updated_file)

            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': Model "' + str(self.identifier_string) + '" trained at "' + str(self.model_updated_time)
                + '" successfully loaded.'
            )
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Model "' + str(self.identifier_string)\
                     + '" failed to load from file "' + str(self.fpath_model)\
                     + '". Got exception ' + str(ex) + '.'
            Log.error(errmsg)
            raise Exception(errmsg)
        finally:
            self.mutex_training.release()

    def persist_model_to_storage(
            self,
            network = None
    ):
        try:
            #
            # Stupid Keras changed behavior in Python 3.8 and need to be a directory instead
            #
            major_version = sys.version_info[0]
            minor_version = sys.version_info[1]
            if (major_version > 3) or ( (major_version == 3) and (minor_version >= 8) ):
                if not os.path.isdir(self.fpath_model):
                    Log.important(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Path "' + str(self.fpath_model) + '" does not exist. Trying to create this directory...'
                    )
                    os.mkdir(path=self.fpath_model)

            self.network.save(self.fpath_model)
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Saved network to file/directory "' + str(self.fpath_model) + '".'
            )

            if self.x_one_hot_dict:
                ModelInterface.safe_dataframe_write(
                    df            = pd.DataFrame({
                        'code': list(self.x_one_hot_dict.keys()),
                        'word': list(self.x_one_hot_dict.values()),
                    }),
                    name_df       = 'x_one_hot_dict',
                    include_index = True,
                    index_label   = 'INDEX',
                    filepath      = self.fpath_model_x_one_hot,
                    log_training  = self.logs_training
                )

            if self.y_one_hot_dict:
                ModelInterface.safe_dataframe_write(
                    df            = pd.DataFrame({
                        'code': list(self.y_one_hot_dict.keys()),
                        'label':  list(self.y_one_hot_dict.values()),
                    }),
                    name_df       = 'y_one_hot_dict',
                    include_index = True,
                    index_label   = 'INDEX',
                    filepath      = self.fpath_model_y_one_hot,
                    log_training  = self.logs_training
                )

            # To allow applications to check if model updated
            # It is important to do it last (and fast), after everything is done
            ModelInterface.safe_file_write(
                dict_obj      = {'timenow': str(datetime.now())},
                name_dict_obj = 'model last updated time',
                filepath      = self.fpath_updated_file,
                write_as_json = False,
                log_training  = self.logs_training
            )

            return
        except Exception as ex_save:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Error saving model "' + str(self.identifier_string) + '": ' + str(ex_save)
            Log.error(errmsg)
            raise Exception(errmsg)


if __name__ == '__main__':
    import nwae.ml.config.Config as cf
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config,
        default_config_file = cf.Config.CONFIG_FILE_DEFAULT
    )
    Log.LOGLEVEL = Log.LOG_LEVEL_INFO

    ut_params = ut.UnitTestParams(
        dirpath_wordlist     = config.get_config(param=cf.Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist     = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_WORDLIST),
        dirpath_app_wordlist = config.get_config(param=cf.Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_app_wordlist = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        dirpath_synonymlist  = config.get_config(param=cf.Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist  = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        dirpath_model        = config.get_config(param=cf.Config.PARAM_MODEL_DIR)
    )
    Log.important('Unit Test Params: ' + str(ut_params.to_string()))

    from nwae.ml.nndense.NnDenseModelUnitTest import NnDenseModelUnitTest
    res_ut = NnDenseModelUnitTest(ut_params=ut_params).run_unit_test()
    exit(res_ut.count_fail)
