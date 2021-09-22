# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from nwae.ml.trainer.TrainerInterface import TrainerInterface
from nwae.ml.modelhelper.TextModelHelper import TextModelHelper
import nwae.ml.TrainingDataModel as tdm
import nwae.math.optimization.Eidf as eidf
from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
import nwae.lang.TextProcessor as txtprocessor
import nwae.lang.nlp.daehua.DaehuaTrainDataModel as dhtdmodel
from nwae.ml.text.EmbeddingParams import EmbeddingParams
from nwae.lang.model.WordFreqDocMatrix import WordFreqDocMatrix


class TextTrainer(TrainerInterface):

    def __init__(
            self,
            identifier_string,
            # Where to keep training data model files
            dir_path_model,
            # Can be in TrainingDataModel type or pandas DataFrame type with 3 columns (Intent ID, Intent, Text Segmented)
            # Preprocessing of text (tokenization, spelling corrections, stemming, etc) is assumed to be already done
            training_data,
            word_freq_model,
            # If training data is None, must pass a training_data_source object with method fetch_data() implemented
            training_data_source = None,
            model_name = None,
            model_params = None,
            # Either 'train_model' (or None), or 'train_nlp_eidf', etc.
            train_mode = TrainerInterface.TRAIN_MODE_MODEL,
            # Train a single y/label ID only, regardless of train mode
            y_id = None,
    ):
        super().__init__(
            identifier_string    = identifier_string,
            dir_path_model       = dir_path_model,
            training_data        = training_data,
            training_data_source = training_data_source,
            model_name           = model_name,
            model_params         = model_params,
            train_mode           = train_mode,
            y_id                 = y_id
        )

        self.word_freq_model = word_freq_model
        allowed_values = WordFreqDocMatrix.MAP_WORD_FREQ_FEATURE_VECT.keys()
        assert self.word_freq_model in allowed_values, \
            'Word freq model "' + str(self.word_freq_model) + '" must be in allowed value ' + str(allowed_values)

        if self.model_name is None:
            self.model_name = TextModelHelper.MODEL_NAME_HYPERSPHERE_METRICSPACE

        if self.train_mode is None:
            self.train_mode = TextTrainer.TRAIN_MODE_MODEL

        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Initialized text trainer for model "' + str(self.model_name)
            + '", word freq model "' + str(self.word_freq_model)
            + '", train mode "' + str(self.train_mode) + '"'
        )

        #
        # TxtDataPreprocessor object passed back to us after fetching and preprocessing
        #
        self.df_training_data_pp = None
        self.embedding_params = EmbeddingParams()

        #
        # Partial/Incremental training mode.
        # In this mode, training model files will only write to sub-folders of the model
        # directory, instead of the final model files in the model directory.
        # It is to speed up the actual model training so that only looks in sub-folders
        # for pre-calculated sub-models.
        #
        self.is_partial_training = (self.train_mode == TextTrainer.TRAIN_MODE_MODEL_BY_LABEL)\
                                   | (self.y_id is not None)
        return

    #
    # If training data is not in desired format, we do some conversion
    #
    def preprocess_training_data(
            self
    ):
        if not self.is_training_data_ready:
            try:
                #
                # The external interface must pass back 2 parameters, a DataFrame of preprocessed training data
                # and Embedding Layer params
                #
                self.df_training_data_pp, self.embedding_params = self.training_data_source.fetch_and_preprocess_data()

                Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Successfully preprocessed training data. Max label val = ' + str(self.embedding_params.max_label_val)
                    + ', max sentence length = ' + str(self.embedding_params.max_sent_len)
                    + ', vocabulary size = ' + str(self.embedding_params.vocab_size)
                    + ', x one hot dict: ' + str(self.embedding_params.x_one_hot_dict)
                )

                self.training_data = TextTrainer.convert_preprocessed_text_to_training_data_model(
                    model_name         = self.model_name,
                    training_dataframe = self.df_training_data_pp,
                    embedding_x        = self.embedding_params.x,
                    embedding_y        = self.embedding_params.y,
                    embedding_x_one_hot_dict = self.embedding_params.x_one_hot_dict,
                    embedding_y_one_hot_dict = self.embedding_params.y_one_hot_dict,
                    word_freq_model    = self.word_freq_model,
                )
            except Exception as ex:
                errmsg = \
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                    + ': Exception calling external object type "' + str(type(self.training_data_source)) \
                    + '" method fetch_and_preprocess_data(), exception msg: ' + str(ex)
                Log.error(errmsg)
                raise Exception(errmsg)

        if type(self.training_data) is not tdm.TrainingDataModel:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': "' + str(self.identifier_string)
                + '": Wrong training data type "' + str(type(self.training_data)) + '".'
            )

        # Train a single y/label ID only, regardless of train mode
        if self.y_id is not None:
            # Filter by this y/label only
            self.training_data.filter_by_y_id(
                y_id = self.y_id
            )

        return

    def train(
            self,
            write_model_to_storage = True,
            write_training_data_to_storage = True
    ):
        if type(self.training_data) not in (tdm.TrainingDataModel, pd.DataFrame):
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Train mode "' + str(self.train_mode) + '", y/label id ' + str(self.y_id)
                + '. Wrong training data type "' + str(type(self.training_data)) + '".'
            )
        else:
            Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Train mode "' + str(self.train_mode) + '", y/label id ' + str(self.y_id)
                + '. Training started for "' + self.identifier_string
                + '", model name "' + str(self.model_name)
                + '" training data type "' + str(type(self.training_data)) + '" initialized.'
                , log_list = self.log_training
            )

        try:
            tdm_object = self.training_data

            Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Train mode "' + str(self.train_mode)
                + '". Training Model using model name "' + str(self.model_name)
                + '". for bot "' + str(self.identifier_string) + '".'
                , log_list = self.log_training
            )
            if self.train_mode == TextTrainer.TRAIN_MODE_MODEL:
                model_obj = TextModelHelper.get_model(
                    model_name             = self.model_name,
                    identifier_string      = self.identifier_string,
                    model_params           = self.model_params,
                    dir_path_model         = self.dir_path_model,
                    training_data          = tdm_object,
                    embed_max_label_val    = self.embedding_params.max_label_val,
                    embed_max_sentence_len = self.embedding_params.max_sent_len,
                    embed_vocabulary_size  = self.embedding_params.vocab_size,
                    is_partial_training    = self.is_partial_training
                )
                model_obj.train(
                    write_model_to_storage = write_model_to_storage,
                    write_training_data_to_storage = write_training_data_to_storage,
                    log_list_to_populate   = self.log_training
                )
            elif self.train_mode == TextTrainer.TRAIN_MODE_MODEL_BY_LABEL:
                # Loop by unique y's
                unique_y_list = list(set(list(self.training_data.get_y())))
                # Keep backup of training data
                x_initial = self.training_data.get_x()
                x_name_initial = self.training_data.get_x_name()
                y_name_initial = self.training_data.get_y_name()
                y_initial = self.training_data.get_y()
                for y_id_item in unique_y_list:
                    # Create new TrainingDataModel object
                    tdm_item = tdm.TrainingDataModel(
                        x      = x_initial.copy(),
                        x_name = x_name_initial.copy(),
                        y      = y_initial.copy(),
                        y_name = y_name_initial.copy()
                    )
                    # Filter by this y/label only
                    tdm_item.filter_by_y_id(
                        y_id = y_id_item
                    )
                    Log.important(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                        + ': [' + str(self.identifier_string) + '] Begin training label ' + str(y_id_item)
                        + ', with data:\n\r' + str(tdm_item.get_print_friendly_x())
                    )
                    model_obj_item = TextModelHelper.get_model(
                        model_name             = self.model_name,
                        model_params           = self.model_params,
                        identifier_string      = self.identifier_string,
                        dir_path_model         = self.dir_path_model,
                        training_data          = tdm_item,
                        embed_max_label_val    = self.embedding_params.max_label_val,
                        embed_max_sentence_len = self.embedding_params.max_sent_len,
                        embed_vocabulary_size  = self.embedding_params.vocab_size,
                        is_partial_training    = True
                    )
                    model_obj_item.train(
                        write_model_to_storage         = write_model_to_storage,
                        write_training_data_to_storage = write_training_data_to_storage,
                        model_params                   = self.model_params,
                        log_list_to_populate           = self.log_training
                    )
                    Log.important(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                        + ': [' + str(self.identifier_string) + '] Done training label ' + str(y_id_item)
                        + '.'
                    )
            elif self.train_mode == TextTrainer.TRAIN_MODE_MODEL_USE_PARTIAL_MODELS:
                model_obj = TextModelHelper.get_model(
                    model_name             = self.model_name,
                    model_params           = self.model_params,
                    identifier_string      = self.identifier_string,
                    dir_path_model         = self.dir_path_model,
                    training_data          = tdm_object,
                    embed_max_label_val    = self.embedding_params.max_label_val,
                    embed_max_sentence_len = self.embedding_params.max_sent_len,
                    embed_vocabulary_size  = self.embedding_params.vocab_size,
                    is_partial_training    = self.is_partial_training
                )
                model_obj.train_from_partial_models(
                    write_model_to_storage = write_model_to_storage,
                    write_training_data_to_storage = write_training_data_to_storage,
                    model_params           = self.model_params,
                    log_list_to_populate   = self.log_training
                )
            elif self.train_mode == TextTrainer.TRAIN_MODE_NLP_EIDF:
                eidf_opt_obj = eidf.Eidf(
                    x      = tdm_object.get_x(),
                    y      = tdm_object.get_y(),
                    x_name = tdm_object.get_x_name()
                )
                info_msg = eidf_opt_obj.optimize(
                    initial_w_as_standard_idf = True,
                    logs = self.log_training
                )
                Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                    + str(info_msg)
                )
                eidf_opt_obj.persist_eidf_to_storage(
                    dir_path_model    = self.dir_path_model,
                    identifier_string = self.identifier_string
                )
            else:
                raise Exception('Invalid train mode "' + str(self.train_mode) + '"!')
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Training exception: ' + str(ex) + '.'
            Log.error(errmsg)
            raise Exception(errmsg)

    @staticmethod
    def convert_preprocessed_text_to_training_data_model(
            model_name,
            training_dataframe,
            embedding_x,
            embedding_y,
            embedding_x_one_hot_dict,
            embedding_y_one_hot_dict,
            word_freq_model,
    ):
        if model_name == TextModelHelper.MODEL_NAME_HYPERSPHERE_METRICSPACE:
            return TextTrainer.__convert_processed_text_to_training_data_model_type_for_hypersphere_metricspace(
                training_dataframe = training_dataframe,
                word_freq_model = word_freq_model,
            )
        else:
            return TextTrainer.__convert_preprocessed_text_to_training_data_model_for_nn_dense(
                embedding_x = embedding_x,
                embedding_y = embedding_y,
                embedding_x_one_hot_dict = embedding_x_one_hot_dict,
                embedding_y_one_hot_dict = embedding_y_one_hot_dict
            )

    @staticmethod
    def __convert_preprocessed_text_to_training_data_model_for_nn_dense(
            embedding_x,
            embedding_y,
            embedding_x_one_hot_dict,
            embedding_y_one_hot_dict,
    ):
        return tdm.TrainingDataModel(
            x = embedding_x,
            y = embedding_y,
            x_one_hot_dict = embedding_x_one_hot_dict,
            y_one_hot_dict = embedding_y_one_hot_dict,
            is_map_points_to_hypersphere = False
        )

    @staticmethod
    def __convert_processed_text_to_training_data_model_type_for_hypersphere_metricspace(
            # pandas DataFrame type with the intent, text, language etc columns
            # Preprocessing of text (tokenization, spelling corrections, stemming, etc) is assumed to be already done
            training_dataframe,
            word_freq_model,
            # How many lines to keep from training data, -1 keep all. Used for mainly testing purpose.
            keep = -1,
    ):
        td = training_dataframe

        # Extract these columns
        classes_id     = td[dhtdmodel.DaehuaTrainDataModel.COL_TDATA_INTENT_ID]
        text_segmented = td[dhtdmodel.DaehuaTrainDataModel.COL_TDATA_TEXT_SEGMENTED]
        classes_name   = td[dhtdmodel.DaehuaTrainDataModel.COL_TDATA_INTENT_NAME]
        lang_detected  = td[dhtdmodel.DaehuaTrainDataModel.COL_TDATA_TEXT_LANG]

        Log.info(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Columns: ' + str(td.columns)
            + '\n\rClasses ID:\n\r' + str(classes_id)
            + '\n\rText Segmented:\n\r' + str(text_segmented)
            + '\n\rClasses name:\n\r' + str(classes_name)
            + '\n\rText Lang Detected:\n\r' + str(lang_detected)
        )

        # Help to keep both linked
        df_classes_id_name = pd.DataFrame({
            'id': classes_id,
            'name': classes_name
        })

        unique_classes_id = list(set(classes_id))
        if keep <= 0:
            keep = len(unique_classes_id)
        else:
            keep = min(keep, len(unique_classes_id))
        unique_classes_trimmed = list(set(classes_id))[0:keep]
        np_unique_classes_trimmed = np.array(unique_classes_trimmed)
        Log.important(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Total unique classes = ' + str(np_unique_classes_trimmed.shape[0])
            + ': ' + str(np_unique_classes_trimmed)
        )

        # True/False series, filter out those x not needed for testing
        np_indexes = np.isin(element=classes_id, test_elements=np_unique_classes_trimmed)
        Log.debugdebug(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Total unique indexes = ' + str(np_indexes.shape[0])
            + ': ' + str(np_indexes)
        )

        df_classes_id_name = df_classes_id_name[np_indexes]
        # This dataframe becomes our map to get the name of y/classes
        df_classes_id_name.drop_duplicates(inplace=True)

        Log.info(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': y FILTERED: ' + str(np_unique_classes_trimmed.tolist())
        )
        Log.debugdebug('y DF FILTERED: :' + str(df_classes_id_name))

        # By creating a new np array, we ensure the indexes are back to the normal 0,1,2...
        np_label_id = np.array(list(classes_id[np_indexes]))
        Log.info(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': np_label_id: ' + str(np_label_id.tolist())
        )
        Log.info(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': text segmented list: ' + str(list(text_segmented[np_indexes]))
        )
        # Convert text to usable array form for further NLP processing
        # Note that we are not doing any encoding yet, string still remains as string,
        # and we are only breaking the already preprocessed text into more convenient arrays of tokens/words
        txtprocessor_obj = txtprocessor.TextProcessor(
            lang_str_or_list = list(lang_detected[np_indexes]),
            text_segmented_list = list(text_segmented[np_indexes])
        )
        text_segmented_list_list = txtprocessor_obj.convert_segmented_text_to_array_form()
        # TODO numpy warning
        #   VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
        #   np_sentences_list = np.array(text_segmented_list_list)
        np_sentences_list = np.array(text_segmented_list_list, dtype=object)

        # Merge to get the label name
        df_tmp_id = pd.DataFrame(data={'id': np_label_id})
        df_tmp_id = df_tmp_id.merge(df_classes_id_name, how='left')
        np_label_name = np.array(df_tmp_id['name'])

        if (np_label_id.shape != np_label_name.shape) or (np_label_id.shape[0] != np_sentences_list.shape[0]):
            raise Exception(
                str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Label ID and name must have same dimensions.\n\r Label ID:\n\r'
                + str(np_label_id)
                + 'Label Name:\n\r'
                + str(np_label_name)
            )

        Log.debugdebug('LABELS ID:\n\r' + str(np_label_id[0:20]))
        Log.debugdebug('LABELS NAME:\n\r' + str(np_label_name[0:20]))
        Log.debugdebug('np TEXT SEGMENTED:\n\r' + str(np_sentences_list[0:20]))
        Log.debugdebug('TEXT SEGMENTED:\n\r' + str(text_segmented[np_indexes]))
        Log.debug('NP SENTENCES LIST:\n\r' + str(np_sentences_list.tolist()))

        #
        # Finally we have our text data in the desired format
        #
        tdm_obj = tdm.TrainingDataModel.unify_word_features_for_text_data(
            label_id       = np_label_id.tolist(),
            label_name     = np_label_name.tolist(),
            sentences_list = np_sentences_list.tolist(),
            keywords_remove_quartile = 0,
            word_frequency_model = word_freq_model,
        )

        Log.important(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Successfully converted text data to training numbers data using word freq model "'
            + str(word_freq_model) + '", sentences shape ' + str(np_sentences_list.shape)
        )

        Log.debugdebug('TDM x:\n\r' + str(tdm_obj.get_x()))
        Log.debugdebug('TDM x_name:\n\r' + str(tdm_obj.get_x_name()))
        Log.debugdebug('TDM y' + str(tdm_obj.get_y()))

        return tdm_obj

