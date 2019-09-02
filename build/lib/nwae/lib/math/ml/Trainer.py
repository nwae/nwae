# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime as dt
import nwae.lib.math.ml.ModelHelper as modelHelper
import nwae.lib.math.ml.TrainingDataModel as tdm
import nwae.lib.math.optimization.Eidf as eidf
import threading
import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo


#
# Helper class to train data using given model
# Supports also text data.
#
class Trainer(threading.Thread):

    #
    # Model Training
    #
    # Train the entire model in one shot
    TRAIN_MODE_MODEL          = 'train_model'
    # In this case the training will loop by y_id and do each partial
    # training one by one, and write only to label specific training files.
    # The purpose is to do incremental training, thus fast
    TRAIN_MODE_MODEL_BY_LABEL = 'train_model_by_label'
    TRAIN_MODE_MODEL_USE_PARTIAL_MODELS = 'train_model_use_partial_models'
    #
    # NLP Training
    #
    TRAIN_MODE_NLP_EIDF = 'train_nlp_eidf'

    #
    # If passing in text training data, then make sure these columns exist in the
    # pandas DataFrame passed in.
    #
    COL_TEXT_TDATA_INTENT = 'Intent'
    COL_TEXT_TDATA_INTENT_ID = 'Intent ID'
    COL_TEXT_TDATA_TEXT_SEGMENTED = 'TextSegmented'

    def __init__(
            self,
            identifier_string,
            # Where to keep training data model files
            dir_path_model,
            # Can be in TrainingDataModel type or pandas DataFrame type with 3 columns (Intent ID, Intent, Text Segmented)
            training_data,
            model_name = None,
            # Either 'train_model' (or None), or 'train_nlp_eidf'
            train_mode = TRAIN_MODE_MODEL,
            # Train a single y/label ID only, regardless of train mode
            y_id = None
    ):
        super(Trainer, self).__init__()

        self.identifier_string = identifier_string
        self.dir_path_model = dir_path_model
        self.training_data = training_data
        if model_name is None:
            model_name = modelHelper.ModelHelper.MODEL_NAME_HYPERSPHERE_METRICSPACE
        self.model_name = model_name

        self.train_mode = train_mode
        self.y_id = y_id

        self.__mutex_training = threading.Lock()
        self.bot_training_start_time = None
        self.bot_training_end_time = None
        self.is_training_done = False

        self.log_training = []

        # Do some conversion if necessary on training data
        self.__pre_process_training_data()
        if type(self.training_data) is not tdm.TrainingDataModel:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': "' + str(self.identifier_string)
                + '": Wrong training data type "' + str(type(self.training_data)) + '".'
            )

        # Partial/Incremental training mode
        self.is_partial_training = (self.train_mode == Trainer.TRAIN_MODE_MODEL_BY_LABEL)\
                                   | (self.y_id is not None)
        # Train a single y/label ID only, regardless of train mode
        if self.y_id is not None:
            # Filter by this y/label only
            self.training_data.filter_by_y_id(
                y_id = self.y_id
            )

        return

    #
    # If training data is not in desired format, we do some conversion
    #
    def __pre_process_training_data(
            self
    ):
        #
        # If not in proper TrainingDataModel type, we assume the training data is legacy text form
        #
        if type(self.training_data) is pd.DataFrame:
            lg.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': Convert pandas DataFrame type to TrainingDataModel type...'
            )
            tdm_object = Trainer.convert_to_training_data_model_type(
                td = self.training_data
            )
            # Reassign back to training data
            self.training_data = tdm_object
        return

    def run(self):
        try:
            self.__mutex_training.acquire()
            self.bot_training_start_time = dt.datetime.now()
            self.log_training = []

            self.train()

            self.bot_training_end_time = dt.datetime.now()
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Training Identifier ' + str(self.identifier_string) + '" training exception: ' + str(ex) + '.'
            lg.Log.critical(errmsg)
            raise Exception(errmsg)
        finally:
            self.is_training_done = True
            self.__mutex_training.release()

        lg.Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Train mode "' + str(self.train_mode)
            + '". Training Identifier ' + str(self.identifier_string) + '" trained successfully.'
        )
        return self.log_training

    def train(
            self,
            write_model_to_storage = True,
            write_training_data_to_storage = False,
            model_params = None
    ):
        if type(self.training_data) not in (tdm.TrainingDataModel, pd.DataFrame):
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Train mode "' + str(self.train_mode) + '", y/label id ' + str(self.y_id)
                + '. Wrong training data type "' + str(type(self.training_data)) + '".'
            )
        else:
            lg.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Train mode "' + str(self.train_mode) + '", y/label id ' + str(self.y_id)
                + '. Training started for "' + self.identifier_string
                + '", model name "' + str(self.model_name)
                + '" training data type "' + str(type(self.training_data)) + '" initialized.'
            )

        try:
            tdm_object = self.training_data

            lg.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Train mode "' + str(self.train_mode)
                + '". Training Model using model name "' + str(self.model_name)
                + '". for bot "' + str(self.identifier_string) + '".'
            )
            if self.train_mode == Trainer.TRAIN_MODE_MODEL:
                model_obj = modelHelper.ModelHelper.get_model(
                    model_name          = self.model_name,
                    identifier_string   = self.identifier_string,
                    dir_path_model      = self.dir_path_model,
                    training_data       = tdm_object,
                    is_partial_training = self.is_partial_training
                )
                model_obj.train(
                    write_model_to_storage = write_model_to_storage,
                    write_training_data_to_storage = write_training_data_to_storage,
                    model_params = model_params
                )
            elif self.train_mode == Trainer.TRAIN_MODE_MODEL_BY_LABEL:
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
                    lg.Log.important(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                        + ': [' + str(self.identifier_string) + '] Begin training label ' + str(y_id_item)
                        + ', with data:\n\r' + str(tdm_item.get_print_friendly_x())
                    )
                    model_obj_item = modelHelper.ModelHelper.get_model(
                        model_name          = self.model_name,
                        identifier_string   = self.identifier_string,
                        dir_path_model      = self.dir_path_model,
                        training_data       = tdm_item,
                        is_partial_training = True
                    )
                    model_obj_item.train(
                        write_model_to_storage         = write_model_to_storage,
                        write_training_data_to_storage = write_training_data_to_storage,
                        model_params                   = model_params
                    )
                    lg.Log.important(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                        + ': [' + str(self.identifier_string) + '] Done training label ' + str(y_id_item)
                        + '.'
                    )
            elif self.train_mode == Trainer.TRAIN_MODE_MODEL_USE_PARTIAL_MODELS:
                model_obj = modelHelper.ModelHelper.get_model(
                    model_name          = self.model_name,
                    identifier_string   = self.identifier_string,
                    dir_path_model      = self.dir_path_model,
                    training_data       = tdm_object,
                    is_partial_training = self.is_partial_training
                )
                model_obj.train_from_partial_models(
                    write_model_to_storage = write_model_to_storage,
                    write_training_data_to_storage = write_training_data_to_storage,
                    model_params = model_params
                )
            elif self.train_mode == Trainer.TRAIN_MODE_NLP_EIDF:
                eidf_opt_obj = eidf.Eidf(
                    x      = tdm_object.get_x(),
                    y      = tdm_object.get_y(),
                    x_name = tdm_object.get_x_name()
                )
                info_msg = eidf_opt_obj.optimize(
                    initial_w_as_standard_idf = True
                )
                self.log_training = eidf_opt_obj.log_training
                lg.Log.info(
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
            lg.Log.error(errmsg)
            raise Exception(errmsg)

    @staticmethod
    def convert_to_training_data_model_type(
            td,
            # How many lines to keep from training data, -1 keep all. Used for mainly testing purpose.
            keep = -1
    ):
        # Extract these columns
        classes_id     = td[Trainer.COL_TEXT_TDATA_INTENT_ID]
        text_segmented = td[Trainer.COL_TEXT_TDATA_TEXT_SEGMENTED]
        classes_name   = td[Trainer.COL_TEXT_TDATA_INTENT]

        lg.Log.debugdebug(
            str(Trainer.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Columns: ' + str(td.columns)
            + '\n\rClasses ID:\n\r' + str(classes_id)
            + '\n\rText Segmented:\n\r' + str(text_segmented)
            + '\n\rClasses name:\n\r' + str(classes_name)
        )

        # Help to keep both linked
        df_classes_id_name = pd.DataFrame({
            'id': classes_id,
            'name': classes_name
        })

        # For unit testing purpose, keep only 10 classes
        unique_classes_id = list(set(classes_id))
        if keep <= 0:
            keep = len(unique_classes_id)
        else:
            keep = min(keep, len(unique_classes_id))
        unique_classes_trimmed = list(set(classes_id))[0:keep]
        np_unique_classes_trimmed = np.array(unique_classes_trimmed)

        # True/False series, filter out those x not needed for testing
        np_indexes = np.isin(element=classes_id, test_elements=np_unique_classes_trimmed)

        df_classes_id_name = df_classes_id_name[np_indexes]
        # This dataframe becomes our map to get the name of y/classes
        df_classes_id_name.drop_duplicates(inplace=True)

        lg.Log.debugdebug('y FILTERED:\n\r' + str(np_unique_classes_trimmed))
        lg.Log.debugdebug('y DF FILTERED:\n\r:' + str(df_classes_id_name))

        # By creating a new np array, we ensure the indexes are back to the normal 0,1,2...
        np_label_id = np.array(list(classes_id[np_indexes]))
        np_text_segmented = np.array(list(text_segmented[np_indexes]))

        # Merge to get the label name
        df_tmp_id = pd.DataFrame(data={'id': np_label_id})
        df_tmp_id = df_tmp_id.merge(df_classes_id_name, how='left')
        np_label_name = np.array(df_tmp_id['name'])

        if (np_label_id.shape != np_label_name.shape) or (np_label_id.shape[0] != np_text_segmented.shape[0]):
            raise Exception(
                str(Trainer.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + 'Label ID and name must have same dimensions.\n\r Label ID:\n\r'
                + str(np_label_id)
                + 'Label Name:\n\r'
                + str(np_label_name)
            )

        lg.Log.debugdebug('LABELS ID:\n\r' + str(np_label_id[0:20]))
        lg.Log.debugdebug('LABELS NAME:\n\r' + str(np_label_name[0:20]))
        lg.Log.debugdebug('np TEXT SEGMENTED:\n\r' + str(np_text_segmented[0:20]))
        lg.Log.debugdebug('TEXT SEGMENTED:\n\r' + str(text_segmented[np_indexes]))

        #
        # Finally we have our text data in the desired format
        #
        tdm_obj = tdm.TrainingDataModel.unify_word_features_for_text_data(
            label_id       = np_label_id.tolist(),
            label_name     = np_label_name.tolist(),
            text_segmented = np_text_segmented.tolist(),
            keywords_remove_quartile = 0
        )

        lg.Log.debugdebug('TDM x:\n\r' + str(tdm_obj.get_x()))
        lg.Log.debugdebug('TDM x_name:\n\r' + str(tdm_obj.get_x_name()))
        lg.Log.debugdebug('TDM y' + str(tdm_obj.get_y()))

        return tdm_obj

