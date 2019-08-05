# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime as dt
import mozg.lib.math.ml.TrainingDataModel as tdm
import mozg.lib.math.ml.metricspace.MetricSpaceModel as msModel
import mozg.lib.math.ml.deeplearning.Keras as krModel
import threading
import mozg.common.util.Log as lg
from inspect import currentframe, getframeinfo


#
# Helper class to train data using given model
#
class Trainer(threading.Thread):

    MODEL_NAME_DEFAULT = 'default'
    MODEL_NAME_KERAS = 'keras'

    COL_TDATA_INTENT = 'Intent'
    COL_TDATA_INTENT_ID = 'Intent ID'
    COL_TDATA_TEXT_SEGMENTED = 'TextSegmented'

    def __init__(
            self,
            identifier_string,
            # Where to keep training data model files
            dir_path_model,
            # Can be in TrainingDataModel type or pandas DataFrame type with 3 columns (Intent ID, Intent, Text Segmented)
            training_data,
            model_name = MODEL_NAME_DEFAULT
    ):
        super(Trainer, self).__init__()

        self.identifier_string = identifier_string
        self.dir_path_model = dir_path_model
        # Can be None type when class initiated
        self.training_data = training_data
        self.model_name = model_name

        self.__mutex_training = threading.Lock()
        self.bot_training_start_time = None
        self.bot_training_end_time = None
        self.is_training_done = False
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

        lg.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Training Identifier ' + str(self.identifier_string) + '" trained successfully.'
        )

    def train(self):
        if type(self.training_data) not in (tdm.TrainingDataModel, pd.DataFrame):
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': Wrong training data type "' + str(type(self.training_data)) + '".'
            )
        else:
            lg.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': Training started for "' + self.identifier_string
                + '", model name "' + str(self.model_name)
                + '" training data type "' + str(type(self.training_data)) + '" initialized.'
            )

        try:
            tdm_object = self.training_data
            if type(self.training_data) is pd.DataFrame:
                lg.Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                    + ': Convert pandas DataFrame type to TrainingDataModel type...'
                )
                tdm_object = self.convert_to_training_data_model_type(
                    td = self.training_data
                )

            if self.model_name == Trainer.MODEL_NAME_KERAS:
                kr_model = krModel.Keras(
                    identifier_string = self.identifier_string,
                    dir_path_model    = self.dir_path_model,
                    training_data     = tdm_object
                )
                kr_model.train()
            else:
                ms_model = msModel.MetricSpaceModel(
                    identifier_string = self.identifier_string,
                    # Directory to keep all our model files
                    dir_path_model    = self.dir_path_model,
                    # Training data in TrainingDataModel class type
                    training_data     = tdm_object,
                    # From all the initial features, how many we should remove by quartile. If 0 means remove nothing.
                    key_features_remove_quartile = 0,
                    # Initial features to remove, should be an array of numbers (0 index) indicating column to delete in training data
                    stop_features = (),
                    # If we will create an "IDF" based on the initial features
                    weigh_idf     = True
                )
                ms_model.train()
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Training exception: ' + str(ex) + '.'
            lg.Log.error(errmsg)
            raise Exception(errmsg)

    def convert_to_training_data_model_type(
            self,
            td,
            # How many lines to keep from training data, -1 keep all. Used for mainly testing purpose.
            keep = -1
    ):
        # Extract these columns
        classes_id     = td[Trainer.COL_TDATA_INTENT_ID]
        text_segmented = td[Trainer.COL_TDATA_TEXT_SEGMENTED]
        classes_name   = td[Trainer.COL_TDATA_INTENT]

        lg.Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
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
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
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

