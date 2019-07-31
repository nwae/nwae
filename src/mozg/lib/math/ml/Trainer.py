# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime as dt
import mozg.lib.math.ml.TrainingDataModel as tdm
import threading
import mozg.common.util.Log as lg
from inspect import currentframe, getframeinfo


class Trainer(threading.Thread):

    TD_TYPE_NORMAL = 'normal'
    TD_TYPE_TEXT = 'text'

    def __init__(
            self,
            identifier_string,
            model_interface,
            training_data,
            training_data_type = TD_TYPE_NORMAL
    ):
        super(Trainer, self).__init__()

        self.identifier_string = identifier_string
        self.model_interface = model_interface
        self.training_data = training_data
        self.training_data_type = training_data_type

        self.__mutex_training = threading.Lock()
        return

    def run(self):
        try:
            self.__mutex_training.acquire()
            self.bot_training_start_time = dt.datetime.now()
            self.log_training = []
            self.train(
                self.keywords_remove_quartile,
                self.stopwords,
                self.weigh_idf
            )
            self.bot_training_end_time = dt.datetime.now()
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Training Identifier ' + str(self.identifier_string) + '" training exception: ' + str(ex) + '.'
            log.Log.critical(errmsg)
            raise Exception(errmsg)
        finally:
            self.is_training_done = True
            self.__mutex_training.release()

        log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Training Identifier ' + str(self.identifier_string) + '" trained successfully.')

    def train(
            self,
            # Object type TrainingDataModel
            tdm_object,
            model_interface
    ):
        ms_model = msModel.MetricSpaceModel(
            identifier_string = self.identifier_string,
            # Directory to keep all our model files
            dir_path_model    = self.dir_path_model,
            # Training data in TrainingDataModel class type
            training_data     = tdm_obj,
            # From all the initial features, how many we should remove by quartile. If 0 means remove nothing.
            key_features_remove_quartile = 0,
            # Initial features to remove, should be an array of numbers (0 index) indicating column to delete in training data
            stop_features = (),
            # If we will create an "IDF" based on the initial features
            weigh_idf     = weigh_idf
        )
        ms_model.train()

    def train_text_data(
            self,
            td
    ):
        # Extract these columns
        classes_id     = td[ctd.ChatTrainingData.COL_TDATA_INTENT_ID]
        text_segmented = td[ctd.ChatTrainingData.COL_TDATA_TEXT_SEGMENTED]
        classes_name   = td[ctd.ChatTrainingData.COL_TDATA_INTENT]

        # Help to keep both linked
        df_classes_id_name = pd.DataFrame({
            'id': classes_id,
            'name': classes_name
        })

        # For unit testing purpose, keep only 10 classes
        unique_classes_id = list(set(classes_id))
        if keep < 0:
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
        print('y FILTERED:\n\r' + str(np_unique_classes_trimmed))
        print('y DF FILTERED:\n\r:' + str(df_classes_id_name))

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

