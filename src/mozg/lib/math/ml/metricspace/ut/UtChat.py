import mozg.common.data.security.Auth as au
import mozg.common.util.Log as log
from inspect import currentframe, getframeinfo
import mozg.lib.chat.classification.training.ChatTrainingData as ctd
import numpy as np
import mozg.lib.math.ml.TrainingDataModel as tdm
import mozg.lib.math.ml.metricspace.MetricSpaceModel as msModel


class UtChat:

    def __init__(self):
        self.topdir = '/Users/mark.tan/git/mozg'
        self.botkey = 'db_mario2.accid4.botid22'
        self.account_id = 4
        self.bot_id = 22
        self.bot_lang = 'cn'
        self.db_profile = 'mario2'

        self.identifier_string = 'demo_msmodel_accid4_botid22'
        self.dir_path_model = self.topdir + '/app.data/models'
        return

    def test_train(self):
        chat_td = ctd.ChatTrainingData(
            use_db     = True,
            db_profile = self.db_profile,
            account_id = self.account_id,
            bot_id     = self.bot_id,
            lang       = self.bot_lang,
            bot_key    = self.botkey,
            dirpath_traindata      = None,
            postfix_training_files = None,
            dirpath_wordlist     = self.topdir + '/nlp.data/wordlist',
            dirpath_app_wordlist = self.topdir + '/nlp.data/app/chats',
            dirpath_synonymlist  = self.topdir + '/nlp.data/app/chats'
        )

        td = chat_td.get_training_data_from_db()
        # Take just ten labels
        unique_classes = td[ctd.ChatTrainingData.COL_TDATA_INTENT_ID]
        text_segmented = td[ctd.ChatTrainingData.COL_TDATA_TEXT_SEGMENTED]

        keep = 10
        unique_classes_trimmed = list(set(unique_classes))[0:keep]
        np_unique_classes_trimmed = np.array(unique_classes_trimmed)
        np_indexes = np.isin(element=unique_classes, test_elements=np_unique_classes_trimmed)

        # By creating a new np array, we ensure the indexes are back to the normal 0,1,2...
        np_label_id = np.array(list(unique_classes[np_indexes]))
        np_text_segmented = np.array(list(text_segmented[np_indexes]))

        print('LABELS:\n\r' + str(np_label_id[0:20]))
        print('np TEXT SEGMENTED:\n\r' + str(np_text_segmented[0:20]))
        print('TEXT SEGMENTED:\n\r' + str(text_segmented[np_indexes]))

        #
        # Finally we have our text data in the desired format
        #
        tdm_obj = tdm.TrainingDataModel.unify_word_features_for_text_data(
            label_id=np_label_id.tolist(),
            text_segmented=np_text_segmented.tolist(),
            keywords_remove_quartile=0
        )

        print('TDM x:\n\r' + str(tdm_obj.get_x()))
        print('TDM x_name:\n\r' + str(tdm_obj.get_x_name()))
        print('TDM y' + str(tdm_obj.get_y()))

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
            weigh_idf     = True
        )
        ms_model.train(
            key_features_remove_quartile=0,
            stop_features=(),
            weigh_idf=True
        )

    def test_predict_classes(self):
        #
        # Now read back params and predict classes
        #
        ms_pc = msModel.MetricSpaceModel(
            identifier_string = self.identifier_string,
            # Directory to keep all our model files
            dir_path_model    = self.dir_path_model,
        )
        ms_pc.load_model_parameters_from_storage(
            dir_model = self.dir_path_model
        )

        x = ms_pc.training_data.get_x()
        x_name = ms_pc.training_data.get_x_name()
        y = ms_pc.training_data.get_y()

        x_classes = ms_pc.predict_classes(x=x)
        print('PREDICTED CLASSES x_classes (type ' + str(type(x_classes)) + '):\n\r' + str(x_classes))

        # Convert to string type
        y_str = np.array([])
        print('ORIGINAL CLASSES y (type ' + str(type(y_str)) + ')\n\r' + str(y_str))

        # Compare with expected
        compare = (x_classes != y)
        print(compare.tolist())
        print('Total Errors = ' + str(np.sum(compare*1)))

        return


if __name__ == '__main__':
    au.Auth.init_instances()
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_DEBUG_1

    obj = UtChat()
    #obj.test_train()
    obj.test_predict_classes()

