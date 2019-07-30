import mozg.common.data.security.Auth as au
import mozg.common.util.Log as log
from inspect import currentframe, getframeinfo
import mozg.lib.chat.classification.training.ChatTrainingData as ctd
import numpy as np
import pandas as pd
import mozg.lib.math.ml.TrainingDataModel as tdm
import mozg.lib.math.ml.metricspace.MetricSpaceModel as msModel
import mozg.lib.math.NumpyUtil as npUtil
import mozg.common.util.Profiling as prf
import mozg.ConfigFile as cf


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

    def test_train(
            self,
            weigh_idf = True,
            # How many of the classes to keep to test training. -1 to keep nothing and train all.
            keep = -1
    ):
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

        print('LABELS ID:\n\r' + str(np_label_id[0:20]))
        print('LABELS NAME:\n\r' + str(np_label_name[0:20]))
        print('np TEXT SEGMENTED:\n\r' + str(np_text_segmented[0:20]))
        print('TEXT SEGMENTED:\n\r' + str(text_segmented[np_indexes]))

        #
        # Finally we have our text data in the desired format
        #
        tdm_obj = tdm.TrainingDataModel.unify_word_features_for_text_data(
            label_id       = np_label_id.tolist(),
            label_name     = np_label_name.tolist(),
            text_segmented = np_text_segmented.tolist(),
            keywords_remove_quartile = 0
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
            weigh_idf     = weigh_idf
        )
        ms_model.train()

    def test_predict_classes(
            self,
            indexes_to_test = None,
            include_rfv = False,
            include_match_details = False,
            top = 5,
            do_profiling = False
    ):
        #
        # Now read back params and predict classes
        #
        ms_pc = msModel.MetricSpaceModel(
            identifier_string = self.identifier_string,
            # Directory to keep all our model files
            dir_path_model    = self.dir_path_model,
            do_profiling      = False
        )
        ms_pc.load_model_parameters()
        ms_pc.load_training_data_from_storage()

        x = ms_pc.training_data.get_x()
        x_name = ms_pc.training_data.get_x_name()
        y = ms_pc.training_data.get_y()

        if indexes_to_test is None:
            indexes_to_test = range(x.shape[0])

        prf_start = prf.Profiling.start()
        count_correct_top = [0]*top
        count_all = 0
        mse = 0
        mse_norm = 0

        for i in indexes_to_test:
            predict_result = ms_pc.predict_class(
                x           = npUtil.NumpyUtil.convert_dimension(arr=x[i],to_dim=2),
                include_rfv = include_rfv,
                include_match_details = include_match_details,
                top = top
            )

            # Just the first row
            y_observed = predict_result.predicted_classes
            top_class_distance = predict_result.top_class_distance
            match_details = predict_result.match_details

            count_all += 1
            ok = [0]*top
            match_in_top = 0
            for top_i in range(top):
                if len(y_observed) < top_i+1:
                    continue
                # Just the top_i predicted ones
                ok[top_i] = (y[i] in y_observed[0:(top_i+1)])
                count_correct_top[top_i] += 1*(ok[top_i])
                if y[i] == y_observed[top_i]:
                    match_in_top = top_i+1

            metric = top_class_distance
            metric_norm = metric / msModel.MetricSpaceModel.HPS_MAX_EUCL_DIST
            mse += metric ** 2
            mse_norm += metric_norm ** 2

            msg = str(i) + '. Expected ' + str(y[i]) + ', got ' + str(y_observed)
            msg += '. Top match position #' + str(match_in_top) + ''
            log.Log.info(msg)

        for top_i in range(top):
            log.Log.info('Top ' + str(top_i+1) + ' correct = '
                         + str(count_correct_top[top_i]) + ' (of ' + str(count_all) + ')')

        log.Log.info('MSE = ' + str(mse))
        log.Log.info('MSE Normalized = ' + str(mse_norm))

        prf_dur = prf.Profiling.get_time_dif(prf_start, prf.Profiling.stop())
        log.Log.important(
            str(self.__class__) + str(getframeinfo(currentframe()).lineno)
            + ' PROFILING ' + str(count_all) + ' calculations: ' + str(round(1000*prf_dur,0))
            + ', or ' + str(round(1000*prf_dur/count_all,0)) + ' milliseconds per calculation'
        )


if __name__ == '__main__':
    cf.ConfigFile.init_config(topdir='/Users/mark.tan/git/mozg')
    au.Auth.init_instances()
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_INFO

    obj = UtChat()
    do_training = True

    if do_training:
        obj.test_train(
            weigh_idf = True,
            # keep      = 10
        )
        exit(0)

    obj.test_predict_classes(
        #indexes_to_test=[107,131],
        include_rfv = False,
        include_match_details = False,
        top = 5
    )


