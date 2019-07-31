# -*- coding: utf-8 -*-

import mozg.common.data.security.Auth as au
import mozg.common.util.Log as log
from inspect import currentframe, getframeinfo
import mozg.lib.chat.classification.training.ChatTrainingData as ctd
import mozg.lib.math.ml.Trainer as trainer
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

        trainer_obj = trainer.Trainer(
            identifier_string = self.identifier_string,
            dir_path_model    = self.dir_path_model,
            training_data     = td
        )

        trainer_obj.train()
        return

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
    do_training = False

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


