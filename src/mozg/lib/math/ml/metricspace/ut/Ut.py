import numpy as np
import pandas as pd
import mozg.lib.math.ml.ModelHelper as modelHelper
import mozg.lib.math.ml.Trainer as trainer
import mozg.lib.math.ml.TrainingDataModel as tdm
import mozg.lib.math.ml.metricspace.MetricSpaceModel as msModel
import mozg.utils.Log as log
from inspect import currentframe, getframeinfo
import mozg.lib.math.NumpyUtil as npUtil
import mozg.ConfigFile as cf
import mozg.utils.Profiling as prf


class Ut:

    DATA_X = np.array(
        [
            # 무리 0
            [1, 2, 1, 1, 0, 0],
            [2, 1, 2, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            # 무리 1
            [0, 1, 2, 1, 0, 0],
            [0, 2, 2, 2, 0, 0],
            [0, 2, 1, 2, 0, 0],
            # 무리 2
            [0, 0, 0, 1, 2, 3],
            [0, 1, 0, 2, 1, 2],
            [0, 1, 0, 1, 1, 2],
            # 무리 3 (mix)
            [1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 2],
            [2, 0, 0, 0, 0, 1],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 2, 1, 1, 0],
            [0, 1, 1, 2, 1, 0]
        ]
    )
    DATA_TEXTS = [
        # 0
        '하나 두 두 셋 넷',
        '하나 하나 두 셋 셋 넷',
        '하나 두 셋 넷',
        # 1
        '두 셋 셋 넷',
        '두 두 셋 셋 넷 넷',
        '두 두 셋 넷 넷',
        # 2
        '넷 다섯 다섯 여섯 여섯 여섯',
        '두 넷 넷 다섯 다섯 여섯 여섯',
        '두 넷 다섯 여섯 여섯',
        # 3
        '하나 여섯',
        '하나 여섯 여섯',
        '하나 하나 여섯',
        '두 셋 넷 다섯',
        '두 셋 셋 넷 다섯',
        '두 셋 넷 넷 다섯'
    ]
    DATA_Y = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    )
    DATA_X_NAME = np.array(['하나', '두', '셋', '넷', '다섯', '여섯'])

    #
    # To test against trained models
    #
    DATA_TEST_X = np.array(
        [
            # 무리 0
            [1.2, 2.0, 1.1, 1.0, 0, 0],
            [2.1, 1.0, 2.4, 1.0, 0, 0],
            [1.5, 1.0, 1.3, 1.0, 0, 0],
            # 무리 1
            [0, 1.1, 2.5, 1.5, 0, 0],
            [0, 2.2, 2.6, 2.4, 0, 0],
            [0, 2.3, 1.7, 2.1, 0, 0],
            # 무리 2
            [0, 0.0, 0, 1.6, 2.1, 3.5],
            [0, 1.4, 0, 2.7, 1.2, 2.4],
            [0, 1.1, 0, 1.3, 1.3, 2.1],
            # 무리 3
            [1.1, 0.0, 0.0, 0.0, 0.0, 1.5],
            [0.0, 1.4, 0.9, 1.7, 1.2, 0.0]
        ]
    )
    DATA_TEST_X_NAME = np.array(['하나', '두', '셋', '넷', '다섯', '여섯', 'xxx'])

    #
    # Layers Design
    #
    NEURAL_NETWORK_LAYERS = [
        {
            'units': 128,
            'activation': 'relu',
            'input_shape': (DATA_X.shape[1],)
        },
        {
            # 4 unique classes
            'units': 4,
            'activation': 'softmax'
        }
    ]

    def __init__(
            self,
            identifier_string,
            model_name
    ):
        self.identifier_string = identifier_string
        self.model_name = model_name

        self.x_expected = Ut.DATA_X
        self.texts = Ut.DATA_TEXTS

        self.y = Ut.DATA_Y
        self.x_name = Ut.DATA_X_NAME
        #
        # Finally we have our text data in the desired format
        #
        y_list = self.y.tolist()
        y_list = list(y_list)
        self.tdm_obj = tdm.TrainingDataModel.unify_word_features_for_text_data(
            label_id       = y_list.copy(),
            label_name     = y_list.copy(),
            text_segmented = self.texts,
            keywords_remove_quartile = 0
        )

        self.x_friendly = self.tdm_obj.get_print_friendly_x()
        print(self.tdm_obj.get_x())
        for k in self.x_friendly.keys():
            print(self.x_friendly[k])
        print(self.tdm_obj.get_x_name())
        print(self.tdm_obj.get_y())

        return

    def unit_test_train(
            self,
            model_params = None
    ):
        trainer_obj = trainer.Trainer(
            identifier_string = self.identifier_string,
            model_name        = self.model_name,
            dir_path_model    = cf.ConfigFile.DIR_MODELS,
            training_data     = self.tdm_obj
        )

        trainer_obj.train(
            write_training_data_to_storage = True,
            model_params = model_params
        )

        # How to make sure order is the same output from TextCluster in unit tests?
        x_name_expected = ['넷' '두' '셋' '여섯' '다섯' '하나']

        sentence_matrix_expected = np.array([
            [0.37796447, 0.75592895, 0.37796447, 0., 0., 0.37796447],
            [0.31622777, 0.31622777, 0.63245553, 0., 0., 0.63245553],
            [0.5, 0.5, 0.5, 0., 0., 0.5],
            [0.40824829, 0.40824829, 0.81649658, 0., 0., 0.],
            [0.57735027, 0.57735027, 0.57735027, 0., 0., 0.],
            [0.66666667, 0.66666667, 0.33333333, 0., 0., 0.],
            [0.26726124, 0., 0., 0.80178373, 0.53452248, 0.],
            [0.5547002, 0.2773501, 0., 0.5547002, 0.5547002, 0.],
            [0.37796447, 0.37796447, 0., 0.75592895, 0.37796447, 0.]
        ])
        for i in range(0, sentence_matrix_expected.shape[0], 1):
            v = sentence_matrix_expected[i]
            ss = np.sum(np.multiply(v, v)) ** 0.5
            print(v)
            print(ss)

        agg_by_labels_expected = np.array([
            [1.19419224, 1.57215671, 1.51042001, 0., 0., 1.51042001],
            [1.65226523, 1.65226523, 1.72718018, 0., 0., 0.],
            [1.19992591, 0.65531457, 0., 2.11241287, 1.46718715, 0.]
        ])

        idf_expected = [0., 0., 0.40546511, 1.09861229, 1.09861229, 1.09861229]

        x_w_expected = [
            [0., 0., 0.34624155, 0., 0., 0.9381454],
            [0., 0., 0.34624155, 0., 0., 0.9381454],
            [0., 0., 0.34624155, 0., 0., 0.9381454],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0.83205029, 0.5547002, 0.],
            [0., 0., 0., 0.70710678, 0.70710678, 0.],
            [0., 0., 0., 0.89442719, 0.4472136, 0.]
        ]
        y_w_expected = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
        x_name_w_expected = ['넷', '두', '셋', '여섯', '다섯', '하나']

        # Cluster value will change everytime! So don't rely on this
        x_clustered_expected = [
            [0., 0., 0.34624155, 0., 0., 0.9381454],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0.86323874, 0.5009569, 0.],
            [0., 0., 0., 0.70710678, 0.70710678, 0.]
        ]
        y_clustered_expected = [1, 2, 3, 3]

    def unit_test_predict_classes(
            self,
            include_rfv = False,
            include_match_details = False,
            top = 5
    ):
        log.Log.info(
            'Test predict classes using model "' + str(self.model_name) + '".'
        )

        model_obj = modelHelper.ModelHelper.get_model(
            model_name        = self.model_name,
            identifier_string = self.identifier_string,
            dir_path_model    = cf.ConfigFile.DIR_MODELS,
            training_data     = None
        )
        model_obj.start()
        model_obj.wait_for_model()
        #model_obj.load_model_parameters()

        test_x = Ut.DATA_TEST_X
        test_x_name = Ut.DATA_TEST_X_NAME
        model_x_name = model_obj.get_model_features()
        if model_x_name is None:
            model_x_name = Ut.DATA_X_NAME

        if model_x_name.ndim == 2:
            model_x_name = model_x_name[0]
        log.Log.info('Model x_name: ' + str(model_x_name))

        # Reorder by model x_name
        df_x_name = pd.DataFrame(data={'word': model_x_name, 'target_order': range(0, len(model_x_name), 1)})
        df_test_x_name = pd.DataFrame(data={'word': test_x_name, 'original_order': range(0, len(test_x_name), 1)})
        # print('**** Target Order: ' + str(model_x_name))
        # print('**** Original order: ' + str(test_x_name))
        # Left join to ensure the order follows target order and target symbols
        df_x_name = df_x_name.merge(df_test_x_name, how='left')
        # print('**** Merged Order: ' + str(df_x_name))
        # Then order by original order
        df_x_name = df_x_name.sort_values(by=['target_order'], ascending=True)
        # Then the order we need to reorder is the target_order column
        reorder = np.array(df_x_name['original_order'])
        log.Log.debugdebug(df_x_name)
        log.Log.debugdebug(reorder)
        log.Log.debugdebug(test_x)

        test_x_transpose = test_x.transpose()
        log.Log.debugdebug(test_x_transpose)

        reordered_test_x = np.zeros(shape=test_x_transpose.shape)
        log.Log.debugdebug(reordered_test_x)

        for i in range(0, reordered_test_x.shape[0], 1):
            reordered_test_x[i] = test_x_transpose[reorder[i]]

        reordered_test_x = reordered_test_x.transpose()
        log.Log.debugdebug(reordered_test_x)

        x_classes_expected = self.y
        # Just the top predicted ones
        all_y_observed_top = []
        all_y_observed = []
        mse = 0
        mse_norm = 0
        count_all = reordered_test_x.shape[0]

        log.Log.info('Predict classes for x:\n\r' + str(reordered_test_x))
        prf_start = prf.Profiling.start()

        for i in range(reordered_test_x.shape[0]):
            v = npUtil.NumpyUtil.convert_dimension(arr=reordered_test_x[i], to_dim=2)
            if self.model_name == modelHelper.ModelHelper.MODEL_NAME_HYPERSPHERE_METRICSPACE:
                predict_result = model_obj.predict_class(
                    x           = v,
                    include_rfv = include_rfv,
                    include_match_details = include_match_details,
                    top = top
                )
            else:
                predict_result = model_obj.predict_class(
                    x           = v
                )
            y_observed = predict_result.predicted_classes
            all_y_observed_top.append(y_observed[0])
            all_y_observed.append(y_observed)
            top_class_distance = predict_result.top_class_distance
            match_details = predict_result.match_details

            log.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Point v ' + str(v) + ', predicted ' + str(y_observed)
                + ', Top Class Distance: ' + str(top_class_distance)
                + ', Match Details: ' + str(match_details)
            )

            if self.model_name == modelHelper.ModelHelper.MODEL_NAME_HYPERSPHERE_METRICSPACE:
                metric = top_class_distance
                metric_norm = metric / msModel.MetricSpaceModel.HPS_MAX_EUCL_DIST
                mse += metric ** 2
                mse_norm += metric_norm ** 2

        prf_dur = prf.Profiling.get_time_dif(prf_start, prf.Profiling.stop())
        log.Log.important(
            str(self.__class__) + str(getframeinfo(currentframe()).lineno)
            + ' PROFILING ' + str(count_all) + ' calculations: ' + str(round(1000*prf_dur,0))
            + ', or ' + str(round(1000*prf_dur/count_all,2)) + ' milliseconds per calculation'
        )

        # Compare with expected
        compare_top_x = {}

        for t in range(1, top + 1, 1):
            compare_top_x[t] = np.array([True] * len(all_y_observed))
            for i in range(len(all_y_observed)):
                matches_i = all_y_observed[i]
                if x_classes_expected[i] in matches_i[0:t]:
                    compare_top_x[t][i] = False
            log.Log.info(compare_top_x[t])
            log.Log.critical(
                'Total Errors (compare top #' + str(t) + ') = ' + str(np.sum(compare_top_x[t] * 1))
            )

        log.Log.info('mse = ' + str(mse))
        log.Log.info('mse_norm = ' + str(mse_norm))

        if self.model_name == modelHelper.ModelHelper.MODEL_NAME_HYPERSPHERE_METRICSPACE:
            predict_result = model_obj.predict_classes(
                    x           = reordered_test_x,
                    include_rfv = include_rfv,
                    include_match_details = include_match_details,
                    top = top
                )
            log.Log.info('Predicted Classes:\n\r' + str(predict_result.predicted_classes))
            log.Log.info('Top class distance:\n\r' + str(predict_result.top_class_distance))
            log.Log.info('Match Details:\n\r' + str(predict_result.match_details))
            log.Log.info('MSE = ' + str(predict_result.mse))
            log.Log.info('MSE Normalized = ' + str(predict_result.mse_norm))
        return


if __name__ == '__main__':
    cf.ConfigFile.get_cmdline_params_and_init_config()

    # Overwrite config file log level
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_INFO

    for model_name in [
            #modelHelper.ModelHelper.MODEL_NAME_HYPERSPHERE_METRICSPACE,
            modelHelper.ModelHelper.MODEL_NAME_KERAS,
    ]:
        obj = Ut(
            identifier_string = 'demo_ut1',
            model_name        = model_name
        )
        if model_name == modelHelper.ModelHelper.MODEL_NAME_KERAS:
            obj.unit_test_train(
                model_params = Ut.NEURAL_NETWORK_LAYERS
            )
        else:
            obj.unit_test_train()

        obj.unit_test_predict_classes(
            include_rfv = False,
            include_match_details = False,
            top = 2
        )

