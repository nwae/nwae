import numpy as np
import pandas as pd
import mozg.lib.math.ml.TrainingDataModel as tdm
import mozg.lib.math.ml.metricspace.MetricSpaceModel as msModel
import mozg.common.util.Log as log
from inspect import currentframe, getframeinfo


class Ut:

    def __init__(
            self
    ):
        self.identifier_string = 'demo_msmodel_testdata'
        self.topdir = '/Users/mark.tan/git/mozg'
        self.dir_path_model = self.topdir + '/app.data/models'

        self.x_expected = np.array(
            [
                # 무리 A
                [1, 2, 1, 1, 0, 0],
                [2, 1, 2, 1, 0, 0],
                [1, 1, 1, 1, 0, 0],
                # 무리 B
                [0, 1, 2, 1, 0, 0],
                [0, 2, 2, 2, 0, 0],
                [0, 2, 1, 2, 0, 0],
                # 무리 C
                [0, 0, 0, 1, 2, 3],
                [0, 1, 0, 2, 1, 2],
                [0, 1, 0, 1, 1, 2]
            ]
        )
        self.texts = [
            # 'A'
            '하나 두 두 셋 넷',
            '하나 하나 두 셋 셋 넷',
            '하나 두 셋 넷',
            # 'B'
            '두 셋 셋 넷',
            '두 두 셋 셋 넷 넷',
            '두 두 셋 넷 넷',
            # 'C'
            '넷 다섯 다섯 여섯 여섯 여섯',
            '두 넷 넷 다섯 다섯 여섯 여섯',
            '두 넷 다섯 여섯 여섯',
        ]

        self.y = np.array(
            ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
        )
        self.x_name = np.array(['하나', '두', '셋', '넷', '다섯', '여섯'])
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
            weigh_idf = False
    ):
        ms_model = msModel.MetricSpaceModel(
            identifier_string = self.identifier_string,
            # Directory to keep all our model files
            dir_path_model    = self.dir_path_model,
            # Training data in TrainingDataModel class type
            training_data     = self.tdm_obj,
            # From all the initial features, how many we should remove by quartile. If 0 means remove nothing.
            key_features_remove_quartile = 0,
            # Initial features to remove, should be an array of numbers (0 index) indicating column to delete in training data
            stop_features = (),
            # If we will create an "IDF" based on the initial features
            weigh_idf     = weigh_idf
        )
        ms_model.train()

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
        y_clustered_expected = ['A', 'B', 'C', 'C']

    def unit_test_predict_classes(
            self,
            include_rfv = False,
            include_match_details = False
    ):
        ms = msModel.MetricSpaceModel(
            identifier_string = self.identifier_string,
            # Directory to keep all our model files
            dir_path_model    = self.dir_path_model,
        )
        ms.load_model_parameters_from_storage()

        test_x = np.array(
            [
                # 무리 A
                [1.2, 2.0, 1.1, 1.0, 0, 0],
                [2.1, 1.0, 2.4, 1.0, 0, 0],
                [1.5, 1.0, 1.3, 1.0, 0, 0],
                # 무리 B
                [0, 1.1, 2.5, 1.5, 0, 0],
                [0, 2.2, 2.6, 2.4, 0, 0],
                [0, 2.3, 1.7, 2.1, 0, 0],
                # 무리 C
                [0, 0.0, 0, 1.6, 2.1, 3.5],
                [0, 1.4, 0, 2.7, 1.2, 2.4],
                [0, 1.1, 0, 1.3, 1.3, 2.1]
            ]
        )
        test_x_name = np.array(['하나', '두', '셋', '넷', '다섯', '여섯'])
        model_x_name = ms.model_data.x_name
        if model_x_name.ndim == 2:
            model_x_name = model_x_name[0]
        print(model_x_name)

        # Reorder by model x_name
        df_x_name = pd.DataFrame(data={'word': model_x_name, 'target_order': range(0, len(model_x_name), 1)})
        df_test_x_name = pd.DataFrame(data={'word': test_x_name, 'original_order': range(0, len(test_x_name), 1)})
        df_x_name = df_x_name.merge(df_test_x_name)
        # Then order by original order
        df_x_name = df_x_name.sort_values(by=['target_order'], ascending=True)
        # Then the order we need to reorder is the target_order column
        reorder = np.array(df_x_name['original_order'])
        print(df_x_name)
        print(reorder)
        print(test_x)

        test_x_transpose = test_x.transpose()
        print(test_x_transpose)

        reordered_test_x = np.zeros(shape=test_x_transpose.shape)
        print(reordered_test_x)

        for i in range(0, reordered_test_x.shape[0], 1):
            reordered_test_x[i] = test_x_transpose[reorder[i]]

        reordered_test_x = reordered_test_x.transpose()
        print(reordered_test_x)

        predict_result = ms.predict_classes(
            x           = reordered_test_x,
            include_rfv = include_rfv,
            include_match_details = include_match_details
        )
        y_observed = predict_result.predicted_classes
        # Just the top predicted ones
        y_observed_top = []
        for item in y_observed:
            y_observed_top.append(item[0])
        top_class_distance = predict_result.top_class_distance
        match_details = predict_result.match_details

        mse = np.sum(np.multiply(top_class_distance, top_class_distance))
        mse_norm = mse / (msModel.MetricSpaceModel.HPS_MAX_EUCL_DIST ** 2)

        print(y_observed)
        print(top_class_distance)
        print(match_details)
        print('MSE = ' + str(mse))
        print('MSE normalized = ' + str(mse_norm))

        # Mean square error

        # Compare with expected
        #x_classes_expected = ms.training_data.get_y()
        x_classes_expected = self.y
        print(x_classes_expected)

        # Compare with expected
        compare = (y_observed_top != x_classes_expected)
        print(compare.tolist())
        print('Total Errors = ' + str(np.sum(compare*1)))


if __name__ == '__main__':
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_INFO
    obj = Ut()
    obj.unit_test_train(weigh_idf=True)
    obj.unit_test_predict_classes(
        include_rfv = False,
        include_match_details = False
    )

