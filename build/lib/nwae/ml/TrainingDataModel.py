# -*- coding: utf-8 -*-

import numpy as np
import nwae.utils.Log as log
from inspect import currentframe, getframeinfo
from nwae.lang.model.WordFreqDocMatrix import WordFreqDocMatrix
import nwae.math.Constants as const
import nwae.math.NumpyUtil as npUtil
from nwae.utils.UnitTest import ResultObj, UnitTest


#
# 데이터는 np array 타입으로 필요합니다
#
class TrainingDataModel:

    def __init__(
            self,
            # np array 타입. Keras 라이브러리에서 x는 데이터를 의미해, Normalized data
            x,
            # np array 타입. Keras 라이브러리에서 y는 태그를 의미해
            y,
            # Maps the integral values in x (if x is integral) to some meaning (e.g. word)
            # For now not used in training, but more useful in the Models
            x_one_hot_dict = None,
            y_one_hot_dict = None,
            # REMEMBER that x_name, y_name are informative & supplementary only,
            # major operations MUST only use x & y
            # np array 형식으호. Имена дименций x
            x_name = None,
            # np array 형식으호
            y_name = None,
            # Will normalize points passed in to the hypershere
            is_map_points_to_hypersphere   = True,
            # Should only do this in desperation, otherwise we should always deal with numbers only
            is_convert_y_label_to_str_type = False,
            custom_data                    = None,
    ):
        # Only positive real values
        self.x = x
        self.y = y
        self.x_one_hot_dict = x_one_hot_dict
        self.y_one_hot_dict = y_one_hot_dict
        self.y_name = y_name

        self.is_map_points_to_hypersphere = is_map_points_to_hypersphere
        self.is_convert_y_label_to_str_type = is_convert_y_label_to_str_type

        # We try to keep the order of x_name as it was given to us, after any kind of processing
        self.x_name_index = np.array(range(0, self.x.shape[1], 1))
        if x_name is None:
            # If no x_name given we just use 0,1,2,3... as column names
            self.x_name = self.x_name_index.copy()
        else:
            self.x_name = x_name

        if type(self.x) is not np.ndarray:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': x must be np.array type, got type "' + str(type(self.x)) + '".'
            )
        if type(self.y) is not np.ndarray:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': x must be np.array type, got type "' + str(type(self.y)) + '".'
            )
        if self.y_name is None:
            self.y_name = np.array(self.y)
        elif type(self.y_name) is not np.ndarray:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': y_name must be np.array type, got type "' + str(type(self.y_name)) + '".'
            )

        log.Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': x shape = ' + str(self.x.shape) + ', y shape = ' + str(self.y.shape)
            + ': x name shape = ' + str(self.x_name.shape) + ', y name shape = ' + str(self.y_name.shape)
        )

        # TODO This is super slow, need to do something else faster
        # Change label to string type
        if self.is_convert_y_label_to_str_type:
            self.y = self.y.astype('str')

        self.custom_data = custom_data

        # Weights (all 1's by default)
        self.w = np.array([1]*self.x_name.shape[0])

        self.__check_xy_consistency()
        if self.is_map_points_to_hypersphere:
            self.x = npUtil.NumpyUtil.normalize(x=self.x)
            self.__remove_points_not_on_hypersphere()

        return

    def __check_xy_consistency(self):
        if (self.x.shape[0] != self.y.shape[0]) or (self.y.shape[0] != self.y_name.shape[0]):
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Number of x training points = ' + str(self.x.shape[0])
                + ' is not equal to number of labels = ' + str(self.y.shape[0])
                + ' or not equal to number of label names = ' + str(self.y_name.shape[0])
            )

        # The x_names are names of the dimension points of x
        # So if x is 2 dimensions, and the columns are of length 10, then x_names must be of length 10
        # If x is 3 dimensions with the 2nd and 3rd dimensions of shape (12,55), then x_names must be (12,55) in shape
        if self.x_name is not None:
            # No need to check for dimensions 3 and above
            for i_dim in range(1, min(2,self.x.ndim), 1):
                if self.x.shape[i_dim] != self.x_name.shape[i_dim-1]:
                    raise Exception(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Number of x dim ' + str(i_dim) + ' = ' + str(self.x.shape[i_dim])
                        + ' is not equal to number of x names dim ' + str(i_dim-1) + ' = ' + str(self.x_name.shape[i_dim-1])
                    )

        log.Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Consistency of training data checked OK.'
        )
        return

    def filter_by_y_id(
            self,
            y_id
    ):
        if type(y_id) in (np.int64, int, str):
            y_id = int(y_id)
        else:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Expected numpy.int64/int/str type, got "' + str(type(y_id)) + '" for y_id "' + str(y_id) + '".'
            )

        cond_y_id = np.isin(element=self.y, test_elements=[y_id])
        # Filter away unneeded ones
        self.x = self.x[cond_y_id]
        self.y = self.y[cond_y_id]
        self.y_name = self.y_name[cond_y_id]
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': After filtering y_id ' + str(y_id)
            + ',\n\rx:\n\r' + str(self.x.tolist()) + ',\n\ry:\n\r' + str(self.y.tolist())
        )

    #
    # The function of weights are just to reduce the meaningful dimensions of x (making some columns 0)
    # This function modifies x, y, y_name, and possibly x_name (if we do column deletion)
    #
    def weigh_x(
            self,
            # Expect a 1-dimensional np array
            w
    ):
        if type(w) is not np.ndarray:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                +': Weight w must be of type numpy ndarray, got type "' + str(type(w)) + '".'
            )

        # Length of w must be same with length of x columns
        pass_condition = (w.ndim == 1) and (w.shape[0] == self.x.shape[1])
        if not pass_condition:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                +': Weight w has wrong dimensions ' + str(w.shape)
                + ', not compatible with x dim ' + str(self.x.shape) + '.'
            )

        self.w = w

        #
        # Weigh x by w
        #
        x_w = np.multiply(self.x, self.w)
        log.Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': x weighted by w:\n\r' + str(x_w)
        )

        #
        # After weighing need to renormalize rows of x and do cleanup if necessary
        #
        for i in range(0, x_w.shape[0], 1):
            p = x_w[i]
            mag = np.sum(np.multiply(p, p)) ** 0.5
            if mag < const.Constants.SMALL_VALUE:
                x_w[i] = p * 0
            else:
                x_w[i] = p / mag

        log.Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': x weighted by w and renormalized:\n\r' + str(x_w)
        )

        self.x = x_w

        # Now remove points no longer lying on the hypersphere
        if self.is_map_points_to_hypersphere:
            self.__remove_points_not_on_hypersphere()

        # #
        # # After weighing, some dimensions may have disappeared (w[i]==0)
        # # But don't remove the columns, it is messy as we have to change w also.
        # #
        # w_indexes = np.array(range(self.w.shape[0]))
        # indexes_zero_columns = w_indexes[ (self.w<const.Constants.SMALL_VALUE) ]
        # self.x = np.delete(self.x, indexes_zero_columns, axis=1)
        # self.x_name = np.delete(self.x_name, indexes_zero_columns, axis=0)
        # log.Log.debug(
        #     str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
        #     + ': Deleted zero column indexes ' + str(indexes_zero_columns)
        #     + '. New x now dimension ' + str(self.x.shape)
        #     + ', x_name dimension ' + str(self.x_name.shape)
        # )
        return

    def __check_x_normalized(self):
        for i in range(0,self.x.shape[0],1):
            p = self.x[i]
            if not npUtil.NumpyUtil.is_normalized(x=p):
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': Tensor x not normalized at row ' + str(i) + '.\n\rPoint=\n\r' + str(p)
                log.Log.warning(errmsg)
                # No need to raise exception, it will be removed when we check for points not on hypersphere
                # raise Exception(errmsg)

    #
    # Remove rows with 0's
    #
    def __remove_points_not_on_hypersphere(self):
        indexes_to_remove = []
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '. x dimension ' + str(self.x.shape) + ', y dimension ' + str(self.y.shape)
        )
        for i in range(0,self.x.shape[0],1):
            p = self.x[i]
            is_normalized = npUtil.NumpyUtil.is_normalized(x=p)
            if (np.sum(p) < const.Constants.SMALL_VALUE) or (not is_normalized):
                indexes_to_remove.append(i)
                log.Log.warning(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Bad (sum to 0 or not normalized) x at index ' + str(i) + ', values ' + str(p)
                )
                continue

        if len(indexes_to_remove) > 0:
            self.x = np.delete(self.x, indexes_to_remove, axis=0)
            self.y = np.delete(self.y, indexes_to_remove, axis=0)
            self.y_name = np.delete(self.y_name, indexes_to_remove, axis=0)

            log.Log.warning(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Deleted row indexes ' + str(indexes_to_remove)
                + '. New x now dimension ' + str(self.x.shape)
                + ', y dimension ' + str(self.y.shape)
            )
        self.__check_xy_consistency()
        return

    #
    # x training data is usually huge, here we print non zero columns only for purposes of saving, etc.
    #
    def get_print_friendly_x(
            self,
            min_value_as_one = False
    ):
        x_dict = {}
        # Loop every sample
        for i in range(0, self.x.shape[0], 1):
            # Extract training data row
            v = self.x[i]
            # Keep only those > 0
            non_zero_indexes = v > 0
            # Extract x and x_name with non-zero x values
            x_name_show = self.x_name[non_zero_indexes]
            v_show = v[non_zero_indexes]
            y_show = self.y[i]

            if min_value_as_one:
                min_v = 0.0
                try:
                    min_v = np.min(v_show)
                    v_show = v_show / min_v
                except Exception as ex:
                    errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                             + ': Cannot get min val for x index ' + str(i)\
                             + ', point ' + str(v)\
                             + ' , nonzero x_name ' + str(x_name_show)\
                             + ', nonzero values ' + str(v_show) + '.'
                    # raise Exception(errmsg)

                v_show = np.round(v_show, 1)

            #
            # To ensure serializable by JSON, need to convert to proper types
            #
            v_show = v_show.astype(float)
            # Single label
            y_show = int(y_show)

            # Column names mean nothing because we convert to values list
            #x_dict[i] = pd.DataFrame(data={'wordlabel': x_name_show, 'fv': v_show}).values.tolist()
            x_dict[str(i)] = {
                'index': int(i),
                'x_name': x_name_show.tolist(),
                'x': v_show.tolist(),
                'y': y_show
            }
        return x_dict

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_x_name(self):
        return self.x_name

    def get_y_name(self):
        return self.y_name

    def get_w(self):
        return self.w

    def get_x_one_hot_dict(self):
        return self.x_one_hot_dict

    def get_y_one_hot_dict(self):
        return self.y_one_hot_dict

    def get_custom_data(self):
        return self.custom_data

    #
    # Помогающая Функция объединить разные свойства в тренинговый данные.
    # Returns sentence matrix array of combined word features
    # After this we will have our x (samples) and y (labels).
    #
    @staticmethod
    def unify_word_features_for_text_data(
            # List of segmented text data (the "x" but not in our unified format yet)
            # This function will convert this into our unified "x".
            # E.g. [ ['this','is','sentence','1','.'] , ['this','is','sent','2','.'], ...]
            sentences_list,
            # List of labels (the "y")
            label_id,
            # y_name. In case label id are not easily readable (e.g. ID from DB), then names for clarity
            label_name,
            keywords_remove_quartile,
            word_frequency_model,
            is_convert_y_label_to_str_type = False,
            add_unknown_word_in_keywords_list = True,
    ):
        log_training = []

        if ( type(label_id) not in (list, tuple) ) \
                or ( type(label_name) not in (list, tuple) ) \
                or ( type(sentences_list) not in (list, tuple) ):
            raise Exception(
                str(TrainingDataModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Label ID/Name and sentences list must be list/tuple type. Got label id type '
                + str(type(label_id)) + ', and text segmented type ' + str(type(sentences_list)) + '.'
            )
        if ( len(label_id) != len(sentences_list) ) or ( len(label_id) != len(label_name) ):
            raise Exception(
                str(TrainingDataModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Label ID length = ' + str(len(label_id))
                + ', label name length = ' + str(len(label_name))
                + ', and Text Segmented length = ' + str(len(sentences_list)) + ' must be equal.'
            )

        log.Log.info(
            str(TrainingDataModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '. Using keywords remove quartile = ' + str(keywords_remove_quartile) + '.'
            , log_list = log_training
        )

        log.Log.debug(
            str(TrainingDataModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Training data text\n\r' + str(sentences_list)
            + ', label IDs\n\r' + str(label_id)
            + ', label names\n\r' + str(label_name)
        )

        #
        # Extract all keywords
        # Our training now doesn't remove any word, uses no stopwords, but uses an IDF weightage to measure
        # keyword value.
        #
        log.Log.important(
            str(TrainingDataModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Starting training data model using word freq model "' + str(word_frequency_model) + '"'
            , log_list = log_training
        )

        # Extract unique Commands/Intents
        log.Log.info(
            str(TrainingDataModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Extracting unique commands/intents..'
            , log_list = log_training
        )
        unique_classes = set(label_id)
        # Change back to list, this list may change due to deletion of invalid commands.
        unique_classes = list(unique_classes)
        log.Log.info(
            str(TrainingDataModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Unique classes:\n\r' + str(unique_classes)
            , log_list = log_training
        )

        #
        # Get RFV for every command/intent, representative feature vectors by command type
        #
        # Get sentence matrix for all sentences first
        log.Log.important(
            str(TrainingDataModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Calculating sentence matrix for all training data...'
            , log_list = log_training
        )
        word_freq_doc_matrix = WordFreqDocMatrix()
        word_doc_matrix, keywords_for_fv = word_freq_doc_matrix.get_word_doc_matrix(
            sentences_list           = sentences_list,
            freq_measure             = word_frequency_model,
            feature_presence_only    = False,
            idf_matrix               = None,
            add_unknown_word_in_list = add_unknown_word_in_keywords_list,
        )
        log.Log.info(
            str(TrainingDataModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Word frequency measure model "' + str(word_frequency_model)
            + '" Keywords extracted as follows:\n\r' + str(keywords_for_fv)
        )

        fv_wordlabels = keywords_for_fv
        sentence_fv = np.transpose(word_doc_matrix)

        # Sanity check
        for i in range(0, sentence_fv.shape[0], 1):
            v = sentence_fv[i]
            if np.sum(v) == 0:
                continue
            if abs(1 - np.sum(np.multiply(v,v))**0.5) > const.Constants.SMALL_VALUE:
                raise Exception(
                    'Feature vector ' + str(v) + ' not normalized!'
                )

        # Check again
        if ( len(label_id) != sentence_fv.shape[0] ) or ( len(label_id) != len(label_name) ):
            raise Exception(
                str(TrainingDataModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Label ID length = ' + str(len(label_id))
                + ', label name length = ' + str(len(label_name))
                + ', and sentence FV shape/length = ' + str(sentence_fv.shape) + ' must be equal.'
            )

        return TrainingDataModel(
            x      = sentence_fv,
            x_name = np.array(fv_wordlabels),
            y      = np.array(label_id),
            y_name = np.array(label_name),
            is_convert_y_label_to_str_type = is_convert_y_label_to_str_type
        )


class TrainingDataModelUnitTest:
    def __init__(self, ut_params):
        return

    def run_unit_test(self):
        res_final = ResultObj(count_ok=0, count_fail=0)

        x = np.array(
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
                [0, 1, 0, 1, 1, 2],
                # Bad row on purpose
                [0, 0, 0, 0, 0, 0],
            ]
        )
        y = np.array(
            [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
        )
        x_name = np.array(['하나', '두', '셋', '넷', '다섯', '여섯'])

        x_model_expected = [
            [0.37796447, 0.75592895, 0.37796447, 0.37796447, 0.        , 0.        ],
            [0.63245553, 0.31622777, 0.63245553, 0.31622777, 0.        , 0.        ],
            [0.5       , 0.5       , 0.5       , 0.5       , 0.        , 0.        ],
            [0.        , 0.40824829, 0.81649658, 0.40824829, 0.        , 0.        ],
            [0.        , 0.57735027, 0.57735027, 0.57735027, 0.        , 0.        ],
            [0.        , 0.66666667, 0.33333333, 0.66666667, 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.26726124, 0.53452248, 0.80178373],
            [0.        , 0.31622777, 0.        , 0.63245553, 0.31622777, 0.63245553],
            [0.        , 0.37796447, 0.        , 0.37796447, 0.37796447, 0.75592895],
        ]
        y_model_expected = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        x_friendly_expected = {
            '0': {'index': 0, 'x_name': ['하나', '두', '셋', '넷'],
                  'x': [0.3779644730092272, 0.7559289460184544, 0.3779644730092272, 0.3779644730092272], 'y': 1},
            '1': {'index': 1, 'x_name': ['하나', '두', '셋', '넷'],
                  'x': [0.6324555320336759, 0.31622776601683794, 0.6324555320336759, 0.31622776601683794], 'y': 1},
            '2': {'index': 2, 'x_name': ['하나', '두', '셋', '넷'],
                  'x': [0.5, 0.5, 0.5, 0.5], 'y': 1},
            '3': {'index': 3, 'x_name': ['두', '셋', '넷'],
                  'x': [0.4082482904638631, 0.8164965809277261, 0.4082482904638631], 'y': 2},
            '4': {'index': 4, 'x_name': ['두', '셋', '넷'],
                  'x': [0.5773502691896258, 0.5773502691896258, 0.5773502691896258], 'y': 2},
            '5': {'index': 5, 'x_name': ['두', '셋', '넷'],
                  'x': [0.6666666666666666, 0.3333333333333333, 0.6666666666666666], 'y': 2},
            '6': {'index': 6, 'x_name': ['넷', '다섯', '여섯'],
                  'x': [0.2672612419124244, 0.5345224838248488, 0.8017837257372732], 'y': 3},
            '7': {'index': 7, 'x_name': ['두', '넷', '다섯', '여섯'],
                  'x': [0.31622776601683794, 0.6324555320336759, 0.31622776601683794, 0.6324555320336759], 'y': 3},
            '8': {'index': 8, 'x_name': ['두', '넷', '다섯', '여섯'],
                  'x': [0.3779644730092272, 0.3779644730092272, 0.3779644730092272, 0.7559289460184544], 'y': 3},
        }

        map_to_hypersphere = True

        obj = TrainingDataModel(
            x      = x,
            y      = y,
            x_name = x_name,
            is_map_points_to_hypersphere   = map_to_hypersphere,
            is_convert_y_label_to_str_type = False
        )

        x_model = obj.get_x()
        for i in range(len(x_model_expected)):
            x_line = np.round(x_model[i], 8).tolist()
            res_final.update_bool(res_bool=UnitTest.assert_true(
                observed = x_line,
                expected = x_model_expected[i],
                test_comment = 'Compare x model ' + str(x_line)
            ))

        y_model = obj.get_y()
        res_final.update_bool(res_bool=UnitTest.assert_true(
            observed = y_model.tolist(),
            expected = y_model_expected,
            test_comment = 'Compare y model ' + str(y_model)
        ))

        x_friendly = obj.get_print_friendly_x()
        # print(x_friendly)
        # for k in x_friendly.keys():
        #     print(str(k) + ': ' + str(x_friendly[k]))
        for i in x_friendly_expected.keys():
            x_line_f = x_friendly[i]
            res_final.update_bool(res_bool=UnitTest.assert_true(
                observed = x_line_f,
                expected = x_friendly_expected[i],
                test_comment = 'Compare x model ' + str(x_line_f)
            ))

        return res_final


if __name__ == '__main__':
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_DEBUG_1
    utest = TrainingDataModelUnitTest(ut_params=None)
    res = utest.run_unit_test()
    exit(res.count_fail)
