# -*- coding: utf-8 -*-

import numpy as np
import mozg.common.util.Log as log
from inspect import currentframe, getframeinfo
import mozg.common.data.security.Auth as au
import mozg.lib.lang.classification.TextClusterBasic as tcb
import mozg.lib.math.Constants as const


#
# 데이터는 np array 형식으로 필요합니다
#
class TrainingDataModel:

    def __init__(
            self,
            # np array 형식으호. Keras 라이브러리에서 x는 데이터를 의미해
            x,
            # np array 형식으호. Keras 라이브러리에서 y는 태그를 의미해
            y,
            # np array 형식으호. Имена дименций x
            x_name = None,
            # np array 형식으호
            y_name = None,
            check_if_x_normalized = False
    ):
        # Only positive real values
        self.x = x
        self.y = y
        self.y_name = y_name

        # We try to keep the order of x_name as it was given to us, after any kind of processing
        self.x_name_index = np.array(range(0, self.x.shape[1], 1))
        if x_name is None:
            # If no x_name given we just use 0,1,2,3... as column names
            self.x_name = self.x_name_index.copy()
        else:
            self.x_name = x_name

        self.check_if_x_normalized = check_if_x_normalized

        if type(self.x) is not np.ndarray:
            raise Exception('x must be np.array type, got type "' + str(type(self.x)) + '".')
        if type(self.y) is not np.ndarray:
            raise Exception('x must be np.array type, got type "' + str(type(self.y)) + '".')
        if self.y_name is None:
            self.y_name = np.array(self.y)
        elif type(self.y_name) is not np.ndarray:
            raise Exception('y_name must be np.array type, got type "' + str(type(self.y_name)) + '".')

        # Change label to string type
        y_str = np.array([])
        for el in self.y:
            el_str = str(el)
            y_str = np.append(y_str, el_str)
        self.y = y_str

        self.__remove_bad_rows()

        if (self.x.shape[0] != self.y.shape[0]) and (self.y.shape[0] != self.y_name.shape[0]):
            raise Exception(
                'Number of x training points = ' + str(self.x.shape[0])
                + ' is not equal to number of labels = ' + str(self.y.shape[0])
                + ' or not equal to number of label names = ' + str(self.y_name.shape[0])
            )

        # The x_names are names of the dimension points of x
        # So if x is 2 dimensions, and the columns are of length 10, then x_names must be of length 10
        # If x is 3 dimensions with the 2nd and 3rd dimensions of shape (12,55), then x_names must be (12,55) in shape
        if self.x_name is not None:
            for i_dim in range(1,self.x.ndim,1):
                if self.x.shape[i_dim] != self.x_name.shape[i_dim-1]:
                    raise Exception(
                        'Number of x dim ' + str(i_dim) + ' = ' + str(self.x.shape[i_dim])
                        + ' is not equal to number of x names dim ' + str(i_dim-1) + ' = ' + str(self.x_name.shape[i_dim-1])
                    )

        return

    def weigh_x(
            self,
            # Expect a 1-dimensional np array
            w
    ):
        if type(w) is not np.ndarray:
            raise Exception('Weight w must be of type numpy ndarray, got type "' + str(type(w)) + '".')

        # Length of w must be same with length of x columns
        pass_condition = (w.ndim == 1) and (w.shape[0] == self.x.shape[1])
        if not pass_condition:
            raise Exception('Weight w has wrong dimensions ' + str(w.shape)
                            + ', not compatible with x dim ' + str(self.x.shape) + '.')

        #
        # Weigh x by w
        #
        x_w = np.multiply(self.x, w)
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': x weighted by w:\n\r' + str(x_w)
        )

        #
        # After weighing need to renormalize and do cleanup if necessary
        #
        for i in range(0, x_w.shape[0], 1):
            p = x_w[i]
            mag = np.sum(np.multiply(p, p)) ** 0.5
            if mag < const.Constants.SMALL_VALUE:
                x_w[i] = p * 0
            else:
                x_w[i] = p / mag

        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': x weighted by w and renormalized:\n\r' + str(x_w)
        )

        self.x = x_w

        # Now redo cleanup
        self.__remove_bad_rows()

    #
    # Remove rows with 0's
    #
    def __remove_bad_rows(self):
        indexes_to_remove = []
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '. x dimension ' + str(self.x.shape) + ', y dimension ' + str(self.y.shape)
        )
        for i in range(0,self.x.shape[0],1):
            p = self.x[i]
            is_not_normalized = abs((np.sum(np.multiply(p,p))**0.5) - 1) > const.Constants.SMALL_VALUE
            if (np.sum(p) < const.Constants.SMALL_VALUE) or (self.check_if_x_normalized and is_not_normalized):
                indexes_to_remove.append(i)
                log.Log.warning(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Bad (sum to 0 or not normalized) x at index ' + str(i) + ', values ' + str(p)
                )
                continue

        if len(indexes_to_remove) > 0:
            self.x = np.delete(self.x, indexes_to_remove, axis=0)
            self.y = np.delete(self.y, indexes_to_remove, axis=0)
            log.Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Deleted indexes ' + str(indexes_to_remove)
                + '. New x now dimension ' + str(self.x.shape) + ', y dimension ' + str(self.y.shape)
            )

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

            min_v = 0.0
            try:
                min_v = np.min(v_show)
            except Exception as ex:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': Cannot get min val for x index ' + str(i)\
                         + ' , nonzero x_name ' + str(x_name_show)\
                         + ', nonzero values ' + str(v_show) + '.'
                raise Exception(errmsg)

            if min_value_as_one:
                v_show = np.round(v_show / min_v, 1)

            # Column names mean nothing because we convert to values list
            #x_dict[i] = pd.DataFrame(data={'wordlabel': x_name_show, 'fv': v_show}).values.tolist()
            x_dict[str(i)] = {
                'index': i,
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

    #
    # Помогающая Функция объединить разные свойства в тренинговый данные.
    # Returns sentence matrix array of combined word features
    # After this we will have our x (samples) and y (labels).
    #
    @staticmethod
    def unify_word_features_for_text_data(
            # List of segmented text data (the "x" but not in our unified format yet)
            # This function will convert this into our unified "x".
            text_segmented,
            # List of labels (the "y")
            label_id,
            # In case label id are not easily readable (e.g. ID from DB), then names for clarity
            label_name,
            keywords_remove_quartile,
            stopwords = ()
    ):
        log_training = []

        if ( type(label_id) not in (list, tuple) ) \
                or ( type(label_name) not in (list, tuple) ) \
                or ( type(text_segmented) not in (list, tuple) ):
            raise Exception(
                str(TrainingDataModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Label ID/Name and Text Segmented must be list/tuple type. Got label id type '
                + str(type(label_id)) + ', and text segmented type ' + str(type(text_segmented)) + '.'
            )
        if ( len(label_id) != len(text_segmented) ) and ( len(label_id) != len(label_name) ):
            raise Exception(
                str(TrainingDataModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Label ID length = ' + str(len(label_id))
                + ', label name length = ' + str(len(label_name))
                + ', and Text Segmented length = ' + str(len(text_segmented)) + ' must be equal.'
            )

        log.Log.info(
            str(TrainingDataModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '. Using keywords remove quartile = ' + str(keywords_remove_quartile)
            + ', stopwords = ' + str(stopwords) + '.'
            , log_list = log_training
        )

        log.Log.debugdebug(
            str(TrainingDataModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Training data text\n\r' + str(text_segmented)
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
            + ': Starting text cluster, calculate top keywords...'
            , log_list = log_training
        )
        textcluster = tcb.TextClusterBasic(
            text      = text_segmented,
            stopwords = stopwords
        )
        textcluster.calculate_top_keywords(
            remove_quartile = keywords_remove_quartile
        )
        log.Log.info(
            str(TrainingDataModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Keywords extracted as follows:\n\r' + str(textcluster.keywords_for_fv)
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
        log.Log.critical(
            str(TrainingDataModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Calculating sentence matrix for all training data...'
            , log_list = log_training
        )
        textcluster.calculate_sentence_matrix(
            freq_measure          = 'normalized',
            feature_presence_only = False,
            idf_matrix            = None
        )

        fv_wordlabels = textcluster.keywords_for_fv
        sentence_fv = textcluster.sentence_matrix

        # Sanity check
        for i in range(0, sentence_fv.shape[0], 1):
            v = sentence_fv[i]
            if np.sum(v) == 0:
                continue
            if abs(1 - np.sum(np.multiply(v,v))**0.5) > 0.000001:
                raise Exception(
                    'Feature vector ' + str(v) + ' not normalized!'
                )

        return TrainingDataModel(
            x      = sentence_fv,
            x_name = np.array(fv_wordlabels),
            y      = np.array(label_id),
            check_if_x_normalized = True
        )


if __name__ == '__main__':
    au.Auth.init_instances()
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_DEBUG_1

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
        ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C']
    )
    x_name = np.array(['하나', '두', '셋', '넷', '다섯', '여섯'])
    obj = TrainingDataModel(
        x = x,
        y = y,
        x_name = x_name
    )
    x_friendly = obj.get_print_friendly_x()
    print(x_friendly)
    for k in x_friendly.keys():
        print(str(k) + ': ' + str(x_friendly[k]))
