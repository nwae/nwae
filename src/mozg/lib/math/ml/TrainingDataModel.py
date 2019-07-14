# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import mozg.lib.chat.classification.training.ChatTrainingData as ctd
import mozg.common.util.Log as log
from inspect import currentframe, getframeinfo
import mozg.common.data.security.Auth as au
import mozg.lib.lang.classification.TextClusterBasic as tcb


#
# 데이터는 np array 형식으로 필요합니다
#
class TrainingDataModel:

    KEY_WORD_LABELS = 'word_labels'
    KEY_SENTENCE_TENSOR = 'sentence_tensor'

    def __init__(
            self,
            # np array 형식으호. Keras 라이브러리에서 x는 데이터를 의미해
            x,
            # np array 형식으호. Keras 라이브러리에서 y는 태그를 의미해
            y
    ):
        self.x = x
        self.y = y
        return

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    #
    # Помогающая Функция объединить разные свойства в тренинговый данные.
    # Returns sentence matrix array of combined word features
    #
    @staticmethod
    def unify_word_features_for_text_data(
            # At least 2 columns must exist 'Intent ID', 'TextSegmented'
            label_id,
            text_segmented,
            keywords_remove_quartile,
            stopwords = (),
    ):
        log_training = []

        if ( type(label_id) not in (list, tuple) ) or ( type(text_segmented) not in (list, tuple) ):
            raise Exception(
                str(TrainingDataModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Label ID and Text Segmented must be list/tuple type. Got label id type '
                + str(type(label_id)) + ', and text segmented type ' + str(type(text_segmented)) + '.'
            )
        if len(label_id) != len(text_segmented):
            raise Exception(
                str(TrainingDataModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Label ID length = ' + str(len(label_id))
                + ' and Text Segmented length = ' + str(len(text_segmented)) + ' not equal.'
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
            + ', labels\n\r' + str(label_id)
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
            if abs(1 - np.sum(np.multiply(v,v))**0.5) > 0.00000000001:
                raise Exception(
                    'Feature vector ' + str(v) + ' not normalized!'
                )

        return {
            TrainingDataModel.KEY_WORD_LABELS: np.array(fv_wordlabels),
            TrainingDataModel.KEY_SENTENCE_TENSOR: sentence_fv
        }


def demo_text_data():
    topdir = '/Users/mark.tan/git/mozg.nlp'
    chat_td = ctd.ChatTrainingData(
        use_db     = True,
        db_profile = 'mario2',
        account_id = 4,
        bot_id     = 22,
        lang       = 'cn',
        bot_key    = 'db_mario2.accid4.botid22',
        dirpath_traindata      = None,
        postfix_training_files = None,
        dirpath_wordlist       = topdir + '/nlp.data/wordlist',
        dirpath_app_wordlist   = topdir + '/nlp.data/app/chats',
        dirpath_synonymlist    = topdir + '/nlp.data/app/chats'
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

    print(np_label_id[0:20])
    print(np_text_segmented[0:20])
    print(np_text_segmented[0])

    retdict = TrainingDataModel.unify_word_features_for_text_data(
        label_id       = np_label_id.tolist(),
        text_segmented = np_text_segmented.tolist(),
        keywords_remove_quartile = 0
    )
    np_wordlabels = retdict[TrainingDataModel.KEY_WORD_LABELS]
    fv = retdict[TrainingDataModel.KEY_SENTENCE_TENSOR]

    error_count = 0
    total_count = fv.shape[0]
    for i in range(0, fv.shape[0], 1):
        v = fv[i]
        print_indexes = v>0
        labels_show = np_wordlabels[print_indexes]
        v_show = v[print_indexes]
        df = pd.DataFrame(data={'wordlabel': labels_show, 'fv': v_show})

        # Compare with original text
        txt = np_text_segmented[i]
        txt = tcb.TextClusterBasic.filter_sentence(
            sentence_text = txt
        )
        txt_arr = txt.split(sep=' ')
        # Filter out words not in wordlabels as we might have removed some quartile
        np_txt_arr = np.array(txt_arr)
        np_txt_arr = np_txt_arr[np.isin(element=np_txt_arr, test_elements=np_wordlabels)]
        txt_arr = np_txt_arr.tolist()
        if len(txt_arr) == 0:
            print('Sentence "' + txt + '" became nothing after removing quartile.')
            continue

        min_freq = 0.0
        try:
            min_freq = np.min(v_show)
        except Exception as ex:
            raise Exception('Cannot get min frequency for sentence "' + str(np_text_segmented[i])
                            + '", values ' + str(v_show) + '.')

        labels_show = labels_show.tolist()
        # Some words need to repeat more than once
        new_labels_show = []
        for j in range(0,df.shape[0],1):
            char = df['wordlabel'].loc[j]
            freq = int(df['fv'].loc[j] / min_freq)
            for k in range(0,freq,1):
                new_labels_show.append(char)

        new_labels_show.sort()
        txt_arr.sort()
        if not (new_labels_show==txt_arr):
            print(df)
            print(new_labels_show)
            print(txt_arr)
            error_count = error_count + 1
            print('WARNING!')
        else:
            print(str(i) + '. CHECK PASSED.')
    print(str(error_count) + ' errors from ' + str(total_count) + ' tests')

    # td.to_csv(path_or_buf='/Users/mark.tan/Downloads/td.csv')


if __name__ == '__main__':
    au.Auth.init_instances()
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_INFO
    demo_text_data()
    exit(0)
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
            [0, 1, 0, 1, 1, 2]
        ]
    )
    y = np.array(
        ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
    )