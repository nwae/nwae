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
    #
    @staticmethod
    def unify_features_for_text_data(
            self,
            # At least 2 columns must exist 'Intent ID', 'TextSegmented'
            training_data,
            keywords_remove_quartile,
            stopwords,
    ):
        td = None
        self.log_training = []

        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + '. Using keywords remove quartile = ' + str(keywords_remove_quartile)
            + ', stopwords = [' + str(stopwords) + ']'
            + ', weigh by IDF = ' + str(weigh_idf)
            , log_list = self.log_training
        )

        log.Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Training data: ' + str(td)
        )

        #
        # Extract all keywords
        # Our training now doesn't remove any word, uses no stopwords, but uses an IDF weightage to measure
        # keyword value.
        #
        log.Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Starting text cluster, calculate top keywords...'
            , log_list = self.log_training
        )
        self.textcluster = tcb.TextClusterBasic(
            text      = list(td[ctd.ChatTrainingData.COL_TDATA_TEXT_SEGMENTED]),
            stopwords = stopwords
        )
        self.textcluster.calculate_top_keywords(
            remove_quartile = keywords_remove_quartile
        )
        log.Log.info(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                         + ': Keywords extracted as follows:' + str(self.textcluster.keywords_for_fv))

        # Extract unique Commands/Intents
        log.Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Extracting unique commands/intents..'
            , log_list = self.log_training
        )
        self.commands = set( list( td[ctd.ChatTrainingData.COL_TDATA_INTENT_ID] ) )
        # Change back to list, this list may change due to deletion of invalid commands.
        self.commands = list(self.commands)
        log.Log.critical(self.commands)

        # Prepare data frames to hold RFV, etc. These also may change due to deletion of invalid commands.
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Preparing RFV data frames...'
            , log_list = self.log_training
        )
        m = np.zeros((len(self.commands), len(self.textcluster.keywords_for_fv)))
        self.df_rfv = pd.DataFrame(m, columns=self.textcluster.keywords_for_fv, index=self.commands)
        self.df_rfv_distance_furthest = pd.DataFrame({
            reffv.RefFeatureVector.COL_COMMAND:list(self.commands),
            reffv.RefFeatureVector.COL_DISTANCE_TO_RFV_FURTHEST:[ChatTraining.MINIMUM_THRESHOLD_DIST_TO_RFV]*len(self.commands),
            reffv.RefFeatureVector.COL_TEXT: ['']*len(self.commands)
        })

        #
        # Get IDF first
        #   We join all text from the same intent, to get IDF
        # TODO: IDF may not be the ideal weights, design an optimal one.
        #
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Joining all training data in the same command/intent to get IDF...'
            , log_list = self.log_training
        )
        i = 0
        text_bycategory = [''] * len(self.commands)
        for com in self.commands:
            # Join all text of the same command/intent together and treat them as one
            is_same_command = td[ctd.ChatTrainingData.COL_TDATA_INTENT_ID]==com
            text_samples = list(td[ctd.ChatTrainingData.COL_TDATA_TEXT_SEGMENTED].loc[is_same_command])
            text_com = ' '.join(text_samples)
            text_bycategory[i] = text_com
            i = i + 1
        log.Log.debug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                          + ': Joined intents: ' + str(text_bycategory))
        # Create a new TextCluster object
        self.textcluster_bycategory = tcb.TextClusterBasic(text=text_bycategory, stopwords=stopwords)
        # Always use the same keywords FV!!
        self.textcluster_bycategory.set_keywords(df_keywords=self.textcluster.df_keywords_for_fv.copy())

        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Calculating sentence matrix of combined Intents to get IDF...'
            , log_list = self.log_training
        )
        self.textcluster_bycategory.calculate_sentence_matrix(
            freq_measure='normalized',
            feature_presence_only=False,
            idf_matrix=None
        )


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
    td.to_csv(path_or_buf='/Users/mark.tan/Downloads/td.csv')


if __name__ == '__main__':
    au.Auth.init_instances()
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