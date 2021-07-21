# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

from nwae.lang.preprocessing.BasicPreprocessor import BasicPreprocessor
import numpy as np
import pandas as pd
import nwae.math.Cluster as clst
import collections
import nwae.lang.characters.LangCharacters as lc
import nwae.lang.model.FeatureVector as fv
import nwae.utils.Log as log
from inspect import currentframe, getframeinfo
from nwae.lang.config.Config import Config
import nwae.utils.UnitTest as ut


#
# This is a basic text clustering algorithm, assuming that text is already nicely
# segmented into words (e.g. '我 昨天 的 流水 是 多少 ？'), uses frequency only as features,
# and relies on stopwords (and/or IDF measure) to increase accuracy.
# For English/Indonesian/Korean or any language with conjugations, root word extraction is a must,
# otherwise it will be inaccurate. Chinese, Thai, Vietnamese don't need this extra complication.
# TODO
#   - Normalized frequency works well for Chinese, but not English, why?
#   - The text processing functions should be moved away to TextProcessor class
#
# NOTE: This Class is NOT Thread Safe
#
class TextClusterBasic:

    #
    # Initialize with a list of text, assumed to be already word separated by space.
    #
    def __init__(
            self,
            # A list of text sentences in list type, already in lowercase and cleaned of None or ''.
            # Preprocessing assumed to be done and no text processing will be done here.
            sentences_list
    ):
        self.sentences_list = sentences_list
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Sentences list (before filter):\n\r' + str(self.sentences_list)
        )
        self.__sanity_check()

        # First we need the keywords for feature vector, then the sentence matrix based on these keywords
        self.df_keywords_for_fv = None
        self.keywords_for_fv = None
        self.df_sentence_matrix = None
        self.sentence_matrix = None
        self.df_idf = None
        self.idf_matrix = None
        return

    #
    # Make sure the sentences list is in correct type and form
    #
    def __sanity_check(
            self
    ):
        new_array = []
        for sent in self.sentences_list:
            if type(sent) not in (list, tuple):
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': Warning line ' + str(sent) + ', sentence not list type but type "'\
                         + str(type(sent)) + '": ' + str(sent)
                log.Log.warning(errmsg)
                raise Exception(errmsg)
            for j in range(len(sent)):
                w = sent[j]
                if type(w) is not str:
                    errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                             + ': Warning line ' + str(sent) + ', have non string type words "' \
                             + str(type(w)) + '": ' + str(w)
                    log.Log.warning(errmsg)
                    raise Exception(errmsg)
        return

    #
    # Option to pre-define keywords instead of extracting from text
    #
    def set_keywords(self, df_keywords):
        self.df_keywords_for_fv = df_keywords
        self.keywords_for_fv = list(df_keywords['Word'])
        return

    def remove_non_keywords(
            self,
            words_list
    ):
        words_list_pure = []
        for word in words_list:
            # Ignore word/sentence separators, punctuations
            if (word in lc.LangCharacters.UNICODE_BLOCK_PUNCTUATIONS) \
                    or (word in lc.LangCharacters.UNICODE_BLOCK_WORD_SEPARATORS) \
                    or (word in lc.LangCharacters.UNICODE_BLOCK_SENTENCE_SEPARATORS):
                continue
            words_list_pure.append(word)
        return words_list_pure

    #
    # TODO: Add POS tagging to word segmentation so that POS tag will be another feature,
    # TODO: and use some kind of keyword extraction algorithm.
    # TODO: TextRank don't work very well at all, need something else.
    #
    def calculate_top_keywords(
            self,
            remove_quartile=0,
            add_unknown_word_in_list=False
    ):
        log.Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ' : Calculating Top Keywords..'
        )

        # Paste all sentences into a single huge vector
        all_words = [w for sent in self.sentences_list for w in sent]
        all_words_pure = self.remove_non_keywords(words_list=all_words)
        col = collections.Counter(all_words_pure)

        # Order by top frequency keywords, and also convert to a Dictionary type (otherwise we can't extract
        # into DataFrame columns later)
        col = col.most_common()
        df_word_freq = pd.DataFrame({'Word': [x[0] for x in col], 'Frequency': [x[1] for x in col]})
        df_word_freq['Prop'] = df_word_freq['Frequency'] / sum(df_word_freq['Frequency'])
        log.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Word Frequency (' + str(df_word_freq.shape[0]) + ' words).'
        )
        log.Log.debugdebug(str(df_word_freq))

        #
        # There will be a lot of words, so we remove (by default) the lower 50% quartile of keywords.
        # This will help wipe out a lot of words, up to 80-90% or more.
        #
        q_remove = 0
        # If user passes in 0, no words will be removed
        if remove_quartile > 0:
            q_remove = np.percentile(df_word_freq['Frequency'], remove_quartile)
            log.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Quartile : ' + str(remove_quartile) + '% is at frequency value ' + str(q_remove) + '.'
            )

        df_word_freq_qt = df_word_freq[df_word_freq['Frequency'] > q_remove]
        df_word_freq_qt = df_word_freq_qt.reset_index(drop=True)
        log.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Word Frequency (' + str(df_word_freq_qt.shape[0]) + ' words). After removing quartile : '
            + str(remove_quartile) + '%.'
        )
        log.Log.debugdebug(str(df_word_freq_qt))

        self.df_keywords_for_fv = df_word_freq_qt
        self.keywords_for_fv = list(df_word_freq_qt['Word'])

        if add_unknown_word_in_list:
            if BasicPreprocessor.W_UNK not in self.keywords_for_fv:
                self.keywords_for_fv.append(BasicPreprocessor.W_UNK)
                log.Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Appended SYMBOL "' + str(BasicPreprocessor.W_UNK)
                    + '" to keywords list for the purpose of unknown words.'
                )

        return

    def calculate_sentence_matrix(
            self,
            freq_measure          = 'tf',
            feature_presence_only = False,
            idf_matrix            = None
    ):
        #
        # 3. Model the sentences into a feature vector, using word frequency, relative positions, etc. as features
        #
        # Using the keywords set in this class, we create a profile template
        no_keywords = len(self.keywords_for_fv)

        model_fv = fv.FeatureVector()
        model_fv.set_freq_feature_vector_template(
            list_symbols = self.keywords_for_fv
        )

        #
        # 4. Link top keywords to sentences in a matrix, used as features in feature vector
        #
        # Create a frequency matrix of keywords by sentences
        # Get feature vector of sentence
        # Number of rows or rank (number of axes) in numpy speak
        nrow = len(self.sentences_list)
        ncol = no_keywords
        # By default this is TF
        sentence_matrix = np.zeros((nrow, ncol))
        sentence_matrix_freq = np.zeros((nrow, ncol))
        # The normalized version, by proportion
        sentence_matrix_norm = np.zeros((nrow, ncol))

        # Fill matrix
        for i in range(0, sentence_matrix.shape[0], 1):
            sent_arr = self.sentences_list[i]
            if len(sent_arr) == 0:
                continue
            df_fv = model_fv.get_freq_feature_vector(
                text_list = sent_arr,
                feature_as_presence_only = feature_presence_only
            )
            # By default, use TF
            sentence_matrix[i] = list(df_fv['TF'])
            sentence_matrix_freq[i] = list(df_fv['Frequency'])
            sentence_matrix_norm[i] = list(df_fv['FrequencyNormalized'])
            log.Log.debugdebug('Sentence ' + str(i) + ': ' + str(sent_arr) + '.')

        #
        # By default, we should use the TF form.
        #
        if freq_measure == 'normalized':
            sentence_matrix = sentence_matrix_norm
        elif freq_measure == 'frequency':
            sentence_matrix = sentence_matrix_freq

        log.Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ' Sentence matrix (type=' + str(type(sentence_matrix)) + '):'
        )
        log.Log.debugdebug(sentence_matrix)

        #
        # Weigh by IDF Matrix if given
        #
        if idf_matrix is not None:
            dim_sentence_matrix = sentence_matrix.shape
            dim_idf_matrix = idf_matrix.shape
            if dim_idf_matrix[0]!=dim_sentence_matrix[1]:
                raise Exception(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': IDF Matrix Dimensions must be (N,1) where N=number of columns in sentence matrix. '
                    + 'Sentence Matrix Dimension = ' + str(dim_sentence_matrix) + '. '
                    + 'IDF Matrix Dimension = ' + str(dim_idf_matrix)
                )
            # This is not matrix multiplication, but should be
            sentence_matrix = np.multiply(sentence_matrix, np.transpose(idf_matrix))
            # Normalize back if initially using a normalized measure
            if freq_measure == 'normalized':
                for j in range(0, sentence_matrix.shape[0], 1):
                    log.Log.debugdebug('For sentence ' + str(j) + '\n\r' + str(sentence_matrix[j]))

                    normalize_factor = np.sum(np.multiply(sentence_matrix[j], sentence_matrix[j])) ** 0.5

                    if normalize_factor > 0:
                        sentence_matrix[j] = sentence_matrix[j] / normalize_factor

            log.Log.debugdebug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ' After IDF weights, sentence matrix:\n\r' +  str(sentence_matrix)
            )

        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Sentence matrix after applying IDF, shape ' + str(sentence_matrix.shape) + '.'
        )

        # # Convert to dataframe with names
        # df_sentence_matrix = pd.DataFrame(data=sentence_matrix, columns=self.keywords_for_fv)
        # self.df_sentence_matrix = df_sentence_matrix
        # # TODO CHange to use np array
        # dm = df_sentence_matrix.shape

        # # TODO Why do this? Since we already have the sentence matrix above. Should just use the values
        # if TextClusterBasic.USE_DEPRECATED_NUMPY_MATRIX:
        #     self.sentence_matrix = np.matrix(df_sentence_matrix.values)
        # else:
        #     self.sentence_matrix = np.array(df_sentence_matrix.values).reshape(dm[0], dm[1])
        self.sentence_matrix = sentence_matrix
        log.Log.debugdebug('Sentence Matrix:\n\r' + str(self.sentence_matrix))
        return

    #
    # Inverse Document Frequency = log(Total_Documents / Number_of_Documents_of_Word_occurrence)
    # This measure doesn't work well in automatic clustering of texts.
    # TODO
    #   Long outdated code
    #
    def calculate_idf(
            self
    ):
        total_columns_or_words = self.sentence_matrix.shape[1]

        word_presence_matrix = (self.sentence_matrix>0)*1
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Word presence = ' + str(word_presence_matrix)
        )

        # Sum columns
        # If using outdated np.matrix, this keyword_presense will be a (1,n) array, but if using np.array, this will be 1-dimensional vector
        keyword_presence = np.sum(word_presence_matrix, axis=0)
        # Make sure idf is not infinity
        keyword_presence[keyword_presence==0] = 1
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Keyword presence = ' + str(keyword_presence)
        )

        # Total document count
        n_documents = self.sentence_matrix.shape[0]
        log.Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Total unique documents/intents to calculate IDF = ' + str(n_documents)
        )

        # If using outdated np.matrix, this IDF will be a (1,n) array, but if using np.array, this will be 1-dimensional vector
        idf = np.log(n_documents / keyword_presence)

        df_idf = None
        if n_documents <= 1:
            # If there is only a single document, there is no IDF as log(1)=0 and all values will be 0, so just set all to 1
            df_idf = pd.DataFrame({'Word': self.keywords_for_fv, 'IDF': 1})
            log.Log.warning(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Only ' + str(n_documents) + ' document in IDF calculation. Setting IDF to 1: '
                + str(df_idf))
        else:
            try:
                log.Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Using Keywords: ' + str(self.keywords_for_fv)
                )
                log.Log.debugdebug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Using IDF type ' + str(type(idf)) + ': ' + str(idf)
                )
                # To make sure idf is in a single array, we convert to values as it could be a single row narray or matrix
                if idf.shape[0] == total_columns_or_words:
                    df_idf = pd.DataFrame({'Word':self.keywords_for_fv, 'IDF':idf.tolist()})
                else:
                    df_idf = pd.DataFrame({'Word':self.keywords_for_fv, 'IDF':idf.tolist()[0]})
            except Exception as ex:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': IDF Data Frame error: ' + str(ex) + '.'
                log.Log.critical(errmsg)
                raise(errmsg)

        self.df_idf = df_idf
        # Make sure to transpose to get a column matrix
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': IDF data frame: ' + str(self.df_idf)
        )
        self.idf_matrix = np.array( df_idf['IDF'].values ).reshape(df_idf.shape[0],1)

        log.Log.debugdebug(str(self.__class__) + 'IDF Matrix as follows: ' + str(self.idf_matrix))
        return

    #
    # The main clustering function
    # TODO
    #   This method already broken
    #
    def cluster_text(
            self,
            ncenters,
            iterations            = 50,
            feature_presence_only = False,
            freq_measure          = 'tf',
            weigh_idf             = False,
            optimal_cluster_threshold_change = clst.Cluster.THRESHOLD_CHANGE_DIST_TO_CENTROIDS
    ):
        # Model sentences into feature vectors
        self.calculate_sentence_matrix(
            freq_measure          = freq_measure,
            feature_presence_only = feature_presence_only,
            idf_matrix            = None
        )

        #
        # From the sentence matrix, we can calculate the IDF
        #
        # Do a redundant multiplication so that a copy is created, instead of pass by reference
        sentence_matrix = self.sentence_matrix * 1
        if weigh_idf:
            self.calculate_idf()
            # Recalculate sentence matrix
            self.calculate_sentence_matrix(
                freq_measure          = freq_measure,
                feature_presence_only = feature_presence_only,
                idf_matrix            = self.idf_matrix.copy()
            )

        #
        # 5. Cluster sentences in matrix
        #
        # Optimal clusters
        # TODO: Improve the optimal cluster algorithm before using the value returned
        if False:
            optimal_clusters = clst.Cluster.get_optimal_cluster(
                matx=sentence_matrix,
                n_tries=10,
                iterations=iterations,
                threshold_change=optimal_cluster_threshold_change,
                verbose=verbose)
            if verbose>=1:
                log.Log.log('Optimal Clusters = ' + str(optimal_clusters))

        retval_cluster = clst.Cluster.cluster(
            matx          = sentence_matrix,
            feature_names = self.keywords_for_fv,
            ncenters      = ncenters,
            iterations    = iterations
        )
        log.Log.debug(
            'Return cluster: ' + str(retval_cluster.np_cluster_labels)
        )
        # It will return a tuple of a single object (numpy ndarray)
        # e.g. ( array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int32), )
        return retval_cluster.np_cluster_labels


if __name__ == '__main__':
    from nwae.lang.classification.TextClusterBasicUnitTest import TextClusterBasicUnitTest
    config = Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class       = Config,
        default_config_file = Config.CONFIG_FILE_PATH_DEFAULT
    )
    ut_params = ut.UnitTestParams(
        dirpath_wordlist     = config.get_config(param=Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist     = config.get_config(param=Config.PARAM_NLP_POSTFIX_WORDLIST),
        dirpath_app_wordlist = config.get_config(param=Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_app_wordlist = config.get_config(param=Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        dirpath_synonymlist  = config.get_config(param=Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist  = config.get_config(param=Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        dirpath_model        = None
    )

    TextClusterBasicUnitTest(
        ut_params = ut_params
    ).run_unit_test()
