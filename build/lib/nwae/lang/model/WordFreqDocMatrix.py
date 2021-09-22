# -*- coding: utf-8 -*-

from nwae.lang.preprocessing.BasicPreprocessor import BasicPreprocessor
import numpy as np
import pandas as pd
import collections
import nwae.lang.characters.LangCharacters as lc
from nwae.lang.model.FeatureVect import FeatureVector
from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
from nwae.utils.UnitTest import ResultObj, UnitTest, UnitTestParams
from nwae.lang.preprocessing.TxtPreprocessor import TxtPreprocessor

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class WordFreqDocMatrix:

    DEFAULT_NON_WORDS_LIST = lc.LangCharacters.UNICODE_BLOCK_PUNCTUATIONS \
                             + lc.LangCharacters.UNICODE_BLOCK_WORD_SEPARATORS \
                             + lc.LangCharacters.UNICODE_BLOCK_SENTENCE_SEPARATORS

    BY_FREQ              = 'freq'
    BY_FREQ_NORM         = 'freq_normalized'
    BY_FREQ_PROB         =  'freq_probility'
    """
    ограничить рост количества слова, помогая нейтрализировать влияния "stop words"
    """
    BY_LOG_FREQ          = 'log_frequency'
    BY_LOG_FREQ_NORM     = 'log_frequency_normalized'
    BY_SIGMOID_FREQ      = 'sigmoid_frequency'
    BY_SIGMOID_FREQ_NORM = 'sigmoid_frequency_normalized'

    LOG_BASE = np.exp(1)

    MAP_WORD_FREQ_FEATURE_VECT = {
        BY_FREQ:              FeatureVector.COL_FREQUENCY,
        BY_FREQ_PROB:         FeatureVector.COL_FREQ_PROB,
        BY_FREQ_NORM:         FeatureVector.COL_FREQ_NORM,
        BY_LOG_FREQ:          FeatureVector.COL_LOG_FREQ,
        BY_LOG_FREQ_NORM:     FeatureVector.COL_LOG_FREQ_NORM,
        BY_SIGMOID_FREQ:      FeatureVector.COL_SIGMOID_FREQ,
        BY_SIGMOID_FREQ_NORM: FeatureVector.COL_SIGMOID_FREQ_NORM
    }

    @staticmethod
    def map_to_feature_vect_word_freq_measure(
            freq_measure,
    ):
        if freq_measure in WordFreqDocMatrix.MAP_WORD_FREQ_FEATURE_VECT.keys():
            return WordFreqDocMatrix.MAP_WORD_FREQ_FEATURE_VECT[freq_measure]
        else:
            raise Exception(
                str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': No such freq measure "' + str(freq_measure) + '"'
            )

    #
    # Initialize with a list of text, assumed to be already word separated by space.
    #
    def __init__(self):
        return

    #
    # Make sure the sentences list is in correct type and form
    #
    def __sanity_check(
            self,
            sentences_list,
    ):
        for sent in sentences_list:
            if type(sent) not in (list, tuple):
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': Warning line ' + str(sent) + ', sentence not list type but type "'\
                         + str(type(sent)) + '": ' + str(sent)
                Log.warning(errmsg)
                raise Exception(errmsg)
            for j in range(len(sent)):
                w = sent[j]
                if type(w) is not str:
                    errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                             + ': Warning line ' + str(sent) + ', have non string type words "' \
                             + str(type(w)) + '": ' + str(w)
                    Log.warning(errmsg)
                    raise Exception(errmsg)
        return

    def remove_non_keywords(
            self,
            words_list,
            non_words_list = DEFAULT_NON_WORDS_LIST,
    ):
        return [w for w in words_list if w not in non_words_list]

    def calculate_top_keywords(
            self,
            sents_list,
            remove_quartile = 0,
            stopwords_list = (),
            add_unknown_word_in_list = False
    ):
        # Paste all sentences into a single huge vector
        all_words = [w for sent in sents_list for w in sent]
        all_words_pure = self.remove_non_keywords(words_list = all_words)
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Remain ' + str(len(all_words_pure))
            + ' words after removing non words (punctuations, etc)'
        )
        if stopwords_list:
            all_words_pure = self.remove_non_keywords(
                words_list = all_words_pure,
                non_words_list = stopwords_list,
            )
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Remain ' + str(len(all_words_pure))
            + ' words after removing non words and stopwords ' + str(stopwords_list)
        )
        w_freq = collections.Counter(all_words_pure)

        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ' : Calculating top keywords, total words (not unique) = ' + str(len(all_words))
            + ', after remove non keywords, total words (not unique) = ' + str(len(all_words_pure))
            + ': ' + str(all_words_pure)
        )

        # Order by top frequency keywords, and also convert to a Dictionary type (otherwise we can't extract
        # into DataFrame columns later)
        w_freq_common = w_freq.most_common()
        df_word_freq = pd.DataFrame({'Word': [x[0] for x in w_freq_common], 'Frequency': [x[1] for x in w_freq_common]})
        df_word_freq['Prop'] = df_word_freq['Frequency'] / sum(df_word_freq['Frequency'])
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Unique Word Frequency (' + str(df_word_freq.shape[0]) + ' words): '
            + str(w_freq_common)
        )

        #
        # There will be a lot of words, so we remove (by default) the lower 50% quartile of keywords.
        # This will help wipe out a lot of words, up to 80-90% or more.
        #
        q_remove = 0
        # If user passes in 0, no words will be removed
        if remove_quartile > 0:
            q_remove = np.percentile(df_word_freq['Frequency'], remove_quartile)
            Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Quartile : ' + str(remove_quartile) + '% is at frequency value ' + str(q_remove) + '.'
            )

        df_word_freq_qt = df_word_freq[df_word_freq['Frequency'] > q_remove]
        df_word_freq_qt = df_word_freq_qt.reset_index(drop=True)
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Unique Word Frequency (' + str(df_word_freq_qt.shape[0]) + ' words). After removing quartile : '
            + str(remove_quartile) + '%.'
        )
        Log.debug(str(df_word_freq_qt))

        df_keywords_for_fv = df_word_freq_qt
        keywords_for_fv = list(df_word_freq_qt['Word'])

        if add_unknown_word_in_list:
            if BasicPreprocessor.W_UNK not in keywords_for_fv:
                keywords_for_fv.append(BasicPreprocessor.W_UNK)
                Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Appended SYMBOL "' + str(BasicPreprocessor.W_UNK)
                    + '" to keywords list for the purpose of unknown words.'
                )
        return keywords_for_fv, df_keywords_for_fv

    def reconstruct_check(self, sent_vec, keywords_list):
        Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Reconstructing ' + str(sent_vec) + ' from keywords ' + str(keywords_list)
        )
        s_reconstruct_arr = []
        for j in range(len(sent_vec)):
            freq = sent_vec[j]
            while freq > 0:
                s_reconstruct_arr.append(keywords_list[j])
                freq = freq - 1
        return s_reconstruct_arr

    def get_word_doc_matrix(
            self,
            # A list of text sentences in list type, already in lowercase and cleaned of None or ''.
            # Preprocessing assumed to be done and no text processing will be done here.
            sentences_list,
            freq_measure             = BY_FREQ,
            feature_presence_only    = False,
            idf_matrix               = None,
            # For keywords list
            remove_quartile_keywords = 0,
            add_unknown_word_in_list = False,
            # Log base has no effect on LogFreqNormalized & LogFreqProbability as it is just a constant factor
            log_base                 = 10.0,
            stopwords_list           = (),
    ):
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Sentences list (before filter):\n\r' + str(sentences_list)
        )
        self.__sanity_check(sentences_list=sentences_list)

        #
        # 3. Model the sentences into a feature vector, using word frequency, relative positions, etc. as features
        #
        # Using the keywords set in this class, we create a profile template
        keywords_for_fv, df_keywords_for_fv = self.calculate_top_keywords(
            sents_list = sentences_list,
            remove_quartile = remove_quartile_keywords,
            add_unknown_word_in_list = add_unknown_word_in_list,
            stopwords_list = stopwords_list,
        )
        no_keywords = len(keywords_for_fv)

        model_fv = FeatureVector()
        model_fv.set_freq_feature_vector_template(
            list_symbols = keywords_for_fv
        )

        #
        # 4. Link top keywords to sentences in a matrix, used as features in feature vector
        #
        # Create a frequency matrix of keywords by sentences
        # Get feature vector of sentence
        # Number of rows or rank (number of axes) in numpy speak
        nrow = len(sentences_list)
        ncol = no_keywords
        sentence_matrix = np.zeros((nrow, ncol))

        # By default, use Probability
        col_to_use = self.map_to_feature_vect_word_freq_measure(freq_measure=freq_measure)
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Word frequency doc matrix using word freq model "' + str(freq_measure)
            + '" or in FeatureVect is "' + str(col_to_use) + '"'
        )

        # Fill matrix
        for i in range(0, sentence_matrix.shape[0], 1):
            sent_arr = sentences_list[i]
            if len(sent_arr) == 0:
                continue
            df_fv = model_fv.get_freq_feature_vector(
                text_list = sent_arr,
                feature_as_presence_only = feature_presence_only,
                log_base = log_base,
            )

            sentence_matrix[i] = list(df_fv[col_to_use])

            Log.debugdebug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Reconstruct "' + str(sentences_list[i]) + '" as "'
                + str(self.reconstruct_check(sent_vec=sentence_matrix[i], keywords_list=keywords_for_fv))
            )
            if np.sum(sentence_matrix[i]) <= 0.0:
                # Can happen if remove keywords by quartile
                # Log.warning(
                #     str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                #     + ': 0 Frequency total for sentence ' + str(i) + ', "' + str(sentences_list[i]) + '"'
                # )
                pass
            Log.debugdebug('Sentence ' + str(i) + ': ' + str(sent_arr) + '.')

        Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ' Sentence matrix (type=' + str(type(sentence_matrix)) + '):'
        )

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
            if freq_measure in [self.BY_FREQ_NORM, self.BY_LOG_FREQ_NORM, self.BY_SIGMOID_FREQ_NORM]:
                for j in range(0, sentence_matrix.shape[0], 1):
                    Log.debugdebug('For sentence ' + str(j) + '\n\r' + str(sentence_matrix[j]))

                    normalize_factor = np.sum(np.multiply(sentence_matrix[j], sentence_matrix[j])) ** 0.5

                    if normalize_factor > 0:
                        sentence_matrix[j] = sentence_matrix[j] / normalize_factor

            Log.debugdebug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ' After IDF weights, sentence matrix:\n\r' +  str(sentence_matrix)
            )

        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Sentence matrix after applying IDF, shape ' + str(sentence_matrix.shape) + '.'
        )

        Log.debugdebug('Sentence Matrix:\n\r' + str(sentence_matrix))
        # Return standard format rows as words, columns as documents like LSA
        word_doc_matrix = np.transpose(sentence_matrix)
        # The word row indexes in word_doc_matrix rows corresponds to the keyword indexes in keywords_for_fv list
        assert word_doc_matrix.shape[0] == len(keywords_for_fv)
        return word_doc_matrix, keywords_for_fv


class WordFreqDocMatrixUnitTest:
    def __init__(self, ut_params):
        self.ut_params = ut_params
        return

    def run_unit_test(self):
        res_final = ResultObj(count_ok=0, count_fail=0)

        sentences_list = [
            'искуссвенном матрице',
            'матрица, симуляция, Вселенная, реальный мир',
            'идеи симуляции',
            'Вселенная искусственна.',
            'мозг симуляцией симуляция симуляции',
            'реальный мир.. именно таковым.',
        ]
        kw_expected = [
            'симуляц', 'матриц', 'вселен', 'реальн', 'мир', 'искуссвен', 'ид', 'искусствен', 'мозг', 'имен', 'таков',
        ]
        t_word_doc_matrix_expected = [
            (WordFreqDocMatrix.BY_FREQ, [
                [0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,],
                [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,],
                [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,],
                [0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,],
                [3., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,],
                [0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1.,],
            ], 10.0),
            # Using log base np.exp(1)
            (WordFreqDocMatrix.BY_LOG_FREQ, [
                [0.,    0.693, 0.,    0.,    0.,    0.693, 0.,    0.,    0.,    0.,    0.,    ],
                [0.693, 0.693, 0.693, 0.693, 0.693, 0.,    0.,    0.,    0.,    0.,    0.,    ],
                [0.693, 0.,    0.,    0.,    0.,    0.,    0.693, 0.,    0.,    0.,    0.,    ],
                [0.,    0.,    0.693, 0.,    0.,    0.,    0.,    0.693, 0.,    0.,    0.,    ],
                [1.386, 0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.693, 0.,    0.,    ],
                [0.,    0.,    0.,    0.693, 0.693, 0.,    0.,    0.,    0.,    0.693, 0.693, ],
            ], np.exp(1)),
            (WordFreqDocMatrix.BY_LOG_FREQ, [
                [0.,    0.301, 0.,    0.,    0.,    0.301, 0.,    0.,    0.,    0.,    0.,    ],
                [0.301, 0.301, 0.301, 0.301, 0.301, 0.,    0.,    0.,    0.,    0.,    0.,    ],
                [0.301, 0.,    0.,    0.,    0.,    0.,    0.301, 0.,    0.,    0.,    0.,    ],
                [0.,    0.,    0.301, 0.,    0.,    0.,    0.,    0.301, 0.,    0.,    0.,    ],
                [0.602, 0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.301, 0.,    0.,    ],
                [0.,    0.,    0.,    0.301, 0.301, 0.,    0.,    0.,    0.,    0.301, 0.301, ],
            ], 10.0),
            # No difference whatever log base for BY_LOG_FREQ_NORM
            (WordFreqDocMatrix.BY_LOG_FREQ_NORM, [
                [0.,    0.707, 0.,    0.,    0.,    0.707, 0.,    0.,    0.,    0.,  0., ],
                [0.447, 0.447, 0.447, 0.447, 0.447, 0.,    0.,    0.,    0.,    0.,  0., ],
                [0.707, 0.,    0.,    0.,    0.,    0.,    0.707, 0.,    0.,    0.,  0., ],
                [0.,    0.,    0.707, 0.,    0.,    0.,    0.,    0.707, 0.,    0.,  0., ],
                [0.894, 0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.447, 0.,  0., ],
                [0.,    0.,    0.,    0.5,   0.5,   0.,    0.,    0.,    0.,    0.5, 0.5, ],
            ], np.exp(1)),
            (WordFreqDocMatrix.BY_LOG_FREQ_NORM, [
                [0.,    0.707, 0.,    0.,    0.,    0.707, 0.,    0.,    0.,    0.,  0., ],
                [0.447, 0.447, 0.447, 0.447, 0.447, 0.,    0.,    0.,    0.,    0.,  0., ],
                [0.707, 0.,    0.,    0.,    0.,    0.,    0.707, 0.,    0.,    0.,  0., ],
                [0.,    0.,    0.707, 0.,    0.,    0.,    0.,    0.707, 0.,    0.,  0., ],
                [0.894, 0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.447, 0.,  0., ],
                [0.,    0.,    0.,    0.5,   0.5,   0.,    0.,    0.,    0.,    0.5, 0.5, ],
            ], 12.567),
            (WordFreqDocMatrix.BY_SIGMOID_FREQ, [
                [0.,    0.462, 0.,    0.,    0.,    0.462, 0.,    0.,    0.,    0.,    0.,    ],
                [0.462, 0.462, 0.462, 0.462, 0.462, 0.,    0.,    0.,    0.,    0.,    0.,    ],
                [0.462, 0.,    0.,    0.,    0.,    0.,    0.462, 0.,    0.,    0.,    0.,    ],
                [0.,    0.,    0.462, 0.,    0.,    0.,    0.,    0.462, 0.,    0.,    0.,    ],
                [0.905, 0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.462, 0.,    0.,    ],
                [0.,    0.,    0.,    0.462, 0.462, 0.,    0.,    0.,    0.,    0.462, 0.462, ],
            ], 10),
            (WordFreqDocMatrix.BY_SIGMOID_FREQ_NORM, [
                [0.,    0.707, 0.,    0.,    0.,    0.707, 0.,    0.,    0.,    0.,  0., ],
                [0.447, 0.447, 0.447, 0.447, 0.447, 0.,    0.,    0.,    0.,    0.,  0., ],
                [0.707, 0.,    0.,    0.,    0.,    0.,    0.707, 0.,    0.,    0.,  0., ],
                [0.,    0.,    0.707, 0.,    0.,    0.,    0.,    0.707, 0.,    0.,  0., ],
                [0.891, 0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.455, 0.,  0., ],
                [0.,    0.,    0.,    0.5,   0.5,   0.,    0.,    0.,    0.,    0.5, 0.5, ],
            ], 10),
        ]

        from nwae.lang.LangFeatures import LangFeatures

        tpp = TxtPreprocessor(
            identifier_string      = 'test',
            # Don't need directory path for model, as we will not do spelling correction
            dir_path_model         = None,
            # Don't need features/vocabulary list from model
            model_features_list    = None,
            lang                   = LangFeatures.LANG_RU,
            dirpath_synonymlist    = self.ut_params.dirpath_synonymlist,
            postfix_synonymlist    = self.ut_params.postfix_synonymlist,
            dir_wordlist           = self.ut_params.dirpath_wordlist,
            postfix_wordlist       = self.ut_params.postfix_wordlist,
            dir_wordlist_app       = self.ut_params.dirpath_app_wordlist,
            postfix_wordlist_app   = self.ut_params.postfix_app_wordlist,
            do_spelling_correction = False,
            do_word_stemming       = True,
            do_profiling           = False,
        )
        sent_processed = tpp.preprocess_list(
            sentences_list = sentences_list,
        )
        Log.info('Processed sentences: ' + str(sent_processed))

        cl = WordFreqDocMatrix()
        for method_expected_base in t_word_doc_matrix_expected:
            method = method_expected_base[0]
            expected_word_doc_matrix = method_expected_base[1]
            base = method_expected_base[2]
            word_doc_matrix, keywords = cl.get_word_doc_matrix(
                sentences_list = sent_processed,
                freq_measure   = method,
                remove_quartile_keywords = 0,
                log_base       = base,
            )
            Log.info('Doc matrix (method=' + str(method) + '): ' + str(np.transpose(word_doc_matrix)))
            Log.info('Keywords (method=' + str(method) + '): ' + str(keywords))
            res_final.update_bool(res_bool=UnitTest.assert_true(
                observed = keywords,
                expected = kw_expected,
                test_comment = 'Method=' + str(method) + ', Keywords ' + str(keywords) + ', expected ' + str(kw_expected)
            ))
            for i in range(len(sentences_list)):
                doc = np.round(word_doc_matrix[:, i], 3)
                doc_expected = expected_word_doc_matrix[i]
                Log.info('Compare ' + str(doc) + ' with ' + str(doc_expected))
                res_final.update_bool(res_bool=UnitTest.assert_true(
                    observed = doc.tolist(),
                    expected = doc_expected,
                    test_comment = 'Method=' + str(method) + ', Document vec # ' + str(i) + ' ' + str(doc) + ', expected ' + str(doc_expected)
                ))

            for i in range(len(sentences_list)):
                s = sent_processed[i]
                s = [w for w in s if w in keywords]
                s_vec = word_doc_matrix[:, i]
                Log.debug('Reconstructing: ' + str(s_vec) + ', length ' + str(len(s_vec)))
                # No point reconstruct check for other than raw frequency count
                if method == WordFreqDocMatrix.BY_FREQ:
                    s_reconstruct = cl.reconstruct_check(sent_vec=s_vec, keywords_list=keywords)
                    Log.debug('   Reconstruct ' + str(i) + ': ' + str(s_reconstruct))
                    Log.debug('   Original: ' + str(s))
                    # Compare words in list
                    s.sort()
                    s_reconstruct.sort()
                    res_final.update_bool(res_bool=UnitTest.assert_true(
                        observed = s_reconstruct,
                        expected = s,
                        test_comment = 'Method=' + str(method) + ', Compare original sentence "' + str(s) + '" with reconstruct "' + str(s_reconstruct) + '"',
                    ))

        return res_final


if __name__ == '__main__':
    Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1

    from nwae.lang.config.Config import Config
    config = Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = Config,
        default_config_file = '/usr/local/git/nwae/nwae.lang/app.data/config/default.cf'
    )
    ut_params = UnitTestParams(
        dirpath_wordlist     = config.get_config(param=Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist     = config.get_config(param=Config.PARAM_NLP_POSTFIX_WORDLIST),
        dirpath_app_wordlist = config.get_config(param=Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_app_wordlist = config.get_config(param=Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        dirpath_synonymlist  = config.get_config(param=Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist  = config.get_config(param=Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        dirpath_model        = None,
    )

    res = WordFreqDocMatrixUnitTest(ut_params=ut_params).run_unit_test()
    print('Total passed ' + str(res.count_ok) + ', total fail ' + str(res.count_fail))
    exit(res.count_fail)
