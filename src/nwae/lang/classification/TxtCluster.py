# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import nwae.math.Cluster as clst
from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
from nwae.lang.model.WordFreqDocMatrix import WordFreqDocMatrix


"""
NOTE: This class is not thread safe
"""
class TxtCluster:

    #
    # Initialize with a list of text, assumed to be already word separated by space.
    #
    def __init__(
            self,
            # A list of text sentences in list type, already in lowercase and cleaned of None or ''.
            # Preprocessing assumed to be done and no text processing will be done here.
            sentences_list,
    ):
        self.sentences_list = sentences_list
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Sentences list (before filter):\n\r' + str(self.sentences_list)
        )
        return

    #
    # The main clustering function
    #
    def cluster_text(
            self,
            ncenters,
            freq_measure    = WordFreqDocMatrix.BY_SIGMOID_FREQ_NORM,
            iterations      = 50,
            words_per_topic = 20,
            stopwords_list  = (),
            # In percent, e.g. 50 for 50%
            remove_quartile_keywords = 0.0,
            feature_presense_only = False,
    ):
        docmodel = WordFreqDocMatrix()
        word_freq_doc_matrix, keywords_list = docmodel.get_word_doc_matrix(
            sentences_list = self.sentences_list,
            # It seems sigmoid-freq-normalized outperforms log-freq-normalized, and for sure the original weak freq
            freq_measure   = freq_measure,
            stopwords_list = stopwords_list,
            remove_quartile_keywords = remove_quartile_keywords,
            feature_presence_only = feature_presense_only,
        )

        #
        # From the sentence matrix, we can calculate the IDF
        #
        # Do a redundant multiplication so that a copy is created, instead of pass by reference
        sentence_matrix = np.transpose(word_freq_doc_matrix)

        retval_cluster = clst.Cluster.cluster(
            matx          = sentence_matrix,
            feature_names = keywords_list,
            ncenters      = ncenters,
            iterations    = iterations
        )
        # Relative to sentence_matrix, shape is (n_documents,)
        # It will return a tuple of a single object (numpy ndarray)
        # e.g. ( array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int32), )
        doc_labels = retval_cluster.np_cluster_labels[0]
        # Relative to keywords_list, shape is (n_clusters, n_words_in_keywords_list)
        doc_centers = retval_cluster.np_cluster_centers

        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Doc labels, shape ' + str(doc_labels.shape) + ': ' + str(doc_labels)
        )
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Doc centers, shape ' + str(doc_centers.shape) + ': ' + str(doc_centers)
        )

        """
        Extract top words from each cluster center
        """
        df = pd.DataFrame(doc_centers)
        df.columns = keywords_list
        # Log.debugdebug(df)

        np_keyword = np.array(keywords_list)
        n_print_how_many = min(20, len(keywords_list))

        # Represent the topic centers in top words like LDA/LSA
        topic_words = []
        words_per_topic = min(words_per_topic, len(keywords_list))
        for center in doc_centers:
            # To sort the weights of keywords descending, we need to flip, as argsort() by default sorts ascending
            center_idx_sort = np.flip(np.argsort(center))
            # Extract the keywords from sorted indexes
            center_idx_sort_keywords = np_keyword[center_idx_sort]
            center_idx_sort_values   = center[center_idx_sort]
            wv = dict(zip(center_idx_sort_keywords[0:words_per_topic].tolist(), center_idx_sort_values[0:words_per_topic].tolist()))
            topic_words.append(wv)
            Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': For center ' + str(center[0:n_print_how_many])
                + '\n\rsorted to\n\r' + str(center_idx_sort[0:n_print_how_many])
                + '\n\rtop keywords-values\n\r' + str(wv)
            )
        return topic_words, doc_labels, doc_centers

    def to_data_frame(
            self,
            # Pre-processed sentences are hard to read, so we use original sentences
            sentences_list_no_preprocessing,
            # List of dictionary (if word-weights) representing a topic
            topic_words,
            # numpy ndarray
            doc_labels,
    ):
        # Convenient data frame for the topics & original documents
        df_classified = pd.DataFrame()
        for doc_idx in range(np.max(doc_labels) + 1):
            Log.debug('Cluster #' + str(doc_idx))
            Log.debug('\tWord-Value Center: ' + str(topic_words[doc_idx]))
            Log.debug('\tWords Center: ' + str(topic_words[doc_idx].keys()))
            cluster_words = str(topic_words[doc_idx].keys())
            topic_sentences = []
            for j in range(len(sentences_list_no_preprocessing)):
                if doc_labels[j] == doc_idx:
                    print('\t\t' + str(sentences_list_no_preprocessing[j]))
                    topic_sentences.append(sentences_list_no_preprocessing[j])
            df_topic = pd.DataFrame({
                'ClusterNo': doc_idx,
                'ClusterTopWords': cluster_words,
                'Sentence': topic_sentences,
            })
            df_classified = df_classified.append(df_topic)

        return df_classified


if __name__ == '__main__':
    from nwae.lang.preprocessing.TxtPreprcsrAllLang import TextPreprscrAllLang

    corpora = 'test'
    max_sentences = 0
    if corpora == 'test':
        from nwae.lang.corpora.TestCorpora import TestCorpora
        sample_fpath = 'sample.txt'
        tc = TestCorpora()
        sentences_list = tc.test_corpora_general(
            data_from_internet = False,
            write_to_file_path = sample_fpath,
            sample_fpath       = sample_fpath
        )
        cluster_count = 10
        stopwords_list = ()
        csv_output = None
        remove_quartile_keywords = 50.0
    else:
        raise Exception('??')

    from nwae.lang.config.Config import Config
    config = Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class       = Config,
        default_config_file = Config.CONFIG_FILE_PATH_DEFAULT
    )
    # Overwrite from config
    Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1

    tpp = TextPreprscrAllLang(
        dir_wordlist         = config.get_config(param=Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist     = config.get_config(param=Config.PARAM_NLP_POSTFIX_WORDLIST),
        dir_app_wordlist     = config.get_config(param=Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_app_wordlist = config.get_config(param=Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        dir_synlist          = config.get_config(param=Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synlist      = config.get_config(param=Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
    )
    sent_processed = tpp.preprocess_list_all_langs(
        sentences_list = sentences_list,
    )
    print('Processed sentences: ' + str(sent_processed))
    cl = TxtCluster(sentences_list = sent_processed)
    topic_words, cluster_labels, cluster_centers = cl.cluster_text(
        freq_measure    = WordFreqDocMatrix.BY_SIGMOID_FREQ_NORM,
        ncenters        = cluster_count,
        words_per_topic = 20,
        stopwords_list  = stopwords_list,
        remove_quartile_keywords = remove_quartile_keywords,
    )

    df_clsfy = cl.to_data_frame(
        sentences_list_no_preprocessing = sentences_list,
        topic_words = topic_words,
        doc_labels  = cluster_labels,
    )
    if csv_output:
        df_clsfy.to_csv(csv_output, sep=',')
    [ print(ww.keys()) for ww in topic_words ]

    # for label_idx in range(np.max(cluster_labels)+1):
    #     print('Cluster #' + str(label_idx))
    #     print('\tWord-Value Center: ' + str(words_values[label_idx]))
    #     for j in range(len(sentences_list)):
    #         if cluster_labels[j] == label_idx:
    #             # s = sent_processed[j].copy()
    #             # s.sort()
    #             # print('\t' + str(s))
    #             print('\t\t' + str(sentences_list[j]))

    exit(0)
