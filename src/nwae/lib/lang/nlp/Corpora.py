# -*- coding: utf-8 -*-

import pickle
from collections import Counter
from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
import nwae.lib.lang.TextProcessor as txtprocessor
import nltk


class Corpora:

    NLTK_COMTRANS = 'comtrans'

    CORPORA_NLTK_TRANSLATED_SENTENCES_EN_DE = 'alignment-de-en.txt'

    _PAD = '_PAD'
    _GO  = '_GO'
    _EOS = '_EOS'
    _UNK = '_UNK'
    _START_VOCAB = [_PAD, _GO, _EOS, _UNK]
    PAD_ID = 0
    GO_ID  = 1
    EOS_ID = 2
    UNK_ID = 3
    OP_DICT_IDS = [PAD_ID, GO_ID, EOS_ID, UNK_ID]

    def __init__(
            self
    ):
        nltk.download(Corpora.NLTK_COMTRANS)
        return

    def retrieve_corpora(
            self,
            corpora_name
    ):
        from nltk.corpus import comtrans
        als = comtrans.aligned_sents(corpora_name)
        sentences_l1 = [sent.words for sent in als]
        sentences_l2 = [sent.mots for sent in als]
        Log.info('Sentences length = ' + str(len(sentences_l1)))

        # Filter length
        (sentences_l1, sentences_l2) = self.filter_pair_sentence_length(
            sentences_arr_l1 = sentences_l1,
            sentences_arr_l2 = sentences_l2,
            max_len = 20
        )
        Log.info('Sentences length after filtering = ' + str(len(sentences_l1)))
        assert len(sentences_l1) == len(sentences_l2)
        return (sentences_l1, sentences_l2)

    def filter_pair_sentence_length(
            self,
            sentences_arr_l1,
            sentences_arr_l2,
            max_len,
            min_len = 0
    ):
        assert len(sentences_arr_l1) == len(sentences_arr_l2)

        filtered_sentences_l1 = []
        filtered_sentences_l2 = []

        for i in range(len(sentences_arr_l1)):
            sent1 = sentences_arr_l1[i]
            sent2 = sentences_arr_l2[i]
            if min_len <= len(sent1) <= max_len and \
                    min_len <= len(sent2) <= max_len:
                filtered_sentences_l1.append(sent1)
                filtered_sentences_l2.append(sent2)

        return (filtered_sentences_l1, filtered_sentences_l2)

    #
    # Order words from highest to lowest frequency, and assign a number to each number
    #
    def create_indexed_dictionary(
            self,
            sentences,
            dict_size = 100,
            storage_path = None
    ):
        count_words = Counter()
        dict_words = {}
        opt_dict_size_initial = len(Corpora.OP_DICT_IDS)
        for sen in sentences:
            for word in sen:
                count_words[word] += 1

        dict_words[Corpora._PAD] = Corpora.PAD_ID
        dict_words[Corpora._GO] = Corpora.GO_ID
        dict_words[Corpora._EOS] = Corpora.EOS_ID
        dict_words[Corpora._UNK] = Corpora.UNK_ID

        # Add to dictionary of words starting from highest term frequency
        for idx, item in enumerate(count_words.most_common(dict_size)):
            Log.info('Doing idx ' + str(idx) + ', item ' + str(item))
            dict_words[item[0]] = idx + opt_dict_size_initial

        if storage_path:
            pickle.dump(dict_words, open(storage_path, 'wb'))
        return dict_words

    def sentences_to_indexes(
            self,
            sentences,
            indexed_dict
    ):
        indexed_sentences = []
        not_found_counter = 0
        for sent in sentences:
            idx_sent = []
            for word in sent:
                try:
                    idx_sent.append(indexed_dict[word])
                except Exception as ex:
                    idx_sent.append(Corpora.UNK_ID)
                    not_found_counter += 1
            indexed_sentences.append(idx_sent)

        return indexed_sentences

    def extract_max_length(self, corpora):
        return max([len(sentence) for sentence in corpora])

    def prepare_sentences(
            self,
            sentences_l1,
            sentences_l2,
            len_l1,
            len_l2
    ):
        assert(len(sentences_l1) == len(sentences_l2))
        data_set = []

        for i in range(len(sentences_l1)):
            padding_l1 = len_l1 - len(sentences_l1[i])
            pad_sentence_l1 = ([Corpora.PAD_ID]*padding_l1) + sentences_l1[i]

            padding_l2 = len_l2 - len(sentences_l2[i])
            pad_sentence_l2 = [Corpora.GO_ID] + sentences_l2[i] + [Corpora.EOS_ID]\
                              + ([Corpora.PAD_ID] * padding_l2)
            data_set.append([pad_sentence_l1, pad_sentence_l2])

        return data_set

    def build_data_set(self):
        tp_obj = txtprocessor.TextProcessor(text_segmented_list=None)

        (sen_l1, sen_l2) = self.retrieve_corpora(
            corpora_name = Corpora.CORPORA_NLTK_TRANSLATED_SENTENCES_EN_DE
        )
        Log.info(
            'Retrieved Corpora lang 1 of length ' + str(len(sen_l1)) + ': ' + str(sen_l1[0:10])
        )
        Log.info(
            'Retrieved Corpora lang 2 of length ' + str(len(sen_l2)) + ': ' + str(sen_l2[0:10])
        )

        clean_sen_l1 = [tp_obj.clean_punctuations_and_convert_to_lowercase(sentence=s) for s in sen_l1]
        clean_sen_l2 = [tp_obj.clean_punctuations_and_convert_to_lowercase(sentence=s) for s in sen_l2]
        Log.info('Cleaned sentence: ' + str(clean_sen_l1[0:10]))
        Log.info('Cleaned sentence: ' + str(clean_sen_l2[0:10]))

        dict_words_l1 = self.create_indexed_dictionary(
            sentences=clean_sen_l1
        )
        Log.info('Dict words lang 1: ' + str(dict_words_l1))
        dict_words_l2 = self.create_indexed_dictionary(
            sentences=clean_sen_l2
        )
        Log.info('Dict words lang 2: ' + str(dict_words_l2))

        idx_sentences_l1 = self.sentences_to_indexes(
            sentences=clean_sen_l1,
            indexed_dict=dict_words_l1
        )
        Log.info('Indexed sentences lang 1: ' + str(idx_sentences_l1[0:10]))
        idx_sentences_l2 = self.sentences_to_indexes(
            sentences=clean_sen_l2,
            indexed_dict=dict_words_l2
        )
        Log.info('Indexed sentences lang 2: ' + str(idx_sentences_l2[0:10]))

        max_len_l1 = self.extract_max_length(corpora=idx_sentences_l1)
        max_len_l2 = self.extract_max_length(corpora=idx_sentences_l2)
        Log.info('Max length l1 = ' + str(max_len_l1) + ', max length l2 = ' + str(max_len_l2))

        data_set = self.prepare_sentences(
            sentences_l1 = idx_sentences_l1,
            sentences_l2 = idx_sentences_l2,
            len_l1       = max_len_l1,
            len_l2       = max_len_l2
        )
        Log.info('Data set: ' + str(data_set[0:10]))

        return data_set


if __name__ == '__main__':
    Corpora().build_data_set()

