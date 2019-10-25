# -*- coding: utf-8 -*-

import pickle
import re
from collections import Counter
from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
import hanziconv as hzc
import nwae.lib.lang.TextProcessor as txtprocessor
import nltk


class Corpora:

    NLTK_COMTRANS = 'comtrans'

    CORPORA_NLTK_TRANSLATED_SENTENCES_EN_DE = 'alignment-de-en.txt'

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


if __name__ == '__main__':
    obj = Corpora()
    tp_obj = txtprocessor.TextProcessor(text_segmented_list=None)

    (sen_l1, sen_l2) = obj.retrieve_corpora(
        corpora_name = Corpora.CORPORA_NLTK_TRANSLATED_SENTENCES_EN_DE
    )
    print(sen_l1[0:10])
    print(sen_l2[0:10])
    print('Corpora length = ' + str(len(sen_l1)))

    clean_sen_l1 = [tp_obj.clean_punctuations_and_convert_to_lowercase(sentence=s) for s in sen_l1]
    clean_sen_l2 = [tp_obj.clean_punctuations_and_convert_to_lowercase(sentence=s) for s in sen_l2]
    print(clean_sen_l1[0:10])
    print(clean_sen_l2[0:10])
