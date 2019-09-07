# -*- coding: utf-8 -*-

import pickle
import re
from collections import Counter
import nwae.utils.Log as log
from inspect import currentframe, getframeinfo
import hanziconv as hzc
import nwae.lib.lang.LangFeatures as lf
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
        return (sentences_l1, sentences_l2)


if __name__ == '__main__':
    obj = Corpora()

    (sen_l1, sen_l2) = obj.retrieve_corpora(
        corpora_name = Corpora.CORPORA_NLTK_TRANSLATED_SENTENCES_EN_DE
    )
    print(sen_l1[0:10])
    print(sen_l2[0:10])