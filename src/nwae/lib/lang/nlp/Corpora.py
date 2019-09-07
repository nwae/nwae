# -*- coding: utf-8 -*-

import pickle
import re
from collections import Counter
import nwae.utils.Log as log
from inspect import currentframe, getframeinfo
import hanziconv as hzc
from nltk.corpus import comtrans
import nwae.lib.lang.LangFeatures as lf


class Corpora:

    CORPORA_NLTK_TRANSLATED_SENTENCES_EN_FR = 'alignment-fr-en.txt'
    CORPORA_NLTK_TRANSLATED_SENTENCES_EN_DE = 'alignment-de-en.txt'

    def __init__(
            self
    ):
        return

    def retrieve_corpora(
            self,
            corpora_name
    ):
        als = comtrans.aligned_sents(corpora_name)
        sentences_l1 = [sent.words for sent in als]
        sentences_l2 = [sent.mots for sent in als]
        return (sentences_l1, sentences_l2)
