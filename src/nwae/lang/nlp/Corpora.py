# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
from nwae.lang.preprocessing.BasicPreprocessor import BasicPreprocessor
import nltk
from nltk.corpus import comtrans
from nwae.lang.LangFeatures import LangFeatures
from nwae.utils.networking.Ssl import Ssl


class Corpora:

    NLTK_COMTRANS = 'comtrans'

    CORPORA_NLTK_TRANSLATED_SENTENCES_EN_DE = 'alignment-de-en.txt'

    def __init__(
            self,
            lang = LangFeatures.LANG_EN
    ):
        self.lang = LangFeatures.map_to_lang_code_iso639_1(
            lang_code = lang
        )
        Ssl.disable_ssl_check()
        try:
            if nltk.download(Corpora.NLTK_COMTRANS):
                Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': NLTK download of "' + Corpora.NLTK_COMTRANS + '" OK.'
                )
            else:
                raise Exception(
                    'Download "' + str(Corpora.NLTK_COMTRANS) + '" returned False'
                )
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': NLTK download of "' + str(Corpora.NLTK_COMTRANS) + '" exception: ' \
                     + str(ex) + '.'
            Log.error(errmsg)
            raise Exception(errmsg)
        return

    def retrieve_corpora(
            self,
            corpora_name
    ):
        try:
            als = comtrans.aligned_sents(corpora_name)
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Comtrans download of corpora "' + str(corpora_name) + '" exception: ' \
                     + str(ex) + '.'
            Log.error(errmsg)
            raise Exception(errmsg)

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
    # The final output is a list of sentence pairs:
    #  - Left sentence pair is padded with initial 0's, right sentence padded at the end
    #  - All words converted numbers
    #
    def build_data_set(self):
        (sen_l1, sen_l2) = self.retrieve_corpora(
            corpora_name = Corpora.CORPORA_NLTK_TRANSLATED_SENTENCES_EN_DE
        )
        Log.info(
            'Retrieved Corpora lang 1 of length ' + str(len(sen_l1)) + ': ' + str(sen_l1[0:10])
        )
        Log.info(
            'Retrieved Corpora lang 2 of length ' + str(len(sen_l2)) + ': ' + str(sen_l2[0:10])
        )

        clean_sen_l1 = [BasicPreprocessor.clean_punctuations(sentence=s) for s in sen_l1]
        clean_sen_l2 = [BasicPreprocessor.clean_punctuations(sentence=s) for s in sen_l2]
        Log.info('Cleaned sentence: ' + str(clean_sen_l1[0:10]))
        Log.info('Cleaned sentence: ' + str(clean_sen_l2[0:10]))

        dict_words_l1 = BasicPreprocessor.create_indexed_dictionary(
            sentences=clean_sen_l1
        )
        Log.info('Dict words lang 1: ' + str(dict_words_l1))
        dict_words_l2 = BasicPreprocessor.create_indexed_dictionary(
            sentences=clean_sen_l2
        )
        Log.info('Dict words lang 2: ' + str(dict_words_l2))

        idx_sentences_l1 = BasicPreprocessor.sentences_to_indexes(
            sentences    = clean_sen_l1,
            indexed_dict = dict_words_l1
        )
        Log.info('Indexed sentences lang 1: ' + str(idx_sentences_l1[0:10]))
        idx_sentences_l2 = BasicPreprocessor.sentences_to_indexes(
            sentences    = clean_sen_l2,
            indexed_dict = dict_words_l2
        )
        Log.info('Indexed sentences lang 2: ' + str(idx_sentences_l2[0:10]))

        max_len_l1 = BasicPreprocessor.extract_max_length(corpora=idx_sentences_l1)
        max_len_l2 = BasicPreprocessor.extract_max_length(corpora=idx_sentences_l2)
        Log.info('Max length l1 = ' + str(max_len_l1) + ', max length l2 = ' + str(max_len_l2))

        data_set = BasicPreprocessor.prepare_sentence_pairs(
            sentences_l1 = idx_sentences_l1,
            sentences_l2 = idx_sentences_l2,
            len_l1       = max_len_l1,
            len_l2       = max_len_l2
        )
        Log.info('Data set: ' + str(data_set[0:10]))

        class retclass:
            def __init__(self, list_sent_pairs, max_len_l1, max_len_l2):
                self.list_sent_pairs = list_sent_pairs
                self.max_len_l1 = max_len_l1
                self.max_len_l2 = max_len_l2

        return retclass(list_sent_pairs=data_set, max_len_l1=max_len_l1, max_len_l2=max_len_l2)


if __name__ == '__main__':
    ret = Corpora().build_data_set()
    print('Data set type "' + str(type(ret.list_sent_pairs)) + '", size = ' + str(len(ret.list_sent_pairs)))
    #for item in ret.data_set:
    #    print(item)

