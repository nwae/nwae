# -*- coding: utf-8 -*-

import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo
import collections
import pickle
from nwae.lang.LangFeatures import LangFeatures
from nwae.lang.preprocessing.BasicPreprocessor import BasicPreprocessor


#
# General processing of text data to usable math forms for NLP processing
#
class TextProcessor:

    def __init__(
            self,
            # We support a single lang or a list of languages, one for each text
            lang_str_or_list,
            # A list of sentences in str format, but split by words either with our
            # default word delimiter DEFAULT_WORD_SPLITTER or space or whatever.
            # Or can also be a list of sentences in already split list format
            text_segmented_list
    ):
        self.lang = lang_str_or_list
        self.text_segmented_list = text_segmented_list

        self.lang_list = None
        if type(self.lang) in (list, tuple):
            self.lang_list = [LangFeatures.map_to_lang_code_iso639_1(lang_code=l) for l in self.lang]
            if len(self.lang_list) != len(self.text_segmented_list):
                raise Exception(
                    str(TextProcessor.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Language list & text segmented list must have same length! '
                )
        else:
            self.lang = LangFeatures.map_to_lang_code_iso639_1(
                lang_code = self.lang
            )
            self.lang_list = [self.lang] * len(self.text_segmented_list)

        lg.Log.debugdebug(
            str(TextProcessor.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Text segmented list: ' + str(self.text_segmented_list)
        )
        return

    #
    # We want to convert a list of segmented text:
    #   [ 'Российский робот "Федор" возвратился на Землю на корабле "Союз МС-14"',
    #     'Корабль "Союз МС-14" с роботом "Федор" был запущен на околоземную орбиту 22 августа.'
    #     ... ]
    #
    # to a list of lists
    #   [ ['Российский', 'робот' ,'"', 'Федор', '"', 'возвратился', 'на', 'Землю', 'на', 'корабле', '"', 'Союз', 'МС-14', '"']
    #     ['Корабль', '"', 'Союз', 'МС-14', '"', 'с', 'роботом', '"', 'Федор', '"', 'был', 'запущен', 'на', 'околоземную', 'орбиту', '22', 'августа', '.' ]
    #     ... ]
    #
    def convert_segmented_text_to_array_form(
            self
    ):
        # A list of sentences in list format
        sentences_list = []
        for i in range(0,len(self.text_segmented_list),1):
            sent = self.text_segmented_list[i]
            sep_lang = BasicPreprocessor.get_word_separator(lang=self.lang_list[i])

            if type(sent) is not str:
                raise Exception(
                    str(TextProcessor.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Sentence "' + str(sent) + '" not string'
                )
            # Try to split by default splitter
            split_arr = sent.split(sep_lang)
            lg.Log.debug(
                str(TextProcessor.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Split sentence by word separator "' + str(sep_lang)
                + '"\n\r   "' + str(sent)
                + '" to:\n\r   ' + str(split_arr)
            )
            # Append to return array
            sentences_list.append(split_arr)

        return sentences_list

    #
    # Order words from highest to lowest frequency, and assign a number to each word
    # We should replace our current way of creating a dictionary using the words itself
    # in TextClusterBasic class.
    #
    def create_indexed_dictionary(
            self,
            # List of sentences (each sentence is a list of words)
            sentences,
            dict_size = 10000,
            storage_path = None
    ):
        count_words = collections.Counter()
        dict_words = {}
        opt_dict_size_initial = len(BasicPreprocessor.OP_DICT_IDS)
        for sen in sentences:
            for word in sen:
                count_words[word] += 1

        #
        # Map words to number
        #
        dict_words[BasicPreprocessor.W_PAD] = BasicPreprocessor.PAD_ID
        dict_words[BasicPreprocessor.W_GO] = BasicPreprocessor.GO_ID
        dict_words[BasicPreprocessor.W_EOS] = BasicPreprocessor.EOS_ID
        dict_words[BasicPreprocessor.W_UNK] = BasicPreprocessor.UNK_ID

        # Add to dictionary of words starting from highest term frequency
        for idx, item in enumerate(count_words.most_common(dict_size)):
            lg.Log.debugdebug('Doing idx ' + str(idx) + ', item ' + str(item))
            dict_words[item[0]] = idx + opt_dict_size_initial

        if storage_path:
            try:
                pickle.dump(dict_words, open(storage_path, 'wb'))
            except Exception as ex:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ': Exception writing indexed dictionary to file: ' + str(ex) + '.'
                lg.Log.error(errmsg)

        return dict_words

    #
    # Based on indexed dictionary, convert sentences of words to sentences of numbers
    #
    def sentences_to_indexes(
            self,
            # List of sentences (each sentence is a list of words)
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
                    idx_sent.append(BasicPreprocessor.UNK_ID)
                    not_found_counter += 1
            indexed_sentences.append(idx_sent)

        return indexed_sentences

    def extract_max_length(
            self,
            corpora
    ):
        return max([len(sentence) for sentence in corpora])

    def prepare_sentence_pairs(
            self,
            # List of sentences (each sentence is a list of words)
            sentences_l1,
            sentences_l2,
            len_l1,
            len_l2
    ):
        assert(len(sentences_l1) == len(sentences_l2))
        data_set = []

        for i in range(len(sentences_l1)):
            padding_l1 = len_l1 - len(sentences_l1[i])
            # For left pair, pad from left
            pad_sentence_l1 = ([BasicPreprocessor.PAD_ID]*padding_l1) + sentences_l1[i]

            padding_l2 = len_l2 - len(sentences_l2[i])
            # For right pair, pad from right
            pad_sentence_l2 = [BasicPreprocessor.GO_ID] + sentences_l2[i] + [BasicPreprocessor.EOS_ID]\
                              + ([BasicPreprocessor.PAD_ID] * padding_l2)
            data_set.append([pad_sentence_l1, pad_sentence_l2])

        return data_set


if __name__ == '__main__':
    sent_list = [
        'Российский робот "Федор" возвратился на Землю на корабле "Союз МС-14"',
        'Корабль "Союз МС-14" с роботом "Федор" был запущен на околоземную орбиту 22 августа.'
        ]

    obj = TextProcessor(
        lang_str_or_list = 'ru',
        text_segmented_list = sent_list
    )
    sent_list_list = obj.convert_segmented_text_to_array_form()
    print(sent_list_list)

    clean_sent = [BasicPreprocessor.clean_punctuations(sentence=s) for s in sent_list_list]
    lg.Log.info('Cleaned sentence: ' + str(clean_sent[0:10]))

    clean_empty_sent = [x for x in clean_sent if x != '']
    lg.Log.info('Final Cleaned sentence: ' + str(clean_sent[0:10]))

    dict_words = obj.create_indexed_dictionary(
        sentences = clean_sent
    )
    lg.Log.info('Dict words lang 1: ' + str(dict_words))

    idx_sentences = obj.sentences_to_indexes(
        sentences    = clean_sent,
        indexed_dict = dict_words
    )
    print('Indexed sentences: ' + str(idx_sentences))
