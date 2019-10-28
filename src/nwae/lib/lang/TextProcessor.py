# -*- coding: utf-8 -*-

import re
import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo
import collections
import pickle


#
# General processing of text data to usable math forms for NLP processing
#
class TextProcessor:

    #
    # Our own default word delimiter
    #
    DEFAULT_WORD_SPLITTER = '--||--'
    DEFAULT_SPACE_SPLITTER = ' '

    # Sentence padding if shorter than min length
    _PAD = '_PAD'
    # Start of sentence
    _GO  = '_GO'
    # End of sentence
    _EOS = '_EOS'
    # Unknown word
    _UNK = '_UNK'
    _START_VOCAB = [_PAD, _GO, _EOS, _UNK]
    PAD_ID = 0
    GO_ID  = 1
    EOS_ID = 2
    UNK_ID = 3
    OP_DICT_IDS = [PAD_ID, GO_ID, EOS_ID, UNK_ID]

    def __init__(
            self,
            # A list of sentences in str format, but split by words either with our
            # default word delimiter DEFAULT_WORD_SPLITTER or space or whatever.
            # Or can also be a list of sentences in already split list format
            text_segmented_list
    ):
        self.text_segmented_list = text_segmented_list
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
            self,
            sep = DEFAULT_WORD_SPLITTER
    ):
        # A list of sentences in list format
        sentences_list = []
        for sent in self.text_segmented_list:
            # Try to split by default splitter
            split_arr = sent.split(sep)
            if len(split_arr) == 1:
                split_arr = sent.split(TextProcessor.DEFAULT_SPACE_SPLITTER)
                lg.Log.warning(
                    str(TextProcessor.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Could not split sentence by default separator "' + str(sep)
                    + '"\n\r   "' + str(sent)
                    + '"\n\rSplitting by space to:\n\r   ' + str(split_arr) + '.'
                )
            else:
                lg.Log.debugdebug(
                    str(TextProcessor.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Split sentence by default separator "' + str(sep)
                    + '"\n\r   "' + str(sent)
                    + '" to:\n\r   ' + str(split_arr)
                )
            # Do some separation of punctuations stuck to a word
            split_arr = self.clean_punctuations_and_convert_to_lowercase(
                sentence = split_arr
            )
            # Remove empty string ''
            split_arr = [ x for x in split_arr if x!='' ]
            # Append to return array
            sentences_list.append(split_arr)

        return sentences_list

    #
    # This is just a very basic function to do some cleaning, it is expected that
    # fundamental cleaning has already been done before coming here.
    #
    def clean_punctuations_and_convert_to_lowercase(
            self,
            sentence
    ):
        try:
            # It is easy to split words in English/German, compared to Chinese, Thai, Vietnamese, etc.
            # Don't include space separator, if you need to split by space, do it before coming here,
            # as we are only cleaning here, and may include languages like Vietnamese, so if we include
            # space here, we are splitting the word into syllables, which will be wrong.
            regex_word_split = re.compile(pattern="([!?.,？。，:;$\"')(])")
            # Split words not already split (e.g. 17. should be '17', '.')
            clean_words = [re.split(regex_word_split, word.lower()) for word in sentence]
            # Return non-empty split values, w
            # Same as:
            # for words in clean_words:
            #     for w in words:
            #         if words:
            #             if w:
            #                 w
            # All None and '' will be filtered out
            return [w for words in clean_words for w in words if words if w]
        except Exception as ex:
            errmsg =\
                str(TextProcessor.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                + ': Could not clean punctuations and convert to lowercase for sentence "'\
                + str(sentence) + '" exception message: ' + str(ex) + '.'
            lg.Log.error(errmsg)
            raise Exception(errmsg)

    #
    # Order words from highest to lowest frequency, and assign a number to each word
    # We should replace our current way of creating a dictionary using the words itself
    # in TextClusterBasic class.
    #
    def create_indexed_dictionary(
            self,
            sentences,
            dict_size = 100,
            storage_path = None
    ):
        count_words = collections.Counter()
        dict_words = {}
        opt_dict_size_initial = len(TextProcessor.OP_DICT_IDS)
        for sen in sentences:
            for word in sen:
                count_words[word] += 1

        #
        # Map words to number
        #
        dict_words[TextProcessor._PAD] = TextProcessor.PAD_ID
        dict_words[TextProcessor._GO] = TextProcessor.GO_ID
        dict_words[TextProcessor._EOS] = TextProcessor.EOS_ID
        dict_words[TextProcessor._UNK] = TextProcessor.UNK_ID

        # Add to dictionary of words starting from highest term frequency
        for idx, item in enumerate(count_words.most_common(dict_size)):
            lg.Log.info('Doing idx ' + str(idx) + ', item ' + str(item))
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
                    idx_sent.append(TextProcessor.UNK_ID)
                    not_found_counter += 1
            indexed_sentences.append(idx_sent)

        return indexed_sentences

    def extract_max_length(
            self,
            corpora
    ):
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
            pad_sentence_l1 = ([TextProcessor.PAD_ID]*padding_l1) + sentences_l1[i]

            padding_l2 = len_l2 - len(sentences_l2[i])
            pad_sentence_l2 = [TextProcessor.GO_ID] + sentences_l2[i] + [TextProcessor.EOS_ID]\
                              + ([TextProcessor.PAD_ID] * padding_l2)
            data_set.append([pad_sentence_l1, pad_sentence_l2])

        return data_set


if __name__ == '__main__':
    sent_list = [
        'Российский робот "Федор" возвратился на Землю на корабле "Союз МС-14"',
        'Корабль "Союз МС-14" с роботом "Федор" был запущен на околоземную орбиту 22 августа.'
        ]

    obj = TextProcessor(
        text_segmented_list = sent_list
    )
    sent_list_list = obj.convert_segmented_text_to_array_form()
    print(sent_list_list)