# -*- coding: utf-8 -*-

import re
import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo


#
# General processing of text data to usable math forms for NLP processing
#
class TextProcessor:

    #
    # Our own default word delimiter
    #
    DEFAULT_WORD_SPLITTER = '--||--'
    DEFAULT_SPACE_SPLITTER = ' '

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
                split_arr = sent.split(' ')
                lg.Log.warning(
                    str(TextProcessor.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Could not split sentence by default separator "' + str(sep)
                    + '"\n\r   "' + str(sent)
                    + '"\n\rSplitting by space to:\n\r   ' + str(split_arr) + '.'
                )
            else:
                lg.Log.info(
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
        # It is easy to split words in English/German, compared to Chinese, Thai, Vietnamese, etc.
        regex_word_split = re.compile(pattern="([!?.,？。，:;$\"')( ])")
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