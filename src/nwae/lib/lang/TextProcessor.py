# -*- coding: utf-8 -*-

import re
import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo
import nwae.lib.lang.characters.LangCharacters as langchar


#
# General processing of text data to usable math forms for NLP processing
#
class TextProcessor:

    DEFAULT_WORD_SPLITTER = '--||--'
    DEFAULT_SPACE_SPLITTER = ' '

    def __init__(
            self
    ):
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
            text_segmented_list,
            sep = DEFAULT_WORD_SPLITTER
    ):
        list_list_text = []
        for sent in text_segmented_list:
            # Try to split using default splitter
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
            # Remove empty string ''
            split_arr = [ x for x in split_arr if x!='' ]
            # Append to return array
            list_list_text.append(split_arr)

        return list_list_text

    def clean_sentence(
            self,
            sentence
    ):
        # It is easy to split words in English/German, compared to Chinese, Thai, Vietnamese, etc.
        regex_word_split = re.compile(pattern="([!?.,:;$\"')( ])")
        # Split words not already split (e.g. 17. should be '17', '.')
        clean_words = [re.split(regex_word_split, word.lower()) for word in sentence]
        # Return non-empty split values, w
        # Same as:
        # for words in clean_words:
        #     for w in words:
        #         if words:
        #             if w:
        #                 w
        return [w for words in clean_words for w in words if words if w]


if __name__ == '__main__':
    sent_list = [
        'Российский робот "Федор" возвратился на Землю на корабле "Союз МС-14"',
        'Корабль "Союз МС-14" с роботом "Федор" был запущен на околоземную орбиту 22 августа.'
        ]

    obj = TextProcessor()
    sent_list_list = obj.convert_segmented_text_to_array_form(
        text_segmented_list = sent_list
    )
    print(sent_list_list)