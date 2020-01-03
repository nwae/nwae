# -*- coding: utf-8 -*-

import nwae.utils.Log as log
from inspect import currentframe, getframeinfo
import nwae.lib.lang.LangFeatures as lf
import re


#
# When model updates, this also need to update. So be careful.
#
class BasicPreprocessor:

    #
    # Our own default word delimiter
    #
    DEFAULT_WORD_SPLITTER = '--||--'
    DEFAULT_SPACE_SPLITTER = ' '

    # Sentence padding if shorter than min length
    W_PAD = '_pad'
    # Start of sentence
    W_GO  = '_go'
    # End of sentence
    W_EOS = '_eos'
    # Unknown word
    W_UNK = '_unk'
    # Other common symbols
    # Number
    W_NUM = '_num'
    # Username or any word with mix of character/numbers/etc
    W_USERNAME = '_username'
    W_USERNAME_NONWORD = '_username_nonword'
    # URL
    W_URI = '_uri'

    ALL_SPECIAL_SYMBOLS = (
        W_PAD, W_GO, W_EOS, W_UNK,
        W_NUM, W_USERNAME, W_USERNAME_NONWORD, W_URI
    )

    _START_VOCAB = [W_PAD, W_GO, W_EOS, W_UNK]
    PAD_ID = 0
    GO_ID  = 1
    EOS_ID = 2
    UNK_ID = 3
    OP_DICT_IDS = [PAD_ID, GO_ID, EOS_ID, UNK_ID]

    #
    # The difference between this and the function get_word_separator_type() in
    # LangFeatures class is that this is OUR OWN encoding, whereas the other one
    # is the natural human written word separator.
    #
    @staticmethod
    def get_word_separator(
            lang
    ):
        word_separator = BasicPreprocessor.DEFAULT_SPACE_SPLITTER
        if lang in (lf.LangFeatures.LANG_VI, lf.LangFeatures.LANG_VN):
            word_separator = BasicPreprocessor.DEFAULT_WORD_SPLITTER
        return word_separator

    #
    # This is just a very basic function to do some cleaning, it is expected that
    # fundamental cleaning has already been done before coming here.
    #
    @staticmethod
    def clean_punctuations(
            # list of words
            sentence,
            punctuations_pattern = '([!?.,？。，:;$"\')(])',
            convert_to_lower_case = True
    ):
        try:
            # Don't include space separator, if you need to split by space, do it before coming here,
            # as we are only cleaning here, and may include languages like Vietnamese, so if we include
            # space here, we are splitting the word into syllables, which will be wrong.
            regex_word_split = re.compile(pattern=punctuations_pattern)
            # Split words not already split (e.g. 17. should be '17', '.')
            # re.split() will add a redundant '' at the end, so we have to remove later.
            if convert_to_lower_case:
                clean_words = [re.split(regex_word_split, word.lower()) for word in sentence]
            else:
                clean_words = [re.split(regex_word_split, word) for word in sentence]
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
                str(BasicPreprocessor.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                + ': Could not clean punctuations and convert to lowercase for sentence "'\
                + str(sentence) + '" exception message: ' + str(ex) + '.'
            log.Log.error(errmsg)
            return None



