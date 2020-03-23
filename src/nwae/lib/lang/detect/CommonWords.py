# --*-- coding: utf-8 --*--

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
from nwae.utils.StringUtils import StringUtils
import re


class CommonWords:

    def __init__(
            self,
            lang
    ):
        self.lang = lang
        self.raw_words = None
        self.common_words = None
        return

    #
    # Minimum intersection with common words given any random English sentence
    #
    def get_min_threshold_intersection_pct(
            self
    ):
        raise Exception(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Must be implemented by child class!'
        )

    def test_lang(
            self,
            word_list
    ):
        lang_intersection = set(word_list).intersection(self.get_common_words())
        pct_intersection = len(lang_intersection) / len(set(word_list))
        Log.debug(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': "' + str(self.lang) + '" intersection = ' + str(pct_intersection)
        )
        if pct_intersection > self.get_min_threshold_intersection_pct():
            return True
        else:
            return False

    def get_common_words(
            self
    ):
        return self.common_words

    def process_common_words(
            self,
            word_split_token = ' '
    ):
        try:
            self.raw_words = StringUtils.trim(self.raw_words)
            self.raw_words = re.sub(
                pattern = '[\xa0\t\n\r]',
                repl    = word_split_token,
                string  = self.raw_words
            )
            self.raw_words = self.raw_words.lower()
        except Exception as ex:
            errmsg = str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Error processing raw words. Exception: ' + str(ex)
            Log.error(errmsg)
            raise Exception(errmsg)

        try:
            self.common_words = self.raw_words.split(word_split_token)
            # Remove None, '', {}, etc.
            self.common_words = [w for w in self.common_words if w]
            self.common_words = sorted(set(self.common_words))
            Log.info(
                str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': Loaded ' + str(len(self.common_words)) + ' common words of lang "' + str(self.lang) + '".'
            )
        except Exception as ex:
            errmsg = str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Error processing common words. Exception: ' + str(ex)
            Log.error(errmsg)
            raise Exception(errmsg)

        return