# --*-- coding: utf-8 --*--

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
from nwae.utils.StringUtils import StringUtils
import re


class CommonWords:

    def __init__(
            self
    ):
        self.raw_words = None
        self.common_words = None
        return

    def get_common_words(
            self
    ):
        return self.common_words

    def process_common_words(
            self
    ):
        try:
            self.raw_words = StringUtils.trim(self.raw_words)
            self.raw_words = re.sub(
                pattern = '[\xa0\t\n\r]',
                repl    = ' ',
                string  = self.raw_words
            )
            self.raw_words = self.raw_words.lower()
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Error processing raw words. Exception: ' + str(ex)
            Log.error(errmsg)
            raise Exception(errmsg)

        try:
            self.common_words = self.raw_words.split(' ')
            self.common_words = sorted(self.common_words)
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Error processing common words. Exception: ' + str(ex)
            Log.error(errmsg)
            raise Exception(errmsg)

        return