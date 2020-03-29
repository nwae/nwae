# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe


#
# Our goal is not 100% correct grammer, rather more like the Porter Stemmer,
# empirical, fast, and extracts stem words which may be different from
# vocabulary.
#
class LemmatizerBase:

    def __init__(
            self,
            # E.g.
            noun_endings,
            verb_endings
    ):
        self.noun_endings = noun_endings
        self.verb_endings = verb_endings

        # Group by length
        self.noun_endings_dict = self.__group_endings_by_len(
            endings_list = self.noun_endings
        )
        Log.debug(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Noun endings dict: ' + str(self.noun_endings_dict)
        )

        self.verb_endings_dict = self.__group_endings_by_len(
            endings_list = self.verb_endings
        )
        Log.debug(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Verb endings dict: ' + str(self.verb_endings_dict)
        )
        return

    def __group_endings_by_len(
            self,
            endings_list
    ):
        endings_dict = {}
        maxlen = 0
        for s in endings_list:
            maxlen = max(maxlen, len(s))
        # Longest to shortest
        for i in range(maxlen,0,-1):
            endings_dict[i] = []
        # Put them in the groups
        for s in endings_list:
            endings_dict[len(s)].append(s)

        return endings_dict

    def stem(
            self,
            word
    ):
        l = len(word)
        if l <= 1:
            return word

        s_noun = self.process_noun(
            word = word
        )
        if s_noun is not None:
            return s_noun
        else:
            return word

    def process_noun(
            self,
            word
    ):
        l = len(word)

        for i in self.noun_endings_dict.keys():
            postfix = word[(l-i):l]
            check = postfix in self.noun_endings_dict[i]
            Log.debugdebug(
                str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Check ' + str(check) + ' for "' + str(postfix)
                + '" in ' + str(self.noun_endings_dict[i])
            )
            if check:
                return word[0:(l - i)]
        return None


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_2

    lmt = LemmatizerBase(
        noun_endings = ('는', '은', '가', '이', '이라면', '라면'),
        verb_endings = ()
    )

    words = ['나는', '했어', '너라면']

    for w in words:
        print('Word "' + str(w) + '" --> "' + str(lmt.stem(word=w)) + '"')
