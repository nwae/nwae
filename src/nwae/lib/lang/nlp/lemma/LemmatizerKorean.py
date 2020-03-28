# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe


class LemmatizerKorean:

    END_NOUN_PARTICLE_SUBJECT = ('는', '은', '가', '이')

    def __init__(
            self
    ):
        return

    def stem(
            self,
            word
    ):
        l = len(word)
        if l <= 1:
            return word
        elif word[l-1] in LemmatizerKorean.END_NOUN_PARTICLE_SUBJECT:
            return self.process_noun_particle_ending(
                word = word
            )
        else:
            return word

    def process_noun_particle_ending(
            self,
            word
    ):
        return word[0:(len(word) - 1)]


if __name__ == '__main__':
    lmt = LemmatizerKorean()

    words = ['나는', '했어']

    for w in words:
        print('Word "' + str(w) + '" --> "' + str(lmt.stem(word=w)) + '"')
