# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
from nwae.lang.nlp.lemma.LemmatizerBase import LemmatizerBase


#
# Our goal is not 100% correct grammer, rather more like the Porter Stemmer,
# empirical, fast, and extracts stem words which may be different from
# vocabulary.
#
class LemmatizerKorean(LemmatizerBase):

    # Часть речи
    NOUN_PARTICLES = (
        '이라면',
        '라면',
        '는', '은', '가', '이',
        '를', '을'
    )

    def __init__(
            self,
            noun_case_endings = NOUN_PARTICLES,
            verb_case_endings = ()
    ):
        super().__init__(
            noun_case_endings = noun_case_endings,
            verb_case_endings = verb_case_endings
        )
        try:
            # Разбить Хангул (한글) слоги на буквы (자모)
            # https://github.com/JDongian/python-jamo, https://python-jamo.readthedocs.io/en/latest/
            from jamo import h2j, j2hcj
        except Exception as ex:
            errmsg = str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Error importing jamo library: ' + str(ex)
            Log.warning(errmsg)
            raise Exception(errmsg)
        return


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_2

    lmt = LemmatizerKorean()

    words = ['나는', '했어', '너라면']

    for w in words:
        print('Word "' + str(w) + '" --> "' + str(lmt.stem(word=w)) + '"')
