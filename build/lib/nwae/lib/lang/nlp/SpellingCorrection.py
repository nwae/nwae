# -*- coding: utf-8 -*-

import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo
import nwae.utils.Profiling as prf
from nltk.metrics.distance import edit_distance


#
# For languages like English where words are already nicely split by space, we can use
# off-the-shelf edit/Levenshtein distance, etc.
# For languages like Thai sadly, it is a completely different story and none of the
# above methods can be used.
#
# Training Data: Fixed Vector Sequences
#    111222333
#    111333222
#    123456789
#
# Consider a given sequence 114222333, 112233
#
# Algorithm:
#
class SpellingCorrection:

    def __init__(
            self,
            reference_sequences
    ):
        self.ref_seqs = reference_sequences
        self.ref_seqs_unicode = []
        # Convert sequence to Unicode
        for seq in self.ref_seqs:
            self.ref_seqs_unicode.append([ord(x) for x in seq])

        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Converted to unicode sequence ' + str(self.ref_seqs_unicode)
        )
        return

    def get_closest_sequence(
            self,
            sentence
    ):
        start_profile_time = prf.Profiling.start()

        sent_unicode = [ord(x) for x in sentence]
        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Sentence unicode: ' + str(sent_unicode)
        )

        #
        # First we filter those reference sequences with enough intersections
        #
        intersections = []
        distances = []
        for sent in self.ref_seqs:
            ed = edit_distance(
                s1 = sentence,
                s2 = sent
            )
            distances.append(ed)

        for seq in self.ref_seqs_unicode:
            intersections.append(len(set(sent_unicode).intersection(seq)))

        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Distances: ' + str(distances)
            + ', Intersections: ' + str(intersections)
        )

        lg.Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Get closest sequence took '
            + str(prf.Profiling.get_time_dif_str(start=start_profile_time, stop=prf.Profiling.stop()))
            + ' secs.'
        )

        return 0


if __name__ == '__main__':
    td = [
        'ปิดปรับปรุงเว็บไซต์ฉุกเฉิน',
        'ลืมรหัสผ่าน',
        'การสมัครกับเว็บไซต์',
        'คูปองอะไรค่ะ'
    ]

    sentences = [
        'ปรับปรุงเวบไฃต',
        'คู่ปองอะไรค่ะ'
    ]

    obj = SpellingCorrection(
        reference_sequences = td
    )

    obj.get_closest_sequence(
        sentence = sentences[0]
    )

    exit(0)
