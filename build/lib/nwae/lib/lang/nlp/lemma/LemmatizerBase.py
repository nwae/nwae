# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe


#
# Our goal is not 100% correct grammer, rather more like the Porter Stemmer,
# empirical, fast, and extracts stem words which may be different from
# vocabulary.
#
class LemmatizerBase:

    # Падежные окончания (case endings) для:
    # 1. Имя сущесвительное
    CE_NOUN        = 'case_ending_noun'
    # 2. Имя прилагательное
    CE_NOUN_ADJ    = 'case_ending_noun_adjective'
    # 3. Имя числительное
    CE_NOUN_NBR    = 'case_ending_noun_number'
    # 4. Местоимение
    CE_NOUN_PFORM  = 'case_ending_pro_form'
    # 5. Глагол
    CE_VERB        = 'case_ending_verb'
    # 6. Наречие
    CE_ADVERB      = 'case_ending_adverb'
    # 7. Предлог (обычно нету)
    CE_PREPOSITION = 'case_ending_preposition'
    # 8. Союз (обычно нету)
    CE_CONJUNCTION = 'case_ending_conjunction'
    # 9. Частица (обычно нету)
    CE_GR_PARTICLE = 'case_ending_grammatical_particle'
    # 10. Междометие
    CE_INTERJ      = 'case_ending_interjection'
    # 11. Причастие
    CE_PARTICIPLE  = 'case_ending_participle'
    # 12. Деепричастие
    CE_DE_PARTICIPLE = 'case_ending_de_participle'

    ALL_CASE_ENDINGS = (
        CE_NOUN, CE_NOUN_ADJ, CE_NOUN_NBR, CE_NOUN_PFORM,
        CE_VERB, CE_ADVERB,
        CE_PREPOSITION, CE_CONJUNCTION, CE_GR_PARTICLE,
        CE_INTERJ, CE_PARTICIPLE, CE_DE_PARTICIPLE
    )

    def __init__(
            self,
            # 1. Имя сущесвительное
            noun_case_endings,
            # 5. Глагол
            verb_case_endings,
            # 2. Имя прилагательное
            noun_adj_case_endings = (),
            # 3. Имя числительное
            noun_nbr_case_endings = (),
            # 4. Местоимение
            noun_proform_case_endings = (),
            # 6. Наречие
            adverb_case_endings = (),
            # 7. Предлог (обычно нету)
            preposition_case_endings = (),
            # 8. Союз (обычно нету)
            conjunction_case_endings = (),
            # 9. Частица (обычно нету)
            grammatical_particle_case_endings = (),
            # 10. Междометие
            interjection_case_endings = (),
            # 11. Причастие
            participle_case_endings = (),
            # 12. Деепричастие
            de_participle_case_endings = (),
    ):
        self.case_endings = {
            LemmatizerBase.CE_NOUN:          noun_case_endings,
            LemmatizerBase.CE_NOUN_ADJ:      noun_adj_case_endings,
            LemmatizerBase.CE_NOUN_NBR:      noun_nbr_case_endings,
            LemmatizerBase.CE_NOUN_PFORM:    noun_proform_case_endings,
            LemmatizerBase.CE_VERB:          verb_case_endings,
            LemmatizerBase.CE_ADVERB:        adverb_case_endings,
            LemmatizerBase.CE_PREPOSITION:   preposition_case_endings,
            LemmatizerBase.CE_CONJUNCTION:   conjunction_case_endings,
            LemmatizerBase.CE_GR_PARTICLE:   grammatical_particle_case_endings,
            LemmatizerBase.CE_INTERJ:        interjection_case_endings,
            LemmatizerBase.CE_PARTICIPLE:    participle_case_endings,
            LemmatizerBase.CE_DE_PARTICIPLE: de_participle_case_endings,
        }

        # Group by length
        self.case_endings_by_len = {}
        for chast_rechi in LemmatizerBase.ALL_CASE_ENDINGS:
            self.case_endings_by_len[chast_rechi] = self.__group_case_endings_by_len(
                endings_list   = self.case_endings[chast_rechi],
                part_of_speech = chast_rechi
            )
        return

    def __group_case_endings_by_len(
            self,
            endings_list,
            part_of_speech
    ):
        endings_by_len = {}
        maxlen = 0
        for s in endings_list:
            maxlen = max(maxlen, len(s))
        # Longest to shortest
        for i in range(maxlen,0,-1):
            endings_by_len[i] = []
        # Put them in the groups
        for s in endings_list:
            endings_by_len[len(s)].append(s)

        Log.debug(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': ' + str(part_of_speech) + ' case endings by length: ' + str(endings_by_len)
        )

        return endings_by_len

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

        ces = self.case_endings_by_len[LemmatizerBase.CE_NOUN]

        for i in ces.keys():
            postfix = word[(l-i):l]
            check = postfix in ces[i]
            Log.debugdebug(
                str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Check ' + str(check) + ' for "' + str(postfix)
                + '" in ' + str(ces[i])
            )
            if check:
                return word[0:(l - i)]
        return None


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_2

    lmt = LemmatizerBase(
        noun_case_endings = ('는', '은', '가', '이', '이라면', '라면'),
        verb_case_endings = ()
    )

    words = ['나는', '했어', '너라면']

    for w in words:
        print('Word "' + str(w) + '" --> "' + str(lmt.stem(word=w)) + '"')
