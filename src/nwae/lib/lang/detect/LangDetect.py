# --*-- coding: utf-8 --*--

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
from nwae.lib.lang.LangFeatures import LangFeatures
from nwae.lib.lang.characters.LangCharacters import LangCharacters
# from iso639 import languages
import numpy as np
import pandas as pd
from nwae.utils.Profiling import Profiling
import random
from nwae.utils.StringUtils import StringUtils
import nwae.utils.UnitTest as ut
from nwae.lib.lang.preprocessing.BasicPreprocessor import BasicPreprocessor
from nwae.lib.lang.detect.comwords.English import English
from nwae.lib.lang.detect.comwords.Indonesian import Indonesian
from nwae.lib.lang.detect.comwords.Vietnamese import Vietnamese


class LangDetect:

    # We break text into these blocks
    TEXT_BLOCK_LEN = 10
    # Default covers 30% of blocks (e.g. if there are 10 blocks, we will randomly pick 3)
    DEFAULT_TEST_COVERAGE_PCT = 0.3
    # Not more than 5 blocks we will test to ensure speed
    DEFAULT_TEST_MAX_RANGE_BLOCKS = 5

    TEST_LATIN_BY_ORDER = [
        LangFeatures.ALPHABET_LATIN_VI,
        LangFeatures.ALPHABET_LATIN_AZ,
        # This Latin that covers all must be last to test
        LangFeatures.ALPHABET_LATIN
    ]
    TEST_CYRILLIC_BY_ORDER = [
        LangFeatures.ALPHABET_CYRILLIC
    ]
    TEST_HANGUL_BY_ORDER = [
        LangFeatures.ALPHABET_HANGUL
    ]
    TEST_CJK_BY_ORDER = [
        LangFeatures.ALPHABET_CJK
    ]
    TEST_THAI_BY_ORDER = [
        LangFeatures.ALPHABET_THAI
    ]

    TESTS_BY_ORDER = TEST_LATIN_BY_ORDER \
            + TEST_CYRILLIC_BY_ORDER \
            + TEST_HANGUL_BY_ORDER \
            + TEST_CJK_BY_ORDER \
            + TEST_THAI_BY_ORDER

    def __init__(
            self
    ):
        self.lang_features = LangFeatures()

        # Map alphabet name to unicode character set array
        self.alphabet_dict = {}
        for alp in LangDetect.TESTS_BY_ORDER:
            self.alphabet_dict[alp] = LangCharacters.get_alphabet_charset(
                alphabet = alp
            )
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Alphabets used: ' + str(self.alphabet_dict.keys())
        )

        self.langs_with_no_word_sep = self.lang_features.get_languages_with_no_word_separator()
        Log.debugdebug('Langs with no word sep: ' + str(self.langs_with_no_word_sep))

        # Load common words
        self.cw_english = English()
        self.cw_indonesian = Indonesian()
        self.cw_vietnamese = Vietnamese()
        return

    #
    # Only for languages with space as word separator
    #
    def __segment_words(
            self,
            text
    ):
        sent = StringUtils.trim(text)
        sent = sent.lower()
        sent = sent.split(' ')
        # Split out punctuations
        sent = BasicPreprocessor.clean_punctuations(
            sentence = sent
        )
        return sent

    #
    # Описание Алгоритма
    #   1. Обнарушение Алфавитов
    #      i) Если приналежит языкам без пробела в качестве разбиение слов или слогов,
    #         это сразу определит тот язык.
    #      ii) Потом Латинские языки, сравнить обычные слова языка с данным текстом
    #
    def detect(
            self,
            text,
            test_coverage_pct = DEFAULT_TEST_COVERAGE_PCT,
            max_test_coverage_len = DEFAULT_TEST_MAX_RANGE_BLOCKS * TEXT_BLOCK_LEN
    ):
        text = str(text)

        if len(text) == 0:
            return None

        alps = self.__detect_alphabet_type(
            text   = text,
            test_coverage_pct = test_coverage_pct,
            max_test_coverage_len = max_test_coverage_len
        )

        # Either None type or empty dict
        if not alps:
            return None

        top_alps = list(alps.keys())
        top_alp = top_alps[0]
        Log.debugdebug('Top alphabet = ' + str(top_alp))

        # Get possible languages for this alphabet
        possible_langs_for_alphabet = self.lang_features.get_languages_for_alphabet_type(alphabet=top_alp)
        Log.debugdebug('Possible languages for alphabet "' + str(top_alp) + '": ' + str(possible_langs_for_alphabet))

        # No dispute when only 1 possible language for given alphabet
        if len(possible_langs_for_alphabet) == 1:
            return possible_langs_for_alphabet

        #
        # Alphabet belongs to Hangul family
        #
        if top_alp in LangDetect.TEST_HANGUL_BY_ORDER:
            return self.detect_lang_from_hangul(text=text)
        elif top_alp in LangDetect.TEST_CYRILLIC_BY_ORDER:
            return self.detect_lang_from_cyrillic(text=text)
        elif top_alp in LangDetect.TEST_THAI_BY_ORDER:
            return self.detect_lang_from_thai_alphabet(text=text)
        #
        # Alphabet belongs to the Latin family
        #
        elif top_alp in LangDetect.TEST_LATIN_BY_ORDER:
            # No extended Latin
            if top_alp in (LangFeatures.ALPHABET_LATIN_AZ, LangFeatures.ALPHABET_LATIN_VI):
                #
                # Check Vietnamese presence, does not have to be the top alphabet,
                # as it is mixed with basic Latin which will usually dominate
                #
                if LangFeatures.ALPHABET_LATIN_VI in top_alps:
                    return [LangFeatures.LANG_VN]

                return self.detect_lang_from_latin_az(text=text)
            else:
                # TODO Support checks for extended Latin
                return None
        elif top_alp == LangFeatures.ALPHABET_CJK:
            # TODO Differentiate Chinese (simplified, traditional, etc.), Japanese, ..
            return [LangFeatures.LANG_CN]

        return None

    def detect_lang_from_hangul(
            self,
            text
    ):
        # TODO Handle different dialects
        return [LangFeatures.LANG_KO]

    def detect_lang_from_cyrillic(
            self,
            text
    ):
        # TODO Handle the whole cyrillic family
        return [LangFeatures.LANG_RU]

    def detect_lang_from_thai_alphabet(
            self,
            text
    ):
        # TODO Handle the different dialects
        return [LangFeatures.LANG_TH]

    def detect_lang_from_latin_az(
            self,
            text
    ):
        sent = self.__segment_words(text=text)

        # TODO Handle also the European languages, Malay, etc.
        if self.cw_vietnamese.test_lang(word_list=sent):
            return [LangFeatures.LANG_VN]
        elif self.cw_english.test_lang(word_list=sent):
            return [LangFeatures.LANG_EN]
        elif self.cw_indonesian.test_lang(word_list=sent):
            return [LangFeatures.LANG_IN]
        else:
            return None

    def __get_text_range_blocks(
            self,
            text
    ):
        # Break into ranges
        range_blocks = []
        i = 0
        len_text = len(text)
        while i < len_text:
            end_range = min(len_text, i+10)
            range_blocks.append(range(i, end_range, 1))
            i = i + 10
        return range_blocks

    def __detect_alphabet_type(
            self,
            text,
            # default coverage
            test_coverage_pct,
            max_test_coverage_len
    ):
        alp_chars = []

        # Return the range blocks of the text
        range_blocks = self.__get_text_range_blocks(text = text)
        n_range = len(range_blocks)
        how_many_range_to_check = min(
            int(test_coverage_pct * n_range),
            int(max_test_coverage_len / LangDetect.TEXT_BLOCK_LEN)
        )

        # Randomly pick the ranges
        random_ranges_index = random.sample(range(n_range), how_many_range_to_check)
        random_ranges_index = sorted(random_ranges_index)
        Log.debugdebug('Random ranges: ' + str(random_ranges_index))

        for rge_idx in random_ranges_index:
            for i in range_blocks[rge_idx]:
                c = text[i]
                # Test Latin first, but from the smaller subsets first
                for alp in LangDetect.TESTS_BY_ORDER:
                    if c in self.alphabet_dict[alp]:
                        alp_chars.append(alp)
                        # Go to next character when found alphabet type
                        break

        if len(alp_chars) == 0:
            return None

        ser = pd.Series(alp_chars)
        vals, counts = np.unique(ser, return_counts=True)
        # We must mup count as key, so that when we sort the paired items later,
        # python will sort by the first index which is the count
        results = dict(zip(counts, vals))

        # Sort ascending
        results_list = sorted(results.items(), reverse=True)
        Log.debug('Results: ' + str(results_list))

        # Reverse back the mapping
        return {kv[1]:kv[0] for kv in results_list}


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1

    text = [
        # Mix
        ("""낮선 곳에서 잠을 자다가 Blessed 中国 are 韩国 those 俄罗斯.., 唧唧复唧唧, 등짝을 훑고 지나가는 지진의 진동""",
         [None]),
        ("""Blessed are those who find wisdom, those who gain understanding""",
         [None]),
        ('Sejumlah pakar kesehatan menyarankan pemerintah Indonesia mempertimbangkan kemungkinan '
         'pembatasan wilayah yang lebih ketat alias lockdown, demi mengantisipasi pertambahan jumlah kasus.',
         [None])
    ]

    ld = LangDetect()
    for s, expected_langs in text:
        start_time = Profiling.start()
        print('Text: ' + str(s))
        lang = ld.detect(
            text   = s,
            test_coverage_pct = 0.5,
            max_test_coverage_len = 30
        )
        timedif = Profiling.get_time_dif_secs(
            start = start_time,
            stop  = Profiling.stop(),
            decimals = 4
        ) * 1000
        print('Lang ' + str(lang) + '. Took ' + str(round(timedif, 2)) + ' ms.')
        print('')
