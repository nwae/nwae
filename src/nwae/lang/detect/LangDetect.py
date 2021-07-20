# --*-- coding: utf-8 --*--

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
from nwae.lang.LangFeatures import LangFeatures
from nwae.lang.characters.LangCharacters import LangCharacters
# from iso639 import languages
import numpy as np
import pandas as pd
from nwae.utils.Profiling import Profiling
import random
import math
from nwae.utils.StringUtils import StringUtils
from nwae.lang.nlp.lemma.Lemmatizer import Lemmatizer
from nwae.lang.preprocessing.BasicPreprocessor import BasicPreprocessor
from nwae.lang.detect.comwords.English import English
from nwae.lang.detect.comwords.Spanish import Spanish
from nwae.lang.detect.comwords.French import French
from nwae.lang.detect.comwords.Indonesian import Indonesian
from nwae.lang.detect.comwords.Vietnamese import Vietnamese


#
# Мы используем примитивный метод, проанализировать часто-используемые слова в языках
#
class LangDetect:

    SUPPORTED_LANGS = (
        LangFeatures.LANG_KO,
        LangFeatures.LANG_RU,
        LangFeatures.LANG_ZH,
        LangFeatures.LANG_TH,
        LangFeatures.LANG_EN, LangFeatures.LANG_ES, LangFeatures.LANG_FR,
        LangFeatures.LANG_VI, LangFeatures.LANG_ID,
    )

    THRESHOLD_PCT_WORDS_IN_MOST_COMMON = 0.15

    # We break text into these blocks
    TEXT_BLOCK_LEN = 10
    # Default covers 30% of blocks (e.g. if there are 10 blocks, we will randomly pick 3)
    DEFAULT_TEST_COVERAGE_PCT = 0.3
    # Not more than 5 blocks we will test to ensure speed
    DEFAULT_TEST_MAX_RANGE_BLOCKS = 5

    TEST_LATIN_BY_ORDER = [
        LangFeatures.ALPHABET_LATIN_AZ,
        # We also detect these special Vietnamese characters, to increase accuracy for Vietnamese
        LangFeatures.ALPHABET_LATIN_VI,
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
        self.common_words = {}
        self.common_words[LangFeatures.LANG_EN] = English()
        self.common_words[LangFeatures.LANG_ES] = Spanish()
        self.common_words[LangFeatures.LANG_FR] = French()
        self.common_words[LangFeatures.LANG_ID] = Indonesian()
        self.common_words[LangFeatures.LANG_VI] = Vietnamese()

        # Load stemmers
        self.word_stemmer = {}
        for lang in LangDetect.SUPPORTED_LANGS:
            lang_have_verb_conj = self.lang_features.have_verb_conjugation(
                lang = lang
            )
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Lang "' + str(lang) + '" verb conjugation = ' + str(lang_have_verb_conj) + '.'
            )
            self.word_stemmer[lang] = None
            if lang_have_verb_conj:
                try:
                    self.word_stemmer[lang] = Lemmatizer(
                        lang = lang
                    )
                    Log.important(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Lang "' + str(lang) + '" stemmer/lemmatizer initialized successfully.'
                    )
                except Exception as ex_stemmer:
                    errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                             + ': Lang "' + str(lang) + ' stemmer/lemmatizer failed to initialize: ' \
                             + str(ex_stemmer) + '.'
                    Log.warning(errmsg)

        return

    #
    # Only for languages with space as word separator
    # Or in the case of Vietnamese, it will split by syllables
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

        detected_top_alps = list(alps.keys())
        Log.debugdebug('Top alphabets = ' + str(detected_top_alps))

        loop_top_x = 2
        loop_counter = 0
        while loop_counter < loop_top_x:
            if len(detected_top_alps) > loop_counter:
                loop_alp = detected_top_alps[loop_counter]
            else:
                break
            loop_counter += 1
            Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Loop ' + str(loop_counter) + ' alphabet "' + str(loop_alp) + '"'
            )

            # Get possible languages for this alphabet
            possible_langs_for_alphabet = self.lang_features.get_languages_for_alphabet_type(
                alphabet = loop_alp
            )
            Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Possible languages for alphabet "' + str(loop_alp)
                + '": ' + str(possible_langs_for_alphabet)
            )

            # No dispute when only 1 possible language for given alphabet
            if len(possible_langs_for_alphabet) == 1:
                Log.debugdebug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Only 1 possible language for alphabet: ' + str(possible_langs_for_alphabet)
                )
                return possible_langs_for_alphabet

            det_langs = []
            #
            # Alphabet belongs to Hangul family
            #
            if loop_alp in LangDetect.TEST_HANGUL_BY_ORDER:
                det_langs = self.detect_lang_from_hangul(text=text)
            elif loop_alp in LangDetect.TEST_CYRILLIC_BY_ORDER:
                det_langs = self.detect_lang_from_cyrillic(text=text)
            elif loop_alp in LangDetect.TEST_THAI_BY_ORDER:
                det_langs = self.detect_lang_from_thai_alphabet(text=text)
            #
            # Alphabet belongs to the Latin family
            #
            elif loop_alp in LangDetect.TEST_LATIN_BY_ORDER:
                # Almost all Latin Family languages will have LatinAZ come out tops first
                if loop_alp == LangFeatures.ALPHABET_LATIN_AZ:
                    det_langs = self.detect_lang_from_latin_az(
                        text = text,
                        detected_alphabets_present = detected_top_alps
                    )

                if not det_langs:
                    # We extend the search to all Latin if can't find anything
                    det_langs = self.detect_lang_from_latin(
                        text = text
                    )
            elif loop_alp == LangFeatures.ALPHABET_CJK:
                det_langs = self.detect_lang_from_cjk(text=text)

            # If have result, return the result and quit the loop
            if det_langs:
                return det_langs

        return []

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

    def detect_lang_from_cjk(
            self,
            text
    ):
        # TODO Differentiate Chinese (simplified, traditional, etc.), Japanese, ..
        return [LangFeatures.LANG_ZH]

    def detect_lang_from_thai_alphabet(
            self,
            text
    ):
        # TODO Handle the different dialects
        return [LangFeatures.LANG_TH]

    def detect_lang_from_latin_az(
            self,
            text,
            detected_alphabets_present
    ):
        sent = self.__segment_words(text=text)

        lang_codes = []
        lang_pct = []

        for lang in (
                LangFeatures.LANG_EN, LangFeatures.LANG_ES, LangFeatures.LANG_FR,
                LangFeatures.LANG_VI, LangFeatures.LANG_ID,
        ):
            lang_codes.append(lang)
            max_word_n_tuple = 1
            if lang == LangFeatures.LANG_VI:
                max_word_n_tuple = 2
            lang_pct.append(self.common_words[lang].get_pct_intersection_with_common_words(
                word_list = sent,
                max_word_n_tuple = max_word_n_tuple
            ))

        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': For sentence ' + str(sent)
            + ' lang codes/pct: ' + str(pd.DataFrame({'code': lang_codes, 'pct': lang_pct}).values)
        )

        if lang_codes:
            idx_max = np.argmax(lang_pct)
            idx_max = int(idx_max)

            if lang_pct[idx_max] > LangDetect.THRESHOLD_PCT_WORDS_IN_MOST_COMMON:
                return [lang_codes[idx_max]]
            else:
                # Check word stems
                for lang in lang_codes:
                    if self.word_stemmer[lang] is None:
                        continue
                    sent_stems = []
                    for w in sent:
                        w_stem = self.word_stemmer[lang].stem(word=w)
                        sent_stems.append(w_stem)
                    if sent_stems == sent:
                        continue
                    Log.debug(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': For lang "' + str(lang)
                        + '", trying stemmed words: ' + str(sent_stems)
                    )
                    pct_int = self.common_words[lang].get_pct_intersection_with_common_words(
                        word_list = sent_stems
                    )
                    if pct_int > LangDetect.THRESHOLD_PCT_WORDS_IN_MOST_COMMON:
                        return [lang]

                # Although French, Spanish could also have these characters, we favor Vietnamese
                if LangFeatures.ALPHABET_LATIN_VI in detected_alphabets_present:
                    return [LangFeatures.LANG_VI]

        return []

    def detect_lang_from_latin(
            self,
            text
    ):
        sent = self.__segment_words(text=text)

        lang_codes = []
        lang_pct = []

        if lang_codes:
            idx_max = np.argmax(lang_pct)
            idx_max = int(idx_max)

            if lang_pct[idx_max] > LangDetect.THRESHOLD_PCT_WORDS_IN_MOST_COMMON:
                return [lang_codes[idx_max]]

        return []

    #
    # Returns tuple of start/end (not inclusive)
    # E.g. [(0,10), (10,20), ..]
    #
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
            # range_blocks.append(range(i, end_range, 1))
            range_blocks.append((i,end_range))
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
        how_many_range_to_check = max(1, min(
            math.ceil(test_coverage_pct * n_range),
            math.ceil(max_test_coverage_len / LangDetect.TEXT_BLOCK_LEN)
        ))

        # Randomly pick the ranges
        random_ranges_index = random.sample(range(n_range), how_many_range_to_check)
        random_ranges_index = sorted(random_ranges_index)
        total_len = 0
        for rg in random_ranges_index:
            start, end = range_blocks[rg]
            total_len += (end - start + 1)

        # Means we got the last truncated block
        if total_len < LangDetect.TEXT_BLOCK_LEN:
            if 0 not in random_ranges_index:
                random_ranges_index = [0] + random_ranges_index

        text_excerps = []
        for rg in random_ranges_index:
            start, end = range_blocks[rg]
            text_excerps.append(text[start:end])

        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Random ranges index: ' + str(random_ranges_index) + ' or: ' + str(text_excerps)
        )

        for rge_idx in random_ranges_index:
            #for i in range_blocks[rge_idx]:
            start, end = range_blocks[rge_idx]
            for i in range(start, end, 1):
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
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Alphabet detection: ' + str(results_list)
        )

        # Reverse back the mapping
        return {kv[1]:kv[0] for kv in results_list}


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_2

    text = [
        # Mix
        ("""낮선 곳에서 잠을 자다가 Blessed 中国 are 韩国 those 俄罗斯.., 唧唧复唧唧, 등짝을 훑고 지나가는 지진의 진동""",
         [None]),
        ("""Blessed are those who find wisdom, those who gain understanding""",
         [None]),
        ('Incrustado en las laderas de unas colinas volcánicas',
         [None]),
        ('นี่คือ minions',
         [None]),
        ('存款10次http://abc.com',
         [None]),
    ]

    ld = LangDetect()
    for s, expected_langs in text:
        start_time = Profiling.start()
        print('Text: ' + str(s))
        lang = ld.detect(
            text   = s,
            #test_coverage_pct = 0.5,
            #max_test_coverage_len = 30
        )
        timedif = Profiling.get_time_dif_secs(
            start = start_time,
            stop  = Profiling.stop(),
            decimals = 4
        ) * 1000
        print('Lang ' + str(lang) + '. Took ' + str(round(timedif, 2)) + ' ms.')
        print('')
