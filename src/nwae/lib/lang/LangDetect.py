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


class LangDetect:

    # We break text into these blocks
    TEXT_BLOCK_LEN = 10
    # Default covers 30% of blocks (e.g. if there are 10 blocks, we will randomly pick 3)
    DEFAULT_TEST_COVERAGE_PCT = 0.3
    # Not more than 5 blocks we will test to ensure speed
    DEFAULT_TEST_MAX_RANGE_BLOCKS = 5

    def __init__(
            self
    ):
        self.lang_features = LangFeatures()

        # Map alphabet name to unicode character set array
        self.alphabet_dict = LangCharacters.get_alphabet_charset_all()

        # # Map number to language, and language to number
        # self.lang_number = {}
        # self.number_lang = {}
        # for lang in self.lang_features.langs.keys():
        #     self.lang_number[lang] = self.lang_features.langs[lang][LangFeatures.C_LANG_NUMBER]
        #     self.number_lang[self.lang_features.langs[lang][LangFeatures.C_LANG_NUMBER]] = lang
        return

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
        if len(text) == 0:
            return None

        alps = self.__detect_alphabet_type(
            text   = text,
            test_coverage_pct = test_coverage_pct,
            max_test_coverage_len = max_test_coverage_len
        )

        return alps

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
        print(random_ranges_index)

        for rge_idx in random_ranges_index:
            for i in range_blocks[rge_idx]:
                c = text[i]
                for alp in self.alphabet_dict.keys():
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
        """낮선 곳에서 잠을 자다가
        Blessed 中国 are 韩国 those 俄罗斯..,
        唧唧复唧唧,
        등짝을 훑고 지나가는 지진의 진동""",
        # ru
        """Умом Россию не понять,
        Аршином общим не измерить:
        У ней особенная стать —В
        Россию можно только верить.""",
        # ko
        """낮선 곳에서 잠을 자다가
        갑자기 들리는 흐르는 물소리
        등짝을 훑고 지나가는 지진의 진동""",
        # en
        """Blessed are those who find wisdom, those who gain understanding,
        14 for she is more profitable than silver and yields better returns than gold.
        15 She is more precious than rubies; nothing you desire can compare with her.
        16 Long life is in her right hand; in her left hand are riches and honor.
        17 Her ways are pleasant ways, and all her paths are peace.
        18 She is a tree of life to those who take hold of her; those who hold her fast will be blessed.""",
        # zh-cn
        """木兰辞 唧唧复唧唧，木兰当户织。……雄兔脚扑朔，雌兔眼迷离，双兔傍地走，安能辨我是雄雌？""",
    ]

    ld = LangDetect()
    for s in text:
        start_time = Profiling.start()
        print('Text: ' + str(s))
        lang = ld.detect(
            text   = s
        )
        timedif = Profiling.get_time_dif_secs(
            start = start_time,
            stop  = Profiling.stop(),
            decimals = 4
        ) * 1000
        print('Lang "' + str(lang) + '". Took ' + str(round(timedif, 2)) + ' ms.')
        print('')
