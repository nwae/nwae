# --*-- coding: utf-8 --*--

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
from nwae.lib.lang.LangFeatures import LangFeatures
from nwae.lib.lang.characters.LangCharacters import LangCharacters
# from iso639 import languages
import numpy as np
import pandas as pd
from nwae.utils.Profiling import Profiling


class LangDetect:

    TEXT_LEN_THRESHOLD_TEST_ALL = 30

    # Scans the whole text
    METHOD_COMPREHENSIVE = 'comprehensive'
    # Scans part of text only, start, middle, end
    METHOD_FAST = 'fast'

    def __init__(
            self
    ):
        self.lang_features = LangFeatures()

        # Map alphabet name to unicode character set array
        self.alphabet_dict = LangCharacters.get_alphabet_charset_all()
        # Map number to alphabet, and alphabet to number
        self.alphabet_number = {}
        self.number_alphabet = {}
        idx = 0
        for alp in self.alphabet_dict.keys():
            self.alphabet_number[alp] = idx
            self.number_alphabet[idx] = alp
            idx += 1

        # Map number to language, and language to number
        self.lang_number = {}
        self.number_lang = {}
        for lang in self.lang_features.langs.keys():
            self.lang_number[lang] = self.lang_features.langs[lang][LangFeatures.C_LANG_NUMBER]
            self.number_lang[self.lang_features.langs[lang][LangFeatures.C_LANG_NUMBER]] = lang
        return

    #
    # Описание Алгоритма
    #   1. Обнарушение языков без пробела в качестве разбиение слов или слогов,
    #      это сразу определит язык.
    #   2. А потом Латинские языки, сравнить обычные слова языка с данным текстом
    #
    def detect(
            self,
            text,
            method = METHOD_COMPREHENSIVE
    ):
        return self.__detect_alphabet_type(
            text   = text,
            method = method
        )

    def __detect_non_space_sep_lang(
            self,
            method = METHOD_COMPREHENSIVE
    ):
        return

    def __detect_alphabet_type(
            self,
            text,
            method
    ):
        alp_chars = []

        len_text = len(text)
        ranges_to_check = [range(len_text)]

        if method == LangDetect.METHOD_FAST:
            if len_text > LangDetect.TEXT_LEN_THRESHOLD_TEST_ALL:
                # Percent of text length to extract excerpt
                pct = 0.1
                # Don't allow more than 20 characters in excerpt
                ml = 20
                excerpt_len = min(ml, int(pct*len_text))
                mid_index = int(len_text / 2)

                range_start = range(0, min(ml, excerpt_len), 1)
                Log.debugdebug('range start: ' + str(range_start))
                range_mid = range(mid_index - int(excerpt_len/2), mid_index + int(excerpt_len/2), 1)
                Log.debugdebug('range mid: ' + str(range_mid))
                range_end = range(len_text - excerpt_len, len_text, 1)
                Log.debugdebug('range end: ' + str(range_end))
                ranges_to_check = [range_start, range_mid, range_end]

        for rge in ranges_to_check:
            for i in rge:
                c = text[i]
                for alp in self.alphabet_dict.keys():
                    if c in self.alphabet_dict[alp]:
                        alp_chars.append(self.alphabet_number[alp])
                        # Go to next character when found alphabet type
                        break

        if len(alp_chars) == 0:
            return None

        ser = pd.Series(alp_chars)
        vals, counts = np.unique(ser, return_counts=True)
        results = dict(zip(vals, counts))

        # Map back number to alphabet name
        ret_results = {}
        for k in results.keys():
            ret_results[self.number_alphabet[k]] = results[k]

        # Sort ascending
        ret_results = sorted(ret_results.items())
        Log.debugdebug('Results: ' + str(ret_results))

        # Using pandas grouping is slower
        # df = pd.DataFrame({'lang': lang_chars, 'count': 1})
        # df_agg = df.groupby(by=['lang']).count()
        # print(df_agg)

        # Return max
        # return self.number_lang[results_sorted[0][0]]
        return ret_results


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1
    method = LangDetect.METHOD_COMPREHENSIVE

    text = [
        # Mix
        """낮선 곳에서 잠을 자다가
        Blessed are those ...,
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
            text   = s,
            method = method
        )
        timedif = Profiling.get_time_dif_secs(
            start = start_time,
            stop  = Profiling.stop(),
            decimals = 4
        ) * 1000
        print('Lang "' + str(lang) + '". Took ' + str(round(timedif, 2)) + ' ms.')
        print('')
