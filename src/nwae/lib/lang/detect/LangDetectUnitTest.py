# --*-- coding: utf-8 --*--

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
from nwae.lib.lang.LangFeatures import LangFeatures
from nwae.lib.lang.detect.LangDetect import LangDetect
import nwae.utils.UnitTest as ut
from nwae.utils.Profiling import Profiling


class LangDetectUnitTest:

    TEST_TEXT_LANG = [
        #
        # Cyrillic
        #
        ('Умом Россию не понять, Аршином общим не измерить: У ней особенная стать —В Россию можно только верить.',
         [LangFeatures.LANG_RU]),
        ('В сужности ведь для интеллигентного человека дурно говорить должно бы считаться таким же неприличием,',
         [LangFeatures.LANG_RU]),
        ('как не уметь читать и писать,',
         [LangFeatures.LANG_RU]),
        ('и в деле образования и воспитания обучение красноречию следовало бы считать неизбежним.',
         [LangFeatures.LANG_RU]),
        #
        # Hangul
        #
        ('낮선 곳에서 잠을 자다가, 갑자기 들리는 흐르는 물소리, 등짝을 훑고 지나가는 지진의 진동',
         [LangFeatures.LANG_KO]),
        ('야!',
         [LangFeatures.LANG_KO]),
        #
        # CJK
        #
        ('木兰辞 唧唧复唧唧，木兰当户织。……雄兔脚扑朔，雌兔眼迷离，双兔傍地走，安能辨我是雄雌？',
         [LangFeatures.LANG_CN]),
        ('湖北 。。。',
         [LangFeatures.LANG_CN]),
        #
        # Latin
        #
        # English
        ('Blessed are those who find wisdom, those who gain understanding',
         [LangFeatures.LANG_EN]),
        ('Blessed are those who find wisdom, those who gain understanding',
         [LangFeatures.LANG_EN]),
        # Vietnamese
        ('bơi cùng cá mập trắng, vảy núi lửa âm ỉ',
         [LangFeatures.LANG_VN]),
        ('boi cung ca map trang, vay nui lua am i',
         [LangFeatures.LANG_VN]),
        # Indonesian
        ('Sejumlah pakar kesehatan menyarankan pemerintah Indonesia mempertimbangkan kemungkinan',
         [LangFeatures.LANG_IN]),
        # Korean in Latin
        ('nanin ramyun joahyo',
         None),
        #
        # Mix
        #
        # As long as less than 5 foreign characters in a block, the randomness
        # will not affect the results.
        ('как не уметь читать и писать 孙子兵法 abcd',
         [LangFeatures.LANG_RU]),
    ]

    def __init__(
            self,
            ut_params
    ):
        self.ut_params = ut_params
        if self.ut_params is None:
            # We only do this for convenience, so that we have access to the Class methods in UI
            self.ut_params = ut.UnitTestParams()
        return

    def run_unit_test(self):
        dt = LangDetect()
        res_final = ut.ResultObj(count_ok=0, count_fail=0)

        for text, expected in LangDetectUnitTest.TEST_TEXT_LANG:
            start_time = Profiling.start()
            observed = dt.detect(
                text = text
            )
            ms = round(1000*Profiling.get_time_dif_secs(start=start_time, stop=Profiling.stop()),2)
            Log.debug('Took ' + str(ms) + ' ms')

            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = observed,
                expected = expected,
                test_comment = 'test lang "' + str(expected) + '"'
            ))

        return res_final


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_2

    LangDetectUnitTest(ut_params=None).run_unit_test()
    exit(0)

    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_2
    text = 'как не уметь читать и писать,'
    ld = LangDetect()
    lang = ld.detect(
        text = text
    )
    print(lang)

