# --*-- coding: utf-8 --*--

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
from nwae.lang.LangFeatures import LangFeatures
from nwae.lang.detect.LangDetect import LangDetect
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
        # These are the Hangul blocks for Unicode block [0xAC00, 0xD7AF]
        ('낮선 곳에서 잠을 자다가, 갑자기 들리는 흐르는 물소리, 등짝을 훑고 지나가는 지진의 진동',
         [LangFeatures.LANG_KO]),
        ('야!',
         [LangFeatures.LANG_KO]),
        # Jamo Unicode block 11xx
        # On the computer, this should print out into Hangul blocks, '자모'
        (chr(0x110c) + chr(0x1161) + chr(0x1106) + chr(0x1169),
         [LangFeatures.LANG_KO]),
        # Jamo Unicode block 31xx
        # On the computer, this should print out the individual Jamos, 'ㅈㅏㅁㅗ'
        (chr(0x3148) + chr(0x314f) + chr(0x3141) + chr(0x3157),
         [LangFeatures.LANG_KO]),
        # Japanese
        ('本日はチャットサービスをご利用いただき、ありがとうございます。オペレーターと接続中です。',
         [LangFeatures.LANG_JA]),
        ('江戸時代には江戸前や江戸前海などの呼び名があった。',
         [LangFeatures.LANG_JA]),
        #
        # CJK
        #
        ('木兰辞 唧唧复唧唧，木兰当户织。……雄兔脚扑朔，雌兔眼迷离，双兔傍地走，安能辨我是雄雌？',
         [LangFeatures.LANG_ZH]),
        ('湖北 。。。',
         [LangFeatures.LANG_ZH]),
        #
        # Thai
        #
        ('ช่วยกันต่อสู้ไวรัส',
         [LangFeatures.LANG_TH]),
        ('ขณะที่มาตรการใหม่จะส่งผลให้บริการต่างๆ',
         [LangFeatures.LANG_TH]),
        ('สวยจัง 10/10.',
         [LangFeatures.LANG_TH]),
        #
        # Latin
        #
        ### English
        ('Blessed are those who find wisdom, those who gain understanding',
         [LangFeatures.LANG_EN]),
        ('Please',
         [LangFeatures.LANG_EN]),
        ('OK',
         [LangFeatures.LANG_EN]),
        ('hi',
         [LangFeatures.LANG_EN]),
        ('I am hungry',
         [LangFeatures.LANG_EN]),
        # This will fail at 1st round and try the stemmed sentence "asdf bank wisdom"
        ('asdf banks wisdoms',
         [LangFeatures.LANG_EN]),
        ### Spanish
        # This will contain both Spanish/French at 25%
        ('Incrustado en las laderas de unas colinas volcánicas',
         [LangFeatures.LANG_ES]),
        ('por su parque natural y por los cercanos establecimiento termales',
         [LangFeatures.LANG_ES]),
        ('A principios de febrero Adriano y Renato, dos vecinos de esta localidad de unos 3.300 habitantes',
         [LangFeatures.LANG_ES]),
        ### French
        ('Le monde est passé en veille ou plutôt tourne à la limite au ralenti',
         [LangFeatures.LANG_FR]),
        ('A travers le monde, les populations sont soumises à d\'énormes restrictions',
         [LangFeatures.LANG_FR]),
        ('Je ne parle pas français.',
         [LangFeatures.LANG_FR]),
        ('Pardon, excusez-moi',
         [LangFeatures.LANG_FR]),
        ### Vietnamese
        ('bơi cùng cá mập trắng, vảy núi lửa âm ỉ',
         [LangFeatures.LANG_VI]),
        # No diacritics
        ('boi cung ca map trang, vay nui lua am i',
         [LangFeatures.LANG_VI]),
        ('toi yeu em',
         [LangFeatures.LANG_VI]),
        # Only the word 'số nhiều' should be detected
        ('rút bao nhiêu số nhiều bao abcdef',
         [LangFeatures.LANG_VI]),
        ### Indonesian
        ('Sejumlah pakar kesehatan menyarankan pemerintah Indonesia mempertimbangkan kemungkinan',
         [LangFeatures.LANG_ID]),
        ('Yang terpenting adalah apa yang kita lakukan.',
         [LangFeatures.LANG_ID]),
        ('Anda tidak bisa memenangi pertandingan sepak bola hanya dengan bertahan.',
         [LangFeatures.LANG_ID]),
        ('Anda juga harus menyerang',
         [LangFeatures.LANG_ID]),
        ('Gianni Infantino ketika meresmikan kampanye melawan virus corona bersama para pesepakbola',
         [LangFeatures.LANG_ID]),
        ### Korean/Thai/etc in Latin, should be nothing
        ('nanin ramyunil joahyo',
         []),
        ('pom mai keui bpai',
         []),
        ('sipeh hochiak',
         []),
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

        start_all_time = Profiling.start()

        for text, expected in LangDetectUnitTest.TEST_TEXT_LANG:
            start_time = Profiling.start()
            observed = dt.detect(
                text = text
            )
            ms = round(1000*Profiling.get_time_dif_secs(start=start_time, stop=Profiling.stop()), 2)
            Log.debug('Took ' + str(ms) + ' ms')

            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = observed,
                expected = expected,
                test_comment = 'test lang "' + str(expected) + '", text "' + str(text) + '"'
            ))

        end_all_time = Profiling.stop()
        avg_per_text_ms = 1000 * Profiling.get_time_dif_secs(
            start = start_all_time,
            stop  = end_all_time
        ) / len(LangDetectUnitTest.TEST_TEXT_LANG)
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Average ' + str(round(avg_per_text_ms,2)) + 'ms per text (total '
            + str(len(LangDetectUnitTest.TEST_TEXT_LANG)) + ' sentences)'
        )

        return res_final


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1

    res = LangDetectUnitTest(ut_params=None).run_unit_test()
    print('Pass=' + str(res.count_ok) + ', Fail=' + str(res.count_fail))
    exit(0)

    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_2
    #text = 'rút bao nhiêu'
    text = 'banks wisdoms asdf'
    ld = LangDetect()
    lang = ld.detect(
        text = text
    )
    print(lang)

