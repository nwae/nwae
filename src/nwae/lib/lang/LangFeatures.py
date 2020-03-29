#!/use/bin/python
# --*-- coding: utf-8 --*--

# !!! Will work only on Python 3 and above

#
# This is the most basic class for all language related classes,
# and should not import any language modules
#

import pandas as pd
from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
# pip install iso-639
# https://www.iso.org/iso-639-language-codes.html
# from iso639 import languages
import nwae.utils.UnitTest as ut


#
# Class LangFeatures
#
#   Helper class to define language properties, such as containing word/syllable separators,
#   alphabet type, etc.
#
class LangFeatures:
    #
    # Latin Type Blocks (English, Spanish, French, Vietnamese, etc.)
    # TODO Break into other language variants
    #
    # This covers all latin, including Spanish, Vietnamese characters
    ALPHABET_LATIN    = 'latin'
    # This covers only the common a-z, A-Z
    ALPHABET_LATIN_AZ = 'latin_az'
    # This covers only the special Vietnamese characters
    ALPHABET_LATIN_VI = 'latin_vi'
    ALPHABET_LATIN_VI_AZ = 'latin_vi_az'
    #
    # CJK Type Blocks (Korean, Chinese, Japanese)
    #
    # TODO Break into Chinese variants (simplified, traditional, etc.),
    #   Japanese, Hanja, etc.
    ALPHABET_HANGUL   = 'hangul'
    ALPHABET_CJK      = 'cjk'
    #
    # Cyrillic Blocks (Russian, Belarusian, Ukrainian, etc.)
    # TODO Break into detailed blocks
    #
    ALPHABET_CYRILLIC = 'cyrillic'
    #
    # Other Blocks
    #
    ALPHABET_THAI     = 'thai'

    ALPHABETS_ALL = [
        ALPHABET_LATIN, ALPHABET_LATIN_AZ, ALPHABET_LATIN_VI, ALPHABET_LATIN_VI_AZ,
        ALPHABET_HANGUL, ALPHABET_CJK,
        ALPHABET_CYRILLIC,
        ALPHABET_THAI,
    ]

    #
    # TODO
    #  Move to use ISO 639-2 standard instead of our own
    #  In the mean time always use map_to_correct_lang_code() to map to the right language code
    #
    # All below follows ISO 639-1 Code
    LANG_CODE_STD_ISO639_1 = 'iso639-1'
    LANG_CODE_STD_ISO639_2 = 'iso639-2'

    #
    # Hangul/CJK Alphabet Family
    #
    # Korean
    LANG_KO = 'ko'
    #
    # CJK Alphabet Family
    #
    # Simplified Chinese
    LANG_CN = 'cn'  # Not ISO 639-1
    LANG_ZH = 'zh'
    # This is actually language code + localisation, not ISO-639-1
    LANG_ZH_CN = 'zh-cn'
    #
    # Cyrillic Alphabet Family
    #
    # Russian
    LANG_RU = 'ru'
    #
    # Thai Alphabet Family
    #
    # Thai
    LANG_TH = 'th'
    #
    # Latin Alphabet Family
    #
    LANG_EN = 'en'
    # Spanish
    LANG_ES = 'es'
    # French
    LANG_FR = 'fr'
    # Vietnamese
    LANG_VN = 'vn'      # Not ISO 639-1
    LANG_VI = 'vi'
    # Indonesian
    LANG_ID = 'id'

    ALL_ISO369_1_SUPPORTED_LANGS = (
        LANG_KO,
        LANG_ZH,
        LANG_RU,
        LANG_TH,
        LANG_EN, LANG_ES, LANG_FR, LANG_VI, LANG_ID
    )

    C_LANG_ID        = 'Language'
    C_LANG_NUMBER    = 'LanguageNo'
    C_LANG_NAME      = 'LanguageName'
    C_HAVE_ALPHABET  = 'Alphabet'
    C_CHAR_TYPE      = 'CharacterType'
    C_HAVE_SYL_SEP   = 'SyllableSep'
    C_SYL_SEP_TYPE   = 'SyllableSepType'
    C_HAVE_WORD_SEP  = 'WordSep'
    C_WORD_SEP_TYPE  = 'WordSepType'
    C_HAVE_VERB_CONJ = 'HaveVerbConjugation'

    T_NONE = ''
    T_CHAR = 'character'
    T_SPACE = 'space'

    LEVEL_ALPHABET = 'alphabet'
    LEVEL_SYLLABLE = 'syllable'
    LEVEL_UNIGRAM  = 'unigram'

    @staticmethod
    def map_to_lang_code_iso639_1(
            lang_code
    ):
        if lang_code in (LangFeatures.LANG_CN, LangFeatures.LANG_ZH_CN):
            return LangFeatures.LANG_ZH
        elif lang_code == LangFeatures.LANG_VN:
            return LangFeatures.LANG_VI
        else:
            if lang_code in LangFeatures.ALL_ISO369_1_SUPPORTED_LANGS:
                return lang_code
            else:
                raise Exception(
                    str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Unsupported language code "' + str(lang_code) + '"'
                )

    # Word lists and stopwords are in the same folder
    def __init__(self):
        #
        # Language followed by flag for alphabet boundary, syllable boundary (either as one
        # character as in Chinese or space as in Korean), then word boundary (space)
        # The most NLP-inconvenient languages are those without word boundary, obviously.
        # Name, Code, Alphabet, CharacterType, SyllableSeparator, SyllableSeparatorType, WordSeparator, WordSeparatorType
        #
        #
        # Hangul/CJK Language Family
        #
        lang_index = 0
        lang_ko = {
            LangFeatures.C_LANG_ID:       LangFeatures.LANG_KO,
            LangFeatures.C_LANG_NUMBER:   lang_index,
            LangFeatures.C_LANG_NAME:     'Hangul',
            LangFeatures.C_HAVE_ALPHABET: True,
            LangFeatures.C_CHAR_TYPE:     LangFeatures.ALPHABET_HANGUL,
            LangFeatures.C_HAVE_SYL_SEP:  True,
            # TODO Not really right to say it is char but rather a "syllable_character"
            LangFeatures.C_SYL_SEP_TYPE:  LangFeatures.T_CHAR,
            LangFeatures.C_HAVE_WORD_SEP: True,
            LangFeatures.C_WORD_SEP_TYPE: LangFeatures.T_SPACE,
            LangFeatures.C_HAVE_VERB_CONJ: True
        }
        #
        # CJK Alphabet Family
        #
        lang_index += 1
        lang_zh = {
            LangFeatures.C_LANG_ID:       LangFeatures.LANG_ZH,
            LangFeatures.C_LANG_NUMBER:   lang_index,
            LangFeatures.C_LANG_NAME:     'Chinese',
            LangFeatures.C_HAVE_ALPHABET: False,
            LangFeatures.C_CHAR_TYPE:     LangFeatures.ALPHABET_CJK,
            LangFeatures.C_HAVE_SYL_SEP:  True,
            LangFeatures.C_SYL_SEP_TYPE:  LangFeatures.T_CHAR,
            LangFeatures.C_HAVE_WORD_SEP: False,
            LangFeatures.C_WORD_SEP_TYPE: LangFeatures.T_NONE,
            LangFeatures.C_HAVE_VERB_CONJ: False
        }
        #
        # Cyrillic Alphabet Family
        #
        lang_index += 1
        lang_ru = {
            LangFeatures.C_LANG_ID:       LangFeatures.LANG_RU,
            LangFeatures.C_LANG_NUMBER:   lang_index,
            LangFeatures.C_LANG_NAME:     'Russian',
            LangFeatures.C_HAVE_ALPHABET: True,
            LangFeatures.C_CHAR_TYPE:     LangFeatures.ALPHABET_CYRILLIC,
            LangFeatures.C_HAVE_SYL_SEP:  False,
            LangFeatures.C_SYL_SEP_TYPE:  LangFeatures.T_NONE,
            LangFeatures.C_HAVE_WORD_SEP: True,
            LangFeatures.C_WORD_SEP_TYPE: LangFeatures.T_SPACE,
            LangFeatures.C_HAVE_VERB_CONJ: True
        }
        #
        # Thai Alphabet Family
        #
        lang_index += 1
        lang_th = {
            LangFeatures.C_LANG_ID:       LangFeatures.LANG_TH,
            LangFeatures.C_LANG_NUMBER:   lang_index,
            LangFeatures.C_LANG_NAME:     'Thai',
            LangFeatures.C_HAVE_ALPHABET: True,
            LangFeatures.C_CHAR_TYPE:     LangFeatures.ALPHABET_THAI,
            LangFeatures.C_HAVE_SYL_SEP:  False,
            LangFeatures.C_SYL_SEP_TYPE:  LangFeatures.T_NONE,
            LangFeatures.C_HAVE_WORD_SEP: False,
            LangFeatures.C_WORD_SEP_TYPE: LangFeatures.T_NONE,
            LangFeatures.C_HAVE_VERB_CONJ: False
        }
        #
        # Latin Alphabet Family
        #
        lang_index += 1
        lang_en = {
            LangFeatures.C_LANG_ID:       LangFeatures.LANG_EN,
            LangFeatures.C_LANG_NUMBER:   lang_index,
            LangFeatures.C_LANG_NAME:     'English',
            LangFeatures.C_HAVE_ALPHABET: True,
            LangFeatures.C_CHAR_TYPE:     LangFeatures.ALPHABET_LATIN_AZ,
            LangFeatures.C_HAVE_SYL_SEP:  False,
            LangFeatures.C_SYL_SEP_TYPE:  LangFeatures.T_NONE,
            LangFeatures.C_HAVE_WORD_SEP: True,
            LangFeatures.C_WORD_SEP_TYPE: LangFeatures.T_SPACE,
            LangFeatures.C_HAVE_VERB_CONJ: True
        }
        lang_index += 1
        lang_es = {
            LangFeatures.C_LANG_ID:       LangFeatures.LANG_ES,
            LangFeatures.C_LANG_NUMBER:   lang_index,
            LangFeatures.C_LANG_NAME:     'Spanish',
            LangFeatures.C_HAVE_ALPHABET: True,
            LangFeatures.C_CHAR_TYPE:     LangFeatures.ALPHABET_LATIN,
            LangFeatures.C_HAVE_SYL_SEP:  False,
            LangFeatures.C_SYL_SEP_TYPE:  LangFeatures.T_NONE,
            LangFeatures.C_HAVE_WORD_SEP: True,
            LangFeatures.C_WORD_SEP_TYPE: LangFeatures.T_SPACE,
            LangFeatures.C_HAVE_VERB_CONJ: True
        }
        lang_index += 1
        lang_fr = {
            LangFeatures.C_LANG_ID:       LangFeatures.LANG_FR,
            LangFeatures.C_LANG_NUMBER:   lang_index,
            LangFeatures.C_LANG_NAME:     'French',
            LangFeatures.C_HAVE_ALPHABET: True,
            LangFeatures.C_CHAR_TYPE:     LangFeatures.ALPHABET_LATIN,
            LangFeatures.C_HAVE_SYL_SEP:  False,
            LangFeatures.C_SYL_SEP_TYPE:  LangFeatures.T_NONE,
            LangFeatures.C_HAVE_WORD_SEP: True,
            LangFeatures.C_WORD_SEP_TYPE: LangFeatures.T_SPACE,
            LangFeatures.C_HAVE_VERB_CONJ: True
        }
        lang_index += 1
        lang_vi = {
            LangFeatures.C_LANG_ID:       LangFeatures.LANG_VI,
            LangFeatures.C_LANG_NUMBER:   lang_index,
            LangFeatures.C_LANG_NAME:     'Vietnamese',
            LangFeatures.C_HAVE_ALPHABET: True,
            LangFeatures.C_CHAR_TYPE:     LangFeatures.ALPHABET_LATIN_VI_AZ,
            LangFeatures.C_HAVE_SYL_SEP:  True,
            LangFeatures.C_SYL_SEP_TYPE:  LangFeatures.T_SPACE,
            LangFeatures.C_HAVE_WORD_SEP: False,
            LangFeatures.C_WORD_SEP_TYPE: LangFeatures.T_NONE,
            LangFeatures.C_HAVE_VERB_CONJ: False
        }
        lang_index += 1
        lang_id = {
            LangFeatures.C_LANG_ID:       LangFeatures.LANG_ID,
            LangFeatures.C_LANG_NUMBER:   lang_index,
            LangFeatures.C_LANG_NAME:     'Indonesian',
            LangFeatures.C_HAVE_ALPHABET: True,
            LangFeatures.C_CHAR_TYPE:     LangFeatures.ALPHABET_LATIN_AZ,
            LangFeatures.C_HAVE_SYL_SEP:  False,
            LangFeatures.C_SYL_SEP_TYPE:  LangFeatures.T_NONE,
            LangFeatures.C_HAVE_WORD_SEP: True,
            LangFeatures.C_WORD_SEP_TYPE: LangFeatures.T_SPACE,
            LangFeatures.C_HAVE_VERB_CONJ: True
        }

        self.langs = {
            # Hangul/CJK
            LangFeatures.LANG_KO: lang_ko,
            # CJK
            LangFeatures.LANG_ZH: lang_zh,
            # Cyrillic
            LangFeatures.LANG_RU: lang_ru,
            # Thai
            LangFeatures.LANG_TH: lang_th,
            # Latin
            LangFeatures.LANG_EN: lang_en,
            LangFeatures.LANG_ES: lang_es,
            LangFeatures.LANG_FR: lang_fr,
            LangFeatures.LANG_VI: lang_vi,
            LangFeatures.LANG_ID: lang_id,
        }
        self.langfeatures = pd.DataFrame(
            self.langs.values()
        )
        return

    def __check_lang(self, lang):
        lang_std = LangFeatures.map_to_lang_code_iso639_1(
            lang_code = lang
        )
        if lang_std not in self.langs.keys():
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': No such language "' + str(lang) + '" in supported languages ' + str(self.langs.keys())
            )
        return lang_std

    def get_word_separator_type(
            self,
            lang
    ):
        lang_std = self.__check_lang(lang = lang)
        lang_dict = self.langs[lang_std]
        return lang_dict[LangFeatures.C_WORD_SEP_TYPE]

    def get_syllable_separator_type(
            self,
            lang
    ):
        lang_std = self.__check_lang(lang = lang)
        lang_dict = self.langs[lang_std]
        return lang_dict[LangFeatures.C_SYL_SEP_TYPE]

    def have_verb_conjugation(
            self,
            lang
    ):
        lang_std = self.__check_lang(lang = lang)
        lang_dict = self.langs[lang_std]
        return lang_dict[LangFeatures.C_HAVE_VERB_CONJ]

    #
    # Means that the smallest token is formed by the character set or alphabet.
    # For example, in English, the language token is the word, formed by latin alphabets,
    # thus the token is a set of alphabets and not the alphabet itself.
    # Same with Korean, an example token '한국어' is a word formed by Hangul alphabets or 자무
    # But Chinese token is the character set itself '我在学中文', where each token is the character
    #
    def is_lang_token_same_with_charset(
            self,
            lang
    ):
        lang_std = self.__check_lang(lang = lang)

        # Languages that have the tokens as the character set, or languages with no syllable or unigram separator
        # Besides cn/th, the same goes for Lao, Cambodian, Japanese, with no spaces to separate syllables/unigrams.
        lf = self.langfeatures
        len = lf.shape[0]
        # First it must not have a word separator
        langindexes = [ x for x in range(0,len,1) if lf[LangFeatures.C_HAVE_WORD_SEP][x]==False ]
        # Second condition is that it doesn't have a syllable separator, or it has a syllable separator which is a character
        langs = [
            lf[LangFeatures.C_LANG_ID][x] for x in langindexes if (
                    lf[LangFeatures.C_HAVE_SYL_SEP][x]==False or
                    ( lf[LangFeatures.C_HAVE_SYL_SEP][x]==True and lf[LangFeatures.C_SYL_SEP_TYPE][x]==LangFeatures.T_CHAR )
            )
        ]
        lang_token_same_with_charset = lang_std in langs
        return lang_token_same_with_charset

    def get_languages_with_word_separator(self):
        len = self.langfeatures.shape[0]
        langs = [ self.langfeatures[LangFeatures.C_LANG_ID][x] for x in range(0,len,1)
                  if self.langfeatures[LangFeatures.C_HAVE_WORD_SEP][x]==True ]
        return langs

    def get_languages_with_syllable_separator(self):
        len = self.langfeatures.shape[0]
        langs = [ self.langfeatures[LangFeatures.C_LANG_ID][x] for x in range(0, len, 1)
                  if self.langfeatures[LangFeatures.C_HAVE_SYL_SEP][x]==True ]
        return langs

    def get_languages_with_only_syllable_separator(self):
        return list(
            set( self.get_languages_with_syllable_separator() ) -\
            set( self.get_languages_with_word_separator() )
        )

    def get_languages_with_no_word_separator(self):
        len = self.langfeatures.shape[0]
        langs = [
            self.langfeatures[LangFeatures.C_LANG_ID][x]
            for x in range(0, len, 1)
            if not self.langfeatures[LangFeatures.C_HAVE_WORD_SEP][x]
        ]
        return langs

    def get_languages_for_alphabet_type(self, alphabet):
        len = self.langfeatures.shape[0]
        langs = [
            self.langfeatures[LangFeatures.C_LANG_ID][x]
            for x in range(0, len, 1)
            if self.langfeatures[LangFeatures.C_CHAR_TYPE][x] == alphabet
        ]
        return langs

    #
    # If separator for either alphabet/syllable/word (we shall refer as token) is None, this means there is no
    # way to identify the token. If the separator is '', means we can identify it by character (e.g. Chinese character,
    # Thai alphabet, Korean alphabet inside a Korean character/syllable).
    #
    def get_split_token(
            self,
            lang,
            level
    ):
        lang_std = self.__check_lang(lang = lang)
        lang_dict = self.langs[lang_std]

        have_alphabet = lang_dict[LangFeatures.C_HAVE_ALPHABET]
        have_syl_sep  = lang_dict[LangFeatures.C_HAVE_SYL_SEP]
        syl_sep_type  = lang_dict[LangFeatures.C_SYL_SEP_TYPE]
        have_word_sep = lang_dict[LangFeatures.C_HAVE_WORD_SEP]
        word_sep_type = lang_dict[LangFeatures.C_WORD_SEP_TYPE]

        if level == LangFeatures.LEVEL_ALPHABET:
            # If a language has alphabets, the separator is by character, otherwise return NA
            if have_alphabet:
                return ''
            else:
                return None
        elif level == LangFeatures.LEVEL_SYLLABLE:
            if have_syl_sep:
                if syl_sep_type == LangFeatures.T_CHAR:
                    return ''
                elif syl_sep_type == LangFeatures.T_SPACE:
                    return ' '
                else:
                    return None
        elif level == LangFeatures.LEVEL_UNIGRAM:
            # Return language specific word separator if exists.
            # Return language specific syllable separator if exists.
            if have_word_sep:
                if word_sep_type == LangFeatures.T_CHAR:
                    return ''
                elif word_sep_type == LangFeatures.T_SPACE:
                    return ' '
                else:
                    return None
            elif have_syl_sep:
                if syl_sep_type == LangFeatures.T_CHAR:
                    return ''
                elif syl_sep_type == LangFeatures.T_SPACE:
                    return ' '
                else:
                    return None

        return None

    def get_alphabet_type(
            self,
            lang
    ):
        lang_std = self.__check_lang(lang = lang)
        # Language index
        lang_index = self.langfeatures.index[self.langfeatures[LangFeatures.C_LANG_ID]==lang_std].tolist()
        if len(lang_index) == 0:
            return None
        lang_index = lang_index[0]

        return self.langfeatures[LangFeatures.C_CHAR_TYPE][lang_index]


class LangFeaturesUnitTest:

    def __init__(
            self,
            ut_params
    ):
        self.ut_params = ut_params
        if self.ut_params is None:
            # We only do this for convenience, so that we have access to the Class methods in UI
            self.ut_params = ut.UnitTestParams()
        return

    def run_unit_test(
            self
    ):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)

        lf = LangFeatures()
        observed = lf.get_languages_with_word_separator()
        observed.sort()
        expected = [
            LangFeatures.LANG_KO, LangFeatures.LANG_RU,
            LangFeatures.LANG_EN, LangFeatures.LANG_ES, LangFeatures.LANG_FR, LangFeatures.LANG_ID
        ]
        expected.sort()

        res_final.update_bool(res_bool=ut.UnitTest.assert_true(
            observed = observed,
            expected = expected,
            test_comment = 'test languages with word separator'
        ))

        observed = lf.get_languages_with_syllable_separator()
        observed.sort()
        expected = [LangFeatures.LANG_ZH, LangFeatures.LANG_KO, LangFeatures.LANG_VI]
        expected.sort()

        res_final.update_bool(res_bool=ut.UnitTest.assert_true(
            observed = observed,
            expected = expected,
            test_comment = 'test languages with syllable separator'
        ))

        observed = lf.get_languages_with_no_word_separator()
        observed.sort()
        expected = [LangFeatures.LANG_ZH, LangFeatures.LANG_TH, LangFeatures.LANG_VI]
        expected.sort()

        res_final.update_bool(res_bool=ut.UnitTest.assert_true(
            observed = observed,
            expected = expected,
            test_comment = 'test languages with no word or syllable separator'
        ))

        observed = lf.get_languages_with_only_syllable_separator()
        observed.sort()
        expected = [LangFeatures.LANG_ZH, LangFeatures.LANG_VI]
        expected.sort()

        res_final.update_bool(res_bool=ut.UnitTest.assert_true(
            observed = observed,
            expected = expected,
            test_comment = 'test languages with ONLY syllable separator'
        ))

        # We could get the languages associated with the alphabet programmatically also,
        # but we do that in the second round
        alphabet_langs = {
            LangFeatures.ALPHABET_HANGUL:   [LangFeatures.LANG_KO],
            LangFeatures.ALPHABET_THAI:     [LangFeatures.LANG_TH],
            LangFeatures.ALPHABET_CYRILLIC: [LangFeatures.LANG_RU],
            LangFeatures.ALPHABET_CJK:      [LangFeatures.LANG_ZH],
            LangFeatures.ALPHABET_LATIN_AZ: [
                LangFeatures.LANG_EN, LangFeatures.LANG_ID,
            ],
            LangFeatures.ALPHABET_LATIN_VI_AZ: [LangFeatures.LANG_VI],
            LangFeatures.ALPHABET_LATIN:    [
                LangFeatures.LANG_ES, LangFeatures.LANG_FR
            ]
        }
        for alp in alphabet_langs.keys():
            observed = lf.get_languages_for_alphabet_type(alphabet=alp)
            observed.sort()
            expected = alphabet_langs[alp]
            expected.sort()

            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = observed,
                expected = expected,
                test_comment = 'R1 test languages for alphabet "' + str(alp) + '"'
            ))

        # In this round we get the languages for an alphabet programmatically
        alphabet_langs = {}
        for alp in LangFeatures.ALPHABETS_ALL:
            alphabet_langs[alp] = lf.get_languages_for_alphabet_type(
                alphabet = alp
            )
        for alp in alphabet_langs.keys():
            observed = lf.get_languages_for_alphabet_type(alphabet=alp)
            observed.sort()
            expected = alphabet_langs[alp]
            expected.sort()

            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = observed,
                expected = expected,
                test_comment = 'R2 test languages for alphabet "' + str(alp) + '"'
            ))

        langs_with_token_same_as_charset = []
        for lang in lf.langs.keys():
            token_same_as_charset = lf.is_lang_token_same_with_charset(
                lang = lang
            )
            if token_same_as_charset:
                langs_with_token_same_as_charset.append(lang)

        res_final.update_bool(res_bool=ut.UnitTest.assert_true(
            observed     = sorted(langs_with_token_same_as_charset),
            expected     = sorted([LangFeatures.LANG_ZH, LangFeatures.LANG_TH]),
            test_comment = 'Test langs with token = charset'
        ))

        return res_final



if __name__ == '__main__':
    def demo_1():
        lf = LangFeatures()
        print ( lf.langfeatures )
        return

    def demo_2():
        lf = LangFeatures()

        for lang in lf.langfeatures[LangFeatures.C_LANG_ID]:
            print ( lang + ':alphabet=[' + str(lf.get_split_token(lang, LangFeatures.LEVEL_ALPHABET)) + ']' )
            print ( lang + ':syllable=[' + str(lf.get_split_token(lang, LangFeatures.LEVEL_SYLLABLE)) + ']' )
            print ( lang + ':unigram=[' + str(lf.get_split_token(lang, LangFeatures.LEVEL_UNIGRAM)) + ']' )
            print ( lang + ':Character Type = ' + lf.get_alphabet_type(lang) )
            print ( lang + ':Token same as charset = ' + str(lf.is_lang_token_same_with_charset(lang=lang)))

    def demo_3():
        lf = LangFeatures()
        print ( lf.langfeatures )

        print ( 'Languages with word separator: ' + str(lf.get_languages_with_word_separator()) )
        print ( 'Languages with syllable separator:' + str(lf.get_languages_with_syllable_separator()) )
        print ( 'Languages with only syllable separator:' + str(lf.get_languages_with_only_syllable_separator()))
        print ( 'Languages with no word or syllable separator:' + str(lf.get_languages_with_no_word_separator()))

    demo_1()
    demo_2()
    demo_3()

    LangFeaturesUnitTest(ut_params=None).run_unit_test()

