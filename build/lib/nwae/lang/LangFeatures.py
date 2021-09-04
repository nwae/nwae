#!/use/bin/python
# --*-- coding: utf-8 --*--

#
# This is the most basic class for all language related classes,
# and should not import any of our own language modules
#
# The existing standards for language codes ISO 639-3/639-1, ISO 15924, do not include
# properties we need (presence & types of syllable/word separators, presence of conjugation, etc),
# At least I am not aware of any ISO standard that includes them
# So we define our own list of properties
# However we also include properties from above standards using open PYPI packages like pycountry
#

import pandas as pd
from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
# pip install iso-639
# https://www.iso.org/iso-639-language-codes.html
# from iso639 import languages
import nwae.utils.UnitTest as ut
try:
    import pycountry
except Exception as ex:
    Log.warning(
        str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
        + ': Cannot import pycountry: ' + str(ex)
    )
    pass


#
# Class LangFeatures
#
#   Helper class to define language properties, such as containing word/syllable separators,
#   alphabet type, etc.
#
#   This most fundamental class for languages tells us:
#
#     1. Alphabet Type
#        What alphabet type a language is written in, either Latin, Cyrillic, etc.
#        This is used for example in LangDetect class.
#
#     2. Separator Properties
#        Whether a language has a natural word separator (e.g. space), syllable separator
#        (e.g. Korean Hangul syllable, Chinese/Japanese character, Vietnamese)
#        This is used for grouping word lists by alphabet/syllable lengths, in word
#        segmentation to know how to move to the next "character" which could be a whole
#        syllable and not a character.
#
#     3. Part of Speech (Часть Речи) Conjugations
#        Whether a language has these conjugations.
#        This is used to check if we need to do stemming or not.
#
class LangFeatures:

    # *** pycountry ***
    # Contains
    #    1. All ISO 639-3 three-letter codes (alpha_3)
    #    2. ISO 639-1 two-letter codes (alpha_2) if exist
    # The returned data structure columns ["alpha_2", "alpha_3", "name", "scope", "type"]
    # follows ISO 639-2 table here https://en.wikipedia.org/wiki/List_of_ISO_639-2_codes
    # For scope=Collective (e.g. Artificial Languages, Australian Languages), they are not included in
    # pycountry, they are specific to ISO 639-2
    # Example with alpha_2 codes (ISO 639-1)
    #   >>> pycountry.languages.get(alpha_2='kr')
    #   Language(alpha_2='kr', alpha_3='kau', name='Kanuri', scope='M', type='L')
    #   >>> pycountry.languages.get(alpha_2='ko')
    #   Language(alpha_2='ko', alpha_3='kor', name='Korean', scope='I', type='L')
    #   >>> pycountry.languages.get(alpha_2='ja')
    #   Language(alpha_2='ja', alpha_3='jpn', name='Japanese', scope='I', type='L')
    # Example with no alpha_2, alpha_3 only
    #   >>> pycountry.languages.get(alpha_3='ace')
    #   Language(alpha_3='ace', name='Achinese', scope='I', type='L')
    #   >>> pycountry.languages.get(alpha_3='ban')
    #   Language(alpha_3='ban', name='Balinese', scope='I', type='L')
    # To minimize dependency, we don't use this for now
    # PYCLANG = pycountry.languages

    #
    # Latin Type Blocks (English, Spanish, French, Vietnamese, etc.)
    # TODO Break into other language variants
    #
    # This covers all latin, including Spanish, Vietnamese characters
    ALPHABET_LATIN         = 'latin'
    # This covers only the common a-z, A-Z
    ALPHABET_LATIN_AZ      = 'latin_az'
    # This covers only the special Vietnamese characters
    ALPHABET_LATIN_VI      = 'latin_vi'
    ALPHABET_LATIN_VI_AZ   = 'latin_vi_az'
    # TODO I guess for these alphabets, can easily do a manual list like Vietnamese below
    ALPHABET_LATIN_FR      = 'latin_fr'
    ALPHABET_LATIN_GERMAN  = 'latin_gr'
    # Czech, Danish, Dutch, Italian, English
    ALPHABET_LATIN_CZECH   = 'latin_ch'
    ALPHABET_LATIN_DANISH  = 'latin_danish'
    ALPHABET_LATIN_DUTCH   = 'latin_dutch'
    ALPHABET_LATIN_ITALY   = 'latin_it'
    ALPHABET_LATIN_SPANISH = 'latin_spanish'
    ALPHABET_LATIN_ENG     = 'latin_en'

    #
    # CJK Type Blocks (Korean, Chinese, Japanese)
    #
    # TODO Break into Chinese variants (simplified, traditional, etc.),
    #   Japanese, Hanja, etc.
    # done
    ALPHABET_HANGUL            = 'hangul'
    ALPHABET_CJK               = 'cjk'
    ALPHABET_CJK_SIMPLIFIED    = 'cjk_simplified'
    ALPHABET_CJK_TRADITIONAL   = 'cjk_traditional'
    # CJK + Hiragana + Katakana
    ALPHABET_HIRAGANA_KATAKANA = 'hiragana_katakana'
    ALPHABET_JAPANESE          = 'japanese'
    # from M: variants for kanji, hanja, chinese (traditional/ simplified)
    ALPHABET_KANJI             = 'kanji'
    ALPHABET_CH_SIMPLIFIED     = 'chinese_simplified'
    ALPHABET_CH_TRADITIONAL    = 'chinese_traditional'
    ALPHABET_HANJA             = 'hanja'
    #
    # Cyrillic Blocks (Russian, Belarusian, Ukrainian, etc.)
    # TODO Break into detailed blocks
    # done
    ALPHABET_CYRILLIC       = 'cyrillic'
    ALPHABET_CYR_RUSSIAN    = 'russian'
    ALPHABET_CYR_UKRAINIAN  = 'ukrainian'
    ALPHABET_CYR_BELORUSIAN = 'belarusian'
    ALPHABET_CYR_BULGARIAN  = 'bulgarian'
    ALPHABET_CYR_MACEDONIAN = 'macedonian'

    #
    # Other Blocks
    #
    ALPHABET_THAI     = 'thai'

    ALPHABETS_ALL = [
        ALPHABET_LATIN, ALPHABET_LATIN_AZ, ALPHABET_LATIN_VI, ALPHABET_LATIN_VI_AZ,
        ALPHABET_HANGUL, ALPHABET_CJK, ALPHABET_HIRAGANA_KATAKANA, ALPHABET_JAPANESE,
        ALPHABET_CYRILLIC, ALPHABET_CYR_BELORUSIAN, ALPHABET_CYR_BULGARIAN, ALPHABET_CYR_MACEDONIAN,
        ALPHABET_CYR_UKRAINIAN, ALPHABET_CYR_RUSSIAN,
        ALPHABET_THAI
    ]

    #
    # TODO
    #  Move to use ISO 639-2 standard instead of our own
    #  In the mean time always use map_to_correct_lang_code() to map to the right language code
    #
    # All below follows ISO 639-1 or ISO 639-3 Code
    LANG_CODE_STD_ISO639_1 = 'iso639-1'
    LANG_CODE_STD_ISO639_2 = 'iso639-2'
    LANG_CODE_STD_ISO639_3 = 'iso639-3'

    #
    # Hangul/CJK Alphabet Family
    #
    # Korean
    LANG_KO  = 'ko'     # ISO-639-1
    LANG_KOR = 'kor'    # ISO-639-3 or PYCLANG.get(alpha_2=LANG_KO).alpha_3
    #
    # CJK Alphabet Family
    #
    # Simplified Chinese
    LANG_CN = 'cn'      # NOT ISO 639-1
    LANG_ZH = 'zh'      # ISO-639-1
    LANG_ZHO = 'zho'    # ISO-639-3 or PYCLANG.get(alpha_2=LANG_ZH).alpha_3
    # This is actually language code + localisation, not ISO-639-1
    LANG_ZH_CN = 'zh-cn'
    # Japanese
    LANG_JA  = 'ja'     # ISO-639-1
    LANG_JPN = 'jpn'    # ISO-639-3 or PYCLANG.get(alpha_2=LANG_JA).alpha_3
    #
    # Cyrillic Alphabet Family
    #
    # Russian
    LANG_RU  = 'ru'     # ISO-639-1
    LANG_RUS = 'rus'    # ISO-639-3 or PYCLANG.get(alpha_2=LANG_RU).alpha_3
    # language code + localisation
    LANG_RU_RU = 'ru-RU'
    # Bulgarian
    LANG_BG  = 'bg'     # ISO-639-1
    LANG_BUL = 'bul'    # ISO-639-3 or PYCLANG.get(alpha_2=LANG_BG).alpha_3
    # Belarusian
    LANG_BE  = 'be'     # ISO-639-1
    LANG_BEL = 'bel'    # ISO-639-3 or PYCLANG.get(alpha_2=LANG_BE).alpha_3
    # Macedonian
    LANG_MK  = 'mk'     # ISO-639-1
    LANG_MKD = 'mkd'    # ISO-639-3 or PYCLANG.get(alpha_2=LANG_MK).alpha_3
    # Ukrainian
    LANG_UK  = 'uk'     # ISO-639-1
    LANG_UKR = 'ukr'    # ISO-639-3 or PYCLANG.get(alpha_2=LANG_UK).alpha_3
    #
    # Thai Alphabet Family
    #
    # Thai
    LANG_TH  = 'th'      # ISO-639-1
    LANG_THA = 'tha'     # ISO-639-3 or PYCLANG.get(alpha_2=LANG_TH).alpha_3
    #
    # Latin Alphabet Family
    #
    LANG_EN  = 'en'      # ISO-639-1
    LANG_ENG = 'eng'     # ISO-639-3 or PYCLANG.get(alpha_2=LANG_EN).alpha_3
    # Spanish
    LANG_ES  = 'es'      # ISO-639-1
    LANG_SPA = 'spa'     # ISO-639-3 or PYCLANG.get(alpha_2=LANG_ES).alpha_3
    # French
    LANG_FR  = 'fr'      # ISO-639-1
    LANG_FRA = 'fra'     # ISO-639-3 or PYCLANG.get(alpha_2=LANG_FR).alpha_3
    # German
    LANG_DE = 'de'       # ISO-639-1
    LANG_DEU = 'deu'      # ISO-639-3 or PYCLANG.get(alpha_2=LANG_FR).alpha_3
    # Italian
    LANG_IT = 'it'  # ISO-639-1
    LANG_ITA = 'ita'  # ISO-639-3 or PYCLANG.get(alpha_2=LANG_FR).alpha_3
    # Dutch
    LANG_NL = 'nl'  # ISO-639-1
    LANG_NLD = 'nld'  # ISO-639-3 or PYCLANG.get(alpha_2=LANG_FR).alpha_3
    # Vietnamese
    LANG_VN  = 'vn'      # Not ISO 639-1
    LANG_VI  = 'vi'      # ISO-639-1
    LANG_VIE = 'vie'     # ISO-639-3 or PYCLANG.get(alpha_2=LANG_VI).alpha_3
    # Indonesian
    LANG_ID  = 'id'      # ISO-639-1
    LANG_IND = 'ind'     # ISO-639-3 or PYCLANG.get(alpha_2=LANG_ID).alpha_3

    ALL_ISO639_1_SUPPORTED_LANGS = (
        LANG_KO, LANG_ZH, LANG_JA, LANG_RU, LANG_TH,
        LANG_EN, LANG_ES, LANG_FR, LANG_VI, LANG_ID,
    )
    ALL_ISO639_3_SUPPORTED_LANGS = (
        LANG_KOR, LANG_ZHO, LANG_JPN, LANG_RUS, LANG_THA,
        LANG_ENG, LANG_SPA, LANG_FRA, LANG_VIE, LANG_IND,
    )

    # For languages with ISO 639-1, we use that, for those without we can use ISO 639-3
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
    # From ISO 639-2
    C_LANG_639_2_ALPHA_2 = 'iso_639_2_alpha_2'
    C_LANG_639_2_ALPHA_3 = 'iso_639_2alpha_3'
    C_LANG_639_2_NAME    = 'iso_639_2name'
    C_LANG_639_2_BIBLIO  = 'iso_639_2bibliographic'
    C_LANG_639_2_SCOPE   = 'iso_639_2scope'
    C_LANG_639_2_TYPE    = 'iso_639_2type'

    T_NONE = ''
    T_CHAR = 'character'
    T_SPACE = 'space'

    LEVEL_ALPHABET = 'alphabet'
    LEVEL_SYLLABLE = 'syllable'
    LEVEL_UNIGRAM  = 'unigram'

    # Map some common errors to correct code
    @staticmethod
    def map_to_lang_code_iso639_1(lang_code):
        return LangFeatures.map_to_correct_lang_code_iso_639_1_or_3(lang_code=lang_code)

    @staticmethod
    def map_to_correct_lang_code_iso_639_1_or_3(
            # 2 character language code
            lang_code
    ):
        # общая ошибка 'cn' вместо 'zh'
        if lang_code in (LangFeatures.LANG_CN, LangFeatures.LANG_ZH_CN):
            return LangFeatures.LANG_ZH
        # общая ошибка 'vn' вместо 'vi'
        elif lang_code == LangFeatures.LANG_VN:
            return LangFeatures.LANG_VI
        else:
            if lang_code in LangFeatures.ALL_ISO639_1_SUPPORTED_LANGS:
                return lang_code
            elif lang_code in LangFeatures.ALL_ISO639_3_SUPPORTED_LANGS:
                return lang_code
            else:
                Log.info(
                    str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Unsupported language code "' + str(lang_code) + '" return unchanged "' + str(lang_code) + '"'
                )
                return lang_code

    # Word lists and stopwords are in the same folder
    def __init__(
            self,
            write_lang_features_to_csv = False
    ):
        #
        # Language followed by flag for alphabet boundary, syllable boundary (either as one
        # character as in Chinese or space as in Korean), then word boundary (space)
        # The most NLP-inconvenient languages are those without word boundary, obviously.
        # Name, Code, Alphabet, CharacterType, SyllableSeparator, SyllableSeparatorType, WordSeparator, WordSeparatorType
        #
        # We need to define our own properties as even ISO 15924 specification does not contain them
        #
        # Hangul/CJK Language Family
        #
        try:
            self.PYCLANG = pycountry.languages
        except Exception as ex:
            Log.warning(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Cannot load pycountry languages: ' + str(ex)
            )
            self.PYCLANG = None

        lang_index = 0
        lang_ko = {
            self.C_LANG_ID:        self.LANG_KO,
            self.C_LANG_NUMBER:    lang_index,
            self.C_LANG_NAME:      'Hangul',
            self.C_HAVE_ALPHABET:  True,
            self.C_CHAR_TYPE:      self.ALPHABET_HANGUL,
            self.C_HAVE_SYL_SEP:   True,
            # TODO Not really right to say it is char but rather a "syllable_character"
            self.C_SYL_SEP_TYPE:   self.T_CHAR,
            self.C_HAVE_WORD_SEP:  True,
            self.C_WORD_SEP_TYPE:  self.T_SPACE,
            self.C_HAVE_VERB_CONJ: True
        }
        #
        # CJK Alphabet Family
        #
        lang_index += 1
        lang_zh = {
            self.C_LANG_ID:        self.LANG_ZH,
            self.C_LANG_NUMBER:    lang_index,
            self.C_LANG_NAME:      'Chinese',
            self.C_HAVE_ALPHABET:  False,
            self.C_CHAR_TYPE:      self.ALPHABET_CJK,
            self.C_HAVE_SYL_SEP:   True,
            self.C_SYL_SEP_TYPE:   self.T_CHAR,
            self.C_HAVE_WORD_SEP:  False,
            self.C_WORD_SEP_TYPE:  self.T_NONE,
            self.C_HAVE_VERB_CONJ: False
        }
        #
        # Japanese Hiragana/Katakana
        #
        lang_index += 1
        lang_ja = {
            self.C_LANG_ID:        self.LANG_JA,
            self.C_LANG_NUMBER:    lang_index,
            self.C_LANG_NAME:      'Japanese',
            self.C_HAVE_ALPHABET:  False,
            self.C_CHAR_TYPE:      self.ALPHABET_JAPANESE,
            self.C_HAVE_SYL_SEP:   True,
            self.C_SYL_SEP_TYPE:   self.T_CHAR,
            self.C_HAVE_WORD_SEP:  False,
            self.C_WORD_SEP_TYPE:  self.T_NONE,
            self.C_HAVE_VERB_CONJ: True
        }
        #
        # Cyrillic Alphabet Family
        #
        lang_index += 1
        lang_ru = {
            self.C_LANG_ID:        self.LANG_RU,
            self.C_LANG_NUMBER:    lang_index,
            self.C_LANG_NAME:      'Russian',
            self.C_HAVE_ALPHABET:  True,
            self.C_CHAR_TYPE:      self.ALPHABET_CYRILLIC,
            self.C_HAVE_SYL_SEP:   False,
            self.C_SYL_SEP_TYPE:   self.T_NONE,
            self.C_HAVE_WORD_SEP:  True,
            self.C_WORD_SEP_TYPE:  self.T_SPACE,
            self.C_HAVE_VERB_CONJ: True
        }

        #
        # Thai Alphabet Family
        #

        lang_index += 1
        lang_th = {
            self.C_LANG_ID:        self.LANG_TH,
            self.C_LANG_NUMBER:    lang_index,
            self.C_LANG_NAME:      'Thai',
            self.C_HAVE_ALPHABET:  True,
            self.C_CHAR_TYPE:      self.ALPHABET_THAI,
            self.C_HAVE_SYL_SEP:   False,
            self.C_SYL_SEP_TYPE:   self.T_NONE,
            self.C_HAVE_WORD_SEP:  False,
            self.C_WORD_SEP_TYPE:  self.T_NONE,
            self.C_HAVE_VERB_CONJ: False
        }
        #
        # Latin Alphabet Family
        #
        lang_index += 1
        lang_en = {
            self.C_LANG_ID:        self.LANG_EN,
            self.C_LANG_NUMBER:    lang_index,
            self.C_LANG_NAME:      'English',
            self.C_HAVE_ALPHABET:  True,
            self.C_CHAR_TYPE:      self.ALPHABET_LATIN_AZ,
            self.C_HAVE_SYL_SEP:   False,
            self.C_SYL_SEP_TYPE:   self.T_NONE,
            self.C_HAVE_WORD_SEP:  True,
            self.C_WORD_SEP_TYPE:  self.T_SPACE,
            self.C_HAVE_VERB_CONJ: True
        }
        lang_index += 1
        lang_es = {
            self.C_LANG_ID:        self.LANG_ES,
            self.C_LANG_NUMBER:    lang_index,
            self.C_LANG_NAME:      'Spanish',
            self.C_HAVE_ALPHABET:  True,
            self.C_CHAR_TYPE:      self.ALPHABET_LATIN,
            self.C_HAVE_SYL_SEP:   False,
            self.C_SYL_SEP_TYPE:   self.T_NONE,
            self.C_HAVE_WORD_SEP:  True,
            self.C_WORD_SEP_TYPE:  self.T_SPACE,
            self.C_HAVE_VERB_CONJ: True
        }
        lang_index += 1
        lang_fr = {
            self.C_LANG_ID:        self.LANG_FR,
            self.C_LANG_NUMBER:    lang_index,
            self.C_LANG_NAME:      'French',
            self.C_HAVE_ALPHABET:  True,
            self.C_CHAR_TYPE:      self.ALPHABET_LATIN,
            self.C_HAVE_SYL_SEP:   False,
            self.C_SYL_SEP_TYPE:   self.T_NONE,
            self.C_HAVE_WORD_SEP:  True,
            self.C_WORD_SEP_TYPE:  self.T_SPACE,
            self.C_HAVE_VERB_CONJ: True
        }
        lang_index += 1
        lang_de = {
            self.C_LANG_ID:        self.LANG_DE,
            self.C_LANG_NUMBER:    lang_index,
            self.C_LANG_NAME:      'German',
            self.C_HAVE_ALPHABET:  True,
            self.C_CHAR_TYPE:      self.ALPHABET_LATIN,
            self.C_HAVE_SYL_SEP:   False,
            self.C_SYL_SEP_TYPE:   self.T_NONE,
            self.C_HAVE_WORD_SEP:  True,
            self.C_WORD_SEP_TYPE:  self.T_SPACE,
            self.C_HAVE_VERB_CONJ: True
        }
        lang_index += 1
        lang_it = {
            self.C_LANG_ID:        self.LANG_IT,
            self.C_LANG_NUMBER:    lang_index,
            self.C_LANG_NAME:      'Italian',
            self.C_HAVE_ALPHABET:  True,
            self.C_CHAR_TYPE:      self.ALPHABET_LATIN,
            self.C_HAVE_SYL_SEP:   False,
            self.C_SYL_SEP_TYPE:   self.T_NONE,
            self.C_HAVE_WORD_SEP:  True,
            self.C_WORD_SEP_TYPE:  self.T_SPACE,
            self.C_HAVE_VERB_CONJ: True
        }
        lang_index += 1
        lang_nl = {
            self.C_LANG_ID:        self.LANG_NL,
            self.C_LANG_NUMBER:    lang_index,
            self.C_LANG_NAME:      'Dutch',
            self.C_HAVE_ALPHABET:  True,
            self.C_CHAR_TYPE:      self.ALPHABET_LATIN,
            self.C_HAVE_SYL_SEP:   False,
            self.C_SYL_SEP_TYPE:   self.T_NONE,
            self.C_HAVE_WORD_SEP:  True,
            self.C_WORD_SEP_TYPE:  self.T_SPACE,
            self.C_HAVE_VERB_CONJ: True
        }

        lang_index += 1
        lang_vi = {
            self.C_LANG_ID:        self.LANG_VI,
            self.C_LANG_NUMBER:    lang_index,
            self.C_LANG_NAME:      'Vietnamese',
            self.C_HAVE_ALPHABET:  True,
            self.C_CHAR_TYPE:      self.ALPHABET_LATIN_VI_AZ,
            self.C_HAVE_SYL_SEP:   True,
            self.C_SYL_SEP_TYPE:   self.T_SPACE,
            self.C_HAVE_WORD_SEP:  False,
            self.C_WORD_SEP_TYPE:  self.T_NONE,
            self.C_HAVE_VERB_CONJ: False
        }
        lang_index += 1
        lang_id = {
            self.C_LANG_ID:        self.LANG_ID,
            self.C_LANG_NUMBER:    lang_index,
            self.C_LANG_NAME:      'Indonesian',
            self.C_HAVE_ALPHABET:  True,
            self.C_CHAR_TYPE:      self.ALPHABET_LATIN_AZ,
            self.C_HAVE_SYL_SEP:   False,
            self.C_SYL_SEP_TYPE:   self.T_NONE,
            self.C_HAVE_WORD_SEP:  True,
            self.C_WORD_SEP_TYPE:  self.T_SPACE,
            self.C_HAVE_VERB_CONJ: True
        }

        self.langs = {
            # Hangul/CJK
            self.LANG_KO: lang_ko,
            self.LANG_JA: lang_ja,
            # CJK
            self.LANG_ZH: lang_zh,
            # Cyrillic
            self.LANG_RU: lang_ru,
            # Thai
            self.LANG_TH: lang_th,
            # Latin
            self.LANG_EN: lang_en,
            self.LANG_ES: lang_es,
            self.LANG_FR: lang_fr,
            self.LANG_DE: lang_de,
            self.LANG_IT: lang_it,
            self.LANG_NL: lang_nl,
            self.LANG_VI: lang_vi,
            self.LANG_ID: lang_id,
        }
        assert lang_index+1 == len(self.langs)

        # Add ISO 639-2 definitions
        for lang in self.langs.keys():
            if self.PYCLANG is not None:
                lang_639 = self.PYCLANG.get(alpha_2=lang)
                self.langs[lang][LangFeatures.C_LANG_639_2_ALPHA_3] = lang_639.alpha_3
                self.langs[lang][LangFeatures.C_LANG_639_2_NAME]    = lang_639.name
                self.langs[lang][LangFeatures.C_LANG_639_2_SCOPE]   = lang_639.scope
                self.langs[lang][LangFeatures.C_LANG_639_2_TYPE]    = lang_639.type
                try:
                    self.langs[lang][LangFeatures.C_LANG_639_2_ALPHA_2] = lang_639.alpha_2
                except Exception:
                    self.langs[lang][LangFeatures.C_LANG_639_2_ALPHA_2] = ''
                try:
                    self.langs[lang][LangFeatures.C_LANG_639_2_BIBLIO] = lang_639.bibliographic
                except Exception:
                    self.langs[lang][LangFeatures.C_LANG_639_2_BIBLIO] = ''
            else:
                self.langs[lang][LangFeatures.C_LANG_639_2_ALPHA_3] = ''
                self.langs[lang][LangFeatures.C_LANG_639_2_NAME]    = ''
                self.langs[lang][LangFeatures.C_LANG_639_2_SCOPE]   = ''
                self.langs[lang][LangFeatures.C_LANG_639_2_TYPE]    = ''
                self.langs[lang][LangFeatures.C_LANG_639_2_ALPHA_2] = ''
                self.langs[lang][LangFeatures.C_LANG_639_2_BIBLIO]  = ''

        # Copy 2-letter keys (ISO 639-1) to also 3-letter keys (ISO 639-3)
        # Means we can access the language structure using either ISO 639-1 or ISO 639-3
        # If engineering standard ISO had been more far-sighted (after all 26*26=676 only)
        # we would not have to do this
        new_items = {}
        for key in self.langs.keys():
            lang_iso_699_3 = self.langs[key][LangFeatures.C_LANG_639_2_ALPHA_3]
            if key != lang_iso_699_3:
                lang_dict = self.langs[key].copy()
                # Change lang id to 3-letter ISO 639-1
                lang_dict[self.C_LANG_ID] = lang_iso_699_3
                new_items[lang_iso_699_3] = lang_dict
        for lang_id3 in new_items:
            self.langs[lang_id3] = new_items[lang_id3]

        self.langfeatures = pd.DataFrame(
            self.langs.values()
        )
        # Конечно более удобно хранить данные в csv файле..
        # но проблема с путем файла и тп будет очень неприятна пользователем
        if write_lang_features_to_csv:
            self.langfeatures = self.langfeatures.sort_values(by=[self.C_LANG_NAME], ascending=True)
            self.langfeatures.to_csv('lang_features.csv', sep=',', index=False)
        return

    def __check_lang(self, lang):
        lang_std = LangFeatures.map_to_correct_lang_code_iso_639_1_or_3(
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
        return lang_dict[self.C_WORD_SEP_TYPE]

    def get_syllable_separator_type(
            self,
            lang
    ):
        lang_std = self.__check_lang(lang = lang)
        lang_dict = self.langs[lang_std]
        return lang_dict[self.C_SYL_SEP_TYPE]

    def have_verb_conjugation(
            self,
            lang
    ):
        lang_std = self.__check_lang(lang = lang)
        lang_dict = self.langs[lang_std]
        return lang_dict[self.C_HAVE_VERB_CONJ]

    #
    # Means that the smallest token is formed by the character set or alphabet.
    # For example, in English, the language token is the word, formed by latin alphabets,
    # thus the token is a set of alphabets and not the alphabet itself.
    # Same with Korean, an example token '한국어' is a word formed by Hangul alphabets or 자무
    # But Chinese (or Japanese) token is the character set itself '我在学中文', where each token
    # is the character.
    # Same thing with Thai, since it has no space at all to split syllables or words, such that
    # the smallest token is the character itself.
    #
    # В языках, есть возможности присутствии
    #    - наглядно разделяемое слово,
    #    - наглядно разделяемый слог,
    #    - или буква
    # "Токен" в нашем применении определяется как один из приведенного выше списка,
    # по порядку сверху вниз, при первом присутствии слова, слога или буквы
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
        langindexes = [ x for x in range(0,len,1) if lf[self.C_HAVE_WORD_SEP][x]==False ]
        # Second condition is that it doesn't have a syllable separator, or it has a syllable separator which is a character
        langs = [
            lf[self.C_LANG_ID][x] for x in langindexes if (
                    lf[self.C_HAVE_SYL_SEP][x]==False or
                    ( lf[self.C_HAVE_SYL_SEP][x]==True and lf[self.C_SYL_SEP_TYPE][x]==self.T_CHAR )
            )
        ]
        lang_token_same_with_charset = lang_std in langs
        return lang_token_same_with_charset

    def get_languages_with_word_separator(self):
        len = self.langfeatures.shape[0]
        langs = [ self.langfeatures[self.C_LANG_ID][x] for x in range(0,len,1)
                  if self.langfeatures[self.C_HAVE_WORD_SEP][x]==True ]
        return langs

    def get_languages_with_syllable_separator(self):
        len = self.langfeatures.shape[0]
        langs = [ self.langfeatures[self.C_LANG_ID][x] for x in range(0, len, 1)
                  if self.langfeatures[self.C_HAVE_SYL_SEP][x]==True ]
        return langs

    def get_languages_with_only_syllable_separator(self):
        return list(
            set( self.get_languages_with_syllable_separator() ) -\
            set( self.get_languages_with_word_separator() )
        )

    def get_languages_with_no_word_separator(self):
        len = self.langfeatures.shape[0]
        langs = [
            self.langfeatures[self.C_LANG_ID][x]
            for x in range(0, len, 1)
            if not self.langfeatures[self.C_HAVE_WORD_SEP][x]
        ]
        return langs

    def get_languages_for_alphabet_type(self, alphabet):
        # Exceptions for alphabets not belonging to any language (either a subset like Hiragana, etc)
        alphabet_exceptions = [self.ALPHABET_HIRAGANA_KATAKANA]
        if alphabet in alphabet_exceptions:
            if alphabet == self.ALPHABET_HIRAGANA_KATAKANA:
                return [LangFeatures.LANG_JA]

        len = self.langfeatures.shape[0]
        langs = [
            self.langfeatures[self.C_LANG_ID][x]
            for x in range(0, len, 1)
            if self.langfeatures[self.C_CHAR_TYPE][x] == alphabet
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

        have_alphabet = lang_dict[self.C_HAVE_ALPHABET]
        have_syl_sep  = lang_dict[self.C_HAVE_SYL_SEP]
        syl_sep_type  = lang_dict[self.C_SYL_SEP_TYPE]
        have_word_sep = lang_dict[self.C_HAVE_WORD_SEP]
        word_sep_type = lang_dict[self.C_WORD_SEP_TYPE]

        if level == self.LEVEL_ALPHABET:
            # If a language has alphabets, the separator is by character, otherwise return NA
            if have_alphabet:
                return ''
            else:
                return None
        elif level == self.LEVEL_SYLLABLE:
            #
            # Syllable split tokens are extremely important for
            #   1. Word list grouping into length groups, for Vietnamese for example, the word
            #      "gam en" is of length 2 (not 6), because each syllable is counted as 1
            #   2. Word segmentation, and how to reconstruct the alphabets or syllables into a word
            #
            if have_syl_sep:
                if syl_sep_type == self.T_CHAR:
                    return ''
                elif syl_sep_type == self.T_SPACE:
                    return ' '
                else:
                    return None
        elif level == self.LEVEL_UNIGRAM:
            #
            # A unigram in our definition is either a syllable or word.
            # It is the biggest unit (but not bigger than a word, not a character) that can be separated.
            # For Chinese since a syllable is also a character, the unigram is thus the syllable and exists.
            # For Thai only alphabet exists, there is no way to split syllable or word, so there is no
            # unigram in Thai.
            #
            # Return language specific word separator if exists.
            # Return language specific syllable separator if exists.
            #
            if have_word_sep:
                if word_sep_type == self.T_CHAR:
                    return ''
                elif word_sep_type == self.T_SPACE:
                    return ' '
                else:
                    return None
            elif have_syl_sep:
                if syl_sep_type == self.T_CHAR:
                    return ''
                elif syl_sep_type == self.T_SPACE:
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
        lang_index = self.langfeatures.index[self.langfeatures[self.C_LANG_ID]==lang_std].tolist()
        if len(lang_index) == 0:
            return None
        lang_index = lang_index[0]

        return self.langfeatures[self.C_CHAR_TYPE][lang_index]


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

        #
        # Syllable split tokens are extremely important for
        #   1. Word list grouping into length groups, for Vietnamese for example, the word
        #      "gam en" is of length 2 (not 6), because each syllable is counted as 1
        #   2. Word segmentation, and how to reconstruct the alphabets or syllables into a word
        #
        lang_syl_split_token = {
            LangFeatures.LANG_KO: '',
            LangFeatures.LANG_RU: None,
            LangFeatures.LANG_ZH: '',
            LangFeatures.LANG_TH: None,
            LangFeatures.LANG_EN: None,
            LangFeatures.LANG_ES: None,
            LangFeatures.LANG_FR: None,
            # Vietnamese is unique splitting by space
            LangFeatures.LANG_VI: ' ',
            LangFeatures.LANG_ID: None,
        }
        for lang in lang_syl_split_token:
            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                observed     = lf.get_split_token(
                    lang  = lang,
                    level = LangFeatures.LEVEL_SYLLABLE
                ),
                expected     = lang_syl_split_token[lang],
                test_comment = 'test lang ' + str(lang) + ' syllable split token'
            ))

        lang_unigram_split_token = {
            LangFeatures.LANG_KO: ' ',
            LangFeatures.LANG_RU: ' ',
            # No word separator, will return the syllable separator
            LangFeatures.LANG_ZH: '',
            LangFeatures.LANG_TH: None,
            LangFeatures.LANG_EN: ' ',
            LangFeatures.LANG_ES: ' ',
            LangFeatures.LANG_FR: ' ',
            # Vietnamese is unique splitting by space
            # No word separator, will return the syllable separator
            LangFeatures.LANG_VI: ' ',
            LangFeatures.LANG_ID: ' ',
        }
        for lang in lang_unigram_split_token:
            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                observed     = lf.get_split_token(
                    lang  = lang,
                    level = LangFeatures.LEVEL_UNIGRAM
                ),
                expected     = lang_unigram_split_token[lang],
                test_comment = 'test lang ' + str(lang) + ' unigram split token'
            ))

        observed = lf.get_languages_with_word_separator()
        observed.sort()
        observed_639_1 = [code for code in observed if len(code)==2]
        observed_639_3 = [code for code in observed if len(code)==3]
        expected = [
            LangFeatures.LANG_KO, LangFeatures.LANG_KOR, LangFeatures.LANG_RU, LangFeatures.LANG_RUS,
            LangFeatures.LANG_EN, LangFeatures.LANG_ENG, LangFeatures.LANG_ES, LangFeatures.LANG_SPA,
            LangFeatures.LANG_FR, LangFeatures.LANG_FRA, LangFeatures.LANG_IT, LangFeatures.LANG_ITA,
            LangFeatures.LANG_NL, LangFeatures.LANG_NLD, LangFeatures.LANG_DE, LangFeatures.LANG_DEU,
            LangFeatures.LANG_ID, LangFeatures.LANG_IND,
        ]
        expected.sort()
        expected_639_1 = [code for code in expected if len(code)==2]
        expected_639_3 = [code for code in expected if len(code)==3]

        res_final.update_bool(res_bool=ut.UnitTest.assert_true(
            observed = observed_639_1,
            expected = expected_639_1,
            test_comment = 'test languages with word separator'
        ))
        if lf.PYCLANG:
            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = observed_639_3,
                expected = expected_639_3,
                test_comment = 'test languages with word separator'
            ))

        observed = lf.get_languages_with_syllable_separator()
        observed.sort()
        observed_639_1 = [code for code in observed if len(code)==2]
        observed_639_3 = [code for code in observed if len(code)==3]
        expected = [
            LangFeatures.LANG_ZH, LangFeatures.LANG_ZHO, LangFeatures.LANG_KO, LangFeatures.LANG_KOR,
            LangFeatures.LANG_JA, LangFeatures.LANG_JPN, LangFeatures.LANG_VI, LangFeatures.LANG_VIE,
        ]
        expected.sort()
        expected_639_1 = [code for code in expected if len(code)==2]
        expected_639_3 = [code for code in expected if len(code)==3]

        res_final.update_bool(res_bool=ut.UnitTest.assert_true(
            observed = observed_639_1,
            expected = expected_639_1,
            test_comment = 'test languages with syllable separator'
        ))
        if lf.PYCLANG:
            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = observed_639_3,
                expected = expected_639_3,
                test_comment = 'test languages with syllable separator'
            ))

        observed = lf.get_languages_with_no_word_separator()
        observed.sort()
        observed_639_1 = [code for code in observed if len(code)==2]
        observed_639_3 = [code for code in observed if len(code)==3]
        expected = [
            LangFeatures.LANG_ZH, LangFeatures.LANG_ZHO, LangFeatures.LANG_JA, LangFeatures.LANG_JPN,
            LangFeatures.LANG_TH, LangFeatures.LANG_THA, LangFeatures.LANG_VI, LangFeatures.LANG_VIE,
        ]
        expected.sort()
        expected_639_1 = [code for code in expected if len(code)==2]
        expected_639_3 = [code for code in expected if len(code)==3]

        res_final.update_bool(res_bool=ut.UnitTest.assert_true(
            observed = observed_639_1,
            expected = expected_639_1,
            test_comment = 'test languages with no word separator'
        ))
        if lf.PYCLANG:
            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = observed_639_3,
                expected = expected_639_3,
                test_comment = 'test languages with no word separator'
            ))

        observed = lf.get_languages_with_only_syllable_separator()
        observed.sort()
        observed_639_1 = [code for code in observed if len(code)==2]
        observed_639_3 = [code for code in observed if len(code)==3]
        expected = [
            LangFeatures.LANG_ZH, LangFeatures.LANG_ZHO, LangFeatures.LANG_JA, LangFeatures.LANG_JPN,
            LangFeatures.LANG_VI, LangFeatures.LANG_VIE,
        ]
        expected.sort()
        expected_639_1 = [code for code in expected if len(code)==2]
        expected_639_3 = [code for code in expected if len(code)==3]

        res_final.update_bool(res_bool=ut.UnitTest.assert_true(
            observed = observed_639_1,
            expected = expected_639_1,
            test_comment = 'test languages with ONLY syllable separator'
        ))
        if lf.PYCLANG:
            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = observed_639_3,
                expected = expected_639_3,
                test_comment = 'test languages with ONLY syllable separator'
            ))

        # We could get the languages associated with the alphabet programmatically also,
        # but we do that in the second round
        alphabet_langs = {
            LangFeatures.ALPHABET_HANGUL:   [LangFeatures.LANG_KO, LangFeatures.LANG_KOR],
            LangFeatures.ALPHABET_THAI:     [LangFeatures.LANG_TH, LangFeatures.LANG_THA],
            LangFeatures.ALPHABET_CYRILLIC: [LangFeatures.LANG_RU, LangFeatures.LANG_RUS],
            LangFeatures.ALPHABET_CJK:      [LangFeatures.LANG_ZH, LangFeatures.LANG_ZHO],
            LangFeatures.ALPHABET_LATIN_AZ: [
                LangFeatures.LANG_EN, LangFeatures.LANG_ENG, LangFeatures.LANG_ID, LangFeatures.LANG_IND,
            ],
            LangFeatures.ALPHABET_LATIN_VI_AZ: [LangFeatures.LANG_VI, LangFeatures.LANG_VIE],
            LangFeatures.ALPHABET_LATIN:    [
                LangFeatures.LANG_ES, LangFeatures.LANG_SPA, LangFeatures.LANG_FR, LangFeatures.LANG_FRA,
                LangFeatures.LANG_DE, LangFeatures.LANG_DEU, LangFeatures.LANG_IT, LangFeatures.LANG_ITA,
                LangFeatures.LANG_NL, LangFeatures.LANG_NLD,
            ]
        }
        for alp in alphabet_langs.keys():
            observed = lf.get_languages_for_alphabet_type(alphabet=alp)
            observed.sort()
            observed_639_1 = [code for code in observed if len(code) == 2]
            observed_639_3 = [code for code in observed if len(code) == 3]
            expected = alphabet_langs[alp]
            expected.sort()
            expected_639_1 = [code for code in expected if len(code) == 2]
            expected_639_3 = [code for code in expected if len(code) == 3]

            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = observed_639_1,
                expected = expected_639_1,
                test_comment = 'R1 test languages for alphabet "' + str(alp) + '"'
            ))
            if lf.PYCLANG:
                res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                    observed = observed_639_3,
                    expected = expected_639_3,
                    test_comment='R1 test languages for alphabet "' + str(alp) + '"'
                ))

        #
        # In this round we get the languages for an alphabet programmatically
        #
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

        observed = sorted(langs_with_token_same_as_charset)
        observed_639_1 = [code for code in observed if len(code) == 2]
        observed_639_3 = [code for code in observed if len(code)==3]

        expected = sorted([
                LangFeatures.LANG_ZH, LangFeatures.LANG_ZHO, LangFeatures.LANG_JA, LangFeatures.LANG_JPN,
                LangFeatures.LANG_TH, LangFeatures.LANG_THA,
            ])
        expected_639_1 = [code for code in expected if len(code) == 2]
        expected_639_3 = [code for code in expected if len(code) == 3]
        res_final.update_bool(res_bool=ut.UnitTest.assert_true(
            observed     = observed_639_1,
            expected     = expected_639_1,
            test_comment = 'Test langs with token = charset'
        ))
        if lf.PYCLANG:
            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = observed_639_3,
                expected = expected_639_3,
                test_comment = 'Test langs with token = charset'
            ))

        return res_final


if __name__ == '__main__':
    Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1

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




