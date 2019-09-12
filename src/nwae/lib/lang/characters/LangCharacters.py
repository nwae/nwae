# -*- coding: utf-8 -*-

import nwae.lib.lang.LangFeatures as lf


#
# Class LangCharacters:
#   This class lays the fundamentals for dealing with characters & strings of multiple languages.
#   We define Unicode blocks for the relevant language characters, including punctuations, etc.
#   Every alphabet or character has a Unicode value (max value is 2^32)
#
#   But when required to store as a string variable, it has to undergo a transformation into say
#   UTF-8. This is purely for compression so we don't store each character as 4 bytes.
#   chr() converts a Unicode value to a Unicode string, e.g. the Unicode value 0x9a6c or 39532
#   is converted to '马' (either stored as UTF-8 or some encoding).
#
#   Another difference with R is that in Python, we always need to convert strings to Unicode form
#   for the above functions to work. In R this is handled transparently.
#
#   The Python function ord() does the opposite, converts '马' back to it's integer Unicode value.
#
# Supports:
#   1. Latin alphabets (English, Vietnamese, Indonesian).
#   2. CJK characters (Chinese). CJK characters are base for Chinese, Korean, Japanese.
#   3. Hangul alphabets & syllables. Hangul syllables have their own unique Unicodes.
#   4. Thai alphabets, numbers, punctuations.
#
# Character Encodings References:
#   https://unicode-table.com/
#   http://jrgraphix.net/research/unicode_blocks.php
#
class LangCharacters(object):

    encoding = 'utf-8'

    def __init__(
            self,
            encoding='utf-8'
    ):
        self.encoding = encoding
        return

    #
    # Converts a unicode type list to a string
    #
    @staticmethod
    def convert_unicode_list_to_string(
            unicode_arr
    ):
        str = ""
        for i in range(0, len(unicode_arr), 1):
            # Handle escape character
            if unicode_arr[i]=='\\':
                str = str + '\\'
            else:
                str = str + unicode_arr[i]
        return str

    #
    # Latin
    #

    # Latin Unicode Block as 'int' list
    UNICODE_BLOCK_ORDINAL_LATIN_BASIC = list( range(0x0041, 0x005A+1, 1) ) +\
                                        list( range(0x0061, 0x007A+1, 1) )
    # Convert to Python Unicode Type list
    UNICODE_BLOCK_LATIN_BASIC = [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_LATIN_BASIC]

    # Latin Extended
    UNICODE_BLOCK_ORDINAL_LATIN_EXTENDED = list( range(0x00C0, 0x00F6+1, 1) ) +\
                                           list( range(0x00F8, 0x01BF+1, 1) ) +\
                                           list( range(0x01C4, 0x024F+1, 1) )
    UNICODE_BLOCK_LATIN_EXTENDED = [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_LATIN_EXTENDED]

    # All Latin
    UNICODE_BLOCK_ORDINAL_LATIN_ALL = UNICODE_BLOCK_ORDINAL_LATIN_BASIC + UNICODE_BLOCK_ORDINAL_LATIN_EXTENDED
    UNICODE_BLOCK_LATIN_ALL = UNICODE_BLOCK_LATIN_BASIC + UNICODE_BLOCK_LATIN_EXTENDED

    # Just Latin specific to Vietnamese
    UNICODE_BLOCK_LATIN_VIETNAMESE =\
        list(u'ăâàằầảẳẩãẵẫáắấạặậêèềẻểẽễéếẹệìỉĩíịôơòồờỏổởõỗỡóốớọộợưùừủửũữúứụựđ')

    #
    # CJK
    #
    UNICODE_BLOCK_ORDINAL_CJK_UNIFIED_IDEOGRAPHS = list( range(0x4E00, 0x9FFF+1, 1) )
    UNICODE_BLOCK_CJK_UNIFIED_IDEOGRAPHS =\
        [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_CJK_UNIFIED_IDEOGRAPHS]

    UNICODE_BLOCK_ORDINAL_CJK_UNIFIED_IDEOGRAPHS_EXT_A = list( range(0x3400, 0x4DBF+1, 1) )
    UNICODE_BLOCK_CJK_UNIFIED_IDEOGRAPHS_EXT_A =\
        [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_CJK_UNIFIED_IDEOGRAPHS_EXT_A]

    UNICODE_BLOCK_ORDINAL_CJK_UNIFIED_IDEOGRAPHS_EXT_B = list( range(0x20000, 0x2A6DF+1, 1) )
    UNICODE_BLOCK_CJK_UNIFIED_IDEOGRAPHS_EXT_B =\
        [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_CJK_UNIFIED_IDEOGRAPHS_EXT_B]

    UNICODE_BLOCK_ORDINAL_CJK_UNIFIED_IDEOGRAPHS_EXT_C = list( range(0x2A700, 0x2B73F+1, 1) )
    UNICODE_BLOCK_CJK_UNIFIED_IDEOGRAPHS_EXT_C =\
        [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_CJK_UNIFIED_IDEOGRAPHS_EXT_C]

    UNICODE_BLOCK_ORDINAL_CJK_UNIFIED_IDEOGRAPHS_EXT_D = list( range(0x2B740, 0x2B81F+1, 1) )
    UNICODE_BLOCK_CJK_UNIFIED_IDEOGRAPHS_EXT_D =\
        [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_CJK_UNIFIED_IDEOGRAPHS_EXT_D]

    UNICODE_BLOCK_ORDINAL_CJK_UNIFIED_IDEOGRAPHS_EXT_E = list( range(0x2B820, 0x2CEAF+1, 1) )
    UNICODE_BLOCK_CJK_UNIFIED_IDEOGRAPHS_EXT_E = \
        [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_CJK_UNIFIED_IDEOGRAPHS_EXT_E]

    UNICODE_BLOCK_ORDINAL_CJK_COMPATIBILITY_IDEOGRAPHS = list( range(0xF900, 0xFAFF+1, 1) )
    UNICODE_BLOCK_CJK_COMPATIBILITY_IDEOGRAPHS = \
        [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_CJK_COMPATIBILITY_IDEOGRAPHS]

    UNICODE_BLOCK_ORDINAL_CJK_COMPATIBILITY_IDEOGRAPHS_SUPP = list( range(0x2F800, 0x2FA1F+1, 1) )
    UNICODE_BLOCK_CJK_COMPATIBILITY_IDEOGRAPHS_SUPP = \
        [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_CJK_COMPATIBILITY_IDEOGRAPHS_SUPP]

    UNICODE_BLOCK_CJK = UNICODE_BLOCK_CJK_UNIFIED_IDEOGRAPHS + UNICODE_BLOCK_CJK_UNIFIED_IDEOGRAPHS_EXT_A +\
                        UNICODE_BLOCK_CJK_COMPATIBILITY_IDEOGRAPHS +\
                        UNICODE_BLOCK_CJK_UNIFIED_IDEOGRAPHS_EXT_B + UNICODE_BLOCK_CJK_UNIFIED_IDEOGRAPHS_EXT_C +\
                        UNICODE_BLOCK_CJK_UNIFIED_IDEOGRAPHS_EXT_D + UNICODE_BLOCK_CJK_UNIFIED_IDEOGRAPHS_EXT_D +\
                        UNICODE_BLOCK_CJK_UNIFIED_IDEOGRAPHS_EXT_E +\
                        UNICODE_BLOCK_CJK_COMPATIBILITY_IDEOGRAPHS_SUPP

    #
    # Hangul
    #
    UNICODE_BLOCK_ORDINAL_HANGUL = list( range(0x1100, 0x11FF+1, 1) )
    UNICODE_BLOCK_HANGUL = [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_HANGUL]

    UNICODE_BLOCK_ORDINAL_HANGUL_SYLLABLE = list( range(0xAC00, 0xD7AF+1, 1) )
    UNICODE_BLOCK_HANGUL_SYLLABLE = [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_HANGUL_SYLLABLE]

    UNICODE_BLOCK_HANGUL_ALL_INCLUDING_SYLLABLE = UNICODE_BLOCK_HANGUL + UNICODE_BLOCK_HANGUL_SYLLABLE

    #
    # Thai
    # From http://sites.psu.edu/symbolcodes/languages/asia/thai/thaichart/
    #
    UNICODE_BLOCK_ORDINAL_THAI_CONSONANTS = list( range(0x0E01, 0x0E2E+1, 1) )
    UNICODE_BLOCK_THAI_CONSONANTS = [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_THAI_CONSONANTS]

    UNICODE_BLOCK_ORDINAL_THAI_VOWELS_AFTER_CONSONANT = list( range(0x0E30, 0x0E3A+1, 1) ) +\
                                                        list( range(0x0E47, 0x0E47+1, 1) )
    UNICODE_BLOCK_THAI_VOWELS_AFTER_CONSONANT =\
        [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_THAI_VOWELS_AFTER_CONSONANT]

    UNICODE_BLOCK_ORDINAL_THAI_VOWELS_BEFORE_CONSONANT = list( range(0x0E40, 0x0E44+1, 1) )
    UNICODE_BLOCK_THAI_VOWELS_BEFORE_CONSONANT = \
        [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_THAI_VOWELS_BEFORE_CONSONANT]

    UNICODE_BLOCK_ORDINAL_THAI_TONEMARKS = list( range(0x0E48, 0x0E4B+1, 1) )
    UNICODE_BLOCK_THAI_TONEMARKS = \
        [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_THAI_TONEMARKS]

    UNICODE_BLOCK_ORDINAL_THAI_NUMBERS = list( range(0x0E50, 0x0E59+1, 1) )
    UNICODE_BLOCK_THAI_NUMBERS =\
        [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_THAI_NUMBERS]

    UNICODE_BLOCK_ORDINAL_THAI_SIGNS_PUNCTUATIONS = list( range(0x0E2F, 0x0E2F+1, 1) ) +\
                                                    list( range(0x0E45, 0x0E46+1, 1) ) +\
                                                    list( range(0x0E4C, 0x0E4F+1, 1) ) +\
                                                    list( range(0x0E5A, 0x0E5B+1, 1) )
    UNICODE_BLOCK_THAI_SIGNS_PUNCTUATIONS = \
        [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_THAI_SIGNS_PUNCTUATIONS]

    UNICODE_BLOCK_THAI = UNICODE_BLOCK_THAI_CONSONANTS +\
                         UNICODE_BLOCK_THAI_VOWELS_AFTER_CONSONANT +\
                         UNICODE_BLOCK_THAI_VOWELS_BEFORE_CONSONANT +\
                         UNICODE_BLOCK_THAI_TONEMARKS +\
                         UNICODE_BLOCK_THAI_NUMBERS +\
                         UNICODE_BLOCK_THAI_SIGNS_PUNCTUATIONS

    #
    # Punctuations, etc.
    #
    UNICODE_BLOCK_WORD_SEPARATORS =\
        list(u' ,!.?') + list( chr(0xFF0C) + chr(0xFF01) + chr(0xFF0E) + chr(0xFF1F) )

    UNICODE_BLOCK_SENTENCE_SEPARATORS =\
        list(u' !.?') + list( chr(0xFF01) + chr(0xFF0E) + chr(0xFF1F) )
    #
    # Numbers: normal Latin and CJK halfwidth/fullwidth
    #
    UNICODE_BLOCK_ORDINAL_NUMBERS = list( range(0x0030, 0x0039+1, 1) ) + list( range(0xFF10, 0xFF19+1, 1) )
    UNICODE_BLOCK_NUMBERS =\
        [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_NUMBERS]

    #
    # Punctuations Only (Half-Width & Full-Width Forms)
    #
    UNICODE_BLOCK_ORDINAL_PUNCTUATIONS = list(range(0x0000, 0x007F+1, 1)) +\
                                         list(range(0x2000, 0x206F+1, 1)) +\
                                         list(range(0x3000, 0x303F+1, 1)) +\
                                         list(range(0xFF00, 0xFF0F+1, 1)) +\
                                         list(range(0xFF1A, 0xFF20+1, 1)) +\
                                         list(range(0xFF3B, 0xFF40+1, 1)) +\
                                         list(range(0xFF5B, 0xFF65+1, 1))
    UNICODE_BLOCK_PUNCTUATIONS =\
        [chr(ordinal) for ordinal in UNICODE_BLOCK_ORDINAL_PUNCTUATIONS]
    # Remove non-punctuations from original list of punctuations
    UNICODE_BLOCK_PUNCTUATIONS = list( set(UNICODE_BLOCK_PUNCTUATIONS) - set(UNICODE_BLOCK_LATIN_ALL) )
    UNICODE_BLOCK_PUNCTUATIONS = list( set(UNICODE_BLOCK_PUNCTUATIONS) - set(UNICODE_BLOCK_WORD_SEPARATORS) )
    UNICODE_BLOCK_PUNCTUATIONS = list( set(UNICODE_BLOCK_PUNCTUATIONS) - set(UNICODE_BLOCK_SENTENCE_SEPARATORS) )
    UNICODE_BLOCK_PUNCTUATIONS = list( set(UNICODE_BLOCK_PUNCTUATIONS) - set(UNICODE_BLOCK_NUMBERS) )

    #
    # Get the valid Unicode Block for a given language
    #
    @staticmethod
    def get_language_charset(lang):
        if lang in [
            lf.LangFeatures.LANG_EN,
            lf.LangFeatures.LANG_VN
        ]:
            return LangCharacters.UNICODE_BLOCK_LATIN_ALL
        if lang == lf.LangFeatures.LANG_CN:
            return LangCharacters.UNICODE_BLOCK_CJK
        if lang == lf.LangFeatures.LANG_TH:
            return LangCharacters.UNICODE_BLOCK_THAI
        if lang == lf.LangFeatures.LANG_KO:
            return LangCharacters.UNICODE_BLOCK_HANGUL_ALL_INCLUDING_SYLLABLE
        else:
            return []

    #
    # Converts a string into a single number for various purposes when dealing with numbers are more convenient.
    # This single number is not necessarily unique.
    #
    def convert_string_to_number(self, s, verbose=0):

        lang = None
        syllable_end = [False]*len(s)

        if s[0] in LangCharacters.UNICODE_BLOCK_THAI and len(s)>1:
            # For Thai, we don't calculate the last syllable character, since that character is highly prone
            # to error. E.g. ส-วัด-ดี (สวัสดี) or ปัน-หา (ปัญหา). This is also our method of Thai spelling correction.
            # Characters that can never be start of syllable
            not_start_syllable_char = LangCharacters.UNICODE_BLOCK_THAI_VOWELS_AFTER_CONSONANT +\
                                      LangCharacters.UNICODE_BLOCK_THAI_TONEMARKS
            lang = lf.LangFeatures.LANG_TH
            char_prev = s[0]
            for i in range(1, len(s)-1, 1):
                char_prev = s[i-1]
                char_cur = s[i]

                # This character can never be start of syllable
                if char_cur not in not_start_syllable_char:
                    continue

                char_next = s[i+1]
                # Case of 'เดือน', 'เมื่อ', 'เลข', etc.
                if char_cur in LangCharacters.UNICODE_BLOCK_THAI_VOWELS_BEFORE_CONSONANT:
                    syllable_end[i-1] = True
                elif char_cur in LangCharacters.UNICODE_BLOCK_THAI_CONSONANTS:
                    # Case of 'การ', 'เดือน', 'ดารา' etc.
                    if ( char_next in LangCharacters.UNICODE_BLOCK_THAI_VOWELS_AFTER_CONSONANT ) and\
                        ( char_prev not in LangCharacters.UNICODE_BLOCK_THAI_VOWELS_BEFORE_CONSONANT):
                        syllable_end[i-1] = True
                    # Case of 'งง', 'สด', etc.
                    #elif ( char_prev in LangCharacters.UNICODE_BLOCK_THAI_TONEMARKS ):
                    #    syllable_end[i-1] = True

            # Last character is always end of syllable
            syllable_end[len(s) - 1] = True

            if verbose >= 1:
                sylsepstring = ''
                for i in range(0, len(s), 1):
                    sylsepstring = sylsepstring + s[i]
                    if syllable_end[i]:
                        sylsepstring = sylsepstring + ' '
                print(sylsepstring)

        x = 0
        index = 1
        # A string "abc" will be calculated as (97 + 2*98 + 3*99), Unicode for 'a' is 97, 'b' is 98, 'c' is 99
        for i in range(0, len(s), 1):
            # We don't include a syllable ending consonant for Thai in the measure, since this character is prone
            # to spelling mistakes
            ignore = False
            if lang == lf.LangFeatures.LANG_TH:
                if s[i] in LangCharacters.UNICODE_BLOCK_THAI_CONSONANTS and syllable_end[i]:
                    # print('Ignore ' + s[i])
                    ignore = True
            if not ignore:
                un = ord(s[i])
                # print('Index ' + str(index) + ', ' + s[i] + ', ' + str(un))
                x = x + index*un
                index = index + 1

        return x

    #
    # Given a string with allowed Unicode block, returns a string with only the allowed Unicode values
    #
    def filter_allowed_characters(self, unicode_list, s):
        # Must convert string to unicode string
        #if type(s) != unicode:
        #s = unicode(s, encoding = self.encoding)

        str_new = u''
        for i in range(0, len(s), 1):
            ch = s[i]
            if ch in unicode_list:
                str_new += ch
        return str_new

    #
    # This function returns whether the written language is normal Vietnamese (a mix of basic and extended Latin)
    # or purely using basic Latin (it is cultural of Vietnamese to leave out all the diacritics and use purely basic
    # Latin)
    #
    def get_vietnamese_type(self, s):
        # Must convert string to unicode string
        #if type(s) != unicode:
        #s = unicode(s, encoding=self.encoding)

        # First we remove the punctuations, numbers, etc.
        remove_block = LangCharacters.UNICODE_BLOCK_PUNCTUATIONS + LangCharacters.UNICODE_BLOCK_NUMBERS + \
                       LangCharacters.UNICODE_BLOCK_WORD_SEPARATORS + LangCharacters.UNICODE_BLOCK_SENTENCE_SEPARATORS
        ss = u''
        for i in range(0, len(s), 1):
            if s[i] not in remove_block:
                ss = ss + s[i]

        is_latin_basic_count = 0
        is_latin_extended_viet_count = 0
        for i in range(0, len(ss), 1):
            latin_basic = ss[i] in LangCharacters.UNICODE_BLOCK_LATIN_BASIC
            latin_extended = ss[i] in LangCharacters.UNICODE_BLOCK_LATIN_EXTENDED
            # print ( ss[i] + " Latin Basic = " + str(latin_basic) + ", Latin Extended = " + str(latin_extended) )
            is_latin_basic_count += 1 * latin_basic
            is_latin_extended_viet_count += 1 * latin_extended

        latin_basic_percent = float( is_latin_basic_count / float(len(ss)) )
        latin_extended_viet_percent = float( is_latin_extended_viet_count / float(len(ss)) )

        if (latin_basic_percent + latin_extended_viet_percent) > 0.5:
            if latin_basic_percent > 0.98:
                return "latin.basic"
            else:
                if latin_extended_viet_percent > 0.1:
                    return "latin.viet"
                else:
                    return "latin.mix"
        else:
            return "other"


if __name__ == '__main__':
    #
    # Check UNICODE BLOCKS
    #
    print('****************************** LATIN ALL ('
          + str(len(LangCharacters.UNICODE_BLOCK_LATIN_ALL))
          + ') BLOCK ******************************')
    print(LangCharacters.UNICODE_BLOCK_LATIN_ALL)

    print('****************************** HANGUL ('
          + str(len(LangCharacters.UNICODE_BLOCK_HANGUL))
          + ') BLOCK ******************************')
    print(LangCharacters.UNICODE_BLOCK_HANGUL)
    print('****************************** HANGUL SYLLABLE BLOCK ('
          + str(len(LangCharacters.UNICODE_BLOCK_HANGUL_SYLLABLE))
          + ') ******************************')
    print(LangCharacters.UNICODE_BLOCK_HANGUL_SYLLABLE)
    print('****************************** HANGUL + SYLLABLE BLOCK ('
          + str(len(LangCharacters.UNICODE_BLOCK_HANGUL_ALL_INCLUDING_SYLLABLE))
          + ') ******************************')
    print(LangCharacters.UNICODE_BLOCK_HANGUL_ALL_INCLUDING_SYLLABLE)

    print('****************************** THAI BLOCK ('
          + str(len(LangCharacters.UNICODE_BLOCK_THAI))
          + ') ******************************')
    print(LangCharacters.UNICODE_BLOCK_THAI)
    print('****************************** THAI CONSONANTS BLOCK ('
          + str(len(LangCharacters.UNICODE_BLOCK_THAI_CONSONANTS))
          + ') ******************************')
    print(LangCharacters.UNICODE_BLOCK_THAI_CONSONANTS)
    print('****************************** THAI NUMBERS BLOCK ('
          + str(len(LangCharacters.UNICODE_BLOCK_THAI_NUMBERS))
          + ') ******************************')
    print(LangCharacters.UNICODE_BLOCK_THAI_NUMBERS)
    print('****************************** THAI TONEMARKS BLOCK ('
          + str(len(LangCharacters.UNICODE_BLOCK_THAI_TONEMARKS))
          + ') ******************************')
    print(LangCharacters.UNICODE_BLOCK_THAI_TONEMARKS)
    print('****************************** THAI VOWELS BEFORE CONSONANT BLOCK ('
          + str(len(LangCharacters.UNICODE_BLOCK_THAI_VOWELS_BEFORE_CONSONANT))
          + ') ******************************')
    print(LangCharacters.UNICODE_BLOCK_THAI_VOWELS_BEFORE_CONSONANT)
    print('****************************** THAI VOWELS AFTER CONSONANT BLOCK ('
          + str(len(LangCharacters.UNICODE_BLOCK_THAI_VOWELS_AFTER_CONSONANT))
          + ') ******************************')
    print(LangCharacters.UNICODE_BLOCK_THAI_VOWELS_AFTER_CONSONANT)

    print('****************************** CJK BLOCK ('
          + str(len(LangCharacters.UNICODE_BLOCK_CJK))
          + ') ******************************')
    print(LangCharacters.UNICODE_BLOCK_CJK)

    print('****************************** VIETNAMESE BLOCK ('
          + str(len(LangCharacters.UNICODE_BLOCK_LATIN_VIETNAMESE))
          + ') ******************************')
    print(LangCharacters.UNICODE_BLOCK_LATIN_VIETNAMESE)

    print('****************************** PUNCTUATIONS BLOCK ('
          + str(len(LangCharacters.UNICODE_BLOCK_PUNCTUATIONS))
          + ') ******************************')
    print(LangCharacters.UNICODE_BLOCK_PUNCTUATIONS)

    s = u'북핵 英国前首мешание相卡梅伦（David Cameron）将要接受一个新的война官方工作，领导一个关于英中合สวยจีง作的英国政府投资计划。'
    print ( s )
    lc = LangCharacters()

    print( lc.filter_allowed_characters(LangCharacters.UNICODE_BLOCK_CJK +
                                       LangCharacters.UNICODE_BLOCK_LATIN_BASIC +
                                       LangCharacters.UNICODE_BLOCK_WORD_SEPARATORS +
                                       LangCharacters.UNICODE_BLOCK_PUNCTUATIONS
                                       , s)
    )

    s = 'สวัสดีการเดือนงงดารา'
    print(lc.convert_string_to_number(s=s, verbose=1))

    s = 'ผู้ว่าฯ ควรเร่งแก้ไขปัญหาจราจรในกรุงเทพฯ และเพิ่มประสิทธิภาพการขนส่งมวลชน'
    print(lc.convert_string_to_number(s=s, verbose=1))

    words = ['สวัสดี', 'สวัดดี', 'วัดดี']
    for w in words:
        print(lc.convert_string_to_number(s=w, verbose=1))
