#!/use/bin/python
# --*-- coding: utf-8 --*--

# !!! Will work only on Python 3 and above

import pandas as pd
# from iso639 import languages


#
# Class LangFeatures
#
#   Helper class to define language properties, such as containing word/syllable separators,
#   alphabet type, etc.
#   Also helps to grab word lists, dictionaries, stop words, of a language.
#
class LangFeatures:

    #
    # TODO
    #  Move to use ISO 639-2 standard instead of our own
    #
    LANG_EN = 'en'
    LANG_CN = 'cn'
    LANG_ZH_CN = 'zh-cn'
    LANG_TH = 'th'
    LANG_VN = 'vn'
    LANG_IN = 'in'
    LANG_KO = 'ko'
    LANG_FR = 'fr'

    ERROR_TOLERANCE = 0.000000000001

    @staticmethod
    def map_to_correct_lang_code(
            lang_code
    ):
        if lang_code == LangFeatures.LANG_CN:
            return LangFeatures.LANG_ZH_CN
        else:
            return lang_code

    # Word lists and stopwords are in the same folder
    def __init__(self):
        #
        # Language followed by flag for alphabet boundary, syllable boundary (either as one
        # character as in Chinese or space as in Korean), then word boundary (space)
        # The most NLP-inconvenient languages are those without word boundary, obviously.
        # Name, Code, Alphabet, CharacterType, SyllableSeparator, SyllableSeparatorType, WordSeparator, WordSeparatorType
        #
        self.langfeatures = pd.DataFrame({
            'Language':        [ 'cn'       , 'th'  , 'vn'        , 'in'        , 'ko'        ],
            'LanguageName':    [ 'Chinese'  , 'Thai', 'Vietnamese', 'Indonesian', 'Hangul'    ],
            'Alphabet':        [ False      , True  , True        , True        , True        ],
            'CharacterType':   [ 'cn'       , 'th'  , 'latin'     , 'latin'     , 'ko'        ],
            'SyllableSep':     [ True       , False , True        , False       , True        ],
            'SyllableSepType': [ 'character', ''    , 'space'     , ''          , 'character' ],
            'WordSep':         [ False      , False , False       , True        , True        ],
            'WordSepType':     [ ''         , ''    , ''          , 'space'     , 'space'     ]
        })

        return

    def is_lang_token_same_with_charset(self, lang):
        # Languages that have the tokens as the character set, or languages with no syllable or unigram separator
        # Besides cn/th, the same goes for Lao, Cambodian, Japanese, with no spaces to separate syllables/unigrams.
        lf = self.langfeatures
        len = lf.shape[0]
        # First it must not have a word separator
        langindexes = [ x for x in range(0,len,1) if lf['WordSep'][x]==False ]
        # Second condition is that it doesn't have a syllable separator, or it has a syllable separator which is a character
        langs = [ lf['Language'][x] for x in langindexes if
                  ( lf['SyllableSep'][x]==False or (lf['SyllableSep'][x]==True and lf['SyllableSepType'][x]=='character') ) ]
        return lang in langs

    def get_languages_with_word_separator(self):
        len = self.langfeatures.shape[0]
        langs = [ self.langfeatures['Language'][x] for x in range(0,len,1) if self.langfeatures['WordSep'][x]==True ]
        return langs

    def get_languages_with_syllable_separator(self):
        len = self.langfeatures.shape[0]
        langs = [ self.langfeatures['Language'][x] for x in range(0, len, 1) if self.langfeatures['SyllableSep'][x]==True ]
        return langs

    def get_languages_with_only_syllable_separator(self):
        return list(
            set( self.get_languages_with_syllable_separator() ) -\
            set( self.get_languages_with_word_separator() )
        )

    #
    # If separator for either alphabet/syllable/word (we shall refer as token) is None, this means there is no
    # way to identify the token. If the separator is '', means we can identify it by character (e.g. Chinese character,
    # Thai alphabet, Korean alphabet inside a Korean character/syllable).
    #
    def get_split_token(self, lang, level):
        # Language index
        lang_index = self.langfeatures.index[self.langfeatures['Language']==lang].tolist()
        if len(lang_index) == 0:
            return None
        lang_index = lang_index[0]

        have_alphabet = self.langfeatures['Alphabet'][lang_index]
        syl_sep = self.langfeatures['SyllableSep'][lang_index]
        syl_sep_type = self.langfeatures['SyllableSepType'][lang_index]
        word_sep = self.langfeatures['WordSep'][lang_index]
        word_sep_type = self.langfeatures['WordSepType'][lang_index]

        if level == 'alphabet':
            # If a language has alphabets, the separator is by character, otherwise return NA
            if have_alphabet:
                return u''
            else:
                return None
        elif level == 'syllable':
            if syl_sep == True:
                if syl_sep_type == 'character':
                    return u''
                elif syl_sep_type == 'space':
                    return u' '
                else:
                    return None
        elif level == 'unigram':
            # Return language specific word separator if exists.
            # Return language specific syllable separator if exists.
            if word_sep == True:
                if word_sep_type == 'character':
                    return u''
                elif word_sep_type == 'space':
                    return u' '
                else:
                    return None
            elif syl_sep == True:
                if syl_sep == True:
                    if syl_sep_type == 'character':
                        return u''
                    elif syl_sep_type == 'space':
                        return u' '
                    else:
                        return None

        return None

    def get_alphabet_type(self, lang):
        # Language index
        lang_index = self.langfeatures.index[self.langfeatures['Language']==lang].tolist()
        if len(lang_index) == 0:
            return None
        lang_index = lang_index[0]

        return self.langfeatures['CharacterType'][lang_index]

    #
    # Alphabet/Syllable detection is the first step in any language detection.
    # Only if alphabet detection is ambiguous, it will resort to unigrams/etc.
    # e.g. Other languages written in Latin characters,
    #       - tummai on ngern mai dai?
    #       - eulmanah ohreh gulrijyo?
    #
    @staticmethod
    def get_alphabet_syllable():
        # We can't use all CJK characters, so we use only the top 120 most frequent Chinese characters
        zhongwen = list(
            (u'的一是不了在人有我他这个们中来上大为和国地到以说时要就出会可' +
            u'也你对生能而子那得于着下自之年过发后作里用道行所然家种事成方' +
            u'多经么去法学如都同现当没动面起看定天分还进好小部其些主样理心' +
            u'她本前开但因只从想实日军者意无力它与长把机十民第公此已工使情')
        )
        rep_cn = []
        for i in range(0, len(zhongwen), 1): rep_cn.append('cn')

        # Most common Korean syllables, because hangul characters are inconvenient to split programmatically.
        # In fact the Hangul syllables made up of Hangul alphabets have their own Unicode, making it separate/unsplittable.
        # These Hangul syllables are inferred from this program, from our own training material
        hangul_syllables = list(
            (u'이다는을가고에지어기스의하통를로은대있테해인도서')
        )
        rep_ko = []
        for i in range(0, len(hangul_syllables), 1): rep_ko.append('ko')

        thai = ie.lang.characters.LangCharacters.LangCharacters.UNICODE_BLOCK_THAI
        rep_th = []
        for i in range(0, len(thai), 1): rep_th.append('th')

        latin = ie.lang.characters.LangCharacters.LangCharacters.UNICODE_BLOCK_LATIN_ALL
        rep_la = []
        for i in range(0, len(latin), 1): rep_la.append('latin')

        return {
            'Token':zhongwen + thai + hangul_syllables + latin,
            'Type':rep_cn + rep_th + rep_ko + rep_la
        }


if __name__ == '__main__':
    def demo_1():
        lf = LangFeatures()
        print ( lf.langfeatures )
        return

    def demo_2():
        lf = LangFeatures()

        for lang in lf.langfeatures['Language']:
            print ( lang + ':alphabet=[' + str(lf.get_split_token(lang, 'alphabet')) + ']' )
            print ( lang + ':syllable=[' + str(lf.get_split_token(lang, 'syllable')) + ']' )
            print ( lang + ':unigram=[' + str(lf.get_split_token(lang, 'unigram')) + ']' )
            print ( lang + ':Character Type = ' + lf.get_alphabet_type(lang) )
            print ( lang + ':Token same as charset = ' + str(lf.is_lang_token_same_with_charset(lang=lang)))

    def demo_3():
        lf = LangFeatures()
        print ( lf.langfeatures )

        print ( lf.get_languages_with_word_separator() )
        print ( lf.get_languages_with_syllable_separator() )
        print ( lf.get_languages_with_only_syllable_separator())


    demo_1()
    demo_2()
    demo_3()
