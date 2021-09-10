# -*- coding: utf-8 -*-

import nwae.lang.characters.LangCharacters as lc
import nwae.lang.LangFeatures as lf
import nwae.lang.nlp.WordList as wl
import nwae.lang.nlp.SynonymList as slist
from nwae.lang.preprocessing.BasicPreprocessor import BasicPreprocessor
from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
# Library to convert Traditional Chinese to Simplified Chinese
import hanziconv as hzc
import nwae.utils.Profiling as prf
import re
try:
    """
    Japanese word segmentation
    """
    import nagisa
except Exception as ex:
    Log.warning(
        str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
        + ': Error importing libraries for japanese tokenization: ' + str(ex)
    )
try:
    """
    Korean word segmentation
    There are many problems with this library kkma, firstly it requires external JVM, it is quite slow
    and will also split wrong (e.g. '탈레반이' will be split wrongly to '탈', '레', '반이') or not in our
    desired application way (e.g. '장악한' split to '장악', '하', 'ㄴ')
    We should write our own, korean language is quite systematic, and we could control the following
      - by default a whole word '탈레반이' if not recognized should just keep as is, and split out only
        particles like '이'
      - naturally in most application the word '장악한' (verb) should not be split to
          ('장악', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD')
        and only stemming or lemmatization should bring it to '장악하다' (verb) and not '장악' (noun)
    """
    from konlpy.tag import Kkma
except Exception as ex:
    Log.warning(
        str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
        + ': Error importing libraries for korean tokenization: ' + str(ex)
    )


#
# Word Segmentation for languages whose word separator is not a space (e.g. Chinese, Thai, Vietnamese)
#   Reason we don't use open source libraries
#     - We don't need perfect word segmentation, and for non-Thai languages, we actually don't need word
#       segmentation at all if Intent Detection is our only goal. However we include for higher accuracy.
#     - We have lots of jargons that these libraries will not split properly.
#     - There is no single library in the same programming language that supports all
#       Chinese, Thai, Vietnamese, Indonesian, Japanese, Korean, etc, which will make the code messy.
#     - We need a mix of math, statistics, custom rules (e.g. conversion to "latin vietnamese"), and
#       language domain knowledge to split nicely, which may need special customization.
#     - Relatively good word segmentation can be achieved with relatively non-complicated algorithms &
#       good language domain knowledge like below.
#
# TODO: Reduce task time from an average of 0.13 secs of an average 10 character Chinese to < 0.05 secs
#
# TODO: Add ability to handle spelling mistakes, using nearest word measure (need to create)
# TODO: Improve on Thai word splitting, by improving word list, algorithm.
# TODO: For Chinese, add additional step to check for maximum likelihood of multiple combinations.
# TODO: Include POS Tagging, NER & Root Word Extraction algorithms within this class.
# TODO: Build on the word measure (LangCharacters) ignoring final consonant (th) syllable on better spelling correction
#
class WordSegmentation(object):

    LOOKFORWARD_MAX_LIMIT = 12
    # Length 4 is good enough to cover 97.95% of Chinese words
    LOOKFORWARD_CN = 4
    # Length 12 is good enough to cover 98.6% of Thai words
    LOOKFORWARD_TH = 12
    # Legnth 20 is good enough to cover 99.36% of Vietnamese words
    # TODO For Vietnamese should use how many spaces, not characters
    LOOKFORWARD_VN = 6

    def __init__(
            self,
            lang,
            dirpath_wordlist,
            postfix_wordlist,
            do_profiling = False,
            # For kkma
            jvmpath = '/Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Home/lib/jli/libjli.dylib',
    ):
        self.lang = lf.LangFeatures.map_to_lang_code_iso639_1(
            lang_code = lang
        )
        self.dirpath_wordlist = dirpath_wordlist
        self.postfix_wordlist = postfix_wordlist
        self.do_profiling = do_profiling
        # Don't use for Korean KKMA for now
        self.use_external_lib = self.lang in [lf.LangFeatures.LANG_JA]
        self.jvmpath = jvmpath

        # Languages not requiring any specialized algorithm to tokenize word like English
        self.have_simple_word_separator = False
        self.simple_word_separator = None

        # For languages which needs to be first split by syllables, punctuations need to
        # be cleaned from the syllables first before tokenization
        self.need_to_split_by_syllables_before_tokenization = False

        self.lang_features = lf.LangFeatures()
        word_sep_type = self.lang_features.get_word_separator_type(lang = self.lang)

        #
        # If the language unigram/word separator is a space, then it is just a simple re.split(),
        # no need to load word lists, etc.
        #
        if self.use_external_lib:
            self.have_simple_word_separator = False
            self.lang_wordlist = None
            self.syl_split_token = self.lang_features.get_split_token(
                lang  = self.lang,
                level = lf.LangFeatures.LEVEL_SYLLABLE
            )
            if self.lang == lf.LangFeatures.LANG_KO:
                try:
                    self.kkma = Kkma(jvmpath=self.jvmpath)
                    self.warn_korean()
                except Exception as ex:
                    errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                             + ': Unable to load Kkma() class, exception: ' + str(ex)
                    Log.error(errmsg)
                    raise Exception(errmsg)
            elif self.lang == lf.LangFeatures.LANG_JA:
                try:
                    import nagisa
                except Exception as ex:
                    errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                             + ': Unable to load nagisa library, exception: ' + str(ex)
                    Log.error(errmsg)
                    raise Exception(errmsg)
        elif word_sep_type == lf.LangFeatures.T_SPACE:
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Ignoring word list for simple language "' + str(self.lang) + '"..'
            )
            self.have_simple_word_separator = True
            self.simple_word_separator = BasicPreprocessor.get_word_separator(lang=self.lang)
            self.lang_wordlist = None
            self.syl_split_token = None
        else:
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Loading word list for complex language "' + str(self.lang)
                + '" using dirpath wordlist "' + str(self.dirpath_wordlist)
                + '", postfix wordlist "' + str(self.postfix_wordlist)
            )
            self.lang_wordlist = wl.WordList(
                lang             = self.lang,
                dirpath_wordlist = self.dirpath_wordlist,
                postfix_wordlist = self.postfix_wordlist
            )

            #
            # We need the language syllable split token. If '' means we look for longest matching
            # by character, else we first split the sentence by the syllable split token first.
            #
            self.syl_split_token = self.lang_features.get_split_token(
                lang  = self.lang,
                level = lf.LangFeatures.LEVEL_SYLLABLE
            )

            if self.syl_split_token is None:
                self.syl_split_token = ''
            self.need_to_split_by_syllables_before_tokenization = self.syl_split_token != ''
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Language "' + str(self.lang) + '" with syllable split token "' + str(self.syl_split_token)
                + '" requires syllable separation & cleaning punctuations stuck to word before tokenization = '
                + str(self.need_to_split_by_syllables_before_tokenization)
            )
        return

    def get_wordlist_length(self):
        if self.use_external_lib:
            return 0
        else:
            return self.lang_wordlist.wordlist.shape[0]

    def warn_korean(self):
        Log.warning(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Korean splitting currently uses kkma which is super slow and unusable for production purposes'
        )

    def convert_to_simplified_chinese(self, text):
        try:
            text_sim = hzc.HanziConv.toSimplified(text)
            return text_sim
        except Exception as ex:
            Log.error(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Error converting "' + str(text) + '" to simplified chinese. Exception ' + str(ex) + '.'
            )
            return text

    def add_wordlist(
            self,
            dirpath,
            postfix,
            array_words = None,
    ):
        if self.use_external_lib:
            Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Not adding word list for language "' + str(self.lang) + '" using external lib'
            )
            return
        self.lang_wordlist.append_wordlist(
            dirpath     = dirpath,
            postfix     = postfix,
            array_words = array_words,
        )

    #
    # Returns possible word matches from first character, no longer than <max_lookforward_chars>
    # So for example if given "冤大头？我有多乐币", this function should return
    # [True, False, True, False, False, False, False, False, False]
    # because possible words from start are "冤" and "冤大头".
    #
    def get_possible_word_separators_from_start(
            self,
            # Can be of type "str", "list" or "tuple"
            text_array,
            # Should be max_lookforward_ngrams
            max_lookforward_chars = 0,
            look_from_longest     = True
    ):
        # TODO Start looking backwards from max_lookforward until 0, to further speed up (for longest match)
        if type(text_array) not in (str, list, tuple):
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Wrong type for text array "' + str(type(text_array))
                + '", text array: ' + str(text_array) + '.'
            )

        if max_lookforward_chars<=0:
            max_lookforward_chars = len(text_array)
        else:
            # Cannot be longer than the length of the array
            max_lookforward_chars = min(len(text_array), max_lookforward_chars)

        # Not more than the longest lookforward we know, which is for Vietnamese
        max_lookforward_chars = min(WordSegmentation.LOOKFORWARD_MAX_LIMIT, max_lookforward_chars)

        tlen = len(text_array)
        # Record word separators
        matches = [False] * max_lookforward_chars
        curpos = 0

        start_range = 0
        end_range = max_lookforward_chars
        step_range = 1
        if look_from_longest:
            start_range = max_lookforward_chars - 1
            end_range = -1
            step_range = -1

        for i_match in range(start_range, end_range, step_range):
            word_tmp = self.syl_split_token.join(text_array[curpos:(curpos + i_match + 1)])

            # How many characters or ngrams (e.g. Vietnamese "gam en" is 2 ngrams)
            n_gram = i_match + 1
            if n_gram not in self.lang_wordlist.ngrams.keys():
                continue
            if word_tmp not in self.lang_wordlist.ngrams[n_gram]:
                Log.debugdebug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': "' + word_tmp + '" not in ' + str(n_gram) + '-gram'
                )
                continue
            else:
                Log.debugdebug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': "' + word_tmp + '" in ' + str(n_gram) + '-gram'
                )

            #
            # Valid Boundary Check:
            # In languages like Thai/Hangul, there are rules as to which alphabets may be start of a word.
            # Thus we need to check for next alphabet, if that is start of a word alphabet or not.
            # This step is the key step for Thai, otherwise the segmentation will be completely unusable.
            #
            if i_match < max_lookforward_chars - 1:
                if self.lang == lf.LangFeatures.LANG_TH:
                    # For Thai, next alphabet must not be a vowel (after consonant) or tone mark
                    alphabets_not_start_of_word =\
                        lc.LangCharacters.UNICODE_BLOCK_THAI_TONEMARKS +\
                        lc.LangCharacters.UNICODE_BLOCK_THAI_VOWELS_AFTER_CONSONANT
                    if text_array[curpos + i_match + 1] in alphabets_not_start_of_word:
                        # Invalid boundary
                        continue

            # Record the match
            matches[i_match] = True
            Log.debugdebug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Word [' + word_tmp + '] = ' + matches[i_match].__str__() + '.'
            )
            if look_from_longest:
                break

        return matches

    #
    # Returns all possible word segmentations, up to max_words
    #
    def get_all_possible_segmentations(self, text, max_words=0):
        # TODO
        return

    #
    # TODO Use WordSegmentionModel.py where we use LSTM to learn any language segmentation
    #
    def segment_words_ml(
            self,
            text,
            look_from_longest = True
    ):
        # TODO
        return

    def get_optimal_lookforward_chars(self, lang):
        # Default to Thai
        lookforward_chars = WordSegmentation.LOOKFORWARD_TH

        if lang == lf.LangFeatures.LANG_ZH:
            lookforward_chars = WordSegmentation.LOOKFORWARD_CN
        elif lang == lf.LangFeatures.LANG_TH:
            lookforward_chars = WordSegmentation.LOOKFORWARD_TH
        elif lang == lf.LangFeatures.LANG_VN:
            lookforward_chars = WordSegmentation.LOOKFORWARD_VN

        return lookforward_chars

    def __is_natural_word_separator(
            self,
            c
    ):
        # Space always represents a word separator (not true for Vietnamese!)
        if (c in (' ','，','。','？','?','"',':',';')) or\
                (c in lc.LangCharacters.UNICODE_BLOCK_WORD_SEPARATORS):
            return True
        else:
            return False

    """Exception that uses external libraries temporarily while we write our own"""
    def segment_ko_ja(
            self,
            text,
            return_array_of_split_words = False
    ):
        try:
            if self.lang in [lf.LangFeatures.LANG_JA]:
                words_postags = nagisa.tagging(text)
                txt_sym_tok = words_postags.words
                txt_sym_postags = words_postags.postags
                Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Japanese segmentation "' + str(txt_sym_tok) + '", word & POS tags: ' + str(words_postags)
                )
                if return_array_of_split_words:
                    return txt_sym_tok
                else:
                    return BasicPreprocessor.get_word_separator(lang=self.lang).join(txt_sym_tok)
            elif self.lang in [lf.LangFeatures.LANG_KO]:
                self.warn_korean()
                words_postags = self.kkma.pos(
                    phrase = text
                )
                txt_sym_tok = [wp[0] for wp in words_postags]
                txt_sym_postags = [wp[1] for wp in words_postags]
                Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Korean segmentation "' + str(txt_sym_tok) + '", word & POS tags: ' + str(words_postags)
                )
                if return_array_of_split_words:
                    return txt_sym_tok
                else:
                    return BasicPreprocessor.get_word_separator(lang=self.lang).join(txt_sym_tok)
            else:
                raise Exception(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': No external library supported for language "' + str(self.lang) + '"'
                )
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Error segmenting lang "' + str(self.lang) + '", text "' + str(text) \
                     + '", exception: ' + str(ex)
            Log.error(errmsg)
            raise Exception(errmsg)
    #
    # Segment by simple word separator
    #
    def segment_words_simple(
            self,
            text,
            return_array_of_split_words = False
    ):
        regex_punctuations = BasicPreprocessor.DEFAULT_PUNCTUATIONS
        if self.lang == lf.LangFeatures.LANG_EN:
            # For English don't include "'" as words like "can't", "don't", etc includes them
            regex_punctuations = re.sub(pattern='[\']', repl='', string=regex_punctuations)
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Using punctuations: ' + str(regex_punctuations)
        )

        word_array = text.split(sep=self.simple_word_separator)
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Split words: ' + str(word_array)
        )

        word_array = BasicPreprocessor.clean_punctuations(
            sentence = word_array,
            punctuations_pattern = regex_punctuations,
            convert_to_lower_case = False
        )
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Split punctuations: ' + str(word_array)
        )

        if return_array_of_split_words:
            return word_array
        else:
            return self.__return_array_words_as_string(array_words=word_array)

    #
    # Segment words based on shortest/longest matching, language specific rules, etc.
    #
    def segment_words(
            self,
            # String type
            text,
            look_from_longest = True,
            # For certain languages like Thai, if a word is split into a single alphabet
            # it certainly has no meaning, and we join them together, until we find a
            # split word of length not 1
            join_single_meaningless_alphabets_as_one = True,
            return_array_of_split_words = False
    ):
        """Exception that uses external libraries temporarily while we write our own"""
        if self.use_external_lib:
            return self.segment_ko_ja(
                text = text,
                return_array_of_split_words = return_array_of_split_words,
            )
        elif self.have_simple_word_separator:
            # For languages not requiring any specialized algorithm for word tokenization
            return self.segment_words_simple(
                text = text,
                return_array_of_split_words = return_array_of_split_words,
            )

        a = prf.Profiling.start()

        if self.lang == lf.LangFeatures.LANG_ZH:
            text_simplified = self.convert_to_simplified_chinese(text = text)
            if text != text_simplified:
                Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Converted to simplified chinese for "' + str(text) + '" to "' + str(text_simplified) + '".'
                )
                if self.do_profiling:
                    Log.important(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Took ' + str(prf.Profiling.get_time_dif_str(start=a, stop=prf.Profiling.stop())) + ' secs.'
                    )
            text = text_simplified

        # string type
        text_array = text

        #
        # If a language has syllables split by a non-empty character, each "character"
        # becomes the syllable, and we split them into a list.
        #
        if self.need_to_split_by_syllables_before_tokenization:
            # E.g. For Vietnamese we break syllables by spaces
            text_array = text.split(sep=self.syl_split_token)
            # If there is a syllable separator, means we require punctuations stuck
            # to words to be separated out before word tokenization like Vietnamese
            tmp_arr = BasicPreprocessor.clean_punctuations(
                sentence = text_array,
                convert_to_lower_case = True
            )
            if type(tmp_arr) in [list, tuple]:
                text_array = tmp_arr
            Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Text "' + str(text) + '" clean punctuations to: ' + str(text_array)
            )

        # Get language charset
        lang_charset = lc.LangCharacters.get_language_charset(self.lang)

        # Default to Thai
        lookforward_chars = self.get_optimal_lookforward_chars(lang = self.lang)

        # Log.debugdebug('Using ' + str(lookforward_chars) + ' lookforward characters')

        tlen = len(text_array)
        Log.debugdebug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Text Length (by syllable): ' + str(tlen)
        )
        if tlen == 0:
            if return_array_of_split_words:
                return []
            else:
                return ''
        word_sep = [False]*tlen
        # End of string is always a word separator
        word_sep[tlen-1] = True
        curpos = 0

        #
        # TODO We can speed up some more here by doing only longest matching, thus only looking from longest.
        #
        while curpos < tlen:
            # Already at last character in text, automatically a word separator
            if curpos == tlen-1:
                word_sep[curpos] = True
                break

            Log.debugdebug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Current position ' + str(curpos) + ', search word "' + str(text_array[curpos:tlen]) + '".'
            )

            lookforward_window = min(lookforward_chars, tlen-curpos)
            match_longest = -1

            # Check if this character is a natural word separator in this language
            if self.__is_natural_word_separator(c=text_array[curpos]):
                Log.debugdebug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + '  Word separator true for character "' + str(text_array[curpos]) + '".'
                )
                match_longest = 0
            # If character is not in language character set
            elif (len(text_array[curpos]) == 1) \
                    and (text_array[curpos] not in lang_charset) \
                    and (self.syl_split_token == ''):       # Only applies to languages with no syllable split token
                # Look for continuous string of foreign characters, no limit up to the end of word
                lookforward_window = tlen - curpos
                match_longest = lookforward_window - 1
                Log.debugdebug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Lookforward Window = ' + str(lookforward_window)
                )

                for i in range(curpos, curpos+lookforward_window, 1):
                    # Found a local character or space
                    Log.debugdebug(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + '   Text "' + str(i) + '"="' + text_array[i]+'"'
                    )
                    if (text_array[i] in lang_charset) or (self.__is_natural_word_separator(c=text_array[i])):
                        # Don't include the local character or space
                        match_longest = i - curpos - 1
                        break
            # Character is in language character set, so we use dictionary longest matching
            else:
                matches = self.get_possible_word_separators_from_start(
                    text_array            = text_array[curpos:tlen],
                    max_lookforward_chars = lookforward_window,
                    look_from_longest     = look_from_longest
                )
                for i_match in range(0,len(matches)):
                    if matches[i_match]:
                        match_longest = i_match

            if match_longest >= 0:
                word_sep[curpos + match_longest] = True

                Log.debugdebug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + '    Found word "'
                    + str(self.syl_split_token.join(text_array[curpos:(curpos+match_longest+1)])) + '".'
                )
                curpos = curpos + match_longest + 1

                # TODO: Improved Segmentation
                # Despite the fact that longest matching works surprisingly well in the majority of cases,
                # there are improvements we can make.
                # Design algorithm to improve on longest matching, by looking forward also considering
                # variants & employing probabilistic techniques of highest likelihood combination.
                # e.g. '人口多' can be split into '人口 多' or '人 口多' depending on context and maximum likelihood.
            else:
                # No separator found, assume just a one character word
                word_sep[curpos] = True
                curpos = curpos + 1

        #
        # Now that we know the word separators already, we can
        #
        array_words = []
        lastpos = 0
        for curpos in range(len(word_sep)):
            if word_sep[curpos]:
                word = self.syl_split_token.join(text_array[lastpos:(curpos+1)])
                lastpos = curpos+1
                if word == ' ':
                    # Don't record space in array
                    continue
                else:
                    array_words.append(word)

        if self.lang == lf.LangFeatures.LANG_TH:
            array_words = self.__join_single_alphabets_as_a_word(
                array_words = array_words
            )

        #
        # Break into array
        #
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Text "' + str(text) + '", separators ' + str(word_sep)
            + '\n\rSplit words: ' + str(array_words)
        )

        if return_array_of_split_words:
            return array_words
        else:
            return self.__return_array_words_as_string(array_words = array_words)

    def __return_array_words_as_string(self, array_words):
        a = prf.Profiling.start()
        print_separator = BasicPreprocessor.get_word_separator(
            lang = self.lang
        )

        s = print_separator.join(array_words)

        if self.do_profiling:
            b = prf.Profiling.stop()
            Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ':      PROFILING Segment Words for [' + text + '] to [' + s
                + '] took ' + prf.Profiling.get_time_dif_str(start=a, stop=b)
            )
        return s

    def __join_single_alphabets_as_a_word(
            self,
            array_words
    ):
        array_words_redo = []
        # Single alphabets have no meaning, so we join them
        join_word = ''
        tlen = len(array_words)
        for i in range(len(array_words)):
            word = array_words[i]
            if join_word == '':
                # Previously no single alphabet
                if (len(word) > 1) or (self.__is_natural_word_separator(c=word)):
                    array_words_redo.append(word)
                else:
                    # Single alphabet word found, join them to previous
                    join_word = join_word + word
            else:
                # Previously has single alphabet
                if (len(word) > 1) or (self.__is_natural_word_separator(c=word)):
                    # Write the joined alphabets, as the sequence has ended
                    array_words_redo.append(join_word)
                    array_words_redo.append(word)
                    join_word = ''
                else:
                    # Single alphabet word found, join them to previous
                    join_word = join_word + word

            # Already at the last position
            if i == tlen - 1:
                if join_word != '':
                    array_words_redo.append(join_word)

        return array_words_redo


if __name__ == '__main__':
    import nwae.lang.config.Config as cf
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class       = cf.Config,
        default_config_file = cf.Config.CONFIG_FILE_PATH_DEFAULT
    )

    lang = lf.LangFeatures.LANG_TH
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_2

    synonymlist_ro = slist.SynonymList(
        lang                = lang,
        dirpath_synonymlist = config.get_config(param=cf.Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_SYNONYMLIST)
    )
    synonymlist_ro.load_synonymlist()

    ws = WordSegmentation(
        lang             = lang,
        dirpath_wordlist = config.get_config(param=cf.Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_WORDLIST),
        do_profiling     = True
    )
    len_before = ws.lang_wordlist.wordlist.shape[0]
    ws.add_wordlist(
        dirpath = config.get_config(param=cf.Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        array_words=list(synonymlist_ro.get_synonym_list_words())
    )
    len_after = ws.lang_wordlist.wordlist.shape[0]
    if len_after - len_before > 0:
        print(": Warning. These words not in word list but in synonym list:")
        words_not_synched = ws.lang_wordlist.wordlist['Word'][len_before:len_after]
        print(words_not_synched)

    text = 'นี่คือ'
    #print(ws.segment_words(text=text, look_from_longest=False))
    print('"' + ws.segment_words(text=text, look_from_longest=True) + '"')

    #
    # Test without any word list
    #
    ws = WordSegmentation(
        lang             = 'ru',
        dirpath_wordlist = None,
        postfix_wordlist = None,
        do_profiling     = True
    )
    text = 'Аккаунт   популярного;южнокорейского чат-бота 이우다   был заблокирован после жалоб на ненавистнические '\
           'высказывания в    адрес   сексуальных меньшинств.'
    print('"' + ws.segment_words(text=text, look_from_longest=True) + '"')

    ws = WordSegmentation(
        lang             = lf.LangFeatures.LANG_KO,
        dirpath_wordlist = None,
        postfix_wordlist = None,
        do_profiling     = True
    )
    text = '구글 드론 배달 서비스의 누적 배달 건수가 10만건을 돌파했다.'
    print('"' + ws.segment_words(text=text, look_from_longest=True) + '"')

    ws = WordSegmentation(
        lang             = lf.LangFeatures.LANG_JA,
        dirpath_wordlist = None,
        postfix_wordlist = None,
        do_profiling     = True
    )
    text = '江戸時代には江戸前や江戸前海などの呼び名があった。'
    print('"' + ws.segment_words(text=text, look_from_longest=True) + '"')

    exit(0)
