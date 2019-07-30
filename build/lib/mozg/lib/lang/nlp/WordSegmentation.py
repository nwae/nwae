#!/usr/bin/python
# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import re
import mozg.lib.lang.characters.LangCharacters as lc
import mozg.lib.lang.LangFeatures as lf
import mozg.lib.lang.nlp.WordList as wl
import mozg.lib.lang.nlp.SynonymList as slist
import mozg.lib.lang.stats.LangStats as ls
import mozg.common.util.Log as log
# Library to convert Traditional Chinese to Simplified Chinese
import hanziconv as hzc
import mozg.common.util.Profiling as prf
from inspect import currentframe, getframeinfo


#
# Word Segmentation
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

    # Length 4 is good enough to cover 97.95% of Chinese words
    LOOKFORWARD_CN = 4
    # Length 12 is good enough to cover 98.6% of Thai words
    LOOKFORWARD_TH = 12
    # Legnth 20 is good enough to cover 97.3% of Vietnamese words
    # TODO For Vietnamese should use how many spaces, not characters
    LOOKFORWARD_VN = 20

    def __init__(
            self,
            lang,
            dirpath_wordlist,
            postfix_wordlist,
            do_profiling = False,
            lang_stats = None,
            verbose = 0
    ):
        self.lang = lang
        self.do_profiling = do_profiling
        self.verbose = verbose

        self.lang_stats = lang_stats
        self.lang_characters = lc.LangCharacters()

        self.lang_features = lf.LangFeatures()
        self.lang_wordlist = wl.WordList(
            lang             = lang,
            dirpath_wordlist = dirpath_wordlist,
            postfix_wordlist = postfix_wordlist,
            verbose          = self.verbose
        )
        self.lang_wordlist.load_wordlist()

        return

    def convert_to_simplified_chinese(self, text):
        text_sim = hzc.HanziConv.toSimplified(text)
        return text_sim

    def add_wordlist(
            self,
            dirpath,
            postfix,
            array_words=None,
    ):
        self.lang_wordlist.append_wordlist(
            dirpath     = dirpath,
            postfix     = postfix,
            array_words = array_words,
        )
        return

    #
    # Returns possible word matches from first character, no longer than <max_lookforward_chars>
    # So for example if given "冤大头？我有多乐币", this function should return
    # [True, False, True, False, False, False, False, False, False]
    # because possible words from start are "冤" and "冤大头".
    #
    def get_possible_word_separators_from_start(
            self,
            text,
            max_lookforward_chars=0,
            look_from_longest=True
    ):
        # Get language wordlist
        wl = self.lang_wordlist.wordlist

        # TODO Start looking backwards from max_lookforward until 0, to further speed up (for longest match)

        if max_lookforward_chars<=0:
            max_lookforward_chars = len(text)
        else:
            # Cannot be longer than the length of the sentence
            max_lookforward_chars = min(len(text), max_lookforward_chars)

        # Not more than the longest lookforward we know, which is for Vietnamese
        max_lookforward_chars = min(WordSegmentation.LOOKFORWARD_VN, max_lookforward_chars)

        tlen = len(text)
        log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Text Length: ' + str(tlen))
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
            word_tmp = text[curpos:(curpos + i_match + 1)]
            # single_number = self.lang_characters.convert_string_to_number(word_tmp)

            n_gram = i_match + 1
            if n_gram not in self.lang_wordlist.ngrams.keys():
                continue
            if word_tmp not in self.lang_wordlist.ngrams[n_gram]:
                log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': [' + word_tmp + '] not in ' + str(n_gram) + '-gram')
                continue
            else:
                log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': [' + word_tmp + '] in ' + str(n_gram) + '-gram')

            #
            # Valid Boundary Check:
            # In languages like Thai/Hangul, there are rules as to which alphabets may be start of a word.
            # Thus we need to check for next alphabet, if that is start of a word alphabet or not.
            # This step is super critical for Thai, otherwise there will be too many segmentation errors.
            #
            if i_match < max_lookforward_chars - 1:
                if self.lang == lf.LangFeatures.LANG_TH:
                    # For Thai, next alphabet must not be a vowel (after consonant) or tone mark
                    alphabets_not_start_of_word = lc.LangCharacters.UNICODE_BLOCK_THAI_TONEMARKS + \
                                                  lc.LangCharacters.UNICODE_BLOCK_THAI_VOWELS_AFTER_CONSONANT
                    if text[curpos + i_match + 1] in alphabets_not_start_of_word:
                        # Invalid boundary
                        continue

            # Record the match
            matches[i_match] = True
            log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Word [' + word_tmp + '] = ' + matches[i_match].__str__() + '.')
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
    # Segment words based on highest likelihood of all possible segmentations.
    # Make sure to call convert_to_simplified_chinese() first for traditional Chinese
    #
    def segment_words_ml(
            self,
            text,
            join_single_alphabets = True,
            look_from_longest     = True
    ):
        # TODO
        return

    def get_optimal_lookforward_chars(self, lang):
        # Default to Thai
        lookforward_chars = WordSegmentation.LOOKFORWARD_TH

        if lang == lf.LangFeatures.LANG_CN:
            lookforward_chars = WordSegmentation.LOOKFORWARD_CN
        elif lang == lf.LangFeatures.LANG_TH:
            lookforward_chars = WordSegmentation.LOOKFORWARD_TH
        elif lang == lf.LangFeatures.LANG_VN:
            lookforward_chars = WordSegmentation.LOOKFORWARD_VN

        return lookforward_chars

    #
    # Segment words based on shortest/longest matching, language specific rules, etc.
    # Make sure to call convert_to_simplified_chinese() first for traditional Chinese
    #
    def segment_words(
            self,
            text,
            join_single_alphabets = True,
            look_from_longest     = True
    ):

        a = None
        if self.do_profiling:
            a = prf.Profiling.start()
            log.Log.info(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                         + '.      PROFILING Segment Words [' + text + '] Start: ' + str(a))

        if self.lang==lf.LangFeatures.LANG_CN or self.lang==lf.LangFeatures.LANG_EN or len(text)<=1:
            join_single_alphabets = False

        # Get language wordlist
        wl = self.lang_wordlist.wordlist
        # Get language charset
        lang_charset = lc.LangCharacters.get_language_charset(self.lang)

        # Default to Thai
        lookforward_chars = self.get_optimal_lookforward_chars(lang = self.lang)

        # log.Log.log('Using ' + str(lookforward_chars) + ' lookforward characters')

        tlen = len(text)
        log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                      + ': Text Length: ' + str(tlen))
        word_sep = [False]*tlen
        curpos = 0

        #
        # TODO We can speed up some more here by doing only longest matching, thus only looking from longest.
        #
        while curpos < tlen:
            # Already at last character in text, automatically a word separator
            if curpos == tlen-1:
                word_sep[curpos] = True
                break

            log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                          + ": Curpos " + str(curpos) + ", Word [" + text[curpos] + "]")

            lookforward_window = min(lookforward_chars, tlen-curpos)
            match_longest = -1

            # Space always represents a word separator
            if text[curpos] == ' ' or (text[curpos] in lc.LangCharacters.UNICODE_BLOCK_WORD_SEPARATORS):
                log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                   + "  Word separator true")
                match_longest = 0
            # If character is not in language character set
            elif text[curpos] not in lang_charset:
                # Look for continuous string of foreign characters, no limit up to the end of word
                lookforward_window = tlen - curpos
                match_longest = lookforward_window - 1
                log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                   + ': Lookforward Window = ' + str(lookforward_window))

                # TODO For Vietnamese, no need to compare until hit space boundary, so can optimize further

                for i in range(curpos, curpos+lookforward_window, 1):
                    # Found a local character or space
                    log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                       + '   Text['+str(i)+']="'+text[i]+'"')
                    if (text[i] in lang_charset) or (text[i] in lc.LangCharacters.UNICODE_BLOCK_WORD_SEPARATORS):
                        # Don't include the local character or space
                        match_longest = i - curpos - 1
                        break
            # Character is in language character set, so we use dictionary longest matching
            else:
                matches = self.get_possible_word_separators_from_start(
                    text = text[curpos:tlen],
                    max_lookforward_chars = lookforward_window,
                    look_from_longest = look_from_longest
                )
                for i_match in range(0,len(matches)):
                    if matches[i_match]:
                        match_longest = i_match

            if match_longest >= 0:
                word_sep[curpos + match_longest] = True
                log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                   + '    Found word [' + text[curpos:(curpos+match_longest+1)] + ']')
                curpos = curpos + match_longest + 1

                # TODO: Improved Segmentation
                # Despite the fact that longest matching works surprisingly well in the majority of cases,
                # there are improvements we can make.
                # Design algorithm to improve on longest matching, by looking forward also considering
                # variants & employing probabilistic techniques of highest likelihood combination.
                # e.g. '人口多' can be split into '人口 多' or '人 口多' depending on context and maximum likelihood.
            else:
                # Resort to statistical method to split characters, no matching found above
                c1 = text[curpos]
                c2 = text[curpos+1]
                if self.lang_stats is not None:
                    prob_post = self.lang_stats.get_collocation_probability(self.lang, c1, c2, 'post')
                    prob_pre = self.lang_stats.get_collocation_probability(self.lang, c1, c2, 'pre')
                    if prob_post is None: prob_post = 0
                    if prob_pre is None: prob_pre = 0
                    # TODO: Design algorithm of word segmentation using purely statistical methods

                # No separator found, assume just a one character word
                word_sep[curpos] = True
                curpos = curpos + 1

        s = ''
        print_separator = '|'
        if self.lang=='cn' or self.lang=='th':
            print_separator = ' '
        for i in range(0, tlen, 1):
            s = s + text[i]
            add_print_separator = False
            if word_sep[i]:
                if not join_single_alphabets:
                    add_print_separator = True
                elif join_single_alphabets:
                    if ( i==0 and not word_sep[i+1] ) or ( i==tlen-1 and not word_sep[i-1] ):
                        add_print_separator = True
                    elif i>=1 and i<tlen-1:
                        # Join together a string of single alphabets
                        if not ( word_sep[i-1] and word_sep[i+1] ):
                            add_print_separator = True
            if add_print_separator:
                s = s + print_separator

        # Collapse print separators
        regex = '[' + print_separator + ']+'
        s = re.sub(regex, print_separator, s)

        log.Log.debug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Sentence split as follows "' + str(s) + '".')

        if self.do_profiling:
            b = prf.Profiling.stop()
            log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                             + ':      PROFILING Segment Words for [' + text + '] to [' + s
                             +'] took ' + prf.Profiling.get_time_dif_str(start=a, stop=b))

        return s


if __name__ == '__main__':
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_DEBUG_2

    topdir = '/Users/mark.tan/git/mozg.nlp'
    lang_stats = ls.LangStats(
        dirpath_traindata   = topdir + '/nlp.data/traindata',
        dirpath_collocation = topdir + '/nlp.output/collocation.stats'
    )
    lang_stats.load_collocation_stats()

    synonymlist_ro = slist.SynonymList(
        lang                = 'cn',
        dirpath_synonymlist = topdir + '/nlp.data/app/chats',
        postfix_synonymlist = '.synonymlist.txt'
    )
    synonymlist_ro.load_synonymlist(verbose=1)

    ws = WordSegmentation(
        lang             = 'cn',
        dirpath_wordlist = topdir + '/nlp.data/wordlist',
        postfix_wordlist = '-wordlist.txt',
        lang_stats       = lang_stats,
        do_profiling     = True,
        verbose          = 0
    )
    len_before = ws.lang_wordlist.wordlist.shape[0]
    ws.add_wordlist(
        dirpath=None,
        postfix=None,
        array_words=list(synonymlist_ro.synonymlist['Word'])
    )
    len_after = ws.lang_wordlist.wordlist.shape[0]
    if len_after - len_before > 0:
        print(": Warning. These words not in word list but in synonym list:")
        words_not_synched = ws.lang_wordlist.wordlist['Word'][len_before:len_after]
        print(words_not_synched)

    text = '谷歌和脸书成了冤大头？我有多乐币 hello world 两间公司合共被骗一亿美元克里斯。happy当只剩两名玩家时，无论是第几轮都可以比牌。'
    #text = 'งานนี้เมื่อต้องขึ้นแท่นเป็นผู้บริหาร แหวนแหวน จึงมุมานะไปเรียนต่อเรื่องธุ'

    text = '入钱'
    #print(ws.segment_words(text=text, join_single_alphabets=True, look_from_longest=False))
    print(ws.segment_words(text=text, join_single_alphabets=True, look_from_longest=True))

