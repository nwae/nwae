#!/usr/bin/python
# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import re
import ie.lib.lang.characters.LangCharacters as lc
import ie.lib.lang.LangFeatures as lf
import ie.lib.lang.nlp.WordList as wl
import ie.lib.util.Log as log
# Library to convert Traditional Chinese to Simplified Chinese
import hanziconv as hzc


#
# Word Segmentation
#   Reason we don't use open source libraries
#     - We have lots of jargons that these libraries will not split properly.
#     - There is no single library in the same programming language that supports all
#       Chinese, Thai, Vietnamese, Indonesian, Japanese, Korean, etc, which will make the code messy.
#     - We need a mix of math, statistics, and language domain knowledge to split nicely, which may
#       need special customization.
#     - Relatively good word segmentation can be achieved with relatively non-complicated algorithms &
#       good language domain knowledge like below.
#
# TODO: Add ability to handle spelling mistakes, using nearest word measure (need to create)
# TODO: Improve on Thai word splitting, by improving word list, algorithm.
# TODO: For Chinese, add additional step to check for maximum likelihood of multiple combinations.
# TODO: Include POS Tagging, NER & Root Word Extraction algorithms within this class.
# TODO: Build on the word measure (LangCharacters) ignoring final consonant (th) syllable on better spelling correction
#
class WordSegmentation(object):

    def __init__(self, lang, dirpath_wordlist, postfix_wordlist, lang_stats=None):
        self.lang = lang
        self.lang_stats = lang_stats
        self.lang_characters = lc.LangCharacters()

        self.lang_features = lf.LangFeatures()
        self.lang_wordlist = wl.WordList(lang=lang, dirpath_wordlist=dirpath_wordlist, postfix_wordlist=postfix_wordlist)
        self.lang_wordlist.load_wordlist()
        return

    def convert_to_simplified_chinese(self, text):
        text_sim = hzc.HanziConv.toSimplified(text)
        return text_sim

    def add_wordlist(self, dirpath, postfix, array_words=None, verbose=0):
        self.lang_wordlist.append_wordlist(dirpath=dirpath, postfix=postfix, array_words=array_words, verbose=verbose)
        return

    #
    # Returns possible word matches from first character, no longer than <max_lookforward_chars>
    # So for example if given "冤大头？我有多乐币", this function should return
    # [True, False, True, False, False, False, False, False, False]
    # because possible words from start are "冤" and "冤大头".
    #
    def get_possible_word_separators_from_start(self, text, max_lookforward_chars=0, verbose=0):
        # Get language wordlist
        wl = self.lang_wordlist.wordlist
        # Get language charset
        lang_charset = lc.LangCharacters.get_language_charset(self.lang)

        if max_lookforward_chars<=0:
            max_lookforward_chars = len(text)
        else:
            # For Thai, default needs to be longer. TODO: Optimize the minimal length via real language stats
            if self.lang == 'th':
                max_lookforward_chars = max(30, max_lookforward_chars)
            max_lookforward_chars = min(len(text), max_lookforward_chars)
        # But don't allow to look forward more than 100 characters, which is unlikely already to find a word beyond that
        # TODO: Optimize this max length via real language stats (e.g. get 99% quartile in our own dictionary)
        max_lookforward_chars = min(100, max_lookforward_chars)

        tlen = len(text)
        if verbose >= 1: log.Log.log('Text Length: ' + str(tlen))
        # Record word separators
        matches = [False] * max_lookforward_chars
        curpos = 0

        for i_match in range(0, max_lookforward_chars, 1):
            word_tmp = text[curpos:(curpos + i_match + 1)]
            # single_number = self.lang_characters.convert_string_to_number(word_tmp)
            index_matches = wl.index[wl['Word'] == word_tmp].tolist()
            if len(index_matches) == 0:
                continue

            #
            # Valid Boundary Check:
            # In languages like Thai/Hangul, there are rules as to which alphabets may be start of a word.
            # Thus we need to check for next alphabet, if that is start of a word alphabet or not.
            # This step is super critical for Thai, otherwise there will be too many segmentation errors.
            #
            if i_match < max_lookforward_chars - 1:
                if self.lang == 'th':
                    # For Thai, next alphabet must not be a vowel (after consonant) or tone mark
                    alphabets_not_start_of_word = lc.LangCharacters.UNICODE_BLOCK_THAI_TONEMARKS + \
                                                  lc.LangCharacters.UNICODE_BLOCK_THAI_VOWELS_AFTER_CONSONANT
                    if text[curpos + i_match + 1] in alphabets_not_start_of_word:
                        # Invalid boundary
                        continue

            # Record the match
            matches[i_match] = True
            if verbose >= 1:
                log.Log.log('  Word [' + word_tmp + '] = ' + matches[i_match].__str__() + '.')

        return matches

    #
    # Returns all possible word segmentations, up to max_words
    #
    def get_all_possible_segmentations(self, text, max_words=0, verbose=0):
        # TODO
        return

    #
    # Segment words based on highest likelihood of all possible segmentations.
    # Make sure to call convert_to_simplified_chinese() first for traditional Chinese
    #
    def segment_words_ml(self, text, join_single_alphabets=True, verbose=0):
        # TODO
        return

    #
    # Segment words based on shortest/longest matching, language specific rules, etc.
    # Make sure to call convert_to_simplified_chinese() first for traditional Chinese
    #
    def segment_words(self, text, join_single_alphabets=True, verbose=0):
        if self.lang=='cn' or self.lang=='en' or len(text)<=1:
            join_single_alphabets = False
        # Get language wordlist
        wl = self.lang_wordlist.wordlist
        # Get language charset
        lang_charset = lc.LangCharacters.get_language_charset(self.lang)

        lookforward_chars = 8
        if self.lang == 'th':
            lookforward_chars = 30

        tlen = len(text)
        if verbose >= 1: log.Log.log('Text Length: ' + str(tlen))
        word_sep = [False]*tlen
        curpos = 0

        while curpos < tlen:
            # Already at last character in text, automatically a word separator
            if curpos == tlen-1:
                word_sep[curpos] = True
                break

            if verbose >= 1:
                log.Log.log("Curpos " + str(curpos) + ", Word [" + text[curpos] + "]")

            lookforward_window = min(lookforward_chars, tlen-curpos)
            match_longest = -1

            # Space always represents a word separator
            if text[curpos] == ' ' or (text[curpos] in lc.LangCharacters.UNICODE_BLOCK_WORD_SEPARATORS):
                if verbose >= 1:
                    log.Log.log("  Word separator true")
                match_longest = 0
            # If character is not in language character set
            elif text[curpos] not in lang_charset:
                # Look for continuous string of foreign characters, no limit up to the end of word
                lookforward_window = tlen - curpos
                match_longest = lookforward_window - 1
                if verbose >= 1: log.Log.log('Lookforward Window = ' + str(lookforward_window))
                for i in range(curpos, curpos+lookforward_window, 1):
                    # Found a local character or space
                    if verbose >= 2: log.Log.log('   Text['+str(i)+']="'+text[i]+'"')
                    if (text[i] in lang_charset) or (text[i] in lc.LangCharacters.UNICODE_BLOCK_WORD_SEPARATORS):
                        # Don't include the local character or space
                        match_longest = i - curpos - 1
                        break
            # Character is in language character set, so we use dictionary longest matching
            else:
                matches = self.get_possible_word_separators_from_start(text=text[curpos:tlen],
                                                                       max_lookforward_chars=lookforward_window,
                                                                       verbose=verbose)
                for i_match in range(0,len(matches)):
                    if matches[i_match]:
                        match_longest = i_match

            if match_longest >= 0:
                word_sep[curpos + match_longest] = True
                if verbose >= 1: log.Log.log('    Found word [' + text[curpos:(curpos+match_longest+1)] + ']')
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

        if verbose >= 1:
            log.Log.log('Sentence split as follows:')
            log.Log.log('[' + s + ']')

        return s
