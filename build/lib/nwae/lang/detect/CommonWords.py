# --*-- coding: utf-8 --*--

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
from nwae.utils.StringUtils import StringUtils
import re
from nwae.lang.LangFeatures import LangFeatures
from nwae.lang.nlp.lemma.Lemmatizer import Lemmatizer


class CommonWords:

    def __init__(
            self,
            lang
    ):
        self.lang = LangFeatures.map_to_lang_code_iso639_1(
            lang_code = lang
        )
        self.raw_words = None
        self.common_words = None

        lfobj = LangFeatures()
        self.lang_have_verb_conj = lfobj.have_verb_conjugation(
            lang = self.lang
        )
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Lang "' + str(self.lang) + '" verb conjugation = ' + str(self.lang_have_verb_conj) + '.'
        )
        self.word_stemmer = None
        if self.lang_have_verb_conj:
            try:
                self.word_stemmer = Lemmatizer(
                    lang = self.lang
                )
                Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Lang "' + str(self.lang) + '" stemmer/lemmatizer initialized successfully.'
                )
            except Exception as ex_stemmer:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                         + ': Lang "' + str(self.lang) + ' stemmer/lemmatizer failed to initialize: ' \
                         + str(ex_stemmer) + '.'
                Log.warning(errmsg)
                self.word_stemmer = None

        return

    #
    # Minimum intersection with common words given any random English sentence
    #
    def get_min_threshold_intersection_pct(
            self
    ):
        raise Exception(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Must be implemented by child class!'
        )

    def get_pct_intersection_with_common_words(
            self,
            word_list,
            # In the case of Vietnamese, we might have to form words from the syllables
            max_word_n_tuple = 1
    ):
        if max_word_n_tuple == 1:
            lang_intersection = set(word_list).intersection(self.get_common_words())
            pct_intersection = len(lang_intersection) / len(set(word_list))
        else:
            # Means we are looking not just at the current token, but form a word from
            # continuous tokens up to max_word_n_tuple (usually not more than 2)
            len_word_list = len(word_list)
            count_int = 0
            cur_index = 0
            actual_word_count = 0
            # Loop by each token in the word list (or rather token list)
            while cur_index < len_word_list:
                max_n_tuple_lookforward = min(max_word_n_tuple, len_word_list-cur_index)
                for j in range(max_n_tuple_lookforward,0,-1):
                    # Look from j tokens ahead
                    end_index = cur_index+j
                    # For the j-tuple word
                    w = ' '.join(word_list[cur_index:end_index])
                    Log.debugdebug(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Test word "' + str(w) + '", cur_index=' + str(cur_index) + ', j=' + str(j))
                    if w in self.get_common_words():
                        count_int += 1
                        # Move forward to the end of the token from the word found
                        cur_index += j-1
                        Log.debugdebug(
                            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': Found word "' + str(w) + '"')
                        break
                cur_index += 1
                actual_word_count += 1

            Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Count Intersection = ' + str(count_int) + ', actual word count = ' + str(actual_word_count)
            )
            pct_intersection = count_int / actual_word_count

        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': "' + str(self.lang) + '" intersection = ' + str(pct_intersection)
        )
        return pct_intersection

    def test_lang(
            self,
            word_list
    ):
        pct_intersection = self.get_pct_intersection_with_common_words(word_list=word_list)
        if pct_intersection > self.get_min_threshold_intersection_pct():
            return True
        else:
            return False

    def get_common_words(
            self
    ):
        return self.common_words

    def process_common_words(
            self,
            word_split_token = ' '
    ):
        try:
            self.raw_words = StringUtils.trim(self.raw_words)
            self.raw_words = re.sub(
                pattern = '[\xa0\t\n\r]',
                repl    = word_split_token,
                string  = self.raw_words
            )
            self.raw_words = self.raw_words.lower()
        except Exception as ex:
            errmsg = str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Error processing raw words. Exception: ' + str(ex)
            Log.error(errmsg)
            raise Exception(errmsg)

        try:
            self.common_words = self.raw_words.split(word_split_token)
            # Remove None, '', {}, etc.
            self.common_words = [w for w in self.common_words if w]

            word_stems = self.add_word_stems()
            if word_stems:
                self.common_words = word_stems + self.common_words

            self.common_words = sorted(set(self.common_words))
            Log.info(
                str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': Loaded ' + str(len(self.common_words)) + ' common words of lang "' + str(self.lang) + '".'
            )
        except Exception as ex:
            errmsg = str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Error processing common words. Exception: ' + str(ex)
            Log.error(errmsg)
            raise Exception(errmsg)

        return

    def add_word_stems(
            self
    ):
        if self.word_stemmer is None:
            return None

        stems = []
        for w in self.common_words:
            w_stem = self.word_stemmer.stem(
                word = w
            )
            if w_stem == w:
                continue
            else:
                stems.append(w_stem)

        stems = sorted(set(stems))
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Loaded ' + str(len(stems)) + ' unique word stems: ' + str(stems)
        )
        return stems