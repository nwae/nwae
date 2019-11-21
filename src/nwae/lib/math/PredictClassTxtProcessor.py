# -*- coding: utf-8 -*-

import nwae.utils.Log as log
import nwae.utils.StringUtils as su
from inspect import currentframe, getframeinfo
import nwae.lib.lang.LangHelper as langhelper
import nwae.lib.lang.LangFeatures as langfeatures
import nwae.lib.lang.nlp.SpellingCorrection as spellcor
import nwae.lib.lang.nlp.lemma.Lemmatizer as lmtz
import nwae.lib.lang.TextProcessor as txtpcsr


class PredictClassTxtProcessor:

    def __init__(
            self,
            identifier_string,
            dir_path_model,
            model_features_list,
            lang,
            dirpath_synonymlist,
            postfix_synonymlist,
            dir_wordlist,
            postfix_wordlist,
            dir_wordlist_app,
            postfix_wordlist_app,
            do_spelling_correction = False,
            do_word_stemming = False,
            do_profiling = False
    ):
        self.identifier_string = identifier_string
        self.dir_path_model = dir_path_model
        self.model_features_list = model_features_list

        self.lang = lang
        self.dirpath_synonymlist = dirpath_synonymlist
        self.postfix_synonymlist = postfix_synonymlist
        self.dir_wordlist = dir_wordlist
        self.postfix_wordlist = postfix_wordlist
        self.dir_wordlist_app = dir_wordlist_app
        self.postfix_wordlist_app = postfix_wordlist_app
        # Allowed root words are just the model features list
        self.allowed_root_words = self.model_features_list
        self.do_spelling_correction = do_spelling_correction
        self.do_word_stemming = do_word_stemming
        self.do_profiling = do_profiling

        #
        # We initialize word segmenter and synonym list after the model is ready
        # because it requires the model features so that root words of synonym lists
        # are only from the model features
        #
        self.wseg = None
        self.synonymlist = None
        self.spell_correction = None
        # Stemmer/Lemmatizer
        self.lang_have_verb_conj = False
        self.word_stemmer_lemmatizer = None

        ret_obj = langhelper.LangHelper.get_word_segmenter(
            lang                 = self.lang,
            dirpath_wordlist     = self.dir_wordlist,
            postfix_wordlist     = self.postfix_wordlist,
            dirpath_app_wordlist = self.dir_wordlist_app,
            postfix_app_wordlist = self.postfix_wordlist_app,
            dirpath_synonymlist  = self.dirpath_synonymlist,
            postfix_synonymlist  = self.postfix_synonymlist,
            # We can only allow root words to be words from the model features
            allowed_root_words   = self.model_features_list,
            do_profiling         = self.do_profiling
        )
        self.wseg = ret_obj.wseg
        self.synonymlist = ret_obj.snnlist

        #
        # For spelling correction
        #
        if self.do_spelling_correction:
            try:
                self.spell_correction = spellcor.SpellingCorrection(
                    lang              = self.lang,
                    words_list        = self.model_features_list,
                    dir_path_model    = self.dir_path_model,
                    identifier_string = self.identifier_string,
                    do_profiling      = self.do_profiling
                )
                log.Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Spelling Correction for model "' + str(self.identifier_string)
                    + '" initialized successfully.'
                )
            except Exception as ex_spellcor:
                self.spell_correction = None
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                         + ': Error initializing spelling correction for model "' \
                         + str(self.identifier_string) \
                         + '", got exception "' + str(ex_spellcor) + '".'
                log.Log.error(errmsg)

        #
        # For stemmer / lemmatization
        #
        if self.do_word_stemming:
            lfobj = langfeatures.LangFeatures()
            self.lang_have_verb_conj = lfobj.have_verb_conjugation(lang=self.lang)
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Lang "' + str(self.lang) + '" verb conjugation = ' + str(self.lang_have_verb_conj) + '.'
            )
            self.word_stemmer_lemmatizer = None
            if self.lang_have_verb_conj:
                try:
                    self.word_stemmer_lemmatizer = lmtz.Lemmatizer(
                        lang=self.lang
                    )
                    log.Log.important(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Lang "' + str(self.lang) + '" stemmer/lemmatizer initialized successfully.'
                    )
                except Exception as ex_stemmer:
                    errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                             + ': Lang "' + str(self.lang) + ' stemmer/lemmatizer failed to initialize: ' \
                             + str(ex_stemmer) + '.'
                    log.Log.error(errmsg)
                    self.word_stemmer_lemmatizer = None

        return

    #
    # Some things we do
    #   1. Segment text or word tokenization
    #   2. Normalize text, replacing synonyms with single word
    #   3. Spelling correction
    #   4. Stemming or Lemmatization
    #
    def process_text(
            self,
            inputtext
    ):
        # Segment words first
        inputtext_trim = su.StringUtils.trim(inputtext)
        # Returns a word array, e.g. ['word1', 'word2', 'x', 'y',...]
        text_segmented_arr = self.wseg.segment_words(
            text = inputtext_trim,
            return_array_of_split_words = True
        )

        #
        # Replace words with root words
        # This step uses synonyms and replaces say "красивая", "милая", "симпатичная", all with "красивая"
        # This will reduce training data without needing to put all versions of the same thing.
        #
        text_normalized_arr = self.synonymlist.normalize_text_array(
            text_segmented_array = text_segmented_arr
        )

        text_normalized_arr_lower = [s.lower() for s in text_normalized_arr]

        log.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Text "' + str(inputtext) + '" segmented to "' + str(text_segmented_arr)
            + '", normalized to "' + str(text_normalized_arr_lower) + '"'
        )

        #
        # Spelling correction
        #
        if self.do_spelling_correction:
            if self.spell_correction is not None:
                text_normalized_arr_lower = self.spell_correction.do_spelling_correction(
                    text_segmented_arr = text_normalized_arr_lower
                )
                log.Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Text "' + str(inputtext) + '" segmented to "' + str(text_segmented_arr)
                    + '", corrected spelling to "' + str(text_normalized_arr_lower) + '".'
                )
        #
        # Stemming / Lemmatization
        #
        if self.do_word_stemming and self.lang_have_verb_conj:
            if self.word_stemmer_lemmatizer:
                for i in range(len(text_normalized_arr_lower)):
                    text_normalized_arr_lower[i] = self.word_stemmer_lemmatizer.stem(
                        word = text_normalized_arr_lower[i]
                    )
                log.Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Text "' + str(inputtext) + '" segmented to "' + str(text_segmented_arr)
                    + '", stemmed to "' + str(text_normalized_arr_lower) + '".'
                )

        #
        # If not in model features, need to replace with '_UNK'
        #
        for i in range(len(text_normalized_arr_lower)):
            if text_normalized_arr_lower[i] not in self.model_features_list:
                text_normalized_arr_lower[i] = txtpcsr.TextProcessor.W_UNK

        log.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Done text processing to: ' + str(text_normalized_arr_lower)
            + ' from "' + str(inputtext) + '".'
        )

        return text_normalized_arr_lower
