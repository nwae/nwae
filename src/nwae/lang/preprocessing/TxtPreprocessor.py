# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
import nwae.utils.StringUtils as su
from inspect import currentframe, getframeinfo
from nwae.lang.preprocessing.BasicPreprocessor import BasicPreprocessor
import nwae.lang.LangHelper as langhelper
import nwae.lang.characters.LangCharacters as langchar
import nwae.lang.LangFeatures as langfeatures
import nwae.lang.nlp.sajun.SpellCheckSentence as spellcor
import nwae.lang.nlp.lemma.Lemmatizer as lmtz
import re
from mex.MexBuiltInTypes import MexBuiltInTypes
from mex.MatchExpression import MatchExpression


#
# Самая важная и основная обработка тесктов в первом этапе любой обработки NLP, Машинного Обучения или
# Искусственного Интеллекта.
# Только предварительная обработка исходных текстовых данных в "чистую" форму текста без кодирования в числа.
#
# The method process_text() takes in a string, and outputs a list of split tokens, and cleaned with the
# listed processes below.
# However these split tokens are still in string form, and further processing is still required to convert
# them to label, one-hot encodings, or word embeddings to be suitable for further ML/AI processing.
#
#   1. Replace special tokens like URLs with special symbols
#   2. Segment text or word tokenization
#      For languages like English, Spanish, Korean, etc, this is easy, only split by space.
#      But for languages with no spaces like Chinese, Japanese, Thai, Lao, Cambodian,
#      or languages that have only syllable boundaries like Vietnamese & Chinese,
#      this is a complicated process.
#   3. Convert to lowercase, clean/separate common punctuations from words
#   4. Normalize text, replacing synonyms with single word
#   5. Spelling correction
#   6. Remove stopwords
#   7. Stemming or Lemmatization
#      Stemming does not necessarily generate a dictionary/valid word, its sole purpose is just to
#      reduce conjugated words to the same root.
#      Thus if the meaning of a word is important, it needs to be lemmatized instead of stemmed
#
# When model updates, this also need to update. So be careful.
#
class TxtPreprocessor:

    def __init__(
            self,
            identifier_string,
            # If None, will not do spelling correction
            dir_path_model,
            # If None, will not replace any word with unknown symbol W_UNK
            model_features_list,
            lang,
            dirpath_synonymlist,
            postfix_synonymlist,
            dir_wordlist,
            postfix_wordlist,
            dir_wordlist_app,
            postfix_wordlist_app,
            # For certain languages like English, it is essential to include this,
            # otherwise predict accuracy will drop drastically.
            # But at the same time must be very careful. By adding manual rules, for
            # example we include words 'it', 'is'.. But "It is" could be a very valid
            # training data that becomes excluded wrongly.
            stopwords_list = None,
            do_spelling_correction = False,
            do_word_stemming = True,
            do_profiling = False
    ):
        self.identifier_string = identifier_string
        self.dir_path_model = dir_path_model
        self.model_features_list = model_features_list
        
        self.lang = langfeatures.LangFeatures.map_to_lang_code_iso639_1(
            lang_code = lang
        )
        self.dirpath_synonymlist = dirpath_synonymlist
        self.postfix_synonymlist = postfix_synonymlist
        self.dir_wordlist = dir_wordlist
        self.postfix_wordlist = postfix_wordlist
        self.dir_wordlist_app = dir_wordlist_app
        self.postfix_wordlist_app = postfix_wordlist_app
        # Allowed root words are just the model features list
        self.allowed_root_words = self.model_features_list
        self.stopwords_list = stopwords_list
        self.do_spelling_correction = do_spelling_correction
        self.do_word_stemming = do_word_stemming
        self.do_profiling = do_profiling

        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Using wordlist dir "' + str(self.dir_wordlist)
            + '", app wordlist dir "' + str(self.dir_wordlist_app)
            + '", synonym dir "' + str(self.dirpath_synonymlist) + '"'
        )

        self.words_no_replace_with_special_symbols = \
            list(langchar.LangCharacters.UNICODE_BLOCK_WORD_SEPARATORS) + \
            list(langchar.LangCharacters.UNICODE_BLOCK_SENTENCE_SEPARATORS) + \
            list(langchar.LangCharacters.UNICODE_BLOCK_PUNCTUATIONS) + \
            list(BasicPreprocessor.ALL_SPECIAL_SYMBOLS)

        self.words_no_replace_with_special_symbols = list(set(self.words_no_replace_with_special_symbols))
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': For model "' + str(self.identifier_string)
            + '", words that will not replace with special symbols: '
            + str(self.words_no_replace_with_special_symbols)
        )

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
                self.spell_correction = spellcor.SpellCheckSentence(
                    lang              = self.lang,
                    words_list        = self.model_features_list,
                    dir_path_model    = self.dir_path_model,
                    identifier_string = self.identifier_string,
                    do_profiling      = self.do_profiling
                )
                Log.important(
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
                Log.error(errmsg)

        #
        # For stemmer / lemmatization
        #
        if self.do_word_stemming:
            lfobj = langfeatures.LangFeatures()
            self.lang_have_verb_conj = lfobj.have_verb_conjugation(lang=self.lang)
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Lang "' + str(self.lang) + '" verb conjugation = ' + str(self.lang_have_verb_conj) + '.'
            )
            self.word_stemmer_lemmatizer = None
            if self.lang_have_verb_conj:
                try:
                    self.word_stemmer_lemmatizer = lmtz.Lemmatizer(
                        lang=self.lang
                    )
                    Log.important(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Lang "' + str(self.lang) + '" stemmer/lemmatizer initialized successfully.'
                    )
                except Exception as ex_stemmer:
                    errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                             + ': Lang "' + str(self.lang) + ' stemmer/lemmatizer failed to initialize: ' \
                             + str(ex_stemmer) + '.'
                    Log.error(errmsg)
                    self.word_stemmer_lemmatizer = None

        self.mex_username_nonword = MatchExpression(
            pattern = 'u,' + MexBuiltInTypes.MEX_TYPE_USERNAME_NONWORD + ',',
            lang    = None
        )
        return

    def __is_username_nonword_type(
            self,
            word
    ):
        params_dict = self.mex_username_nonword.get_params(
            sentence=word,
            return_one_value=True  # if False will return both (left,right) values
        )
        is_username_nonword_type = params_dict['u'] is not None
        return is_username_nonword_type

    def __remove_stopwords(
            self,
            word_list
    ):
        if self.stopwords_list:
            word_list_remove = []
            for w in word_list:
                if w not in self.stopwords_list:
                    word_list_remove.append(w)
            Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Lang "' + str(self.lang) + '", Word list "' + str(word_list)
                + '", removed stopwords to "' + str(word_list_remove) + '".'
            )
            return word_list_remove
        else:
            return word_list

    def preprocess_list(
            self,
            sentences_list,
    ):
        return [self.process_text(inputtext=s, return_as_string=False) for s in sentences_list]

    #
    # Some things we do
    #   1. Replace special tokens like URLs with special symbols
    #   2. Segment text or word tokenization
    #   3. Convert to lowercase, clean/separate common punctuations from words
    #   4. Normalize text, replacing synonyms with single word
    #   5. Spelling correction
    #   6. Remove stopwords
    #   7. Stemming or Lemmatization
    #
    def process_text(
            self,
            inputtext,
            return_as_string = False,
            use_special_symbol_username_nonword = False
    ):
        #
        # 1st Round replace with very special symbols first, that must be done before
        # word segmentation or cleaning.
        # Be careful here, don't simply replace things.
        # For symbols that can wait until after word segmentation like numbers, unknown
        # words, we do later.
        #
        pat_rep_list = [
            {
                'pattern': MexBuiltInTypes.REGEX_URI,
                'repl': ' ' + BasicPreprocessor.W_URI + ' '
            },
        ]
        inputtext_sym =  su.StringUtils.trim(str(inputtext))
        for pat_rep in pat_rep_list:
            inputtext_sym = re.sub(
                pattern = pat_rep['pattern'],
                repl    = pat_rep['repl'],
                string  = inputtext_sym
            )
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Lang "' + str(self.lang) + '", Text "' + str(inputtext)
            + '" special pattern replacement to: ' + str(inputtext_sym)
        )

        #
        # Segment words
        #
        # Returns a word array, e.g. ['word1', 'word2', 'x', 'y',...]
        text_segmented_arr = self.wseg.segment_words(
            text = inputtext_sym,
            return_array_of_split_words = True
        )
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Lang "' + str(self.lang) + '", Text "' + str(inputtext)
            + '" segmented to: ' + str(text_segmented_arr)
        )

        #
        # Remove basic punctuations stuck to word
        #
        # Will return None on error
        tmp_arr = BasicPreprocessor.clean_punctuations(
            sentence = text_segmented_arr
        )
        if type(tmp_arr) in [list, tuple]:
            text_segmented_arr = tmp_arr
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Lang "' + str(self.lang) + '", Text "' + str(inputtext)
            + '" clean punctuations to: ' + str(text_segmented_arr)
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

        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Lang "' + str(self.lang) + '", Text "' + str(inputtext)
            + '", normalized to "' + str(text_normalized_arr_lower) + '"'
        )

        #
        # Spelling correction
        #
        if self.do_spelling_correction:
            if self.spell_correction is not None:
                text_normalized_arr_lower = self.spell_correction.check(
                    text_segmented_arr = text_normalized_arr_lower
                )
                Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Lang "' + str(self.lang) + '", Text "' + str(inputtext)
                    + '", corrected spelling to "' + str(text_normalized_arr_lower) + '".'
                )

        #
        # Remove stopwords before stemming
        #
        text_normalized_arr_lower = self.__remove_stopwords(
            word_list = text_normalized_arr_lower
        )

        #
        # Stemming / Lemmatization
        #
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Lang "' + str(self.lang)
            + '" do stemming = ' + str(self.do_word_stemming)
            + ', have verb conjugation = ' + str(self.lang_have_verb_conj)
        )
        if self.do_word_stemming and self.lang_have_verb_conj:
            if self.word_stemmer_lemmatizer:
                for i in range(len(text_normalized_arr_lower)):
                    text_normalized_arr_lower[i] = self.word_stemmer_lemmatizer.stem(
                        word = text_normalized_arr_lower[i]
                    )
                Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Lang "' + str(self.lang) + '", Text "' + str(inputtext)
                    + '", stemmed to "' + str(text_normalized_arr_lower) + '".'
                )

        #
        # 2nd round replace with special symbols for numbers, unrecognized vocabulary, etc.
        # MUST NOT accidentally replace our earlier special symbols like _uri, etc.
        #
        for i in range(len(text_normalized_arr_lower)):
            word = text_normalized_arr_lower[i]

            #
            # Punctuations, special symbols themselves, etc, will not undergo this process
            #
            if word in self.words_no_replace_with_special_symbols:
                continue

            # Check numbers first, re.match() is fast enough
            # Replace numbers with separate symbol
            if re.match(pattern='^[0-9]+$', string=word):
                Log.debugdebug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Found for number in word "' + str(word) + '"'
                )
                text_normalized_arr_lower[i] = BasicPreprocessor.W_NUM
            elif self.model_features_list is not None:
                if word not in self.model_features_list:
                    text_normalized_arr_lower[i] = BasicPreprocessor.W_UNK
                    if use_special_symbol_username_nonword:
                        # Check if it is a username_nonword form
                        if self.__is_username_nonword_type(word=word):
                            text_normalized_arr_lower[i] = BasicPreprocessor.W_USERNAME_NONWORD
            else:
                if use_special_symbol_username_nonword:
                    if self.__is_username_nonword_type(word=word):
                        text_normalized_arr_lower[i] = BasicPreprocessor.W_USERNAME_NONWORD

        #
        # Remove stopwords again after stemming
        #
        text_normalized_arr_lower = self.__remove_stopwords(
            word_list = text_normalized_arr_lower
        )

        #
        # Finally remove empty words in array
        #
        text_normalized_arr_lower = [x for x in text_normalized_arr_lower if x != '']

        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Lang "' + str(self.lang)
            + '", Done text processing to: ' + str(text_normalized_arr_lower)
            + ' from "' + str(inputtext) + '".'
        )

        if return_as_string:
            print_separator = BasicPreprocessor.get_word_separator(
                lang = self.lang
            )
            return print_separator.join(text_normalized_arr_lower)
        else:
            return text_normalized_arr_lower


if __name__ == '__main__':
    from nwae.lang.config.Config import Config
    config = Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = Config,
        default_config_file = '/usr/local/git/nwae/nwae.lang/app.data/config/default.cf'
    )

    from nwae.lang.LangFeatures import LangFeatures
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_2

    obj = TxtPreprocessor(
        identifier_string      = 'test',
        # Don't need directory path for model, as we will not do spelling correction
        dir_path_model         = None,
        # Don't need features/vocabulary list from model
        model_features_list    = None,
        lang                   = LangFeatures.LANG_TH,
        dirpath_synonymlist    = config.get_config(param=Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist    = config.get_config(param=Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        dir_wordlist           = config.get_config(param=Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist       = config.get_config(param=Config.PARAM_NLP_POSTFIX_WORDLIST),
        dir_wordlist_app       = config.get_config(param=Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_wordlist_app   = config.get_config(param=Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        do_spelling_correction = False,
        do_word_stemming       = False,
        do_profiling           = False
    )

    texts = [
        'ปั่นสล็อต100ครั้ง',
        'อูเสอgeng.mahk_mahk123ได้'
    ]

    for txt in texts:
        obj.process_text(inputtext = txt)

    #
    # Must be able to handle without word lists
    #
    obj = TxtPreprocessor(
        identifier_string      = 'test',
        # Don't need directory path for model, as we will not do spelling correction
        dir_path_model         = None,
        # Don't need features/vocabulary list from model
        model_features_list    = None,
        lang                   = LangFeatures.LANG_RU,
        dirpath_synonymlist    = None,
        postfix_synonymlist    = None,
        dir_wordlist           = None,
        postfix_wordlist       = None,
        dir_wordlist_app       = None,
        postfix_wordlist_app   = None,
        do_spelling_correction = False,
        do_word_stemming       = False,
        do_profiling           = False,
    )
    texts = [
        'Аккаунт популярного южнокорейского чат-бота был заблокирован',
        'после жалоб на ненавистнические высказывания в адрес сексуальных меньшинств',
    ]
    for txt in texts:
        obj.process_text(inputtext = txt)

    texts = [
        '存款10次http://abc.com',
    ]
    obj = TxtPreprocessor(
        identifier_string      = 'test',
        # Don't need directory path for model, as we will not do spelling correction
        dir_path_model         = None,
        # Don't need features/vocabulary list from model
        model_features_list    = None,
        lang                   = LangFeatures.LANG_EN,
        dirpath_synonymlist    = config.get_config(param=Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist    = config.get_config(param=Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        dir_wordlist           = config.get_config(param=Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist       = config.get_config(param=Config.PARAM_NLP_POSTFIX_WORDLIST),
        dir_wordlist_app       = config.get_config(param=Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_wordlist_app   = config.get_config(param=Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        do_spelling_correction = False,
        do_word_stemming       = False,
        do_profiling           = False
    )
    for txt in texts:
        obj.process_text(inputtext = txt)

    exit(0)
