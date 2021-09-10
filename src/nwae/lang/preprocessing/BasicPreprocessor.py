# -*- coding: utf-8 -*-

import nwae.utils.Log as log
from inspect import currentframe, getframeinfo
import nwae.lang.LangFeatures as lf
import re
import nwae.utils.UnitTest as ut


#
# This should be a text processing class, but mixed with some very basic text encoding methods.
# It is just a text utility class, that is all, no complete text pre-processing or any proper encoding.
# If you application is really simple, you may just use this.
# Otherwise you will need the more complete text preprocessor and proper encoding (e.g. text embedding) classes.
#
class BasicPreprocessor:

    #
    # Our own default word delimiter
    #
    DEFAULT_WORD_SPLITTER = '--||--'
    DEFAULT_SPACE_SPLITTER = ' '

    # Need '◦'?
    DEFAULT_PUNCTUATIONS = '([!?.,/？。，:;$“"\')(\[\]{}«»、])'

    # Sentence padding if shorter than min length
    W_PAD = '_pad'
    # Start of sentence
    W_GO  = '_go'
    # End of sentence
    W_EOS = '_eos'
    # Unknown word
    W_UNK = '_unk'
    # Other common symbols
    # Number
    W_NUM = '_num'
    # Username or any word with mix of character/numbers/etc
    W_USERNAME = '_username'
    W_USERNAME_NONWORD = '_username_nonword'
    # URL
    W_URI = '_uri'

    ALL_SPECIAL_SYMBOLS = (
        W_PAD, W_GO, W_EOS, W_UNK,
        W_NUM, W_USERNAME, W_USERNAME_NONWORD, W_URI
    )

    _START_VOCAB = [W_PAD, W_GO, W_EOS, W_UNK]
    PAD_ID = 0
    GO_ID  = 1
    EOS_ID = 2
    UNK_ID = 3
    OP_DICT_IDS = [PAD_ID, GO_ID, EOS_ID, UNK_ID]
    INDEXED_DICT_BASE = {
        W_PAD: PAD_ID, W_GO: GO_ID, W_EOS: EOS_ID, W_UNK: UNK_ID
    }

    #
    # The difference between this and the function get_word_separator_type() in
    # LangFeatures class is that this is OUR OWN encoding, whereas the other one
    # is the natural human written word separator.
    #
    @staticmethod
    def get_word_separator(
            lang
    ):
        lang_std = lf.LangFeatures.map_to_lang_code_iso639_1(
            lang_code = lang
        )
        word_separator = BasicPreprocessor.DEFAULT_SPACE_SPLITTER
        if lang_std == lf.LangFeatures.LANG_VI:
            word_separator = BasicPreprocessor.DEFAULT_WORD_SPLITTER
        return word_separator

    #
    # This is just a very basic function to do some cleaning, it is expected that
    # fundamental cleaning has already been done before coming here.
    #
    @staticmethod
    def clean_punctuations(
            # list of words
            sentence,
            punctuations_pattern = DEFAULT_PUNCTUATIONS,
            convert_to_lower_case = True
    ):
        try:
            # Don't include space separator, if you need to split by space, do it before coming here,
            # as we are only cleaning here, and may include languages like Vietnamese, so if we include
            # space here, we are splitting the word into syllables, which will be wrong.
            regex_word_split = re.compile(pattern=punctuations_pattern)
            # Split words not already split (e.g. 17. should be '17', '.')
            # re.split() will add a redundant '' at the end, so we have to remove later.
            if convert_to_lower_case:
                clean_words = [re.split(regex_word_split, word.lower()) for word in sentence]
            else:
                clean_words = [re.split(regex_word_split, word) for word in sentence]
            # Return non-empty split values, w
            # Same as:
            # for words in clean_words:
            #     for w in words:
            #         if words:
            #             if w:
            #                 w
            # All None and '' will be filtered out
            return [w for words in clean_words for w in words if words if w]
        except Exception as ex:
            errmsg =\
                str(BasicPreprocessor.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                + ': Could not clean punctuations and convert to lowercase for sentence "'\
                + str(sentence) + '" exception message: ' + str(ex) + '.'
            log.Log.error(errmsg)
            return None

    @staticmethod
    def create_indexed_dictionary(
            # list of list of words
            sentences,
            include_indexed_dict_base = True
    ):
        if include_indexed_dict_base:
            indexed_dict = BasicPreprocessor.INDEXED_DICT_BASE.copy()
            max_index = max(indexed_dict.values())
        else:
            indexed_dict = {}
            max_index = -1

        for i in range(len(sentences)):
            sent = sentences[i]
            for j in range(len(sent)):
                word = sent[j]
                if word not in indexed_dict.keys():
                    max_index += 1
                    indexed_dict[word] = max_index
        return indexed_dict

    @staticmethod
    def sentences_to_indexes(
            sentences,
            indexed_dict
    ):
        encoded_sentences = []

        for sent in sentences:
            encoded_sent = []
            log.Log.debugdebug('Processing sentence: ' + str(sent))
            for w in sent:
                if w in indexed_dict.keys():
                    encoded_sent.append(indexed_dict[w])
                else:
                    encoded_sent.append(BasicPreprocessor.UNK_ID)
            encoded_sentences.append(encoded_sent)
            log.Log.debugdebug('Encoded sentence: ' + str(encoded_sent))
        return encoded_sentences

    @staticmethod
    def indexes_to_sentences(
            indexes,
            indexed_dict
    ):
        original_sentences = []
        inv_map = {v: k for k, v in indexed_dict.items()}

        for sent in indexes:
            original_sent = []
            log.Log.debug('Processing indexed sentence: ' + str(sent))
            for idx in sent:
                if idx in inv_map.keys():
                    original_sent.append(inv_map[idx])
                else:
                    original_sent.append(BasicPreprocessor.W_UNK)
            original_sentences.append(original_sent)
            log.Log.debug('Original sentence: ' + str(original_sent))
        return original_sentences

    @staticmethod
    def extract_max_length(
            corpora
    ):
        max_len = 0
        for sent in corpora:
            max_len = max(max_len, len(sent))
        return max_len

    @staticmethod
    def prepare_sentence_pairs(
            sentences_l1,
            sentences_l2,
            len_l1,
            len_l2
    ):
        assert len(sentences_l1) == len(sentences_l2)
        data_set = []
        for i in range(len(sentences_l1)):
            padding_l1 = len_l1 - len(sentences_l1[i])
            pad_sentence_l1 = ([BasicPreprocessor.PAD_ID]*padding_l1) + sentences_l1[i]

            padding_l2 = len_l2 - len(sentences_l2[i])
            pad_sentence_l2 = [BasicPreprocessor.GO_ID] + sentences_l2[i] + ([BasicPreprocessor.PAD_ID]*padding_l2)

            data_set.append([pad_sentence_l1, pad_sentence_l2])
        log.Log.info(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Appended ' + str(len(data_set)) + ' sentence pairs to data set.'
        )
        return data_set


class BasicPreprocessorUnitTest:
    def __init__(self, ut_params):
        self.ut_params = ut_params
        if self.ut_params is None:
            self.ut_params = ut.UnitTestParams()
        return

    def run_unit_test(self):
        res = ut.ResultObj(count_ok=0, count_fail=0)

        sentences = [
            (['Capital', '[tesT]', 'ok?'],
             ['capital', '[', 'test', ']', 'ok', '?']),   # Expected cleaned result
            (['I', 'like', '{coding}', 'la!'],
             ['i', 'like', '{', 'coding', '}', 'la', '!']),   # Expected cleaned result
            (['Самый', 'лучщий', 'филмь', 'всей', 'истории', '"등등등"'],
             ['самый', 'лучщий', 'филмь', 'всей', 'истории', '"', '등등등', '"'])  # Expected cleaned result
        ]
        cleaned_sentences = [BasicPreprocessor.clean_punctuations(sentence=s[0]) for s in sentences]
        expected_cleaned_sentences = [s[1] for s in sentences]
        log.Log.debugdebug(cleaned_sentences)

        #
        # Test cleaning process of separating punctuations, lowercase, etc.
        #
        for i in range(len(cleaned_sentences)):
            observed_sent = cleaned_sentences[i]
            expected_sent = expected_cleaned_sentences[i]
            res.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = observed_sent,
                expected = expected_sent,
                test_comment = 'test sentence "' + str(observed_sent) + '"'
            ))

        index_dict = BasicPreprocessor.create_indexed_dictionary(sentences=cleaned_sentences)
        log.Log.debugdebug(index_dict)

        encoded_sentences = BasicPreprocessor.sentences_to_indexes(
            sentences=cleaned_sentences,
            indexed_dict=index_dict
        )
        log.Log.debugdebug(encoded_sentences)

        # When revert back to original sentence, should be the same with original sentence
        original_sentences = BasicPreprocessor.indexes_to_sentences(
            indexes=encoded_sentences,
            indexed_dict=index_dict
        )
        log.Log.debugdebug(original_sentences)

        #
        # Test encoded sentences and decoding back to original sentence
        #
        for i in range(len(original_sentences)):
            observed_sent = original_sentences[i]
            expected_sent = cleaned_sentences[i]
            res.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = observed_sent,
                expected = expected_sent,
                test_comment = 'test sentence "' + str(observed_sent) + '"'
            ))
        log.Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Passed ' + str(res.count_ok) + ', Failed ' + str(res.count_fail)
        )
        return res


if __name__ == '__main__':
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_DEBUG_1
    res = BasicPreprocessorUnitTest(ut_params=None).run_unit_test()
    exit(res.count_fail)