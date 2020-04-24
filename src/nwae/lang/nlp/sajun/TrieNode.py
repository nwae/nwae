# -*- coding: utf-8 -*-
# By Steve Hanov, 2011. Released to the public domain
# Code modified from http://stevehanov.ca/blog/?id=114 extend functionality to support
# transposition Damerau-Levenschtein, and separated the math from the recursion

import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo
import time
import nwae.lang.nlp.WordList as wl
from nwae.lang.LangFeatures import LangFeatures
import nwae.utils.UnitTest as ut
from nwae.lang.config.Config import Config
from nwae.lang.nlp.sajun.EditDistance import EditDistance
import numpy as np


#
# The Trie data structure keeps a set of words, organized with one node (one TrieNode object)
# for each letter.
# Some nodes may have a word attribute, some may not.
# We support both the Damerau-Levenshtein & Levenstein algorithms for edit distance measure
#
class TrieNode:

    # Keep some interesting statistics
    NODE_COUNT = 0
    WORD_COUNT = 0

    @staticmethod
    def build_trie_node(
            words
    ):
        start = time.time()

        trie = TrieNode()
        # read dictionary file into a trie
        for word in words:
            TrieNode.WORD_COUNT += 1
            trie.insert(word)

        end = time.time()
        lg.Log.important(
            str(TrieNode.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Successfully build Trie Node of ' + str(TrieNode.NODE_COUNT)
            + ' nodes, and ' + str(TrieNode.WORD_COUNT) + ' total words. Took '
            + str(round((end - start), 5)) + ' secs.'
        )
        return trie

    def __init__(
            self,
    ):
        # Some nodes are not words (e.g. "w", "wo", "wor" from "word"), so default to None
        self.word = None
        # Branch off here with more TrieNode class objects
        self.children = {}
        # Need to count using global variable as this is a linked set of TrieNode objects
        TrieNode.NODE_COUNT += 1
        return

    def insert(
            self,
            word
    ):
        node = self
        #
        # Create new nodes if needed, and at the end record the word
        #
        for letter in word:
            if letter not in node.children:
                # New branch
                node.children[letter] = TrieNode()
            # Where we are now
            node = node.children[letter]
        # At the final point, record the word
        node.word = word
        return

    # The search function returns a list of all words that are less than the given
    # maximum distance from the target word
    @staticmethod
    def search_close_words(
            trie,
            word,
            max_cost = 1,
            # 'levenshtein', 'damerau–levenshtein'
            edit_distance_algo = EditDistance.EDIT_DIST_ALGO_DAMLEV,
            use_numpy = False
    ):
        edit_dist = EditDistance(
            algo = edit_distance_algo
        )

        # build first row. 0,1,2,3...,len(word)
        start_row = np.array(
            [ list( range( len(word) + 1 ) ) ]
        )

        results = []

        # recursively search each branch of the trie
        for letter in trie.children:
            TrieNode.search_close_words_recursive(
                node         = trie.children[letter],
                word_part    = letter,
                # 2nd reference word, laid out on the "top row"
                word         = word,
                previous_row = start_row,
                results      = results,
                max_cost     = max_cost,
                edit_dist    = edit_dist,
                use_numpy    = use_numpy,
                prev_word_part_letters = None
            )

        return results

    #
    # The most general among the <edit distance> family of algorithm is
    # the Damerau–Levenshtein distance, that allows all below
    #   1. Insertion
    #   2. Deletion
    #   3. Substitution
    #   4. Transposition of 2 adjacent characters
    #
    # This recursive helper is used by the search function above. It assumes that
    # the previousRow has been filled in already.
    #
    @staticmethod
    def search_close_words_recursive(
            node,
            # A few letters from the "left column" word
            # Index 0 is the current letter, index 1 is one letter back, index 2 is two letters back, ...
            word_part,
            # The full "top row" word
            word,
            # nd.ndarray, 2 dimensional, shape (,len(word)+1). can be more than 1 row
            previous_row,
            results,
            max_cost,
            edit_dist,
            prev_word_part_letters = None,
            # Numpy is slower!
            use_numpy = False
    ):
        retc = edit_dist.calculate(
            word_1 = word_part,
            word_2 = word,
            start_rows_word_1 = previous_row,
            start_letters_word_1 = prev_word_part_letters,
            use_np = use_numpy
        )
        dist_matrix = retc.distance_matrix
        optimal_cost = retc.optimal_cost
        # Pass at least 2 rows if possible so we can do the full Damerau-Levenshtein distance
        if use_numpy:
            len_dm = dist_matrix.shape[0]
        else:
            len_dm = len(dist_matrix)
        current_row = dist_matrix[(len_dm-2):len_dm]

        # if the last entry in the row indicates the optimal cost is less than the
        # maximum cost, and there is a word in this trie node, then add it.
        if optimal_cost <= max_cost:
            if node.word is not None:
                results.append( (node.word, optimal_cost) )

        # if any entries in the row are less than the maximum cost, then
        # recursively search each branch of the trie
        if use_numpy:
            min_value_current_row = np.min(current_row[-1])
        else:
            min_value_current_row = min(current_row[-1])

        if min_value_current_row <= max_cost:
            for letter in node.children:
                if use_numpy:
                    # Create a new object so we don't modify it, we still need it for next letter
                    prev_rows = np.array(current_row)
                else:
                    # Create a new object so we don't modify it, we still need it for next letter
                    prev_rows = np.array(current_row).tolist()

                TrieNode.search_close_words_recursive(
                    node         = node.children[letter],
                    word_part    = letter,
                    word         = word,
                    previous_row = prev_rows,
                    results      = results,
                    max_cost     = max_cost,
                    edit_dist    = edit_dist,
                    prev_word_part_letters = word_part
                )
        return


class TrieNodeUnitTest:

    def __init__(
            self,
            ut_params
    ):
        self.ut_params = ut_params
        if self.ut_params is None:
            # We only do this for convenience, so that we have access to the Class methods in UI
            self.ut_params = ut.UnitTestParams()

    def run_unit_test(self):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)

        lang = LangFeatures.LANG_TH
        # read dictionary file into a trie
        wl_obj = wl.WordList(
            lang             = lang,
            dirpath_wordlist = self.ut_params.dirpath_wordlist,
            postfix_wordlist = self.ut_params.postfix_wordlist
        )
        words = wl_obj.wordlist[wl.WordList.COL_WORD].tolist()

        trie = TrieNode.build_trie_node(
            words = words
        )
        lg.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ": Read %d words into %d nodes" % (TrieNode.WORD_COUNT, TrieNode.NODE_COUNT)
        )

        test_data = [
            ('เงน', {
             EditDistance.EDIT_DIST_ALGO_DAMLEV: (
                ('เกน', 1), ('เขน', 1), ('เคน', 1), ('เงก', 1), ('เงย', 1), ('เงา', 1), ('เงิน', 1), ('เง็น', 1),
                ('เง้', 1), ('เจน', 1), ('เชน', 1), ('เซน', 1), ('เดน', 1), ('เถน', 1), ('เบน', 1), ('เมน', 1),
                ('เลน', 1), ('เวน', 1), ('เสน', 1), ('เอน', 1), ('โงน', 1), ('ฉงน', 1)
             ),
             EditDistance.EDIT_DIST_ALGO_LEV: (
                ('เกน', 1), ('เขน', 1), ('เคน', 1), ('เงก', 1), ('เงย', 1), ('เงา', 1), ('เงิน', 1), ('เง็น', 1),
                ('เง้', 1), ('เจน', 1), ('เชน', 1), ('เซน', 1), ('เดน', 1), ('เถน', 1), ('เบน', 1), ('เมน', 1),
                ('เลน', 1), ('เวน', 1), ('เสน', 1), ('เอน', 1), ('โงน', 1), ('ฉงน', 1)
             )}
            ),
            ('ถนอ', {
                EditDistance.EDIT_DIST_ALGO_DAMLEV: (
                    ('เนอ', 1), ('ถนน', 1), ('ถนป', 1), ('ถนอม', 1), ('ถนะ', 1), ('ถนำ', 1), ('ถมอ', 1), ('ถอน', 1),
                    ('ถือ', 1), ('ถ่อ', 1), ('ถ้อ', 1), ('นอ', 1), ('หนอ', 1)
                ),
                # Should be missing ('ถอน', 1)
                EditDistance.EDIT_DIST_ALGO_LEV: (
                    ('เนอ', 1), ('ถนน', 1), ('ถนป', 1), ('ถนอม', 1), ('ถนะ', 1), ('ถนำ', 1), ('ถมอ', 1),
                    ('ถือ', 1), ('ถ่อ', 1), ('ถ้อ', 1), ('นอ', 1), ('หนอ', 1)
                )}
            )
        ]

        for algo in [EditDistance.EDIT_DIST_ALGO_LEV, EditDistance.EDIT_DIST_ALGO_DAMLEV]:
            edit_dist_calculator = EditDistance(
                algo=algo
            )
            for use_numpy in [True, False]:
                for i in range(len(test_data)):
                    word_corrections_tuple = test_data[i]
                    word = word_corrections_tuple[0]
                    corrections = word_corrections_tuple[1][algo]

                    start = time.time()
                    results = TrieNode.search_close_words(
                        trie = trie,
                        word = word,
                        max_cost = 1,
                        edit_distance_algo = algo,
                        use_numpy = use_numpy
                    )
                    end = time.time()
                    lg.Log.debug('algo="' + str(algo) + '", numpy=' + str(use_numpy) + ', Results of search: ' + str(results))
                    lg.Log.debug('algo="' + str(algo) + '", numpy=' + str(use_numpy) + ', Search took ' + str(end - start) + 's.')

                    res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                        observed     = len(results),
                        expected     = len(corrections),
                        test_comment = 'algo="' + str(algo) + '", numpy=' + str(use_numpy)
                                       + ', test search results count ' + str(len(results))
                    ))
                    for observed_word_dist_tuple in results:
                        observed_word = observed_word_dist_tuple[0]
                        observed_dist = observed_word_dist_tuple[1]

                        res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                            observed = observed_word_dist_tuple in corrections,
                            expected = True,
                            test_comment = 'algo="' + str(algo) + '", numpy=' + str(use_numpy)
                                           + ', test word #' + str(i) + ' "' + str(word)
                                           + '" tuple (word, edit-distance) ' + str(observed_word_dist_tuple)
                        ))
                        # Confirm using edit distance
                        res = edit_dist_calculator.calculate(
                            word_1 = observed_word,
                            word_2 = word,
                            use_np = use_numpy
                        )
                        res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                            observed = res.optimal_cost,
                            expected = observed_dist,
                            test_comment = 'algo="' + str(algo) + '", numpy=' + str(use_numpy)
                                           + ', test distance word #' + str(i) + ' "' + str(observed_word)
                                           + '" and "' + str(word) + '" = ' + str(res.optimal_cost)
                        ))

        return res_final


if __name__ == '__main__':
    config = Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class       = Config,
        default_config_file = Config.CONFIG_FILE_PATH_DEFAULT
    )
    lg.Log.LOGLEVEL = lg.Log.LOG_LEVEL_DEBUG_1

    ut_params = ut.UnitTestParams(
        dirpath_wordlist     = config.get_config(param=Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist     = config.get_config(param=Config.PARAM_NLP_POSTFIX_WORDLIST),
        dirpath_app_wordlist = config.get_config(param=Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_app_wordlist = config.get_config(param=Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        dirpath_synonymlist  = config.get_config(param=Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist  = config.get_config(param=Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        dirpath_model        = None,
        dirpath_sample_data  = config.get_config(param=Config.PARAM_NLP_DIR_SAMPLE_DATA),
    )

    res = TrieNodeUnitTest(ut_params=ut_params).run_unit_test()
    exit(res.count_fail)


