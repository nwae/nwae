# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
import time
import nwae.utils.UnitTest as ut
from nwae.lang.config.Config import Config
import numpy as np


class RetClass:
    def __init__(self, distance_matrix, optimal_cost, algo):
        self.distance_matrix = distance_matrix
        self.optimal_cost = optimal_cost
        self.algo = algo


#
# We support both the Damerau-Levenshtein & Levenstein algorithms for edit distance measure
#
class EditDistance:

    # Using numpy is about 5 times slower!!
    USE_NUMPY = False

    EDIT_DIST_ALGO_DAMLEV = 'damerau–levenshtein'
    EDIT_DIST_ALGO_LEV    = 'levenshtein'

    def __init__(
            self,
            algo = EDIT_DIST_ALGO_DAMLEV
    ):
        self.algo = algo
        return

    def __calculate_use_numpy(
            self,
            # The "left column" word
            word_1,
            # The "top row" word
            word_2,
            # Option to provide a start row, e.g. for recursive calculations
            # Column length must be consistent with word_2 1 + length of word_2
            # At most we will just take the last 2 rows
            start_rows_word_1 = None,
            # If we specify our own start rows, means there must be start word or letters.
            # At most we will just take the last 2 letters
            start_letters_word_1 = None
    ):
        # Put word 2 on the "top row"
        n_columns = len(word_2) + 1
        # Put word 1 on the "left column". We want to transform word 1 to word 2
        n_rows = len(word_1) + 1

        excess_rows = 0
        word_1_edited = word_1
        # If the start rows are passed in (number of rows >= 2), and not using numpy, we will have this
        prev_prev_row = None

        if start_rows_word_1 is not None:
            ok = True
            len_rows = start_rows_word_1.shape[0]
            len_cols = start_rows_word_1.shape[1]
            ok = (ok) and (len_cols == n_columns)
            if not ok:
                raise Exception(
                    'Provided start row: ' + str(start_rows_word_1)
                    + ' column length must be consistent with word_2 "' + str(word_2)
                    + '" length plus 1 = ' + str(n_columns)
                )

            # If we extract more than 1 row, this will change the size of the numpy
            # distance_matrix array, with more rows than n_rows
            excess_rows = min(1, len_rows-1)
            prev_row = start_rows_word_1[(len_rows-1-excess_rows):len_rows]
            if excess_rows > 1:
                for i in range(excess_rows, 0, -1):
                    word_1_edited = start_letters_word_1[-i] + word_1_edited
        else:
            # The usual empty letter row in standard edit distance calculation
            prev_row = np.array( [ list(range(n_columns)) ] )

        # Append given start rows
        distance_matrix = np.append(prev_row, np.zeros((n_rows - 1, n_columns)), axis=0)

        Log.debugdebug(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Top row word "' + str(word_2) + '" ' + str(n_columns) + ' columns, '
            + ' left column word "' + str(word_1) + '" ' + str(n_rows) + ' rows.'
            + ' Edited word 1 = ' + str(word_1_edited)
        )

        for i_row in range(1+excess_rows, n_rows+excess_rows, 1):
            letter = word_1_edited[i_row - 1]

            distance_matrix[i_row][0] = distance_matrix[i_row - 1][0] + 1

            # Build one row for the letter, with a column for each letter in the target
            # word, plus one for the empty string at column 0
            for i_col in range( 1, n_columns, 1 ):

                # +1 cost from the cost of the letter to the left of word_1
                insertCost = distance_matrix[i_row][i_col - 1] + 1
                # +1 cost for deleting the letter from word_2
                deleteCost = distance_matrix[i_row-1][i_col] + 1

                # Compare with current letter (index 0)
                if word_2[i_col - 1] != letter:
                    replaceCost = distance_matrix[i_row-1][ i_col - 1 ] + 1
                else:
                    Log.debugdebug(
                        str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Same "' + str(word_2[i_col-1]) + '" and "' + str(letter) + '"'
                    )
                    replaceCost = distance_matrix[i_row-1][ i_col - 1 ]

                # Levenshtein stops here
                min_cost = min(insertCost, deleteCost, replaceCost)

                # Transposition of 2 adjacent characters
                if self.algo == EditDistance.EDIT_DIST_ALGO_DAMLEV:
                    transpositionCost = max(insertCost, deleteCost, replaceCost)
                    if i_row >= 2 and i_col >= 2:
                        if word_1_edited[i_row-2]==word_2[i_col-1]\
                                and word_1_edited[i_row-1]==word_2[i_col-2]:
                            Log.debugdebug(
                                str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                + ': Transposition between ' + str(word_1_edited[(i_row-2):i_row])
                                + ' and ' + str(word_2[(i_col-2):i_col])
                            )
                            # Add +1 (single exchange) to cost of 2 rows back and 2 columns back
                            transpositionCost = distance_matrix[i_row-2][i_col-2] + 1

                    min_cost = min(min_cost, transpositionCost)

                distance_matrix[i_row][i_col] = min_cost

        Log.debugdebug(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Edit Distance Matrix: ' + str(distance_matrix)
        )

        optimal_cost = distance_matrix[-1][-1]
        return RetClass(
            distance_matrix = distance_matrix,
            optimal_cost    = optimal_cost,
            algo            = self.algo
        )

    def __calculate_use_basic_list(
            self,
            # The "left column" word
            word_1,
            # The "top row" word
            word_2,
            # Option to provide a start row, e.g. for recursive calculations
            # Column length must be consistent with word_2 1 + length of word_2
            # At most we will just take the last 2 rows
            start_rows_word_1 = None,
            # If we specify our own start rows, means there must be start word or letters.
            # At most we will just take the last 2 letters
            start_letters_word_1 = None
    ):
        # Put word 2 on the "top row"
        n_columns = len(word_2) + 1
        # Put word 1 on the "left column". We want to transform word 1 to word 2
        n_rows = len(word_1) + 1

        excess_rows = 0
        word_1_edited = word_1
        # If the start rows are passed in (number of rows >= 2), and not using numpy, we will have this
        prev_prev_row = None

        if start_rows_word_1 is not None:
            ok = True
            len_rows = len(start_rows_word_1)
            # By right we should check the column lengths of all rows,
            # but we just check the last for simplicity
            for row in start_rows_word_1:
                len_cols = len(row)
                ok = ok and (len_cols == n_columns)
            if not ok:
                raise Exception(
                    'Provided start row: ' + str(start_rows_word_1)
                    + ' column length must be consistent with word_2 "' + str(word_2)
                    + '" length plus 1 = ' + str(n_columns)
                )

            prev_row = list(start_rows_word_1[-1])
            if len_rows >= 2:
                excess_rows = 1
                prev_prev_row = list(start_rows_word_1[-2])
                for i in range(1, 0, -1):
                    word_1_edited = start_letters_word_1[-i] + word_1_edited
        else:
            # The usual empty letter row in standard edit distance calculation
            prev_row = list(range(n_columns))

        distance_matrix = [ prev_row ]

        Log.debugdebug(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Top row word "' + str(word_2) + '" ' + str(n_columns) + ' columns, '
            + ' left column word "' + str(word_1) + '" ' + str(n_rows) + ' rows.'
            + ' Edited word 1 = ' + str(word_1_edited)
        )

        cur_row = None

        for i_row in range(1+excess_rows, n_rows+excess_rows, 1):
            letter = word_1_edited[i_row - 1]

            cur_row = [ prev_row[0]+1 ]

            # Build one row for the letter, with a column for each letter in the target
            # word, plus one for the empty string at column 0
            for i_col in range( 1, n_columns, 1 ):

                # +1 cost from the cost of the letter to the left of word_2
                insertCost = cur_row[i_col - 1] + 1
                # +1 cost for deleting the letter from word_2
                deleteCost = prev_row[i_col] + 1

                # Compare with current letter (index 0)
                if word_2[i_col - 1] != letter:
                    replaceCost = prev_row[ i_col - 1 ] + 1
                else:
                    Log.debugdebug(
                        str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Same "' + str(word_2[i_col-1]) + '" and "' + str(letter) + '"'
                    )
                    replaceCost = prev_row[ i_col - 1 ]

                # Levenshtein stops here
                min_cost = min(insertCost, deleteCost, replaceCost)

                # Transposition of 2 adjacent characters
                if self.algo == EditDistance.EDIT_DIST_ALGO_DAMLEV:
                    transpositionCost = max(insertCost, deleteCost, replaceCost)
                    if i_row >= 2 and i_col >= 2:
                        if word_1_edited[i_row-2]==word_2[i_col-1]\
                                and word_1_edited[i_row-1]==word_2[i_col-2]:
                            Log.debugdebug(
                                str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                                + ': Transposition between ' + str(word_1_edited[(i_row-2):i_row])
                                + ' and ' + str(word_2[(i_col-2):i_col])
                            )
                            # Add +1 (single exchange) to cost of 2 rows back and 2 columns back
                            transpositionCost = prev_prev_row[i_col-2] + 1

                    min_cost = min(min_cost, transpositionCost)

                cur_row.append(min_cost)

            prev_prev_row = prev_row
            prev_row = cur_row
            distance_matrix.append(cur_row)

        Log.debugdebug(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Edit Distance Matrix: ' + str(distance_matrix)
        )

        optimal_cost = cur_row[-1]
        return RetClass(
            distance_matrix = distance_matrix,
            optimal_cost    = optimal_cost,
            algo            = self.algo
        )

    def calculate(
            self,
            # The "left column" word
            word_1,
            # The "top row" word
            word_2,
            # Option to provide a start row, e.g. for recursive calculations
            # Column length must be consistent with word_2 1 + length of word_2
            # At most we will just take the last 2 rows
            start_rows_word_1 = None,
            # If we specify our own start rows, means there must be start word or letters.
            # At most we will just take the last 2 letters
            start_letters_word_1 = None,
            use_np = USE_NUMPY,
    ):
        # if type(start_rows_word_2) is np.ndarray:
        #     # We do this in case the user has passed in something he needs to still use, like
        #     # from the TrieNode recursive function.
        #     # If we don't copy it, we will modify the reference causing havoc in the caller
        #     start_rows_word_2 = np.array(start_rows_word_2)
        if use_np:
            return self.__calculate_use_numpy(
                word_1  = word_1,
                word_2  = word_2,
                start_rows_word_1    = start_rows_word_1,
                start_letters_word_1 = start_letters_word_1
            )
        else:
            return self.__calculate_use_basic_list(
                word_1  = word_1,
                word_2  = word_2,
                start_rows_word_1    = start_rows_word_1,
                start_letters_word_1 = start_letters_word_1
            )


class EditDistanceUnitTest:

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

        test_data = [
            # 0: words to compare, 1: expected dist using Damerau-Levenshtein, 2: expected distance using Levenshtein
            (('เงน','เงิน'), 1, 1),
            (('ถนอ', 'ถอน'), 1, 2),
            (('ธรรมะ', 'ธรา'), 3, 3),
            # For Lev, 3 edit distance by
            #   1. Deleting 'ธ' to get 'รรมะ'
            #   2. Inserting 'ธ' to get 'รธมะ'
            #   3. Replacing 'ะ' with 'ร' to get 'รธมร'
            (('ธรรมะ', 'รธมร'), 3, 3),
        ]
        test_algos = [EditDistance.EDIT_DIST_ALGO_DAMLEV, EditDistance.EDIT_DIST_ALGO_LEV]

        for use_numpy in [True, False]:
            for i in range(len(test_data)):
                word1, word2 = test_data[i][0]
                expected_dist = [test_data[i][1], test_data[i][2]]

                for j_algo in range(len(test_algos)):
                    algo = test_algos[j_algo]

                    start = time.time()
                    retc = EditDistance(algo=algo).calculate(
                        word_1 = word1,
                        word_2 = word2,
                        use_np = use_numpy,
                    )
                    dist = retc.optimal_cost
                    end = time.time()
                    Log.debug('Calculated distance: ' + str(dist))
                    Log.debug("Search took " + str(round(1000*(end - start), 2)) + 'ms.')

                    res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                        observed = dist,
                        expected = expected_dist[j_algo],
                        test_comment = 'numpy= ' + str(use_numpy) + ', test word ' + str(i)
                                       + ' "' + str(word1) + '" and "' + str(word2) + '"'
                    ))

        return res_final


if __name__ == '__main__':
    config = Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class       = Config,
        default_config_file = Config.CONFIG_FILE_PATH_DEFAULT
    )
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1

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

    res = EditDistanceUnitTest(ut_params=ut_params).run_unit_test()
    exit(res.count_fail)


