# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import numpy as np
from collections import Counter
import nwae.utils.UnitTest as ut


class Rank:

    #
    # Given a sorted list
    #    [a, a, a, b, b, c, c, c, c, c]
    # It will return the rank by item as
    #    [1, 2, 3, 1, 2, 1, 2, 3, 4, 5]
    #
    @staticmethod
    def rank_sorted_list_by_unique_items(
            sorted_list
    ):
        cntr = Counter(sorted_list)
        max_item_count = max(cntr.values())
        Log.debugdebug(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Max unique item count = ' + str(max_item_count)
        )

        #
        # Another way of not getting the max item count above is to loop until no changes
        # occur to the ranking list.
        # However that will take up memory if the list given is huge, so we prefer knowing
        # in advance how many max loops to take
        #
        len_list = len(sorted_list)
        # Start with 1 rank
        item_rank = np.array([1] * len_list)
        for i in range(max_item_count):
            shift = i + 1
            # Shift down
            sorted_list_shift = np.append(np.array(shift*[None]), sorted_list[0:(len_list-shift)], axis=0)
            Log.debugdebug(
                str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ' Shift #' + str(shift) + ': ' + str(sorted_list_shift)
            )
            # If the rank is the previous rank and member code is the same, means we add 1 to the rank
            condition = (sorted_list == sorted_list_shift) & (item_rank == shift)
            item_rank[condition] = item_rank[condition]+1
        return item_rank.tolist()


class RankUnitTest:
    def __init__(self, ut_params):
        self.ut_params = ut_params
        return

    def run_unit_test(self):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)

        tests = [
            [ [1,1,1,2,2,2,2,2,2,3,3,4,5,5,5,5],
              [1,2,3,1,2,3,4,5,6,1,2,1,1,2,3,4]],
            [['ok','bb','bb','bb','ikra','ikra','pah','pah','pah','pah','pah'],
             [1,   1,   2,   3,   1,     2,     1,    2,    3,    4,    5]],
        ]
        for i in range(len(tests)):
            sorted_list = tests[i][0]
            expected = tests[i][1]
            item_rank = Rank.rank_sorted_list_by_unique_items(
                sorted_list = sorted_list
            )
            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = item_rank,
                expected = expected,
                test_comment = 'Test ' + str(sorted_list) + ' got ' + str(item_rank)
            ))
        return res_final


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_2
    exit(RankUnitTest(ut_params=None).run_unit_test().count_fail)
