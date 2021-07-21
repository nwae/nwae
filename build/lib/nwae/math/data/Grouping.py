# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import numpy as np
import re
import nwae.utils.UnitTest as ut


class Grouping:

    #
    # Group messy text data by regex into more manageable categories
    #
    @staticmethod
    def group_into_classes_by_regex(
            # List of strings to be classified
            x_array,
            # A list of [ (pattern_1, category_1), (pattern_2, category_2) ... ]
            pattern_repl_label,
            # Option to provide start/end regex for all the patterns provided
            # Means start with anything and then some delimiter (space, tab, etc) before the word, or nothing
            start_regex = '([ \t,.:;*()]+)',
            end_regex = '([ \t,.:;*()]+)',
            # For recording log info
            log_list = None
    ):
        # We need to wrap the values in space because we expect the word to be surrounded by delimiters
        x_array_processed = [' ' + x + ' ' for x in x_array]
        # Turn to lowercase
        x_array_processed = np.array([x.lower() for x in x_array_processed if x])

        grouped_x = x_array_processed.copy()

        if len(grouped_x) == 0:
            return grouped_x

        # So that later we push everything with label 0 to "other"
        grouped_x_label = np.array([0]*len(grouped_x))
        for i in range(len(pattern_repl_label)):
            label_index = i+1
            tpl = pattern_repl_label[i]

            # Pattern to match
            pattern = tpl[0]
            # Category to classify to
            category = tpl[1]

            if pattern is not None:
                pattern = start_regex + pattern + end_regex
                Log.debug(
                    str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': ' + str(label_index) + '. Matching pattern "' + str(pattern)
                    + '" for category "' + str(category) + '"'
                )
                try:
                    matched_pattern = np.array([
                        bool(re.search(pattern=pattern, string=x, flags=re.IGNORECASE)) for x in x_array_processed
                        if x
                    ])
                except Exception as ex_search:
                    errmsg = str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                             + ': Error for pattern "' + str(pattern) + '": ' + str(ex_search)
                    Log.error(errmsg)
                    raise Exception(errmsg)

                # Record how many matched for pattern
                matched_count_all = np.sum(matched_pattern*1)

                # Make sure we don't re-classify already classified rows with label=0
                matched_pattern = np.logical_and(matched_pattern, grouped_x_label == 0)

                # Record how many matched for new ones
                matched_count_new = np.sum(matched_pattern*1)

                Log.important(
                    s = str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': ' + str(label_index) + '. Matched pattern "' + str(pattern)
                        + '" for category "' + str(category) + '" to ' + str(matched_count_all)
                        + ' all, and ' + str(matched_count_new)
                        + ' new. Strings matched new: ' + str(list(np.unique(grouped_x[matched_pattern == True]))),
                    log_list = log_list
                )

                # Classify all positive matches
                grouped_x[matched_pattern == True] = category
                # Label by number this group league
                grouped_x_label[matched_pattern == True] = label_index
            else:
                count_strings_sinked_to_others = np.sum(1*(grouped_x_label==0))
                Log.important(
                    s = str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': ' + str(label_index) + '. None pattern "' + str(pattern)
                        + '" for category "' + str(category) + '", sinking all to "' + str(category)
                        + '". ' + str(count_strings_sinked_to_others) + ' strings sinked: '
                        + str(list(np.unique(grouped_x[grouped_x_label == 0]))),
                    log_list = log_list
                )
                # If None as pattern, means it goes to everything not yet classified
                grouped_x[grouped_x_label == 0] = category

        Log.debug('Grouped X (' + str(len(grouped_x)) + '):')
        return grouped_x


class GroupingUnitTest:
    def __init__(self, ut_params=None):
        return

    def run_unit_test(self):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)

        x_array = [
            'cat', 'cat  xx', 'xx   cat', 'xx cat yy', 'xcat', 'catx',
            'Samsung', '*삼성 * 전자*', '삼성xx 전자', '전자 삼성',
            '---'
        ]

        grouped_x = Grouping.group_into_classes_by_regex(
            x_array            = x_array,
            pattern_repl_label = (
                ('(cat)', '*#cat'),
                ('(삼성.*전자|samsung)', '*#Samsung'),
                (None, "*#Другие")
            )
        )
        # print(grouped_x.tolist())
        res_final.update_bool(res_bool=ut.UnitTest.assert_true(
            observed = grouped_x.tolist(),
            expected = [
                '*#cat', '*#cat', '*#cat', '*#cat', '*#Другие', '*#Другие',
                '*#Samsung', '*#Samsung', '*#Samsung', '*#Другие',
                '*#Другие'
            ],
            test_comment = 'Test grouping of ' + str(x_array)
        ))

        return res_final

if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_2

    res = GroupingUnitTest().run_unit_test()
    exit(res.count_fail)
