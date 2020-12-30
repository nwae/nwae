# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier
import pydotplus


class DecisionTree:

    def __init__(
            self,
            # pandas dataframe of features
            df_X,
            # pandas dataframe of a single output column
            df_y
    ):
        self.df_X = df_X
        self.df_y = df_y
        self.feature_names = list(self.df_X.columns)
        return

    #
    # Convert sklearn DecisionTreeClassifier to code
    #
    def tree_to_code(
            self,
            tree,
            feature_names,
            newline,
    ):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        code_str = "def tree({}):".format(", ".join(feature_names)) + newline

        def recurse(node, depth, code_string):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]

                code_string += "{}if {} <= {}:".format(indent, name, threshold) + newline
                code_string = recurse(node=tree_.children_left[node], depth=depth + 1, code_string=code_string)

                code_string += "{}else:  # if {} > {}".format(indent, name, threshold) + newline
                code_string = recurse(node=tree_.children_right[node], depth=depth + 1, code_string=code_string)
            else:
                code_string += "{}return {}".format(indent, tree_.value[node]) + newline
            return code_string

        return recurse(0, 1, code_str)

    def fit(
            self,
            # 'entropy' (information concept) or 'gini' (impurity concept)
            criterion = 'gini',
            max_tree_depth = 10,
            min_samples_split = 20,
            min_impurity_decrease = 0.0,
            output_graph_path = None,
            output_code_path = None,
            output_code_newline = '\n'
    ):
        dtree = DecisionTreeClassifier(
            criterion = criterion,
            max_depth = max_tree_depth,
            min_samples_split = min_samples_split,
            min_impurity_decrease = min_impurity_decrease,
        )
        dtree = dtree.fit(self.df_X, self.df_y)
        data = tree.export_graphviz(
            dtree,
            out_file = None,
            feature_names = self.feature_names,
        )
        code = self.tree_to_code(
            tree = dtree,
            feature_names = self.feature_names,
            newline = output_code_newline
        )
        if output_code_path is not None:
            f = open(output_code_path, 'w')
            f.write(code)
            f.close()

        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Decision tree built successfully: ' + str(data) + ', tree converted to code:\n\r' + str(code)
        )

        if output_graph_path is not None:
            graph = pydotplus.graph_from_dot_data(data)
            graph.write_png(output_graph_path)
        return dtree


if __name__ == '__main__':
    df = pd.DataFrame({
        # Sex '1' have more preference to get a job despite lower IQ
        'iq':  [100, 100,  60,  60,  50,  50,  10,  10],
        'sex': [  1,   2,   1,   2,   1,   2,   1,   2],
        'job': [  1,   1,   1,   1,   1,   0,   0,   0]
    })
    DecisionTree(
        df_X = df[['iq', 'sex']],
        df_y = df[['job']]
    ).fit(
        criterion = 'entropy',
        max_tree_depth = 10,
        min_samples_split = 2,
        min_impurity_decrease = 0.0,
        output_graph_path='dt.png',
        output_code_path='dt_code.txt',
        output_code_newline='\n'
    )
    exit(0)
