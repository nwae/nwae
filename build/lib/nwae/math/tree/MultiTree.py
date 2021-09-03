# --*-- coding: utf-8 --*--

from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
from nwae.math.tree.MultiTreeNode import MultiTreeNode
import nwae.utils.UnitTest as ut


"""
Представляет структуры отдельных деревьев
"""
class MultiTree:

    def __init__(
            self,
    ):
        self.reset_tree()
        return

    def reset_tree(self):
        # All nodes. Key is name
        self.tree_nodes = {}
        # All nodes that are roots. Key is name
        self.tree_roots = {}
        self.tree_roots_depth = {}

    """
    Пример входных данных
    {
        1: [2, 3, 4],
        2: [5, 6, 7],
        3: [7, 8],
        4: [9],
        # Recursive 5 point back to 1 not allowed
        5: [10, 11, 1],
    }
    Алгоритм построить дерево последовательно
      - отдельное дерево только включает один корень
      - повторяющиеся точки в одном дереве не создают бесконечных циклов
    """
    def build_tree(self, dict_parent_childs):
        self.reset_tree()
        for parent_key in dict_parent_childs.keys():
            child_keys = dict_parent_childs[parent_key]
            Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Doing for line ' + str(parent_key) + ': ' + str(child_keys)
            )
            if parent_key not in self.tree_nodes.keys():
                Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Create new parent ' + str(parent_key)
                )
                parent = MultiTreeNode(name=parent_key, dead_node=False)
                self.tree_nodes[parent_key] = parent
            else:
                parent = self.tree_nodes[parent_key]
                Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Retrieved parent ' + str(parent.name)
                )

            for child_k in child_keys:
                if child_k == parent_key:
                    Log.warning(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Child "' + str(child_k) + '" same as parent "' + str(parent_key) + '" Ignoring...'
                    )
                    continue
                Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Doing child ' + str(child_k) + ' for parent ' + str(parent_key)
                )
                if child_k not in self.tree_nodes.keys():
                    Log.debug(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Create new child ' + str(child_k)
                    )
                    child = MultiTreeNode(name=child_k, dead_node=False)
                    self.tree_nodes[child_k] = child
                else:
                    child = self.tree_nodes[child_k]
                    Log.debug(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Retrieved child ' + str(child.name)
                    )

                Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Child ' + str(child.name) + ' adding parent ' + str(parent.name)
                )
                child.add_parent(parent=parent)

        self.build_tree_roots()
        return self.tree_nodes

    def build_tree_roots(self):
        # Find root tree nodes
        self.tree_roots = {}
        for name in self.tree_nodes.keys():
            Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Checking if ' + str(name) + ' is a tree root...'
            )
            node = self.tree_nodes[name]
            if not node.is_dead_node():
                if node.is_tree_root():
                    self.tree_roots[name] = node
                    self.tree_roots_depth[name] = self.calculate_tree_depth(node=node)
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Found ' + str(len(self.tree_roots)) + ' tree roots'
        )
        return

    # Depth of children
    def calculate_tree_depth(self, node, depth=0):
        depth += 1
        child_depths = []
        for i in range(len(node.children)):
            d = self.calculate_tree_depth(node=node.children[i], depth=depth)
            child_depths.append(d)
        if len(child_depths) == 0:
            return depth
        else:
            return max(child_depths)

    def is_node_in_treenode(self, node, treenode):
        if node.name == treenode.name:
            return True
        for child in treenode.children:
            if self.is_node_in_treenode(node=node, treenode=child):
                return True
            else:
                # Keep looking
                continue
        return False

    def find_all_tree_roots_that_contains_node(self, node):
        trees_dict = {}
        for root in self.tree_roots.values():
            if self.is_node_in_treenode(node=node, treenode=root):
                trees_dict[root.name] = root
        return trees_dict

    def print_tree(self, level, tnode, max_levels=8, newline='\n\r', tabchar='\t'):
        if level > max_levels:
            Log.warning(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Level too much > ' + str(max_levels) + ' for "' + str(tnode.name) + '"'
            )
            return ''
        tabstr = ''
        for i in range(level):
            tabstr += tabchar
        string_to_print = str(tabstr) + 'Level ' + str(level) + ': ' + str(tnode.name) + str(newline)
        for child in tnode.children:
            string_to_print += self.print_tree(
                level=level+1, tnode=child, max_levels=max_levels, newline=newline, tabchar=tabchar
            )
        return string_to_print

    def reverse_build_native_dict(self, node, dict_tree={}):
        if node.name not in dict_tree.keys():
            if len(node.children_names) > 0:
                dict_tree[node.name] = node.children_names.copy()
        for c in node.children:
            self.reverse_build_native_dict(node=c, dict_tree=dict_tree)
        return dict_tree


class MultiTreeUnitTest:
    def __init__(self, ut_params=None):
        return

    def run_unit_test(self):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)

        SAMPLE_DATA = {
            # 7 repeats many times, but consistent
            1: [2, 3, 4],
            2: [5, 6, 7],
            3: [7, 8, 7],  # Repeat 7 will not be added
            4: [9],
            5: [10, 11, 1,7],  # Recursive 5 point back to 1 not allowed
            7: [7, 3],  # Recursive not allowed
            # 7: [3],  # Recursive not allowed
            # Separate tree
            100: [101, 102, 103, 104],
            101: [110, 111, 112],
            # Separate tree
            1000: [1001, 1002],
            1002: [1100, 1101, 1102, 1103, 1000],  # Recursive 1002 point back to 1000 not allowed
            1100: [1200, 1201, 1202],
            1202: [1300, 1301, 1002], # Recursive 1202 point back to 1002 not allowed
            # Separate tree
            999: []
        }

        expected_reverse_root_dicts = {
            '1': {'1':['2','3','4'], '2':['5','6','7'], '3':['7','8'], '4':['9'], '5':['10','11','7']},
            '100': {'100': ['101', '102', '103', '104'], '101': ['110', '111', '112']},
            '1000': {'1000': ['1001', '1002'], '1002': ['1100', '1101', '1102', '1103'], '1100': ['1200', '1201', '1202'], '1202': ['1300', '1301']},
            '999': {},
        }

        mt = MultiTree()
        mt.build_tree(dict_parent_childs=SAMPLE_DATA)
        for t in mt.tree_roots.values():
            s = mt.print_tree(level=0, tnode=t, max_levels=100, newline='\n\r', tabchar='   ')
            #print(s)
            reverse_dict = mt.reverse_build_native_dict(node=t, dict_tree={})
            #print(reverse_dict)
            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = reverse_dict,
                expected = expected_reverse_root_dicts[t.name],
                test_comment = 'Check ' + str(reverse_dict) + ' for tree root ' + str(t.name)
            ))

        return res_final

if __name__ == '__main__':
    Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1

    MultiTreeUnitTest().run_unit_test()

    exit(0)
