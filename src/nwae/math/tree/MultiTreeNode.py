# --*-- coding: utf-8 --*--

from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo


"""
Представляет структуры дерева
Каждый класс представляет только одного узла, либо корневого, внутреннего, или терминального (лист),
но связан с другими объектов, в результате создающий несколько отдельных деревьев
"""
class MultiTreeNode:

    def __init__(
            self,
            name,
            # If dead node, cannot add child, or no one can add it as parent
            dead_node,
    ):
        # Standardize to string
        self.name = str(name)
        self.dead_node = dead_node
        # If more than 1 parent, each parent represents a different tree
        self.parents = []
        self.parent_names = []
        # All children are of the same tree
        self.children = []
        self.children_names = []

        self.children_depth = []
        self.parent_depth = []
        return

    def is_dead_node(self):
        return self.dead_node

    def is_tree_root(self):
        return len(self.parents) == 0

    def update(self):
        self.parent_names = [c.name for c in self.parents]
        self.children_names = [c.name for c in self.children]

    def is_higher_level(self, node, supposed_child_node):
        Log.debug(
            '***** check if "' + str(supposed_child_node.name) + '" is higher level than "'
            + str(node.name) + '", parents: ' + str(node.parent_names)
        )
        if supposed_child_node.name in node.parent_names:
            Log.warning(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Node "' + str(self.name) + '" cannot add "' + str(supposed_child_node.name)
                + '" as child. Node "' + str(supposed_child_node.name)
                + '" is already a higher level parent node to "' + str(self.name) + '"'
            )
            return True
        for par in node.parents:
            if self.is_higher_level(node=par, supposed_child_node=supposed_child_node):
                return True
            else:
                continue
        return False

    def add_parent(self, parent):
        if parent.dead_node:
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Parent "' + str(parent.name)
                + '" is dead node (cant have children), not adding parent for node "' + str(self.name) + '"'
            )
            return

        assert type(parent) is MultiTreeNode
        if parent.name in self.parent_names:
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': For node "' + str(self.name) + '" parent "' + str(parent.name) + '" already exists'
            )
        else:
            # Don't add if already exists as parent, anywhere higher up the tree hierarchy
            if self.is_higher_level(node=parent, supposed_child_node=self):
                return
            # Update for both parent and child
            self.parents.append(parent)
            self.update()
            parent.children.append(self)
            parent.update()
            Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': For node "' + str(self.name) + '" successfully added parent "' + str(parent.name) + '"'
            )


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1
    Log.DEBUG_PRINT_ALL_TO_SCREEN = True

    SAMPLE_DATA = {
        1: [2,3,4],
        2: [5,6,7],
        3: [7,8],
        4: [9],
        # Recursive 5 point back to 1 not allowed
        5: [10,11,1,7],
    }

    trees = {}
    for k in SAMPLE_DATA.keys():
        # standardize to string
        k = str(k)
        if k not in trees.keys():
            parent = MultiTreeNode(name=k, dead_node=False)
            trees[k] = parent
        else:
            parent = trees[k]
        child_keys = SAMPLE_DATA[int(k)]
        for child_k in child_keys:
            child_k = str(child_k)
            if child_k not in trees.keys():
                child = MultiTreeNode(name=child_k, dead_node=False)
                trees[child_k] = child
            else:
                child = trees[child_k]
            child.add_parent(parent=parent)

    print('***** Parent: Children *****')
    [print(str(k) + ': ' + str(trees[k].children_names)) for k in trees.keys()]
    print('***** Child: Parents *****')
    [print(str(k) + ': ' + str(trees[k].parent_names)) for k in trees.keys()]

    exit(0)
