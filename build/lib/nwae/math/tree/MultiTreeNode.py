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
        self.name = name
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
                + '" is dead node, not adding parent for node "' + str(self.name) + '"'
            )
            return

        assert type(parent) is MultiTreeNode
        self.update()
        if parent.name in self.parent_names:
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': For node "' + str(self.name) + '" parent "' + str(parent.name) + '" already exists'
            )
        else:
            # Don't add if already exists as parent
            if self.is_higher_level(node=parent, supposed_child_node=self):
                return
            self.parents.append(parent)
            self.update()
            Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': For node "' + str(self.name) + '" successfully added parent "' + str(parent.name) + '"'
            )

    def add_child(self, child):
        if self.dead_node:
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Node "' + str(self.name)
                + '" is dead node, not adding child node "' + str(child.name) + '"'
            )
            return

        assert type(child) is MultiTreeNode
        self.update()
        if child.name in self.children_names:
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': For node "' + str(self.name) + '" child "' + str(child.name) + '" already exists'
            )
        else:
            # Don't add if already exists as parent
            if self.is_higher_level(node=self, supposed_child_node=child):
                return
            self.children.append(child)
            self.update()
            Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': For node "' + str(self.name) + '" successfully added child "' + str(child.name) + '"'
            )


if __name__ == '__main__':
    SAMPLE_DATA = {
        1: [2,3,4],
        2: [5,6,7],
        3: [7,8],
        4: [9],
        # Recursive 5 point back to 1 not allowed
        5: [10,11,1],
    }

    trees = {}
    for k in SAMPLE_DATA.keys():
        if k not in trees.keys():
            parent = MultiTreeNode(name=k, dead_node=False)
            trees[k] = parent
        else:
            parent = trees[k]
        child_keys = SAMPLE_DATA[k]
        for child_k in child_keys:
            if child_k not in trees.keys():
                child = MultiTreeNode(name=child_k, dead_node=False)
                trees[child_k] = child
                child.add_parent(parent=parent)
                parent.add_child(child=child)

    exit(0)
