# -*- coding: utf-8 -*-
#By Steve Hanov, 2011. Released to the public domain

import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo
import time
import nwae.lib.lang.nlp.WordList as wl
import nwae.lib.lang.LangFeatures as langfeatures


#
# The Trie data structure keeps a set of words, organized with one node for
# each letter. Each node has a branch for each letter that may follow it in the
# set of words.
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
            self
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
    def search(
            self,
            word,
            max_cost = 1
    ):
        # build first row. 0,1,2,3...,len(word)
        currentRow = range( len(word) + 1 )

        results = []

        # recursively search each branch of the trie
        for letter in trie.children:
            self.searchRecursive(
                node        = trie.children[letter],
                letter      = letter,
                word        = word,
                previousRow = currentRow,
                results     = results,
                max_cost    = max_cost
            )

        return results

    # This recursive helper is used by the search function above. It assumes that
    # the previousRow has been filled in already.
    def searchRecursive(
            self,
            node,
            letter,
            word,
            previousRow,
            results,
            max_cost
    ):
        columns = len( word ) + 1
        # West, north-west don't exist, so take only north
        currentRow = [ previousRow[0] + 1 ]

        # Build one row for the letter, with a column for each letter in the target
        # word, plus one for the empty string at column 0
        for column in range( 1, columns ):
            # west
            insertCost = currentRow[column - 1] + 1
            # north
            deleteCost = previousRow[column] + 1

            # north-west
            if word[column - 1] != letter:
                replaceCost = previousRow[ column - 1 ] + 1
            else:
                replaceCost = previousRow[ column - 1 ]

            currentRow.append( min( insertCost, deleteCost, replaceCost ) )

        # if the last entry in the row indicates the optimal cost is less than the
        # maximum cost, and there is a word in this trie node, then add it.
        if currentRow[-1] <= max_cost and node.word != None:
            results.append( (node.word, currentRow[-1] ) )

        # if any entries in the row are less than the maximum cost, then
        # recursively search each branch of the trie
        if min( currentRow ) <= max_cost:
            for letter in node.children:
                self.searchRecursive(
                    node        = node.children[letter],
                    letter      = letter,
                    word        = word,
                    previousRow = currentRow,
                    results     = results,
                    max_cost    = max_cost
                )
        return


if __name__ == '__main__':
    import nwae.config.Config as cf
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config
    )

    lang = langfeatures.LangFeatures.LANG_TH

    # read dictionary file into a trie
    wl_obj = wl.WordList(
        lang             = lang,
        dirpath_wordlist = config.get_config(cf.Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist = config.get_config(cf.Config.PARAM_NLP_POSTFIX_WORDLIST)
    )
    words = wl_obj.wordlist[wl.WordList.COL_WORD].tolist()

    trie = TrieNode.build_trie_node(
        words = words
    )
    print("Read %d words into %d nodes" % (TrieNode.WORD_COUNT, TrieNode.NODE_COUNT))

    start = time.time()
    results = trie.search(
        word = 'เงน'
    )
    end = time.time()

    for result in results:
        print(result)

    print("Search took " + str(end - start) + 's.')