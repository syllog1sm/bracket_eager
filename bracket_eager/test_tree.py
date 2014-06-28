import read_ptb
import tree

import pytest

from os import path

ptb_loc = '/usr/local/data/Penn3'


@pytest.fixture
def ptb_text():
    return open(path.join(ptb_loc, 'parsed', 'mrg', 'wsj', '00', 'wsj_0001.mrg')).read()


@pytest.fixture
def sentence_strings(ptb_text):
    return read_ptb.split_sentences(ptb_text)


@pytest.fixture
def pierre_tree(sentence_strings):
    words, brackets = read_ptb.get_brackets(sentence_strings[0])
    return tree.from_brackets(words, brackets)


@pytest.fixture
def pierre_string():
    return ("Pierre Vinken , 61 years old , will join the board as a nonexecutive "
            "director Nov. 29 .")
 
def test_branches(pierre_tree):
    assert pierre_tree.label == 'S'

def test_leaves(pierre_tree, pierre_string):
    words = pierre_string.split()
    leaves = pierre_tree.leaves()
    assert len(leaves) == len(words)


def test_depth_first(pierre_tree):
    nodes = pierre_tree.depth_list()
    seen_end = 0
    for node in nodes:
        assert node.end >= seen_end
        seen_end = node.end
