from grammar import get_valid_stacks, rules_from_trees
import read_ptb
import tree

import pytest
from os import path

@pytest.fixture
def grammar():
    rules = {
        'TOP': [('S',), ('PP',)],
        'S': [('NP', 'VP')],
        'VP': [('VBZ', 'NP', 'PP'), ('VBZ', 'NP')],
        'PP': [('IN', 'NP'),],
    }
    return rules


@pytest.fixture
def recursive_grammar():
    rules = {
        'TOP': [('PP',)],
        'PP': [('IN', 'NP'),],
        'NP': [('NP', 'PP'),],
    }
    return rules


def test_prefix(grammar):
    stacks = get_valid_stacks(grammar)
    assert ('NP', 'VP') in stacks
    assert ('NP',) in stacks
    assert ('VP',) not in stacks
    assert ('IN',) in stacks

def test_recursion(recursive_grammar):
    stacks = get_valid_stacks(recursive_grammar, max_len=100)
    assert max(len(stack) for stack in stacks) < 100


ptb_loc = '/usr/local/data/Penn3'


@pytest.fixture
def ptb_text():
    return open(path.join(ptb_loc, 'parsed', 'mrg', 'wsj', '00', 'wsj_0001.mrg')).read()


@pytest.fixture
def trees(ptb_text):
    tree_list = []
    for sent_str in read_ptb.split_sentences(ptb_text):
        words, brackets = read_ptb.get_brackets(sent_str)
        tree_list.append(tree.from_brackets(words, brackets))
    return tree_list


def test_get_rules(trees):
    rules = rules_from_trees(trees)
    assert len(rules) == 5

