import read_ptb
import tree

import pytest

from os import path

ptb_loc = path.join(path.dirname(tree.__file__), '../testsdata/WSJ_0001.MRG')


@pytest.fixture
def ptb_text():
    return open(path.join(ptb_loc)).read()


@pytest.fixture
def sentence_strings(ptb_text):
    return read_ptb.split_sentences(ptb_text)


def test_split(sentence_strings):
    assert len(sentence_strings) == 2
    assert sentence_strings[0].startswith('( (S\n    (NP-SBJ')
    assert sentence_strings[0].endswith('(. .) ))')
    assert sentence_strings[1].startswith('( (S\n    (NP-SBJ')
    assert sentence_strings[1].endswith('(. .) ))')


def test_tree_read(sentence_strings):
    words, brackets = read_ptb.get_brackets(sentence_strings[0])
    assert len(brackets) == 29
    string = ("Pierre Vinken , 61 years old , will join the board as a nonexecutive "
              "director Nov. 29 .")
    word_strings = string.split()
    starts = [s for l, s, e in brackets]
    ends = [e for l, s, e in brackets]
    assert min(starts) == 0
    assert max(ends) == len(words)
