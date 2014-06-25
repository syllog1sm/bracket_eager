import pytest

from transition_system import *


@pytest.fixture
def words():
    return 'water meter cover lining'.split()


@pytest.fixture
def tags():
    return 'NN NN NN NN'.split()


@pytest.fixture
def start_state(words, tags):
    stack, queue = get_start_state(words, tags)
    assert len(queue) == 4
    assert [w.lex for w in queue] == words
    assert [w.label for w in queue] == tags
    assert stack == []
    return stack, queue


@pytest.fixture
def actions():
    m = {}
    m[SHIFT] = DoShift(0)
    m[BRACKET] = DoBracket(1, label='NP')
    m[MERGE] = DoMerge(2)
    return m


@pytest.fixture
def g_01_23(start_state):
    _, queue = start_state
    b1 = tree.Bracket(queue[0], label='NP1')
    b1.children.append(queue[1])
    b2 = tree.Bracket(queue[2], label='NP2')
    b2.children.append(queue[3])
    top = tree.Bracket(b1, label='TOP')
    top.children.append(b2)
    return [b1, b2, top]
    

def test_end_state(start_state):
    stack, queue = start_state
    shift = DoShift(0)
    assert shift.is_valid(stack, queue)
    shift.apply(stack, queue)
    shift.apply(stack, queue)
    shift.apply(stack, queue)
    shift.apply(stack, queue)
    assert is_end_state(stack, queue)
    assert not shift.is_valid(stack, queue)


def test_shift(start_state):
    stack, queue = start_state
    shift = DoShift(0)
    assert shift.is_valid(stack, queue)
    shift.apply(stack, queue)
    assert len(stack) == 1
    assert len(queue) == 3
    assert stack[0].lex == 'water'
    assert stack[0].label == 'NN'
    assert stack[0].start == 0
    assert stack[0].end == 1


def test_bracket_validity(start_state):
    stack, queue = start_state
    bracket = DoBracket(0, label='NP')
    assert not bracket.is_valid(stack, queue)
    shift = DoShift(1)
    shift.apply(stack, queue)
    assert bracket.is_valid(stack, queue)


def test_merge_validity(start_state):
    stack, queue = start_state
    bracket = DoBracket(0, label='NP')
    shift = DoShift(1)
    merge = DoMerge(2)
    assert not merge.is_valid(stack, queue)
    shift.apply(stack, queue)
    assert not merge.is_valid(stack, queue)
    shift.apply(stack, queue)
    bracket.apply(stack, queue)
    assert merge.is_valid(stack, queue)


def goto(actions, stack, queue, hist):
    for move in hist:
        actions[move].apply(stack, queue)


@pytest.fixture
def s_state(start_state, actions):
    stack, queue = start_state
    goto(actions, stack, queue, [SHIFT])
    return (stack, queue)


@pytest.fixture
def sb_state(start_state, actions):
    stack, queue = start_state
    goto(actions, stack, queue, [SHIFT, BRACKET])
    return (stack, queue)


@pytest.fixture
def sbs_state(start_state, actions):
    stack, queue = start_state
    goto(actions, stack, queue, [SHIFT, BRACKET, SHIFT])
    return (stack, queue)


@pytest.fixture
def ssb_state(start_state, actions):
    stack, queue = start_state
    goto(actions, stack, queue, [SHIFT, SHIFT, BRACKET])
    return (stack, queue)


def test_bracket(s_state, actions):
    stack, queue = s_state
    assert len(stack) == 1
    assert stack[0].start == 0
    assert stack[0].end == 1
    assert stack[0].production == ('NN', tuple())
    actions[BRACKET].apply(stack, queue)
    assert stack[0].label == 'NP'
    assert stack[0].production == ('NP', ('NN',))
 

def test_merge(ssb_state, actions):
    stack, queue = ssb_state
    assert len(stack) == 2
    assert stack[0].start == 0
    assert stack[0].end == 1
    assert stack[1].start == 1
    assert stack[1].end == 2
    m = actions[MERGE]
    q_before = list(queue)
    assert stack[1].production == ('NP', ('NN',))
    m.apply(stack, queue)
    assert len(stack) == 1
    assert queue == q_before
    assert stack[0].start == 0
    assert stack[0].end == 2
    assert stack[0].production == ('NP', ('NN', 'NN'))
    assert not m.is_valid(stack, queue) 


def test_shift_oracle(start_state, g_01_23, actions):
    stack, queue = start_state
    assert len(g_01_23) == 3
    s = actions[SHIFT]
    assert s.is_valid(stack, queue)
    assert SHIFT in oracle(actions.values(), stack, g_01_23)
    s.apply(stack, queue)
    assert s.is_valid(stack, queue)
    assert SHIFT in oracle(actions.values(), stack, g_01_23)
    s.apply(stack, queue)
    assert s.is_valid(stack, queue)
    assert SHIFT not in oracle(actions.values(), stack, g_01_23)
    s.apply(stack, queue)
    assert SHIFT in oracle(actions.values(), stack, g_01_23)


def test_bracket_oracle(start_state, g_01_23):
    s = DoShift(0)
    b1 = DoBracket(1, label='NP1')
    b2 = DoBracket(2, label='NP2')
    b_top = DoBracket(3, label='TOP')
    actions = [s, b1, b2]
    stack, queue = start_state
    s.apply(stack, queue)
    assert len(g_01_23) == 3
    assert b1.is_valid(stack, queue)
    g = oracle(actions, stack, g_01_23)
    assert len(g_01_23) == 3
    assert b1 not in g
    assert b2 not in g
    assert b_top not in g
    s.apply(stack, queue)
    g = oracle(actions, stack, g_01_23)
    assert len(g_01_23) == 3
    assert b1.is_valid(stack, queue)
    assert b1 in g
    assert b2 not in g
    assert b_top not in g
    b1.apply(stack, oracle)
    g = oracle(actions, stack, g_01_23)
    assert b1 not in g
    assert b2 not in g
    assert b_top not in g
    assert SHIFT not in oracle(actions, stack, list(g_01_23))
    s.apply(stack, queue)
    g = oracle(actions, stack, g_01_23)
    assert s.is_valid(stack, queue)
    assert SHIFT in oracle(actions, stack, list(g_01_23))
    assert b1 not in g
    assert b2 not in g
    assert b_top not in g
 
