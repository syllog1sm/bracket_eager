import pytest

from transition_system import *
import read_ptb
import tree


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
    m[SHIFT] = DoShift()
    m[BRACKET] = DoBracket(label='NP')
    m[MERGE] = DoMerge()
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
    

@pytest.fixture
def g_012_3(start_state):
    _, queue = start_state
    b1 = tree.Bracket(queue[0], label='NP')
    b1.children.append(queue[1])
    b1.children.append(queue[2])
    top = tree.Bracket(b1, label='TOP')
    top.children.append(queue[3])
    return [b1, top]
    

def test_end_state(start_state):
    stack, queue = start_state
    shift = DoShift()
    assert shift.is_valid(stack, queue)
    shift.apply(stack, queue)
    shift.apply(stack, queue)
    shift.apply(stack, queue)
    shift.apply(stack, queue)
    b = DoBracket()
    m = DoMerge()
    b.apply(stack, queue)
    m.apply(stack, queue)
    assert m.is_valid(stack, queue)
    m.apply(stack, queue)
    assert is_end_state(stack, queue)
    assert not shift.is_valid(stack, queue)


def test_shift(start_state):
    stack, queue = start_state
    shift = DoShift()
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
    bracket = DoBracket()
    assert not bracket.is_valid(stack, queue)
    shift = DoShift()
    shift.apply(stack, queue)
    shift.apply(stack, queue)
    assert bracket.is_valid(stack, queue)


def test_merge_validity(start_state):
    stack, queue = start_state
    b = DoBracket()
    s = DoShift()
    m = DoMerge()
    assert not m.is_valid(stack, queue)
    s.apply(stack, queue)
    assert not m.is_valid(stack, queue)
    s.apply(stack, queue)
    s.apply(stack, queue)
    b.apply(stack, queue)
    assert m.is_valid(stack, queue)


def goto(actions, stack, queue, hist):
    for move in hist:
        actions[move].apply(stack, queue)


@pytest.fixture
def s_state(start_state, actions):
    stack, queue = start_state
    goto(actions, stack, queue, [SHIFT])
    return (stack, queue)

@pytest.fixture
def ss_state(start_state, actions):
    stack, queue = start_state
    goto(actions, stack, queue, [SHIFT, SHIFT])
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
def sssb_state(start_state, actions):
    stack, queue = start_state
    goto(actions, stack, queue, [SHIFT, SHIFT, SHIFT, BRACKET])
    return (stack, queue)


def test_bracket(ss_state, actions):
    stack, queue = ss_state
    assert len(stack) == 2
    assert stack[0].start == 0
    assert stack[0].end == 1
    assert stack[0].production == ('NN', tuple())
    actions[BRACKET].apply(stack, queue)
    assert stack[0].label == 'NP'
    assert stack[0].production == (None, ('NN', 'NN'))
 

def test_merge(sssb_state, actions):
    stack, queue = sssb_state
    assert len(stack) == 2
    assert stack[0].start == 0
    assert stack[0].end == 1
    assert stack[1].start == 1
    assert stack[1].end == 3
    m = actions[MERGE]
    q_before = list(queue)
    assert stack[1].production == (None, ('NN', 'NN'))
    m.apply(stack, queue)
    assert len(stack) == 1
    assert queue == q_before
    assert stack[0].start == 0
    assert stack[0].end == 3
    assert stack[0].production == (None, ('NN', 'NN', 'NN'))
    assert not m.is_valid(stack, queue) 


def test_shift_oracle(start_state, g_01_23, actions):
    assert len(g_01_23) == 3
    stack, queue = start_state
    gold = iter_gold(stack, queue, g_01_23)
    s = actions[SHIFT]
    assert s.is_gold(stack, queue, gold.next())
    s.apply(stack, queue)
    assert s.is_gold(stack, queue, gold.next())
    s.apply(stack, queue)
    assert not s.is_gold(stack, queue, gold.next())
    s.apply(stack, queue)
    assert s.is_gold(stack, queue, gold.next())


def test_bracket_oracle(start_state, g_01_23):
    assert len(g_01_23) == 3
    s = DoShift()
    b1 = DoBracket(label='NP1')
    b2 = DoBracket(label='NP2')
    b_top = DoBracket(label='TOP')
    stack, queue = start_state
    golds = iter_gold(stack, queue, g_01_23)
    s.apply(stack, queue)
    assert not b1.is_gold(stack, queue, golds.next())
    assert not b2.is_gold(stack, queue, golds.next())
    assert not b_top.is_gold(stack, queue, golds.next())
    s.apply(stack, queue)
    assert b1.is_gold(stack, queue, golds.next())
    assert not b2.is_gold(stack, queue, golds.next())
    assert not b_top.is_gold(stack, queue, golds.next())
    b1.apply(stack, queue)
    assert stack[-1].start == 0
    assert stack[-1].end == 2
    assert not b1.is_gold(stack, queue, golds.next())
    assert not b2.is_gold(stack, queue, golds.next())
    assert not b_top.is_gold(stack, queue, golds.next())
    s.apply(stack, queue)
    assert s.is_gold(stack, queue, golds.next())
    assert not b1.is_gold(stack, queue, golds.next())
    assert not b2.is_gold(stack, queue, golds.next())
    assert not b_top.is_gold(stack, queue, golds.next())
 
 
def test_merge_oracle(start_state, g_012_3):
    assert len(g_012_3) == 2
    s = DoShift()
    b = DoBracket(label='NP')
    b_top = DoBracket(label='TOP')
    m = DoMerge()
    stack, queue = start_state
    golds = iter_gold(stack, queue, g_012_3)
    s.apply(stack, queue)
    s.apply(stack, queue)
    s.apply(stack, queue)
    b.apply(stack, queue)
    assert not s.is_gold(stack, queue, golds.next())
    assert not b.is_gold(stack, queue, golds.next())
    assert not b_top.is_gold(stack, queue, golds.next())
    assert m.is_gold(stack, queue, golds.next())
    m.apply(stack, queue)
    s.apply(stack, queue)
    assert not m.is_valid(stack, queue)
    assert not b.is_gold(stack, queue, golds.next())
    assert not b.is_gold(stack, queue, golds.next())
    assert b_top.is_gold(stack, queue, golds.next())
    b_top.apply(stack, queue)


def test_gold_top(start_state, g_01_23):
    assert len(g_01_23) == 3
    s = DoShift()
    b1 = DoBracket(label='NP1')
    b2 = DoBracket(label='NP2')
    b_top = DoBracket(label='TOP')
    m = DoMerge()
    stack, queue = start_state
    golds = iter_gold(stack, queue, g_01_23)
    s.apply(stack, queue)
    s.apply(stack, queue)
    b1.apply(stack, queue)
    s.apply(stack, queue)
    s.apply(stack, queue)
    assert not queue
    assert not s.is_valid(stack, queue)
    assert not b_top.is_gold(stack, queue, golds.next())
    assert b2.is_gold(stack, queue, golds.next())
    b2.apply(stack, queue)
    g = golds.next()
    assert not m.is_gold(stack, queue, golds.next())
    assert not b1.is_gold(stack, queue, golds.next())
    assert not b2.is_gold(stack, queue, golds.next())
    assert b_top.is_gold(stack, queue, golds.next())


def test_unary_oracle_case():
    ptb_str = """
( (S
    (NP-SBJ
      (VP (VBG telling)
        (NP (NNS lies) )))
    (VP (VBZ is)
      (ADJP-PRD (JJ wrong) ))
  (. .) ))""".strip()
    words, bare_brackets = read_ptb.get_brackets(ptb_str)
    assert words == ['telling', 'lies', 'is', 'wrong', '.']
    assert len(bare_brackets) == 11
    top = tree.from_brackets(words, bare_brackets)
    assert len(top.leaves()) == len(words), [l.lex for l in top.leaves()]
    assert len(top.iter_nodes()) == 11
    assert len(top.depth_list()) == 6
    leaves = top.leaves()

    stack, queue = get_start_state([w.lex for w in leaves], [w.label for w in leaves])
    golds = iter_gold(stack, queue, top.depth_list())

    s = DoShift()
    m = DoMerge()
    np = DoBracket()
    vp = DoBracket()
    adjp = DoBracket()
    u = DoUnary()

    s.apply(stack, queue)
    s.apply(stack, queue)
    next_gold = golds.next()
    assert next_gold.is_unary
    assert u.is_valid(stack, queue)
    assert u.is_gold(stack, queue, next_gold)
    u.apply(stack, queue)
    assert vp.is_gold(stack, queue, golds.next())
    vp.apply(stack, queue)
    next_gold = golds.next()
    assert u.is_gold(stack, queue, next_gold)
    u.apply(stack, queue)
    assert not u.is_valid(stack, queue)


def test_np_to_np():
    ptb_str = """
( (S 
    (NP-SBJ (NNP Mr.) (NNP Vinken) )
    (VP (VBZ is) 
      (NP-PRD 
        (NP (NN chairman) )
        (PP (IN of) 
          (NP 
            (NP (NNP Elsevier) (NNP N.V.) )
            (, ,) 
            (NP (DT the) (NNP Dutch) (VBG publishing) (NN group) )))))
    (. .) ))"""
    words, bare_brackets = read_ptb.get_brackets(ptb_str)
    top = tree.from_brackets(words, bare_brackets)
    leaves = top.leaves()

    st, q = get_start_state([w.lex for w in leaves], [w.label for w in leaves])
    g = iter_gold(st, q, top.depth_list())

    s = DoShift()
    m = DoMerge()
    np = DoBracket(label='NP')
    vp = DoBracket(label='VP')
    pp = DoBracket(label='PP')

    s.apply(st, q)
    s.apply(st, q)
    np.apply(st, q)
    m.apply(st, q)
    s.apply(st, q)
    s.apply(st, q)
    assert np.is_gold(st, q, g.next())
    np.apply(st, q)
    s.apply(st, q)
    s.apply(st, q)
    s.apply(st, q)
    assert np.is_gold(st, q, g.next())
    np.apply(st, q)
    assert m.is_gold(st, q, g.next())
    m.apply(st, q)
    s.apply(st, q)
    assert not np.is_gold(st, q, g.next())
    s.apply(st, q)
    s.apply(st, q)
    s.apply(st, q)
    s.apply(st, q)
    assert np.is_gold(st, q, g.next())
    np.apply(st, q)
    m.apply(st, q)
    m.apply(st, q)
    m.apply(st, q)
    next_gold = g.next()
    assert np.is_gold(st, q, next_gold), 'Stack: %s, Gold: %s' % (st[-1], next_gold)
