import pytest

from transition_system2 import *
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
    state = ParserState.from_words_and_tags(words, tags)
    assert len(state.queue) == 4
    assert [w.lex for w in state.queue] == words
    assert [w.label for w in state.queue] == tags
    assert state.stack == []
    return state


@pytest.fixture
def actions():
    m = {}
    m[SHIFT] = DoShift()
    m[BRACKET] = DoBracket()
    m[MERGE] = DoMerge()
    return m


@pytest.fixture
def g_01_23(start_state):
    queue = start_state.queue
    b1 = tree.Bracket(queue[0], label='NP1')
    b1.children.append(queue[1])
    b2 = tree.Bracket(queue[2], label='NP2')
    b2.children.append(queue[3])
    top = tree.Bracket(b1, label='TOP')
    top.children.append(b2)
    return [b1, b2, top]
    

@pytest.fixture
def g_012_3(start_state):
    queue = start_state.queue
    b1 = tree.Bracket(queue[0], label='NP')
    b1.children.append(queue[1])
    b1.children.append(queue[2])
    top = tree.Bracket(b1, label='TOP')
    top.children.append(queue[3])
    return [b1, top]
    

def test_end_state(start_state):
    shift = DoShift()
    assert shift.is_valid(start_state)
    state = start_state
    state = shift.apply(state)
    state = shift.apply(state)
    state = shift.apply(state)
    state = shift.apply(state)
    b = DoBracket()
    m = DoMerge()
    state = b.apply(state)
    state = m.apply(state)
    assert m.is_valid(state)
    state = m.apply(state)
    state = m.apply(state)
    assert state.is_end_state()
    assert not shift.is_valid(state)


def test_shift(start_state):
    state = start_state
    shift = DoShift()
    assert shift.is_valid(state)
    state = shift.apply(state)
    assert len(state.stack) == 1
    assert len(state.queue) == 3
    stack = state.stack
    assert stack[0].lex == 'water'
    assert stack[0].label == 'NN'
    assert stack[0].start == 0
    assert stack[0].end == 1


def test_bracket_validity(start_state):
    state = start_state
    bracket = DoBracket()
    assert not bracket.is_valid(state)
    shift = DoShift()
    state = shift.apply(state)
    state = shift.apply(state)
    assert bracket.is_valid(state)

def test_merge_validity(start_state):
    state = start_state
    b = DoBracket()
    s = DoShift()
    m = DoMerge()
    assert not m.is_valid(state)
    state = s.apply(state)
    assert not m.is_valid(state)
    state = s.apply(state)
    state = s.apply(state)
    state = b.apply(state)
    assert m.is_valid(state)


def goto(actions, state, hist):
    for move in hist:
        state = actions[move].apply(state)
    return state

@pytest.fixture
def s_state(start_state, actions):
    state = goto(actions, start_state, [SHIFT])
    return state

@pytest.fixture
def ss_state(start_state, actions):
    state = goto(actions, start_state, [SHIFT, SHIFT])
    return state


@pytest.fixture
def sb_state(start_state, actions):
    state = goto(actions, start_state, [SHIFT, BRACKET])
    return state


@pytest.fixture
def sbs_state(start_state, actions):
    return goto(actions, start_state, [SHIFT, BRACKET, SHIFT])


@pytest.fixture
def sssb_state(start_state, actions):
    return goto(actions, start_state, [SHIFT, SHIFT, SHIFT, BRACKET])


def test_bracket(ss_state, actions):
    stack, queue = ss_state.stack, ss_state.queue
    assert len(stack) == 2
    assert stack[0].start == 0
    assert stack[0].end == 1
    assert stack[0].production == ('NN', tuple())
    ssb = actions[BRACKET].apply(ss_state)
    assert len(ssb.stack[-1].children) == 1
    assert ssb.stack[-1].production == (None, ('NN',))
 

def test_merge(sssb_state, actions):
    m = actions[MERGE]
    state = m.apply(sssb_state)
    assert len(state.stack) == 2
    assert state.stack[0].start == 0
    assert state.stack[0].end == 1
    assert state.stack[1].start == 1
    assert state.stack[1].end == 3
    q_before = list(state.queue)
    assert state.stack[1].production == (None, ('NN', 'NN'))
    state = m.apply(state)
    assert len(state.stack) == 1
    assert state.queue == q_before
    assert state.stack[0].start == 0
    assert state.stack[0].end == 3
    assert state.stack[0].production == (None, ('NN', 'NN', 'NN'))
    assert not m.is_valid(state)


def test_shift_oracle(start_state, g_01_23, actions):
    assert len(g_01_23) == 3
    oracle = Oracle(g_01_23[-1])
    s = actions[SHIFT]
    assert s in oracle.next_actions(start_state)
    state = s.apply(start_state)
    assert s in oracle.next_actions(state)
    state = s.apply(state)
    assert not s in oracle.next_actions(state)
    state = s.apply(state)
    assert s in oracle.next_actions(state)


def test_bracket_oracle(start_state, g_01_23):
    state = start_state
    assert len(g_01_23) == 3
    s = DoShift()
    b = DoBracket('NP1')
    m = DoMerge()
    oracle = Oracle(g_01_23[-1])
    state = s.apply(state)
    assert not b in oracle.next_actions(state)
    state = s.apply(state)
    assert b in oracle.next_actions(state)
    state = b.apply(state)
    assert state.stack[-1].start == 1
    assert state.stack[-1].end == 2
    assert state.stack[-1].is_unary
    assert not b in oracle.next_actions(state)
    state = m.apply(state)
    state = s.apply(state)
    assert s in oracle.next_actions(state)
    assert not b in oracle.next_actions(state)
 
 
def test_merge_oracle(start_state, g_012_3):
    assert len(g_012_3) == 2
    s = DoShift()
    b = DoBracket('NP')
    b_top = DoBracket('TOP')
    m = DoMerge()
    state = start_state
    oracle = Oracle(g_012_3[-1])
    state = s.apply(state)
    state = s.apply(state)
    state = s.apply(state)
    state = b.apply(state)
    state = m.apply(state)
    assert not s in oracle.next_actions(state)
    assert not b in oracle.next_actions(state)
    assert m in oracle.next_actions(state)
    state = m.apply(state)
    state = s.apply(state)
    assert not m in oracle.next_actions(state)
    assert b_top in oracle.next_actions(state)
    state = b.apply(state)


def test_gold_top(start_state, g_01_23):
    assert len(g_01_23) == 3
    s = DoShift()
    b1 = DoBracket('NP1')
    b2 = DoBracket('NP2')
    b_top = DoBracket('TOP')
    m = DoMerge()
    state = start_state
    oracle = Oracle(g_01_23[-1])
    state = s.apply(state)
    state = s.apply(state)
    state = b1.apply(state)
    state = m.apply(state)
    state = s.apply(state)
    state = s.apply(state)
    assert not state.queue
    assert not s.is_valid(state)
    assert b2 in oracle.next_actions(state)
    state = b2.apply(state)
    state = m.apply(state)
    assert not m in oracle.next_actions(state)
    assert len(state.stack) == 2
    assert state.stack[-1].end == oracle.next_bracket.end
    assert b_top in oracle.next_actions(state)


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

    state = ParserState.from_words_and_tags([w.lex for w in leaves], [w.label for w in leaves])
    oracle = Oracle(top)

    s = DoShift()
    m = DoMerge()
    np = DoBracket('NP')
    vp = DoBracket('VP')
    adjp = DoBracket('ADJP')

    state = s.apply(state)
    state = s.apply(state)
    assert oracle.next_bracket.is_unary
    state = np.apply(state)
    #assert not golds.next()[0].is_unary
    assert vp in oracle.next_actions(state)
    assert not m in oracle.next_actions(state)
    state = vp.apply(state)
    state = m.apply(state)
    assert np in oracle.next_actions(state)
    state = np.apply(state)



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

    state = ParserState.from_words_and_tags([w.lex for w in leaves], [w.label for w in leaves])
    oracle = Oracle(top)

    s = DoShift()
    m = DoMerge()
    b = DoBracket()

    state = s.apply(state)
    state = s.apply(state)
    state = b.apply(state)
    state = m.apply(state)
    state = s.apply(state)
    state = s.apply(state)
    state = b.apply(state)
    state = s.apply(state)
    state = s.apply(state)
    state = s.apply(state)
    assert DoBracket in [type(x) for x in oracle.next_actions(state)]
    state = b.apply(state)
    state = m.apply(state)
    state = s.apply(state)
    assert not DoBracket in [type(x) for x in oracle.next_actions(state)]
    state = s.apply(state)
    state = s.apply(state)
    state = s.apply(state)
    state = s.apply(state)
    assert DoBracket in [type(x) for x in oracle.next_actions(state)]
    state = b.apply(state)
    state = m.apply(state)
    state = m.apply(state)
    state = m.apply(state)
    assert DoBracket in [type(x) for x in oracle.next_actions(state)]


def test_m_overpredict():
    ptb_str = """
( (S 
    (NP-SBJ-1 (DT The) (NN bill) )
    (VP (VBZ intends) 
      (S 
        (NP-SBJ (-NONE- *-1) )
        (VP (TO to) 
          (VP (VB restrict) 
            (NP (DT the) (NNP RTC) )
            (PP-CLR (TO to) 
              (NP (NNP Treasury) (NNS borrowings) (RB only) )))))
      (, ,) 
      (SBAR-ADV (IN unless) 
        (S 
          (NP-SBJ (DT the) (NN agency) )
          (VP (VBZ receives) 
            (NP (JJ specific) (JJ congressional) (NN authorization) )))))
    (. .) ))""".strip()

    words, bare_brackets = read_ptb.get_brackets(ptb_str)
    top = tree.from_brackets(words, bare_brackets)
    leaves = top.leaves()

    state = ParserState.from_words_and_tags([w.lex for w in leaves], [w.label for w in leaves])
    oracle = Oracle(top)

    s = DoShift()
    m = DoMerge()
    np = DoBracket('NP')
    vp = DoBracket('VP')
    pp = DoBracket('PP')
    np = DoBracket('NP')
    sbar = DoBracket('SBAR')
    b_s = DoBracket('S')

    state = s.apply(state) # the
    state = s.apply(state) # bill
    state = np.apply(state)
    state = m.apply(state); assert len(state.stack) == 1
    state = s.apply(state) # intends
    state = s.apply(state) # to
    state = s.apply(state) # restrict
    state = s.apply(state) # the
    state = s.apply(state) # rtc
    state = np.apply(state)
    state = m.apply(state); assert len(state.stack) == 5
    
    assert not vp in oracle.next_actions(state)
    
    state = vp.apply(state)
    state = m.apply(state)
    assert not m in oracle.next_actions(state)


def test_nested_b():
    ptb_str = """
( (S 
    (NP-SBJ 
      (NP (JJ Influential) (NNS members) )
      (PP (IN of) 
        (NP (DT the) (NNP House) (NNP Ways) 
          (CC and)
          (NNP Means) (NNP Committee) )))
    (VP (VBD introduced) 
      (NP 
        (NP (NN legislation) ) ))))""".strip()

    words, bare_brackets = read_ptb.get_brackets(ptb_str)
    top = tree.from_brackets(words, bare_brackets)
    leaves = top.leaves()

    state = ParserState.from_words_and_tags([w.lex for w in leaves], [w.label for w in leaves])
    oracle = Oracle(top)

    s = DoShift()
    m = DoMerge()
    np = DoBracket('NP')
    vp = DoBracket('VP')
    pp = DoBracket('PP')
    np = DoBracket('NP')
    sbar = DoBracket('SBAR')
    b_s = DoBracket('S')

    state = s.apply(state)
    state = s.apply(state); assert np in oracle.next_actions(state)
    state = np.apply(state); assert not np in oracle.next_actions(state)
    state = m.apply(state)


    state = s.apply(state)
    state = s.apply(state)
    state = s.apply(state)
    state = s.apply(state)
    state = s.apply(state)
    state = s.apply(state); assert s in oracle.next_actions(state)
    state = s.apply(state); assert np in oracle.next_actions(state)
    state = np.apply(state); assert not np in oracle.next_actions(state)
    
    assert state.stack[-1].end == 9
    assert state.stack[-1].start == 8 # committee
    
    state = m.apply(state) # means
    state = m.apply(state) # and
    state = m.apply(state) # ways
    state = m.apply(state) # house
    state = m.apply(state) # the

    assert state.stack[-1].start == 3
    assert pp in oracle.next_actions(state)
    state = pp.apply(state)
    #assert g.next()[0].child_match(st[-1])
    assert state.stack[-1].is_unary
    assert m in oracle.next_actions(state)
