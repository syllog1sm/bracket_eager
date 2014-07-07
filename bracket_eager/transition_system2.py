from collections import defaultdict
import itertools

import tree
from .grammar import get_valid_stacks


SHIFT = 0; BRACKET = 1; MERGE = 2
MOVES = (SHIFT, BRACKET, MERGE)
MOVE_NAMES = ('S', 'B', 'M')


class ParserState(object):
   def __init__(self, stack, queue):
      self.stack = stack
      self.queue = queue

   @classmethod
   def from_words_and_tags(cls, words, tags):
      queue = [tree.Word(i, w, t) for i, (w, t) in enumerate(zip(words, tags))]
      stack = []
      return ParserState(stack, queue)

   def is_end_state(self):
      return len(self.stack) == 1 and not self.queue

   def get_parse_from_state(self):
      return '(TOP ' + self.stack[0].to_ptb() + ' )'

def get_actions(node_labels, rules):
   actions = [DoShift(), DoMerge()]
   for label in sorted(node_labels):
      actions.append(DoBracket(label=label))
   return actions

class Action(object):
    name_to_nr = defaultdict(itertools.count(0).next)
    def __init__(self, label=None, rules=None):
        self.rules = rules
        self.name = MOVE_NAMES[self.move]
        if label:
            try:
                self.name += '-' + label
            except TypeError:
                raise TypeError("Label not stringish or falsey: " + repr(label))
        self.label = label

    @property
    def i(self):
       return Action.name_to_nr[self.name]

    def __str__(self):
        return self.name

    def is_gold(self, stack, queue, golds):
        if not self.is_valid(stack, queue):
            return False
        return self._is_gold(stack[-1] if stack else None, golds)

    def check_grammar(self, result, *child_nodes):
        if not self.rules:
            return True
        child_labels = tuple(n.label for n in child_nodes)
        return result in self.rules and child_labels in self.rules[result]

    def is_grammatical(self, stack, queue):
        return self.is_valid(stack, queue)

    def __eq__(self, other):
       return self.i == other.i


class DoShift(Action):
    move = SHIFT

    def apply(self, state):
       assert(self.is_valid(state)), (self, state.stack, state.queue)
       new_stack = state.stack + [state.queue[0]]
       new_queue = state.queue[1:]
       return ParserState(new_stack, new_queue)

    def is_valid(self, state):
        return bool(state.queue)

class DoMerge(Action):
    move = MERGE

    def apply(self, state):
       s0 = state.stack[-1]
       s1 = state.stack[-2]
       newstack = state.stack[:-2]
       newstack.append(tree.MakeBracket(s1, s0.children, s0.label))
       return ParserState(newstack, state.queue)

    def is_valid(self, state):
        if len(state.stack) < 2:
            return False
        if state.stack[-1].is_leaf:
            return False
        return True

class DoBracket(Action):
    move = BRACKET

    def apply(self, state):
       s0 = state.stack[-1]
       bracket = tree.Bracket(s0, label=self.label)
       return ParserState(state.stack[:-1] + [bracket], state.queue)

    def is_valid(self, state):
        if len(state.stack) < 1:
            return False
        if state.stack[-1].unary_depth >= 3:
            #print "DEEP UNARY"
            return False
        #    return False
        return True

# NOTE: 
#   the _cumloss field is not reliable/accurate. Problem - no loss for extra 
#   bracket action. (because we can always recover with another
#   bracket, and no loss for precision error)
DEBUG=False
class Oracle(object):
   def __init__(self, gold_tree):
      self.gold_tree = gold_tree
      self.gold_brackets = gold_tree.depth_list()
      self.next_bracket_i = 0
      self.next_bracket = self.gold_brackets[0]
      self._cumloss = 0

   def _advance_bracket(self):
      self.next_bracket_i += 1
      self.next_bracket = self.gold_brackets[self.next_bracket_i]

   def _is_reachable(self, span, stack, buffer):
      if not stack: return span.start <= buffer[0].start
      if span.start >= stack[-1].end: return True
      if stack[-1].end > span.end: return False
      for x in stack:
         if x.start == span.start: return True
      return False

   def next_actions(self, state):
      stack, queue = state.stack, state.queue
      if not stack: return [DoShift()]
      if stack[-1] == self.next_bracket:
         self._advance_bracket()
         # unary chain of same type
         if stack[-1] == self.next_bracket:
            return [DoBracket(self.next_bracket.label)]
      while (not self._is_reachable(self.next_bracket, stack, buffer)):
         self._cumloss += 1
         self._advance_bracket()
      # if we landed on a bracket we already have, but with the wrong internal
      # structure (for example, we wanted to build (VP (PRT S)) but got (VP (V S)) instead)
      while stack[-1] == self.next_bracket:
         self._advance_bracket()
      target = self.next_bracket
      if DEBUG:
         print "target:",target
         print "stack:", stack[-1]
         print "eq:",stack[-1] == target, stack[-1].start,stack[-1].end,stack[-1].label,target.start,target.end,target.label
      if stack[-1].start == target.start and stack[-1].end == target.end:
         # we know they are not equal
         assert(stack[-1].label != target.label)
         return [DoBracket(target.label)]
      if stack[-1].end == target.end:
         # if the label is not correct, and its reachable, we need
         # to add the bracket
         if stack[-1].label != target.label:
            return [DoBracket(target.label)]
         # we know it's reachable, so Merge is fine
         return [DoMerge()]
      else:
         assert stack[-1].end < target.end
         return [DoShift()]

   
