import tree
from .grammar import get_valid_stacks

SHIFT = 0; BRACKET = 1; MERGE = 2
MOVES = (SHIFT, BRACKET, MERGE)
MOVE_NAMES = ('S', 'B', 'M')


def get_start_state(words, tags):
    queue = [tree.Word(i, w, t) for i, (w, t) in enumerate(zip(words, tags))]
    stack = []
    return stack, queue


def is_end_state(stack, queue):
    return len(stack) == 1 and not queue


def get_parse_from_state(stack, queue):
    return '(TOP ' + stack[0].to_ptb() + ' )'


def get_actions(node_labels, rules):
    actions = [DoShift(), DoMerge()]
    for label in sorted(node_labels):
        actions.append(DoBracket(label=label))
    return actions


class Action(object):
    nr_actions = 0
    def __init__(self, label=None, rules=None):
        self.rules = rules
        self.name = MOVE_NAMES[self.move]
        if label:
            try:
                self.name += '-' + label
            except TypeError:
                raise TypeError("Label not stringish or falsey: " + repr(label))
        self.label = label
        self.i = Action.nr_actions
        Action.nr_actions += 1

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


class DoShift(Action):
    move = SHIFT

    def apply(self, stack, queue):
        stack.append(queue.pop(0))

    def is_valid(self, stack, queue):
        return bool(queue)

    def _is_gold(self, s0, golds):
        if not s0:
            return True
        return s0.end != golds[0].end


class DoMerge(Action):
    move = MERGE

    def apply(self, stack, queue):
        stack[-1].children.insert(0, stack.pop(-2))

    def is_valid(self, stack, queue):
        if len(stack) < 2:
            return False
        if stack[-1].is_leaf:
            return False
        return True

    def _is_gold(self, s0, golds):
        starts = [n.start for n in golds]
        next_gold = golds[0]
        if s0.unary_depth >= 3:
            return True
        if s0.span_match(next_gold):
            return False
        if s0.start in starts:
            return False
        return not _need_new_bracket(s0, next_gold)


class DoBracket(Action):
    move = BRACKET

    def apply(self, stack, queue):
        s0 = stack.pop()
        bracket = tree.Bracket(s0, label=self.label)
        stack.append(bracket)

    def is_valid(self, stack, queue):
        if len(stack) < 1:
            return False
        if stack[-1].unary_depth >= 3:
            return False
        return True

    def _is_gold(self, s0, golds):
        next_gold = golds[0]
        # Do we end a bracket here? No? Okay, don't add one
        if s0.end != next_gold.end:
            return False
        # If we're doing labelled bracketing, we need to care that _this_
        # DoBracket instance is the right one
        if self.label and self.label != next_gold.label:
            return False
        return _need_new_bracket(s0, next_gold)


def _need_new_bracket(s0, next_gold):
    """Break this out, because M and B are mutually exclusive here"""
    # Case 1: We're a leaf (okay, add bracket)
    if s0.is_leaf:
        return True
    # Case 2: We match the gold's child
    elif next_gold.child_match(s0):
        # ...But do we have an unnecessary unary?
        if s0.is_unary and not next_gold.children[-1].is_unary:
            return False
        else:
            return True
    else:
        return False


def iter_gold(stack, queue, golds):
    """Iterate through the golds for the oracle, discarding golds whose cost
    is sunk. The stack/queue will be modified in-place by the outside context,
    as parsing proceeds."""
    golds = list(golds)
    while golds:
        starts = set([n.start for n in stack])
        if not stack:
            yield golds
        elif golds[0] == stack[-1]:
            golds.pop(0)
        elif golds[0].span_match(stack[-1]) and not golds[0].is_unary:
            golds.pop(0)
        elif stack[-1].is_unary and golds[0].is_unary:
            golds.pop(0)
        elif golds[0].start >= stack[-1].end:
            yield golds
        elif golds[0].end < stack[-1].end or golds[0].start not in starts:
            golds.pop(0)
        else:
            yield golds
