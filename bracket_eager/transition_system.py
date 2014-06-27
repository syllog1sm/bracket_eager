import tree

SHIFT = 0; BRACKET = 1; MERGE = 2; UNARY = 3
MOVES = (SHIFT, BRACKET, MERGE, UNARY)
MOVE_NAMES = ('S', 'B', 'M', 'U')


def get_start_state(words, tags):
    queue = [tree.Word(i, w, t) for i, (w, t) in enumerate(zip(words, tags))]
    stack = []
    return stack, queue


def is_end_state(stack, queue):
    return len(stack) == 1 and not queue


def get_parse_from_state(stack, queue):
    return '(TOP ' + stack[0].to_ptb() + ' )'


def get_actions(node_labels):
    actions = [DoShift(), DoMerge()]
    for label in sorted(node_labels):
        actions.append(DoBracket(label=label))
    for label in sorted(node_labels):
        actions.append(DoUnary(label=label))
    return actions


class Action(object):
    nr_actions = 0
    def __init__(self, label=None):
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

    def is_gold(self, stack, queue, next_gold):
        if not self.is_valid(stack, queue):
            return False
        return self._is_gold(stack[-1] if stack else None, next_gold)


class DoShift(Action):
    move = SHIFT

    def apply(self, stack, queue):
        stack.append(queue.pop(0))

    def is_valid(self, stack, queue):
        return bool(queue)

    def _is_gold(self, s0, next_gold):
        if not s0:
            return True
        return s0.end != next_gold.end


class DoMerge(Action):
    move = MERGE

    def apply(self, stack, queue):
        stack[-1].children.insert(0, stack.pop(-2))

    def is_valid(self, stack, queue):
        if len(stack) < 2:
            return False
        if len(stack[-1].children) < 2:
            return False
        return True

    def _is_gold(self, s0, next_gold):
        if s0.span_match(next_gold):
            return False
        if next_gold.children and s0.span_match(next_gold.children[-1]):
            return False
        return True


class DoBracket(Action):
    move = BRACKET

    def apply(self, stack, queue):
        s0 = stack.pop()
        s1 = stack.pop()
        bracket = tree.Bracket(s1, label=self.label)
        bracket.children.append(s0)
        stack.append(bracket)

    def is_valid(self, stack, queue):
        if len(stack) < 2:
            return False
        return True

    def _is_gold(self, s0, next_gold):
        if self.label and self.label != next_gold.label:
            return False
        if s0.end != next_gold.end:
            return False
        # Can we just M the bracket on the stack?
        if len(s0.children) >= 2 and \
          not (next_gold.children and s0.span_match(next_gold.children[-1])):
            return False
        if s0.span_match(next_gold) and next_gold.is_unary and not s0.is_unary:
            return False
        return True


class DoUnary(Action):
    move = UNARY

    def apply(self, stack, queue):
        stack.append(tree.Bracket(stack.pop(), label=self.label))

    def is_valid(self, stack, gold):
        if not stack:
            return False
        if stack[-1].is_unary:
            return False
        if self.label and stack[-1].label == self.label:
            return False
        return True

    def _is_gold(self, s0, gold_parent):
        if not gold_parent.is_unary:
            return False
        if not gold_parent.span_match(s0):
            return False
        if self.label and self.label != gold_parent.label:
            return False
        return True


def iter_gold(stack, queue, golds):
    """Iterate through the golds for the oracle, discarding golds whose cost
    is sunk. The stack/queue will be modified in-place by the outside context,
    as parsing proceeds."""
    golds = list(golds)
    while golds:
        starts = set([n.start for n in stack])
        if not stack:
            yield golds[0]
        elif golds[0] == stack[-1]:
            golds.pop(0)
        elif golds[0].span_match(stack[-1]) and not golds[0].is_unary:
            golds.pop(0)
        elif stack[-1].is_unary and golds[0].is_unary:
            golds.pop(0)
        elif golds[0].start >= stack[-1].end:
            yield golds[0]
        elif golds[0].end < stack[-1].end or golds[0].start not in starts:
            golds.pop(0)
        else:
            yield golds[0]
