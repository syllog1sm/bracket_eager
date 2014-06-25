import tree

SHIFT = 0; BRACKET = 1; MERGE = 2;
MOVES = (SHIFT, BRACKET, MERGE)


def get_start_state(words, tags):
    queue = [tree.Word(i, w, t) for i, (w, t) in enumerate(zip(words, tags))]
    stack = []
    return stack, queue


def is_end_state(stack, queue):
    return True if not queue else False


def get_parse_from_state(stack, queue):
    pass


def get_actions(node_labels):
    actions = [DoShift(0), DoMerge(1)]
    for label in labels:
        actions.append(DoBracket(len(actions), label=label))
    return actions


class Action(object):
    def __init__(self, i, label):
        self.i = i
        self.label = label


class DoShift(Action):
    def __init__(self, i, label=None):
        Action.__init__(self, i, label)
        self.move = SHIFT
        self.label = label

    def apply(self, stack, queue):
        if self.label is not None:
            queue[0].label = self.label
        stack.append(queue.pop(0))

    def is_valid(self, stack, queue):
        return True if queue else False

    def is_gold(self, stack, queue, next_gold):
        if not self.is_valid(stack, queue):
            return False
        if not stack:
            return True
        return stack[-1].end != next_gold.end



class DoMerge(Action):
    def __init__(self, i, label=None):
        Action.__init__(self, i, label)
        self.move = MERGE

    def apply(self, stack, queue):
        if self.label is not None:
            stack[-1].label = self.label
        stack[-1].children.insert(0, stack.pop(-2))

    def is_valid(self, stack, queue):
        return len(stack) >= 2 and isinstance(stack[-1], tree.Bracket)

    def is_gold(self, stack, queue, next_gold):
        if not self.is_valid(stack, queue):
            return False
        return stack[-1].edges_match(next_gold)


class DoBracket(Action):
    def __init__(self, i, label=None):
        Action.__init__(self, i, label)
        self.move = BRACKET

    def apply(self, stack, queue):
        stack.append(tree.Bracket(stack.pop(), label=self.label))

    def is_valid(self, stack, queue):
        return bool(stack)

    def is_gold(self, stack, queue, next_gold):
        if not self.is_valid(stack, queue):
            return False
        s0 = stack[-1]
        return s0.end == next_gold[-1].end and s0.label != next_gold.label


def iter_golds(stack, queue, golds):
    """Iterate through the golds for the oracle, discarding golds whose cost
    is sunk. The stack/queue will be modified in-place by the outside context,
    as parsing proceeds."""
    golds = list(golds)
    while golds:
        if not stack:
            yield golds[0]
        starts = set([n.start for n in stack]
        if golds[0] == stack[-1]:
            golds.pop(0)
        elif golds[0].start >= stack[-1].start:
            yield golds[0]
        if golds[0].end < stack[-1].end or golds[0].start not in starts:
            golds.pop(0)


def oracle(moves, stack, golds):
    if not stack or not golds:
        return set([SHIFT])
    if golds[0].end > stack[-1].end:
        return set([SHIFT])
    print 'First gold', golds[0].start, golds[0].end
    print 'Stack:', stack[-1].start, stack[-1].end
    # Stop considering golds that we've advanced past the end of
    while golds and golds[0].end < stack[-1].end:
        golds.pop()
    if not golds:
        return set([SHIFT])
    # Stop considering golds we can't match the start of
    while golds and golds[-1].start not in [n.start for n in stack]:
        golds.pop()

    g = golds[0]
    g2 = golds[1] if len(golds) >= 2 else None
    s0 = stack[-1]
    # Sunk bracket: Advance, or merge
    if g.end != s0.end:
        return set([MERGE, SHIFT])
    # Label is wrong, so introduce a new bracket
    elif g.label != s0.label:
        return set(a for a in moves if a.move == BRACKET and a.label == g.label)
    elif g.start < s0.start:
        return set([MERGE])
    elif g2 and g2.end == s0.end:
        return set(a for a in actions if a.move == BRACKET and a.label == g.label)
    else:
        return set([SHIFT])
