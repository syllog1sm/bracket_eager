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
    for label in node_labels:
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
        if next_gold.children[-1] == stack[-1]:
            return False
        return stack[-1].start != next_gold.start or stack[-1].end != next_gold.end


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
        if next_gold is None:
            return False
        s0 = stack[-1]
        assert s0 != next_gold
        return s0.end == next_gold.end and \
               s0.label != next_gold.label and \
               self.label == next_gold.label


def iter_gold(stack, queue, golds):
    """Iterate through the golds for the oracle, discarding golds whose cost
    is sunk. The stack/queue will be modified in-place by the outside context,
    as parsing proceeds."""
    golds = list(golds)
    while True:
        starts = set([n.start for n in stack])
        if not stack:
            yield golds[0]
        elif golds[0] == stack[-1]:
            golds.pop(0)
        elif golds[0].start >= stack[-1].start:
            yield golds[0]
        elif golds[0].end < stack[-1].end or golds[0].start not in starts:
            golds.pop(0)
        else:
            yield golds[0]
