import tree

SHIFT = 0; BRACKET = 1; MERGE = 2;
MOVES = (SHIFT, BRACKET, MERGE)


def get_start_state(words, tags):
    queue = [tree.Word(i, w, t) for i, (w, t) in enumerate(zip(words, tags))]
    stack = []
    return stack, queue


def is_end_state(stack, queue):
    return len(stack) == 1 and not queue


def get_parse_from_state(stack, queue):
    return stack[0].to_ptb()


def get_actions(node_labels):
    actions = [DoShift(0), DoMerge(1)]
    for label in sorted(node_labels):
        actions.append(DoBracket(len(actions), label=label))
    return actions


class Action(object):
    def __init__(self, i, label):
        self.i = i
        self.label = label

    def __str__(self):
        return self.name


class DoShift(Action):
    def __init__(self, i, label=None):
        Action.__init__(self, i, label)
        self.move = SHIFT
        self.label = label
        self.name = 'S'

    def apply(self, stack, queue):
        if self.label is not None:
            queue[0].label = self.label
        stack.append(queue.pop(0))

    def is_valid(self, stack, queue):
        if stack and stack[-1].children and stack[-1] == stack[-1].children[-1]:
            return False
        return bool(queue)

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
        self.name = 'M'

    def apply(self, stack, queue):
        if self.label is not None:
            stack[-1].label = self.label
        stack[-1].children.insert(0, stack.pop(-2))

    def is_valid(self, stack, queue):
        return len(stack) >= 2 and isinstance(stack[-1], tree.Bracket)

    def is_gold(self, stack, queue, next_gold):
        if not self.is_valid(stack, queue):
            return False
        # If the production on the stack is self-unary, we must merge
        if stack[-1].children and stack[-1] == stack[-1].children[-1]:
            return True
        if next_gold.children[-1] == stack[-1]:
            return False
        return stack[-1].start != next_gold.start or stack[-1].end != next_gold.end


class DoBracket(Action):
    def __init__(self, i, label=None):
        Action.__init__(self, i, label)
        self.move = BRACKET
        self.name = 'B-%s' % (label if label is not None else '?')

    def apply(self, stack, queue):
        stack.append(tree.Bracket(stack.pop(), label=self.label))

    def is_valid(self, stack, queue):
        # Disallow unary chains of length 4. If we have 3 unaries in a row,
        # we can't add another.
        if not stack:
            return False
        if len(stack) == 1 and self.label == stack[-1].label:
            return False
        if stack[-1].children and stack[-1] == stack[-1].children[-1]:
            return False
        return True

    def is_gold(self, stack, queue, next_gold):
        if not self.is_valid(stack, queue):
            return False
        s0 = stack[-1]
        if s0.label == next_gold.label:
            if s0 == next_gold.children[-1] and s0 != next_gold.children[-1].children[-1]:
                need_new_bracket = True
            else:
                need_new_bracket = False
        else:
            need_new_bracket = True
        assert s0 != next_gold
        return s0.end == next_gold.end and \
               need_new_bracket and \
               self.label == next_gold.label


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
        elif golds[0].start >= stack[-1].end:
            yield golds[0]
        elif golds[0].end < stack[-1].end or golds[0].start not in starts:
            golds.pop(0)
        else:
            yield golds[0]
