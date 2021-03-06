def from_brackets(words, brackets):
    seen_words = set()
    children = []
    for label, start, end in brackets:
        # Trim labels
        if not label.startswith('-'):
            assert label.split('-')[0], label
            label = label.split('-')[0].split('=')[0]
        if (start + 1) == end and start not in seen_words:
            children.append(Word(start, words[start], label=label))
            seen_words.add(start)
        else:
            for i, first_child in enumerate(children):
                if start == first_child.start:
                    node = Bracket(first_child, label=label)
                    # We already added first_child, at i
                    node.children.extend(children[i+1:])
                    children = children[:i] + [node]
                    break
            else:
                print [(n.label, n.start, n.end) for n in children]
                raise StandardError
    top = children[0]
    for node in top.iter_nodes():
        node.prune_traces()
    top.prune_empty()
    for i, w in enumerate(top.leaves()):
        w.i = i
        w.start = i
        w.end = i+1
    return top


class Node(object):
    def __init__(self, label=None):
        self.children = []
        self.label = label

    def span_match(self, o):
        return self.start == o.start and self.end == o.end

    def child_match(self, other):
        if not self.children:
            return False
        if self.children[-1].is_leaf != other.is_leaf:
            return False
        return self.children[-1].span_match(other)

    def child_eq(self, other):
        if not self.children:
            return False
        return self.children[-1] == other


    @property
    def unary_depth(self):
        d = 0
        n = self
        while len(n.children) == 1:
            d += 1
            n = n.children[0]
        return d

    def depth_list(self):
        nodes = []
        for node in self.children:
            nodes.extend(node.depth_list())
        nodes.append(self)
        return nodes


class Word(Node):
    def __init__(self, i, lex, label=None):
        Node.__init__(self, label=label)
        self.i = i
        self.lex = lex
        self.start = i
        self.end = i + 1
        self.production = (self.label, tuple())
        self.is_leaf = True
        self.is_unary = False
        self.depth = 0
        self.head = self

    def __repr__(self):
        return '%s_%d' % (self.lex, self.i)

    def __eq__(self, o):
        if not isinstance(o, Word):
            return False
        return self.i == o.i

    def to_ptb(self, indent=0):
        return '(%s %s)' % (self.label, self.lex)

    def depth_list(self):
        return []

    def prune_traces(self):
        pass

    def prune_empty(self):
        pass

    def leaves(self):
        return [self]


class Bracket(Node):
    def __init__(self, child, label=None):
        Node.__init__(self, label=label)
        self.children.append(child)
        self.is_leaf = False

    def __repr__(self):
        child_labels = []
        for n in self.children:
            if n.is_leaf:
                child_labels.append('%s_%d' % (n.lex, n.i))
            else:
                child_labels.append('%s_%d-%d' % (n.label, n.start, n.end))
        return '%s --> %s' % (self.label, ' '.join(child_labels))

    def __eq__(self, o):
        if self.label and o.label and self.label != o.label:
            return False
        if self.start != o.start:
            return False
        if self.end != o.end:
            return False
        return True

    def to_ptb(self, indent=0):
        pieces = []
        for child in self.children:
            pieces.append(child.to_ptb(indent+1))
        return '(%s ' % self.label + ' '.join(pieces) + ' )'

    @property
    def is_unary(self):
        return not self.is_leaf and len(self.children) == 1

    @property
    def production(self):
        return (self.label, tuple(n.label for n in self.children))

    @property
    def start(self):
        return self.children[0].start

    @property
    def end(self):
        return self.children[-1].end

    @property
    def head(self):
        if not self.children:
            return None
        else:
            return self.children[-1].head

    def prune_traces(self):
        self.children = [n for n in self.children if n.label != '-NONE-']

    def prune_empty(self):
        for node in self.children:
            node.prune_empty()
        self.children = [n for n in self.children
                         if isinstance(n, Word) or n.children and n.start != n.end]

    def iter_nodes(self):
        queue = [self]
        for node in queue:
            queue.extend(node.children)
        return queue

    def leaves(self):
        leaves = []
        for node in self.children:
            leaves.extend(node.leaves())
        return leaves
