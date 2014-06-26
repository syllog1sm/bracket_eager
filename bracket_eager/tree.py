def from_brackets(words, brackets):
    seen_words = set()
    children = []
    for label, start, end in brackets:
        # Trim labels
        if label != '-NONE-':
            assert label.split('-')[0], label
            label = label.split('-')[0]
        if (start + 1) == end and start not in seen_words:
            children.append(Word(start, words[start], label=label))
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
    return top


class Node(object):
    def __init__(self, label=None):
        self.children = []
        self.label = label

    def __eq__(self, o):
        return False

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

    def __repr__(self):
        return '%s_%d' % (self.lex, self.i)

    def __eq__(self, o):
        if not isinstance(o, Word):
            return False
        return self.i == o.i

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

    def __repr__(self):
        child_labels = ' '.join('%s_%d' % (n.label, n.end) for n in self.children)
        return '%s_%d --> %s' % (self.label, self.start, child_labels)

    @property
    def production(self):
        return (self.label, tuple(n.label for n in self.children))

    @property
    def start(self):
        return self.children[0].start

    @property
    def end(self):
        return self.children[-1].end

    def __eq__(self, o):
        if not isinstance(o, Bracket):
            return False
        return self.start == o.start and self.end == o.end and self.label == o.label

    def prune_traces(self):
        if len(self.children) == 1 and self.children[0].label == '-NONE-':
            self.children = []

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
