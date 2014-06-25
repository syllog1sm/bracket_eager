class Node(object):
    def __init__(self, label=None):
        self.children = []
        self.label = label

    def __eq__(self, o):
        return False


class Word(Node):
    def __init__(self, i, lex, label=None):
        Node.__init__(self, label=label)
        self.i = i
        self.lex = lex
        self.start = i
        self.end = i + 1
        self.production = (self.label, tuple())

    def __eq__(self, o):
        if not isinstance(o, Word):
            return False
        return self.i == o.i


class Bracket(Node):
    def __init__(self, child, label=None):
        Node.__init__(self, label=label)
        self.children.append(child)

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
