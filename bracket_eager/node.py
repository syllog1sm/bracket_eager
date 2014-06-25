class Node(object):
    def __init__(self, label=label):
        self.children = []

    def __eq__(self, o):
        return False


class Word(Node):
    def __init__(self, i, lex, tag=None):
        Node.__init__(self)
        self.i = i
        self.lex = lex
        self.tag = tag

    def __eq__(self, o):
        if not isinstance(o, Word):
            return False
        return self.i == o.i


class Bracket(Node):
    def __init__(self, child, label=None):
        Node.__init__(self, label=label)
        self.children.append(child)

    def __eq__(self, o):
        if not isinstance(o, Bracket):
            return False
        return self.start == o.start and self.end == o.end and self.label == o.label
