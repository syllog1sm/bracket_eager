from os import path
import json


class Config(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def write(cls, model_dir, name, **kwargs):
        open(path.join(model_dir, '%s.json' % name), 'w').write(json.dumps(kwargs))

    @classmethod
    def read(cls, model_dir, name):
        return cls(**json.load(open(path.join(model_dir, '%s.json' % name))))


def get_node_labels(sentences):
    labels = {} # Use dict not set for json serialisability
    for sent in sentences:
        queue = [sent]
        for node in queue:
            if not node.is_leaf:
                labels[node.label] = True
                queue.extend(node.children)
    return labels
