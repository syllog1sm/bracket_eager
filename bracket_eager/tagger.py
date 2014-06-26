import os
from os import path
from collections import defaultdict
from . import util
from .perceptron import Perceptron

START = ['-START-', '-START2-']
END = ['-END-', '-END2-']
 
 
class DefaultList(list):
    """A list that returns a default value if index out of bounds."""
    def __init__(self, default=None):
        self.default = default
        list.__init__(self)
 
    def __getitem__(self, index):
        try:
            return list.__getitem__(self, index)
        except IndexError:
            return self.default
 
 
def setup_dir(model_dir, sentences):
    classes, tagdict = _make_tagdict(sentences)
    util.Config.write(model_dir, 'tagger', tagdict=tagdict, tags=classes)
    

def _make_tagdict(sentences):
    '''Make a tag dictionary for single-tag words.'''
    counts = defaultdict(lambda: defaultdict(int))
    classes = {}
    for sent in sentences:
        for word in sent:
            counts[word.lex][word.label] += 1
            classes[word.label] = True
    freq_thresh = 20
    ambiguity_thresh = 0.97
    tagdict = {}
    for word, tag_freqs in counts.items():
        tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
        n = sum(tag_freqs.values())
        # Don't add rare words to the tag dictionary
        # Only add quite unambiguous words
        if n >= freq_thresh and (float(mode) / n) >= ambiguity_thresh:
            tagdict[word] = tag
    return classes, tagdict


class Tagger(object):
    '''Greedy Averaged Perceptron tagger'''
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.cfg = util.Config.read(model_dir, 'tagger')
        self.tagdict = self.cfg.tagdict
        self.model = Perceptron(self.cfg.tags)
        if path.exists(path.join(model_dir, 'tagger.pickle')):
            self.model.load(path.join(model_dir, 'tagger.pickle'))

    def tag(self, words, tokenize=True):
        prev, prev2 = START
        tags = DefaultList('') 
        context = START + [self._normalize(w) for w in words] + END
        for i, word in enumerate(words):
            tag = self.tagdict.get(word)
            if not tag:
                features = self._get_features(i, word, context, prev, prev2)
                tag = self.model.predict(features)
            tags.append(tag)
            prev2 = prev; prev = tag
        return tags

    def save(self):
        self.model.save(path.join(self.model_dir, 'tagger.pickle'))

    def train_one(self, words, tags):
        prev, prev2 = START
        context = START + [self._normalize(w) for w in words] + END
        for i, word in enumerate(words):
            guess = self.tagdict.get(word)
            if not guess:
                feats = self._get_features(i, word, context, prev, prev2)
                guess = self.model.predict(feats)
                self.model.update(tags[i], guess, feats)
            prev2 = prev; prev = guess

    def load(self, loc):
        w_td_c = pickle.load(open(loc, 'rb'))
        self.model.weights, self.tagdict, self.classes = w_td_c
        self.model.classes = self.classes

    def _normalize(self, word):
        if '-' in word and word[0] != '-':
            return '!HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return '!YEAR'
        elif word[0].isdigit():
            return '!DIGITS'
        else:
            return word.lower()

    def _get_features(self, i, word, context, prev, prev2):
        '''Map tokens into a feature representation, implemented as a
        {hashable: float} dict. If the features change, a new model must be
        trained.'''
        def add(name, *args):
            features[' '.join((name,) + tuple(args))] += 1

        i += len(START)
        features = defaultdict(int)
        # It's useful to have a constant feature, which acts sort of like a prior
        add('bias')
        add('i suffix', word[-3:])
        add('i pref1', word[0])
        add('i-1 tag', prev)
        add('i-2 tag', prev2)
        add('i tag+i-2 tag', prev, prev2)
        add('i word', context[i])
        add('i-1 tag+i word', prev, context[i])
        add('i-1 word', context[i-1])
        add('i-1 suffix', context[i-1][-3:])
        add('i-2 word', context[i-2])
        add('i+1 word', context[i+1])
        add('i+1 suffix', context[i+1][-3:])
        add('i+2 word', context[i+2])
        return features


