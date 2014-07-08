import os
from os import path
from collections import defaultdict
from . import util
from ml import sml as ml

START = ['-START-', '-START2-']
END = ['-END-', '-END2-']
 
 
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
        self.classes = list(sorted(self.cfg.tags))
        self.class_map = {c:i for i,c in enumerate(self.classes)}
        if path.exists(path.join(model_dir, 'tagger.pickle')):
            fname = path.join(model_dir, 'tagger.pickle')
            self.model = ml.SparseMulticlassModel(file(fname),True)
        else:
            self.model = ml.SparseMultitronParameters(len(self.cfg.tags))
        self._nr_updates = 0
        self._nr_correct = 0

    def clear_stats(self):
        self._nr_updates = 0
        self._nr_correct = 0

    @property
    def accuracy_string(self):
       return str(self._nr_correct / float(self._nr_correct + self._nr_updates))


    def tag(self, words, tokenize=True):
        prev, prev2 = START
        tags = []
        context = START + [self._normalize(w) for w in words] + END
        for i, word in enumerate(words):
            tag = self.tagdict.get(word)
            if not tag:
                features = self._get_features(i, word, context, prev, prev2)
                tag, scores = self.model.predict(features)
                tag = self.classes[tag]
            tags.append(tag)
            prev2 = prev; prev = tag
        return tags

    def save(self):
        self.model.dump_fin(file(path.join(self.model_dir, 'tagger.pickle'),'w'),True)

    def train_one(self, words, tags):
        prev, prev2 = START
        context = START + [self._normalize(w) for w in words] + END
        for i, word in enumerate(words):
            guess = self.tagdict.get(word)
            if not guess:
                feats = self._get_features(i, word, context, prev, prev2)
                guess = self.model.update(self.class_map[tags[i]], feats)
                guess = self.classes[guess]
                if guess == tags[i]:
                   self._nr_correct += 1
                else:
                   self._nr_updates += 1
            prev2 = prev; prev = guess

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
        def add(f, name, *args):
            f('_'.join((name,) + tuple(args)))

        i += len(START)
        features = []
        f = features.append
        # It's useful to have a constant feature, which acts sort of like a prior
        add(f, 'bias')
        add(f, 'isuffix', word[-3:])
        add(f, 'ipref1', word[0])
        add(f, 'i-1tag', prev)
        add(f, 'i-2tag', prev2)
        add(f, 'itag+i-2tag', prev, prev2)
        add(f, 'iword', context[i])
        add(f, 'i-1tag+iword', prev, context[i])
        add(f, 'i-1word', context[i-1])
        add(f, 'i-1suffix', context[i-1][-3:])
        add(f, 'i-2word', context[i-2])
        add(f, 'i+1word', context[i+1])
        add(f, 'i+1suffix', context[i+1][-3:])
        add(f, 'i+2word', context[i+2])
        return features

