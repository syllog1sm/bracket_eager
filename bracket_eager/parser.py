"""A simple implementation of a greedy transition-based parser. Released under BSD license."""
from os import path
import os
import sys
from collections import defaultdict
import random
import time
import pickle
import shutil

from . import util
from . import tagger
from .transition_system import get_start_state
from .transition_system import is_end_state
from .transition_system import get_parse_from_state
from .transition_system import get_actions
from .transition_system import iter_gold
from .perceptron import Perceptron
from .features import extract_features


def setup_dir(model_dir, sentences, **kwargs):
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    node_labels = util.get_node_labels(sentences)
    util.Config.write(model_dir, 'config', node_labels=node_labels, **kwargs)
 

def train(model_dir, sentences, nr_iter=15):
    setup_dir(model_dir, sentences, nr_iter=nr_iter)
    tagger.setup_dir(model_dir, [sent.leaves() for sent in sentences])
    parser = Parser(model_dir)
    for itn in range(nr_iter):
        #random.shuffle(sentences)
        for sentence in sentences:
            words = sentence.leaves()
            assert words
            parser.train_one(itn, [w.lex for w in words], sentence)
            parser.tagger.train_one([w.lex for w in words], [w.label for w in words])
        print itn, 'Parse:', parser.model.accuracy_string,
        print 'Tag:', parser.tagger.model.accuracy_string
        parser.model.nr_correct = 0
        parser.model.nr_total = 0
    parser.model.average_weights()
    parser.tagger.model.average_weights()
    parser.save()


class Parser(object):
    def __init__(self, model_dir):
        assert os.path.exists(model_dir) and os.path.isdir(model_dir)
        self.model_dir = model_dir
        self.cfg = util.Config.read(model_dir, 'config')
        self.actions = get_actions(self.cfg.node_labels)
        self.model = Perceptron([a.i for a in self.actions])
        if os.path.exists(path.join(model_dir, 'parser.pickle')):
            self.model.load(path.join(model_dir, 'parser.pickle'))
        self.tagger = tagger.Tagger(model_dir)

    def save(self):
        self.model.save(path.join(self.model_dir, 'parser.pickle'))
        self.tagger.save()
    
    def parse(self, word_strings):
        # Getting passed a string when you want a list sucks to debug.
        assert not isinstance(word_strings, str)
        assert not isinstance(word_strings, unicode)
        # This closure is called by the "max" function. It returns a comparison
        # key, which makes max return the best-scoring valid action.
        def cmp_valid(action):
            return (action.is_valid(stack, queue), scores[action.i])

        tags = self.tagger.tag(word_strings)
        stack, queue = get_start_state(word_strings, tags)
        while not is_end_state(stack, queue):
            features = extract_features(stack, queue)
            scores = self.model.score(features)
            # Get highest scoring valid action
            best_action = max(self.actions, key=cmp_valid) 
            assert best_action.is_valid(stack, queue)
            best_action.apply(stack, queue)
        return get_parse_from_state(stack, queue)

    def train_one(self, itn, word_strings, gold_tree):
        # This closure is called by the "max" function. It returns a comparison
        # key, which makes max return the best-scoring valid action.
        def score_if_valid(action):
            is_valid = action.is_valid(stack, queue)
            return (is_valid, scores[action.i])

        def score_if_gold(action):
            is_gold = action.is_gold(stack, queue, target_bracket)
            return (is_gold, scores[action.i])
        tags = self.tagger.tag(word_strings)
        stack, queue = get_start_state(word_strings, tags)
        golds = iter_gold(stack, queue, gold_tree.depth_list())
        nr_moves = 0
        nr_correct = 0
        while not is_end_state(stack, queue):
            target_bracket = golds.next()
            features = extract_features(stack, queue)
            scores = self.model.score(features)
            guess = max(self.actions, key=score_if_valid)
            assert guess.is_valid
            gold = max(self.actions, key=score_if_gold)
            if not gold.is_gold(stack, queue, target_bracket):
                print
                print [n.start for n in stack]
                print 'Stack:', stack[-1]
                print 'Target', target_bracket
                print target_bracket.children[-1]
                raise StandardError
            self.model.update(gold.i, guess.i, features)
            guess.apply(stack, queue)
