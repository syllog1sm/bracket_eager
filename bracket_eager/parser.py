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

from .grammar import rules_from_trees


def setup_dir(model_dir, sentences, **kwargs):
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    node_labels = util.get_node_labels(sentences)
    rules = rules_from_trees(sentences)
    util.Config.write(model_dir, 'config', node_labels=node_labels,
                      **kwargs)
    pickle.dump(rules, open(path.join(model_dir, 'rules.pickle'), 'w'))
 

def train(model_dir, sentences, nr_iter=15):
    setup_dir(model_dir, sentences, nr_iter=nr_iter)
    tagger.setup_dir(model_dir, [sent.leaves() for sent in sentences])
    parser = Parser(model_dir)
    for itn in range(nr_iter):
        random.shuffle(sentences)
        for sentence in sentences:
            words = sentence.leaves()
            assert words
            parser.train_one(itn, [w.lex for w in words], sentence)
            parser.tagger.train_one([w.lex for w in words], [w.label for w in words])
        print itn, 'Parse:', parser.model.accuracy_string,
        print 'Tag:', parser.tagger.model.accuracy_string
        parser.model.nr_correct = 0
        parser.model.nr_total = 0
        parser.tagger.model.nr_correct = 0
        parser.tagger.model.nr_total = 0
    parser.model.average_weights()
    parser.tagger.model.average_weights()
    parser.save()


class Parser(object):
    def __init__(self, model_dir):
        assert os.path.exists(model_dir) and os.path.isdir(model_dir)
        self.model_dir = model_dir
        self.cfg = util.Config.read(model_dir, 'config')
        self.rules = pickle.load(open(path.join(model_dir, 'rules.pickle')))
        self.actions = get_actions(self.cfg.node_labels, self.rules)
        self.model = Perceptron([a.i for a in self.actions])
        if os.path.exists(path.join(model_dir, 'parser.pickle')):
            self.model.load(path.join(model_dir, 'parser.pickle'))
        self.tagger = tagger.Tagger(model_dir)

    def save(self):
        self.model.save(path.join(self.model_dir, 'parser.pickle'))
        self.tagger.save()
 
    def debug_parse(self, word_strings):
        # Getting passed a string when you want a list sucks to debug.
        assert not isinstance(word_strings, str)
        assert not isinstance(word_strings, unicode)
        # This closure is called by the "max" function. It returns a comparison
        # key, which makes max return the best-scoring valid action.
        def cmp_valid(action):
            return (action.is_valid(stack, queue), scores[action.i])

        tags = self.tagger.tag(word_strings)
        stack, queue = get_start_state(word_strings, tags)
        actions_by_name = dict((a.name, a) for a in self.actions)
        print queue
        while not is_end_state(stack, queue):
            user_action = None
            features = extract_features(stack, queue)
            scores = self.model.score(features)
            # Get highest scoring valid action
            print format_state(stack, queue) 
            best_action = max(self.actions, key=cmp_valid)
            print 'P:', best_action, scores[best_action.i]
            cmd = raw_input('>: ')
            if cmd.strip():
                user_action = actions_by_name[cmd.strip()]
                print format_weights(best_action.i, features, self.model.weights)
                print format_weights(user_action.i, features, self.model.weights)
            if user_action is None:
                best_action.apply(stack, queue)
            else:
                user_action.apply(stack, queue)
        return get_parse_from_state(stack, queue)

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
        b_vp = [a for a in self.actions if a.move == 1 and a.label == 'VP'][0]
        while not is_end_state(stack, queue):
            features = extract_features(stack, queue)
            scores = self.model.score(features)
            # Get highest scoring valid action
            best_action = max(self.actions, key=cmp_valid) 
            assert best_action.is_valid(stack, queue)
            best_action.apply(stack, queue)
        return get_parse_from_state(stack, queue)

    def train_one(self, itn, word_strings, gold_tree):
        tags = self.tagger.tag(word_strings)
        stack, queue = get_start_state(word_strings, tags)
        golds = iter_gold(stack, queue, gold_tree.depth_list())
        nr_moves = 0
        nr_correct = 0
        max_stack_len = 0
        while not is_end_state(stack, queue):
            target_bracket = golds.next()
            features = extract_features(stack, queue)
            scores = self.model.score(features)
            actions = [a for a in self.actions if a.is_valid(stack, queue)]
            guess = max(actions, key=lambda a: scores[a.i])
            assert guess.is_valid(stack, queue)
            actions = [a for a in self.actions if a.is_gold(stack, queue, golds.next())]
            if not actions:
                print target_bracket[0]
                print format_state(stack, queue)
                raise StandardError
            gold = max(actions, key=lambda a: scores[a.i])
            assert gold.is_gold(stack, queue, target_bracket)
            self.model.update(gold.i, guess.i, features)
            guess.apply(stack, queue)
            max_stack_len = max(len(stack), max_stack_len)


def format_weights(clas, features, weights):
    features = [(f, weights[f][clas]) for f in features
                if weights.get(f, {}).get(clas) if f != 'Bias']
    features.sort(reverse=True, key=lambda w: abs(w[1]))
    return ', '.join('%s=%d' % (f, w) for f, w in features[:5])


def format_state(stack, queue):
    return ', '.join(repr(s) for s in stack) +  ' || ' +  ' '.join(repr(q) for q in queue[:2])
