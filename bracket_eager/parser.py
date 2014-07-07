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
from .perceptron import Perceptron
from .features import extract_features

from .grammar import rules_from_trees

#from .transition_system import get_start_state
#from .transition_system import is_end_state
#from .transition_system import get_parse_from_state
#from .transition_system import get_actions
#from .transition_system import iter_gold
from transition_system2 import ParserState, get_actions, Oracle, DoBracket

def setup_dir(model_dir, sentences, **kwargs):
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    node_labels = util.get_node_labels(sentences)
    rules = rules_from_trees(sentences)
    util.Config.write(model_dir, 'config', node_labels=node_labels, **kwargs)
    with open(path.join(model_dir, 'rules.pickle'), 'w') as rules_file:
        pickle.dump(rules, rules_file)
 

def train(model_dir, sentences, nr_iter=15):
    setup_dir(model_dir, sentences, nr_iter=nr_iter)
    tagger.setup_dir(model_dir, [sent.leaves() for sent in sentences])
    parser = Parser(model_dir)
    for itn in range(nr_iter):
        ls = 0
        random.shuffle(sentences)
        for sentence in sentences:
            words = sentence.leaves()
            assert words
            ls += parser.train_one(itn, [w.lex for w in words], sentence)
            parser.tagger.train_one([w.lex for w in words], [w.label for w in words])
        print itn, 'Parse:', parser.model.accuracy_string,
        print 'Tag:', parser.tagger.model.accuracy_string
        print "Cumloss (sort of):",ls
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
        with open(path.join(model_dir, 'rules.pickle')) as rules_file:
            self.rules = pickle.load(rules_file)
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
        state = ParserState.from_words_and_tags(word_strings, tags)
        actions_by_name = dict((a.name, a) for a in self.actions)
        #print queue
        while not state.is_end_state():
            user_action = None
            stack, queue = state.stack, state.queue
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
        return state.get_parse_from_state()

    def parse(self, word_strings):
        # Getting passed a string when you want a list sucks to debug.
        assert not isinstance(word_strings, str)
        assert not isinstance(word_strings, unicode)
        # This closure is called by the "max" function. It returns a comparison
        # key, which makes max return the best-scoring valid action.
        def cmp_valid(action):
            return (action.is_valid(state), scores[action.i])

        tags = self.tagger.tag(word_strings)
        state = ParserState.from_words_and_tags(word_strings, tags)
        while not state.is_end_state():
            features = extract_features(state.stack, state.queue)
            scores = self.model.score(features)
            # Get highest scoring valid action
            best_action = max(self.actions, key=cmp_valid) 
            assert best_action.is_valid(state)
            newstate = best_action.apply(state)
            state = newstate
        return state.get_parse_from_state()

    def train_one(self, itn, word_strings, gold_tree):
        #print "t",
        tags = self.tagger.tag(word_strings)
        state = ParserState.from_words_and_tags(word_strings, tags)
        oracle = Oracle(gold_tree)
        unary_chain = 0
        while not state.is_end_state():
            #print format_state(state.stack, state.queue)
            features = extract_features(state.stack, state.queue)
            scores = self.model.score(features)
            actions = [a for a in self.actions if a.is_valid(state)]
            #if unary_chain > 2:
            #   actions = [a for a in self.actions if not isinstance(a, DoBracket)]
            guess = max(actions, key=lambda a: scores[a.i])
            #if isinstance(guess, DoBracket): unary_chain += 1
            #else: unary_chain = 0
            oracle_actions = oracle.next_actions(state)
            oracle_max = max(oracle_actions, key=lambda a: scores[a.i])
            self.model.update(oracle_max.i, guess.i, features)
            if False and itn == 0:
               state = oracle_max.apply(state)
            else:
               state = guess.apply(state)
        return oracle._cumloss

def format_weights(clas, features, weights):
    features = [(f, weights[f][clas]) for f in features
                if weights.get(f, {}).get(clas) if f != 'Bias']
    features.sort(reverse=True, key=lambda w: abs(w[1]))
    return ', '.join('%s=%d' % (f, w) for f, w in features[:5])


def format_state(stack, queue):
    return ', '.join(repr(s) for s in stack) +  ' || ' +  ' '.join(repr(q) for q in queue[:2])
