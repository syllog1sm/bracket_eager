"""A simple implementation of a greedy transition-based parser. Released under BSD license."""
from os import path
import os
import sys
from collections import defaultdict
import random
import time
import pickle


from .transition_system import get_start_state
from .transition_system import is_end_state
from .transition_system import get_parse_from_state
from .transition_system import oracle
from .transition_system import get_actions


class Parser(object):
    def __init__(self, load=True):
        model_dir = os.path.dirname(__file__)
        self.model = Perceptron(MOVES)
        if load:
            self.model.load(path.join(model_dir, 'parser.pickle'))
        self.tagger = PerceptronTagger(load=load)

    def save(self):
        self.model.save(path.join(os.path.dirname(__file__), 'parser.pickle'))
        self.tagger.save()
    
    def parse(self, word_strings):
        # This closure is called by the "max" function. It returns a comparison
        # key, which makes max return the best-scoring valid action.
        def cmp_valid(action):
            return (action.is_valid(stack, queue), scores[action.i])

        tags = self.tagger.tag(word_strings)
        stack, queue = get_start_state(word_strings, tags)
        while not is_end_start(stack, queue):
            features = extract_features(stack, queue)
            scores = self.model.score(features)
            # Get highest scoring valid action
            best_action = max(self.actions, key=cmp_valid) 
            best_action.apply(stack, queue)
        return get_parse_from_state(stack, queue)

    def train_one(self, itn, word_strings, gold_brackets):
        # This closure is called by the "max" function. It returns a comparison
        # key, which makes max return the best-scoring valid action.
        def cmp_valid(action):
            return (action.is_valid(stack, queue), scores[action.i])

        tags = self.tagger.tag(word_strings)
        stack, queue = get_start_state(word_strings, tags)
        while not is_end_state(stack, queue):
            predicted = self.predict(stack, queue)
            features = extract_features(stack, queue)
            scores = self.model.score(features)
            guess = max(self.actions, key=score_of_valid)
            gold = max(oracle(moves, stack, queue), key=lambda a: scores[a.i])
            self.model.update(best.i, guess.i, features)
            guess.apply(stack, queue)
   

def train(parser, sentences, nr_iter):
    parser.tagger.start_training(sentences)
    for itn in range(nr_iter):
        corr = 0; total = 0
        random.shuffle(sentences)
        for words, gold_tags, gold_brackets in sentences:
            corr += parser.train_one(itn, words, gold_tags, gold_parse)
            if itn < 5:
                parser.tagger.train_one(words, gold_tags)
            total += len(words)
        print itn, '%.3f' % (float(corr) / float(total))
        if itn == 4:
            parser.tagger.model.average_weights()
    print 'Averaging weights'
    parser.model.average_weights()


def main(model_dir, train_loc, heldout_in, heldout_gold):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    input_sents = list(read_pos(heldout_in))
    parser = Parser(load=False)
    sentences = list(read_conll(train_loc))
    train(parser, sentences, nr_iter=15)
    parser.save()
    c = 0
    t = 0
    gold_sents = list(read_conll(heldout_gold))
    t1 = time.time()
    for (words, tags), (_, _, gold_heads, gold_labels) in zip(input_sents, gold_sents):
        _, heads = parser.parse(words)
        for i, w in list(enumerate(words))[1:-1]:
            if gold_labels[i] in ('P', 'punct'):
                continue
            if heads[i] == gold_heads[i]:
                c += 1
            t += 1
    t2 = time.time()
    print 'Parsing took %0.3f ms' % ((t2-t1)*1000.0)
    print c, t, float(c)/t


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
