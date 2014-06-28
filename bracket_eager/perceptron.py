from collections import defaultdict
import pickle


import numpy


class Perceptron(object):
    def __init__(self, classes):
        self.classes = classes
        self.class_map = dict((clas, i) for i, clas in enumerate(sorted(classes)))
        self.nr_class = len(self.classes)
        self.weights = {}
        # The accumulated values, for the averaging. These will be keyed by
        # feature/clas tuples
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/clas tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0
        self.nr_correct = 0
        self.nr_total = 0

    @property
    def accuracy_string(self):
        acc = float(self.nr_correct) / self.nr_total
        return '%.2f' % (acc * 100)

    def predict(self, features):
        '''Dot-product the features and current weights and return the best class.'''
        scores = self.score(features)
        # Do a secondary alphabetic sort, for stability
        return max(self.classes, key=lambda clas: (scores[clas], clas))

    def score(self, features):
        all_weights = self.weights
        scores_array = numpy.zeros(self.nr_class)
        for feat, value in features.iteritems():
            if value == 0:
                continue
            if feat not in all_weights:
                continue
            scores_array += all_weights[feat]
        scores = dict((clas, scores_array[self.class_map[clas]])
                      for clas in self.classes)
        return scores

    def update(self, truth, guess, features):       
        self.i += 1
        self.nr_total += 1
        if truth == guess:
            self.nr_correct += 1
            return None
        truth = self.class_map[truth]
        guess = self.class_map[guess]
        for f in features:
            if f not in self.weights:
                weights = numpy.zeros(self.nr_class)
                self.weights[f] = weights
            else:
                weights = self.weights[f]
            param = (f, truth)
            self._totals[param] += (self.i - self._tstamps[param]) * weights[truth]
            self._tstamps[param] = self.i
            weights[truth] += 1
            param = (f, guess)
            self._totals[param] += (self.i - self._tstamps[param]) * weights[guess]
            self._tstamps[param] = self.i
            weights[guess] -= 1

    def average_weights(self):
        for feat, weights in self.weights.iteritems():
            new_feat_weights = numpy.zeros(self.nr_class)
            for clas, weight in enumerate(weights):
                param = (feat, clas)
                total = self._totals[param]
                total += (self.i - self._tstamps[param]) * weight
                averaged = round(total / float(self.i), 3)
                if averaged:
                    new_feat_weights[clas] = averaged
            self.weights[feat] = new_feat_weights

    def save(self, path):
        print "Saving model to %s" % path
        pickle.dump(self.weights, open(path, 'w'))

    def load(self, path):
        self.weights = pickle.load(open(path))
