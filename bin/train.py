#!/usr/bin/env python
import sys
sys.path.append(".")
import plac
from bracket_eager import parser
from bracket_eager import read_ptb
import os.path

import random
random.seed(1)

@plac.annotations(
    model_dir=("Place to save parser", "positional"),
    #ptb_dir=("Path to Penn Treebank root directory", "positional"),
    ptb_file=("Path to brackets trees file, one tree per line.", "positional"),
    size=("Number of sentences to train from", "option", "n", int)
)
def main(model_dir, ptb_file, size=0):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    #sents = read_ptb.read_sentences(ptb_dir, n=size)
    sents = list(read_ptb.read_oneperline(ptb_file, n=size))
    parser.train(model_dir, sents)


if __name__ == '__main__':
    plac.call(main)
