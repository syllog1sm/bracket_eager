#!/usr/bin/env python
import sys
sys.path.append(".")
import plac

from bracket_eager import read_ptb
from bracket_eager import tree

def main():
    text = sys.stdin.read()
    sent_strs = read_ptb.split_sentences(text)
    for sent_str in sent_strs:
        words, brackets = read_ptb.get_brackets(sent_str)
        top = tree.from_brackets(words, brackets)
        leaves = top.leaves()
        print ' '.join(w.lex for w in leaves if w.label != '-NONE-')


if __name__ == '__main__':
    plac.call(main)
