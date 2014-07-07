import re
import os
from os import path

import tree


def get_brackets(sent_text):
    sent_text = sent_text.strip()
    assert sent_text and sent_text.startswith('(')
    open_brackets = []
    brackets = []
    bracketsRE = re.compile(r'(\()([^\s\)\(]+)|([^\s\)\(]+)?(\))')
    word_i = 0
    words = []
    # Remove outermost bracket
    if sent_text.startswith('(('):
        sent_text = sent_text.replace('((', '( (', 1)
    for match in bracketsRE.finditer(sent_text[2:-1]):
        open_, label, text, close = match.groups()
        if open_:
            assert not close
            assert label.strip()
            open_brackets.append((label, word_i))
        else:
            assert close
            label, start = open_brackets.pop()
            assert label.strip()
            if text:
                words.append(text)
                word_i += 1
            brackets.append((label, start, word_i))
    return words, brackets


def split_sentences(text):
    sentences = []
    current = []
    for line in text.strip().split('\n'):
        line = line.rstrip()
        if not line:
            continue
        # Detect the start of sentences by line starting with (
        # This is messy, but it keeps bracket parsing at the sentence level
        if line.startswith('(') and current:
            sentences.append('\n'.join(current))
            current = []
        current.append(line)
    if current:
        sentences.append('\n'.join(current))
    return sentences


def read_sentences(ptb_dir, n=0):
    wsj_dir = path.join(ptb_dir, 'parsed', 'mrg', 'wsj')
    sents = []
    for subdir in os.listdir(wsj_dir):
        if subdir.endswith('.LOG'):
            continue
        for filename in os.listdir(path.join(wsj_dir, subdir)):
            if not filename.endswith('.mrg'):
                continue
            strings = split_sentences(open(path.join(wsj_dir, subdir, filename)).read())
            for words, brackets in (get_brackets(s) for s in strings):
                try:
                    sents.append(tree.from_brackets(words, brackets))
                except:
                    print words
                    print brackets
                    raise
            if n != 0 and len(sents) >= n:
                return sents
    return sents

def read_oneperline(fname, n = None):
   for i,line in enumerate(file(fname)):
      if n and i > n: break
      words, brackets = get_brackets(line)
      s = tree.from_brackets(words, brackets)
      #for l in s.leaves():
      #   l.end = l.start
      yield s
