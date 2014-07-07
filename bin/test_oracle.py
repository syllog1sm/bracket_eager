import random

import sys
sys.path.append(".")
import plac
from bracket_eager import parser
from bracket_eager import read_ptb
from bracket_eager.transition_system2 import *
import os.path

def pp(sent, state):
   print "GOLD:",sent.to_ptb()
   print "GOT :",state.stack[-1].to_ptb()

DEBUG=False
size = 0
sents = read_ptb.read_oneperline("data/wsj.train", n=size)

for sent in sents:
   if DEBUG: print "NEW SENT"
   o = Oracle(sent)
   words = sent.leaves()
   tags = [w.label for w in words]
   forms = [w.lex for w in words]
   state = ParserState.from_words_and_tags(forms, tags)
   while not state.is_end_state():
      a = o.next_actions(state)[0]
      if DEBUG:
         print "target:", o.next_bracket
         print "stack:", state.stack[-1:]
         print a
      assert a.is_valid(state), ((a, state.stack), sent.to_ptb())
      state = a.apply(state)
   assert o._cumloss == 0
   assert state.stack[-1].to_ptb().strip() == sent.to_ptb().strip(), pp(sent, state)


#print sents

