#!/usr/bin/env python

import sys
import plac
import bracket_eager.parser


def main(model_dir, loc):
    parser = bracket_eager.parser.Parser(model_dir)
    lines = open(loc).read().strip().split('\n')
    for line in lines:
        if not line.strip():
            continue
        words = line.split()
        print parser.debug_parse(line.split())


if __name__ == '__main__':
    plac.call(main)
