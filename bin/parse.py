#!/usr/bin/env python

import sys
import plac
sys.path.append(".")
import bracket_eager.parser


def main(model_dir):
    parser = bracket_eager.parser.Parser(model_dir)
    for line in sys.stdin:
        if not line.strip():
            continue
        words = line.split()
        print parser.parse(line.split())


if __name__ == '__main__':
    plac.call(main)
