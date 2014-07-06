
all: data

data data/wsj.train data/wsj.dev data/wsj.test:
	mkdir data
	cat ptb/0[2-9]/*.MRG ptb/1[0-9]/*.MRG ptb/2[01]/*.MRG |python bin/get_parse.py > data/wsj.train
	cat ptb/22/*.MRG |python bin/get_parse.py > data/wsj.dev
	cat ptb/23/*.MRG |python bin/get_parse.py > data/wsj.test

