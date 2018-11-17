import os
from CorpusPreprocess import gci
from CorpusPreprocess import create_data
from CreateVector import create_word2vec_corpus
from CreateVector import pretrain_word2vec

RAW_CORPUS = 'E:\\chinese word segmentation\\corpus\\PeopleDaily2014'
RAW_DATA_DIR = 'raw_data'
WORD2VEC_DIR = 'word2vec'
WORD2VEC_FILE = WORD2VEC_DIR + 'word_embedding_300dim.txt'
TRAIN_FILE = 'data/train.txt'
VALID_FILE = 'data/valid.txt'
TEST_FILE = 'data/test.txt'


if __name__ == '__main__':
    # process corpus
    # gci(RAW_CORPUS)
    # create_data(RAW_DATA_DIR)
    # train the word vectors
    create_word2vec_corpus(TRAIN_FILE, "", "", 'word2vec/corpus.txt')
    pretrain_word2vec('word2vec/corpus.txt')

