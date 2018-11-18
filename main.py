import os
import pickle
import json
from CorpusPreprocess import gci
from CorpusPreprocess import create_data
from CreateVector import create_word2vec_corpus
from CreateVector import pretrain_word2vec
from CreateMap import create_map
from CreateTfrecord import create_tfrecords
import GlobalParameter


if __name__ == '__main__':
    '''
    process corpus
    '''
    # gci(RAW_CORPUS)
    # create_data(RAW_DATA_DIR)
    # train the word vectors
    # create_word2vec_corpus(TRAIN_FILE, "", "", 'word2vec/corpus.txt')
    # pretrain_word2vec('word2vec/corpus.txt')
    '''
    create map file
    '''
    # create_map()
    '''
    create TFRecords
    '''
    # create_tfrecords()
    '''
    train
    '''

    '''
    export the model
    '''


