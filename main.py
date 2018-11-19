import os
import pickle
import json
from CorpusPreprocess import gci
from CorpusPreprocess import create_data
from CreateVector import create_word2vec_corpus
from CreateVector import pretrain_word2vec
from CreateMap import create_map
from CreateTfrecord import create_tfrecords
from Model import Model_train
import GlobalParameter
from DataUtils import SegBatcher
import tensorflow as tf

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
    # valid_batcher = SegBatcher(GlobalParameter.VALID_FILE, GlobalParameter.batch_size, epochs_num=1)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(tf.local_variables_initializer())
    #     tags, words, sent_lens = sess.run(valid_batcher.next_batch_op)
    #     print(words)
    Model_train()
    '''
    export the model
    '''


