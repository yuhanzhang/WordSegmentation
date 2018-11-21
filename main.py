import os
import pickle
import json
from CorpusPreprocess import gci
from CorpusPreprocess import create_data
from CreateVector import create_word2vec_corpus
from CreateVector import pretrain_word2vec
from CreateMap import create_map
from CreateTfrecord import create_tfrecords
from Model import Model_train, Model_export
from Predictor import cut
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
    # Model_train()
    '''
    export the model
    '''
    # out_names = ['project/output/predict', 'project/logits', "loss/transitions"]
    # Model_export(GlobalParameter.CHECKPOINT_DIR, os.path.join(GlobalParameter.MODEL_DIR, "modle.pb"), out_names)
    '''
    predict test
    '''


