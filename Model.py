import numpy as np
import tensorflow as tf
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from BiLSTM import BiLSTM


def set_model_config(vocabulary_size, tag_size):
    model_config = OrderedDict()
    model_config['lstm_hidden_size'] = 100
    model_config['word_size'] = 100
    model_config['learninng_rate'] = 0.001
    model_config['input_dropout_keep'] = 1.0
    model_config['dropout'] = 0.5
    model_config['max_epoch'] = 20
    model_config['batch_size'] = 32
    model_config['optimizer'] = 'adam'
    model_config['clip'] = 5

    model_config['words_num'] = vocabulary_size
    model_config['tags_num'] = tag_size

    return model_config

def Model_train():
    # TODO: get word2id, tag2id and id2tag
    word2id, id2word, tag2id, id2tag = [1, 2, 3, 4]
    model_config = set_model_config(len(word2id), len(tag2id))
    print(model_config)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # return


def Model_export():
    return


def Model_load():
    return




