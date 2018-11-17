import re
import math
import random
import json
import pickle
import numpy as np
import tensorflow as tf


class BatchManager:
    """
    划分batch数据，进行填充以及排序
    """
    def __init__(self, data, batch_size):
        batch_num = int(math.ceil(len(data) / batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(batch_num):
            batch_data.append(self.__pad_data(sorted_data[i*batch_size:(i+1)*batch_size]))
        self.batch_data = batch_data
        self.data_len = len(self.batch_data)

    def __pad_data(self, data):
        sentences = []
        targets = []
        max_length = max([len(temp[0]) for temp in data])
        for line in data:
            sentence, target = line
            padding = [0] * (max_length - len(sentence))
            sentences.append(sentence + padding)
            targets.append(target + padding)
        return [sentences, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for index in range(self.data_len):
            yield self.batch_data[index]


class SegBatcher:

    def __init__(self, record_file_name, batch_size, epochs_num=None):
        self._batch_size = batch_size
        self._epoch = 0
        self._step = 1
        self.epochs_num = epochs_num
        self.next_batch_op = self.input_pipeline(record_file_name, self._batch_size, self.epochs_num)

    def __parser(self, filename_queue):
        reader = tf.TFRecordReader()
        key, record_string = reader.read(filename_queue)

        features = {
            'labels': tf.FixedLenSequenceFeature([], tf.int64),
            'word_list': tf.FixedLenSequenceFeature([], tf.int64),
            'sent_len': tf.FixedLenSequenceFeature([], tf.int64),
        }

        _, example = tf.parse_single_example(serialized=record_string, sequence_features=features)
        labels = example['labels']
        word_list = example['word_list']
        sent_len = example['sent_len']
        return labels, word_list, sent_len

    def input_pipeline(self, filenames, batch_size, epochs_num=None):
        filename_queue = tf.train.string_input_producer([filenames], num_epochs=epochs_num, shuffle=True)
        labels, word_list, sent_len = self.__parser(filename_queue)

        min_after_dequeue = 10000
        capacity = min_after_dequeue + 12 * batch_size
        next_batch = tf.train.batch([labels, word_list, sent_len], batch_size=batch_size, capacity=capacity,
                                    dynamic_pad=True, allow_smaller_final_batch=True)
        return next_batch


def load_size_file(size_file):
    with open(size_file, 'r') as f:
        print(size_file)
        num_obj = json.load(f)
        return num_obj


def load_map_file(map_file):
    vocabulary, tag2id, id2tag = pickle.load(open(map_file, 'rb'))
    return vocabulary, tag2id, id2tag

