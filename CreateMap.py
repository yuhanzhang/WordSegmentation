import pickle
import json

OOV = '<OOV>'
PAD = 'PAD'


def create_word2id(embedding_file):
    vocabulary = {}
    vocabulary[OOV] = 0
    vocabulary[PAD] = 1
    f = open(embedding_file)
    m, n = f.readline().split(' ')
    m = int(m)
    print('preembedding size : %d.' % m)
    for i, line in enumerate(f):
        word = line.split()[0]
        if not word:
            continue
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)
    print('vocabulary size : %d.' % len(vocabulary))
    return vocabulary


def create_tag_and_id(train_file, tag_index=-1):
    tag2id = {
        'S':0,
        'B':1,
        'M':2,
        'E':3
    }
    id2tag = {v: k for k, v in tag2id.items()}
    return tag2id, id2tag


