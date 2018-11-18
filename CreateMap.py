import pickle
import json
import GlobalParameter


OOV = '<OOV>'
PAD = 'PAD'


def create_word2id(embedding_file):
    vocabulary = {}
    vocabulary[OOV] = 0
    vocabulary[PAD] = 1
    f = open(embedding_file, encoding='utf-8')
    m, n = f.readline().split(' ')
    for i, line in enumerate(f):
        word = line.split()[0]
        if not word:
            continue
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)
    return vocabulary


def create_tag_and_id():
    tag2id = {
        'S': 0,
        'B': 1,
        'M': 2,
        'E': 3
    }
    id2tag = {v: k for k, v in tag2id.items()}
    return tag2id, id2tag


def create_map():
    word2id = create_word2id(GlobalParameter.WORD2VEC_FILE)
    tag2id, id2tag = create_tag_and_id()
    f = open(GlobalParameter.MAP_FILE, 'wb')
    pickle.dump((word2id, tag2id, id2tag), f)
    f.close()
    vocab_size = len(word2id)
    num_class = len(tag2id)
    f = open(GlobalParameter.SIZE_FILE, 'w')
    json.dump({'words_num': vocab_size, 'tags_num': num_class}, f)
    f.close()
