import tensorflow as tf
import os
import json
from DataUtils import load_map_file
import GlobalParameter


def seg2tfrecords(text_file, map_file, columns=[0, 1]):
    filename = os.path.basename(text_file).split('.')[0]
    out_filename = os.path.join('data/', filename+'.tfrecord')
    word2id, tag2id, id2tag = load_map_file(map_file)
    writer = tf.python_io.TFRecordWriter(out_filename)

    num_sample = 0
    all_oov = 0
    total_word = 0
    with open(text_file, 'r', encoding='utf-8') as f:
        sentence = []
        for _, line in enumerate(f):
            line = line.strip('\n')
            if not line:
                # 一个句子输入完毕
                word_count, oov_count = create_example(writer, sentence, word2id, tag2id)
                total_word += word_count
                all_oov += oov_count
                num_sample += 1
                sentence.clear()
                continue
            temp = line.split('\t')
            information = [temp[i] for i in columns]
            sentence.append(information)
    # print('oov rate: %f', 1.0*oov_count/total_word)
    return num_sample


def create_example(writer, sentence, word2id, tag2id):

    word_list = []
    tag_list = []
    oov_count = 0
    word_count = 0
    for word in sentence:
        char = word[0]
        label = word[1]
        word_count += 1
        if char in word2id:
            word_list.append(word2id[char])
        else:
            word_list.append(word2id['<OOV>'])
            oov_count += 1
        tag_list.append(tag2id[label])

    example = tf.train.SequenceExample()
    sentence_len = GlobalParameter.MAX_LENGTH if len(sentence) > GlobalParameter.MAX_LENGTH else len(tag_list)
    fl_tags = example.feature_lists.feature_list['tags']
    for l in tag_list[:sentence_len]:
        fl_tags.feature.add().int64_list.value.append(l)
    fl_words = example.feature_lists.feature_list['words']
    for l in word_list[:sentence_len]:
        fl_words.feature.add().int64_list.value.append(l)
    fl_length = example.feature_lists.feature_list['length']
    for l in [sentence_len]:
        fl_length.feature.add().int64_list.value.append(l)
    writer.write(example.SerializeToString())
    return word_count, oov_count


def create_tfrecords():
    train_num = seg2tfrecords(GlobalParameter.TRAIN_FILE, GlobalParameter.MAP_FILE)
    valid_num = seg2tfrecords(GlobalParameter.VALID_FILE, GlobalParameter.MAP_FILE)
    # test_num = seg2tfrecords(GlobalParameter.TEST_FILE, GlobalParameter.MAP_FILE)
    with open(GlobalParameter.SIZE_FILE, 'r') as f:
        size_obj = json.load(f)

    with open(os.path.join(GlobalParameter.SIZE_FILE), 'w') as f:
        size_obj['train_num'] = train_num
        size_obj['valid_num'] = valid_num
        # size_obj['test_num'] = test_num
        json.dump(size_obj, f)
