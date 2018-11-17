import numpy as np
import tensorflow as tf
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from BiLSTM import BiLSTM
from DataUtils import SegBatcher


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
    model_config['eval_step'] = 20

    model_config['words_num'] = vocabulary_size
    model_config['tags_num'] = tag_size

    return model_config

def Model_train():
    # TODO: get word2id, tag2id and id2tag
    word2id, id2word, tag2id, id2tag = [1, 2, 3, 4]

    # TODO：get the value of the 3 variables
    train_num = 0
    dev_num = 0
    test_num = 0
    model_config = set_model_config(len(word2id), len(tag2id))
    print(model_config)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Graph().as_default():
        print('load pre word2vec ...')
        word_vec = {}
        word_embedding = tf.constant(word_vec, dtype=tf.float32)
        model = BiLSTM(model_config, word_embedding)
        train_file = 'data/train.txt'
        valid_file = 'data/valid.txt'
        test_file = 'data/test.txt'
        batch_size = model_config['batch_size']
        out_path = 'model'
        train_batcher = SegBatcher(train_file, batch_size)
        valid_batcher = SegBatcher(valid_file, batch_size)
        test_batcher = SegBatcher(test_file, batch_size)
        tf.global_variables_initializer()
        sv = tf.train.Supervisor(logdir=out_path, save_model_secs=10)

        with sv.managed_session() as sess:
            sess.as_default()
            threads = tf.train.start_queue_runners(sess=sess)
            loss = []

            def evaluation():
                return

            def test():
                return

            best_acc = 0.0
            valid_batches = []
            end = False
            print('load all valid set batches to memory')

            while not end:
                try:
                    tags, words, sent_lens = sess.run(valid_batcher.next_batch_op)
                    valid_batches.append((words, tags, sent_lens))
                except Exception:
                    end = True

            print('start to train the model ...')
            early_stop = False
            for step in range(model_config['max_epoch']):
                if sv.should_stop():
                    test()
                    break
                examples = 0

                while examples < train_num:
                    if early_stop:
                        break
                    try:
                        tags, words, sent_lens = sess.run(train_batcher.next_batch_op)
                    except Exception:
                        break

                    feed_dict = {
                        model.inputs: words,
                        model.label: tags,
                        model.dropout: model_config['dropout'],
                        model.length: sent_lens.reshape(-1, 1)
                    }
                    global_step, batch_loss, _ = sess.run([model.g_step, model.loss, model.train_op], feed_dict=feed_dict)

                    print('%d iteration, %d valid acc: %f' % (step, ))





def Model_export():
    return


def Model_load():
    return




