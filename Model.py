import numpy as np
import tensorflow as tf
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from BiLSTM import BiLSTM
from DataUtils import SegBatcher, load_map_file, load_size_file
import word2vec
import GlobalParameter


def set_model_config(vocabulary_size, tag_size):
    model_config = OrderedDict()
    model_config['lstm_hidden_size'] = GlobalParameter.lstm_hidden_size
    model_config['word_size'] = GlobalParameter.word_size
    model_config['learninng_rate'] = GlobalParameter.learning_rate
    model_config['input_dropout_keep'] = GlobalParameter.input_dropout_keep
    model_config['dropout'] = GlobalParameter.dropout
    model_config['max_epoch'] = GlobalParameter.max_epoch
    model_config['batch_size'] = GlobalParameter.batch_size
    model_config['optimizer'] = 'adam'
    model_config['clip'] = GlobalParameter.clip
    model_config['eval_step'] = GlobalParameter.eval_step

    model_config['words_num'] = vocabulary_size
    model_config['tags_num'] = tag_size

    return model_config


def Model_train():
    word2id, tag2id, id2tag = load_map_file(GlobalParameter.MAP_FILE)
    id2word = {v: k for k, v in word2id}

    size_dict = load_size_file(GlobalParameter.SIZE_FILE)
    train_num = size_dict['train_num']
    valid_num = size_dict['valid_num']
    # test_num = size_dict['test_num']
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

                    print('%d iteration, %d valid acc: %f' % (step, global_step, batch_loss))


def Model_export():
    return


def Model_load():
    return

