import numpy as np
import tensorflow as tf
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from BiLSTM import BiLSTM
from DataUtils import SegBatcher, load_map_file, load_size_file
from CreateVector import Word2vec
import GlobalParameter


def set_model_config(vocabulary_size, tag_size):
    model_config = OrderedDict()
    model_config['lstm_hidden_size'] = GlobalParameter.lstm_hidden_size
    model_config['word_size'] = GlobalParameter.word_size
    model_config['learning_rate'] = GlobalParameter.learning_rate
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
    id2word = {v: k for k, v in word2id.items()}

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
        word_vec = Word2vec()
        word_embedding = tf.constant(word_vec.load_word2vec(GlobalParameter.WORD2VEC_FILE, id2word), dtype=tf.float32)
        model = BiLSTM(model_config, word_embedding)
        train_file = GlobalParameter.TRAIN_FILE.split('.')[0] + '.tfrecord'
        valid_file = GlobalParameter.VALID_FILE.split('.')[0] + '.tfrecord'
        test_file = GlobalParameter.TEST_FILE.split('.')[0] + '.tfrecord'
        batch_size = model_config['batch_size']
        train_batcher = SegBatcher(train_file, batch_size, epochs_num=model_config['max_epoch'])
        valid_batcher = SegBatcher(valid_file, batch_size, epochs_num=1)
        test_batcher = SegBatcher(test_file, batch_size, epochs_num=1)
        tf.global_variables_initializer()
        tf.local_variables_initializer()
        sv = tf.train.Supervisor(logdir=GlobalParameter.CHECKPOINT_DIR, save_model_secs=10)

        with sv.managed_session() as sess:
            sess.as_default()
            threads = tf.train.start_queue_runners(sess=sess)
            loss = []


            def evaluation(dev_batches, report=False):
                """
                Evaluates model on a valid set
                """
                preds = []
                true_tags = []
                tmp_x = []
                for x_batch, y_batch, sent_len in dev_batches:
                    feed_dict = {
                        model.inputs: x_batch,
                        model.label: y_batch,
                        model.length: sent_len.reshape(-1, ),
                        model.dropout: 1.0
                    }

                    step, loss, logits, lengths, trans = sess.run(
                        [model.global_step, model.loss, model.logits, model.length, model.trans], feed_dict)

                    index = 0
                    small = -1000.0
                    start = np.asarray([[small] * model_config['tags_num'] + [0]])

                    for score, length in zip(logits, lengths):
                        score = score[:length]
                        pad = small * np.ones([length, 1])
                        logit = np.concatenate([score, pad], axis=1)
                        logit = np.concatenate([start, logit], axis=0)
                        path, _ = tf.contrib.crf.viterbi_decode(logit, trans)
                        preds.append(path[1:])
                        tmp_x.append(x_batch[index][:length])
                        index += 1

                    for y, length in zip(y_batch, lengths):
                        y = y.tolist()
                        true_tags.append(y[: length])

                # if FLAGS.debug and len(tmp_x) > 5:
                #     print(tag_to_id)
                #
                #     for j in range(5):
                #         sent = [id_to_word.get(i, "<OOV>") for i in tmp_x[j]]
                #         print("".join(sent))
                #         print("pred:", preds[j])
                #         print("true:", true_tags[j])

                preds = np.concatenate(preds, axis=0)
                true_tags = np.concatenate(true_tags, axis=0)

                if report:
                    print(classification_report(true_tags, preds))

                acc = accuracy_score(true_tags, preds)
                return acc

            def test():
                print("start run test ......")
                test_batches = []
                done = False
                print("load all test batches to memory")

                while not done:
                    try:
                        tags, chars, sent_lens = sess.run(test_batcher.next_batch_op)
                        test_batches.append((chars, tags, sent_lens))
                    except:
                        done = True
                test_acc = evaluation(test_batches, True)
                print("test accc %f" % (test_acc))

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
                    print('sv should_stop')
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
                        model.length: sent_lens.reshape(-1, )
                    }
                    global_step, batch_loss, _ = sess.run([model.global_step, model.loss, model.train_op], feed_dict=feed_dict)

                    print('%d iteration, %d valid loss: %f' % (step, global_step, batch_loss))
                    if global_step % GlobalParameter.eval_step == 0:
                        acc = evaluation(valid_batches)
                        print("%d iteration , %d dev acc: %f " % (step, global_step, acc))
                        if best_acc - acc > 0.01:
                            print("stop training ealy ... best dev acc " % (best_acc))
                            early_stop = True
                            break

                        elif best_acc < acc:
                            best_acc = acc
                            sv.saver.save(sess, GlobalParameter.CHECKPOINT_DIR+'/model', global_step=global_step)
                            print("%d iteration , %d global step best dev acc: %f " % (step, global_step, best_acc))
                    loss.append(batch_loss)
                    examples += batch_size
            sv.saver.save(sess, GlobalParameter.CHECKPOINT_DIR+'/model', global_step=global_step)
            # test()
        sv.coord.request_stop()
        sv.coord.join(threads)
        sess.close()


def Model_export():
    return


def Model_load():
    return

