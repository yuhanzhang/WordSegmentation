import tensorflow as tf
import numpy as np
import pickle
import os
import GlobalParameter
from tensorflow.contrib.crf import viterbi_decode


class Predictor:
    def __init__(self, model_file, map_file):
        word2id, tag2id, id2tag = pickle.load(open(map_file, 'rb'))
        self.word2id = word2id
        self.tag2id = tag2id
        self.id2tag = id2tag
        with tf.gfile.GFile(model_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="prefix")
        self.graph = graph
        self.input = self.graph.get_tensor_by_name("prefix/inputs:0")
        self.length = self.graph.get_tensor_by_name("prefix/length:0")
        self.dropout = self.graph.get_tensor_by_name("prefix/dropout:0")
        self.logits = self.graph.get_tensor_by_name("prefix/project/logits:0")
        self.trans = self.graph.get_tensor_by_name("prefix/loss/transitions:0")

        self.sess = tf.Session(graph=self.graph)
        self.sess.as_default()
        self.num_class = len(self.id2tag)

    def __decode(self, logits, trans, sequence_lengths, tag_num):
        viterbi_sequences = []
        small = -1000.0
        start = np.asarray([[small] * tag_num + [0]])
        for logit, length in zip(logits, sequence_lengths):
            score = logit[:length]
            pad = small * np.ones([length, 1])
            score = np.concatenate([score, pad], axis=1)
            score = np.concatenate([start, score], axis=0)
            viterbi_seq, viterbi_score = viterbi_decode(score, trans)
            viterbi_sequences.append(viterbi_seq[1:])
        return viterbi_sequences

    def predict(self, sentences):
        inputs = []
        lengths = [len(text) for text in sentences]
        max_len = max(lengths)

        for sent in sentences:
            sent_ids = [self.word2id.get(w) if w in self.word2id else self.word2id.get("<OOV>") for w in sent]
            padding = [0] * (max_len - len(sent_ids))
            sent_ids += padding
            inputs.append(sent_ids)
        inputs = np.array(inputs, dtype=np.int32)

        feed_dict = {
            self.input: inputs,
            self.length: lengths,
            self.dropout: 1.0
        }

        logits, trans = self.sess.run([self.logits, self.trans], feed_dict=feed_dict)
        path = self.__decode(logits, trans, lengths, self.num_class)
        tags = [[self.id2tag.get(l) for l in p] for p in path]
        return tags


def cut(sentences):
    predictor = Predictor(os.path.join(GlobalParameter.MODEL_DIR, "modle.pb"), GlobalParameter.MAP_FILE)
    all_labels = predictor.predict(sentences)
    sent_words = []
    for ti, text in enumerate(sentences):
        words = []
        N = len(text)
        seg_labels = all_labels[ti]
        tmp_word = ""
        for i in range(N):
            label = seg_labels[i]
            w = text[i]
            if label == "B":
                tmp_word += w
            elif label == "M":
                tmp_word += w
            elif label == "E":
                tmp_word += w
                words.append(tmp_word)
                tmp_word = ""
            else:
                tmp_word = ""
                words.append(w)
        if tmp_word:
            words.append(tmp_word)
        sent_words.append(words)
    return sent_words