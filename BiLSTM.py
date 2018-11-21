import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.layers.python.layers import initializers


class BiLSTM:
    def __init__(self, config, embeddings):
        # basic configuration
        self.config = config
        self.lstm_hidden_size = config['lstm_hidden_size']
        self.tags_num = config['tags_num']
        self.word_size = config['word_size']
        self.words_num = config['words_num']
        self.learning_rate = config['learning_rate']

        self.word_embedding = tf.get_variable(name='word_embedding', initializer=embeddings)
        self.initializer = initializers.xavier_initializer()
        self.global_step = tf.Variable(0, trainable=False)

        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='inputs')
        self.label = tf.placeholder(dtype=tf.int32, shape=[None, None], name='label')
        self.dropout = tf.placeholder(dtype=tf.float32, name='dropout')
        self.length = tf.placeholder(dtype=tf.int32, shape=[None, ], name='length')

        self.batch_size = tf.shape(self.inputs)[0]
        self.steps_num = tf.shape(self.inputs)[-1]

        self.input_dropout_keep_prob = tf.placeholder_with_default(config['input_dropout_keep'], [],
                                                                   name='input_dropout_keep_prob')
        # forward propagation
        embedding_output = self.__embedding_layer(self.inputs)
        lstm_input = tf.nn.dropout(embedding_output, self.input_dropout_keep_prob)
        lstm_output = self.__bilstm_layer(lstm_input)
        self.__project_layer(lstm_output)
        self.loss = self.__loss_layer(self.logits, self.length)

        # backward propagation
        self.__optimizer()

    def __embedding_layer(self, layer_inputs):
        with tf.variable_scope('embedding'):
            layer_output = tf.nn.embedding_lookup(self.word_embedding, layer_inputs)
        return layer_output

    def __bilstm_layer(self, layer_inputs, name=None):
        with tf.variable_scope('bilstm' if not name else name):
            lstm_forward_cell = rnn.BasicLSTMCell(self.lstm_hidden_size, state_is_tuple=True)
            lstm_backward_cell = rnn.BasicLSTMCell(self.lstm_hidden_size, state_is_tuple=True)
            layer_outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_forward_cell, lstm_backward_cell,
                                                                           layer_inputs, dtype=tf.float32,
                                                                           sequence_length=self.length)
            return tf.concat(layer_outputs, axis=2)

    def __project_layer(self, layer_input, name=None):
        with tf.variable_scope('project' if not name else name):
            with tf.variable_scope('hidden'):
                w_hidden = tf.get_variable(name='w_hidden', shape=[self.lstm_hidden_size*2, self.lstm_hidden_size],
                                         dtype=tf.float32, initializer=self.initializer,
                                         regularizer=l2_regularizer(0.001))
                b_hidden = tf.get_variable(name='b_hidden', shape=[self.lstm_hidden_size], dtype=tf.float32,
                                         initializer=tf.zeros_initializer())
                output = tf.reshape(layer_input, shape=[-1, self.lstm_hidden_size*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, weights=w_hidden, biases=b_hidden))
                drop_hidden = tf.nn.dropout(hidden, self.dropout)

            with tf.variable_scope('output'):
                w_out = tf.get_variable(name='w_out', shape=[self.lstm_hidden_size, self.tags_num], dtype=tf.float32,
                                        initializer=self.initializer, regularizer=l2_regularizer(0.001))
                b_out = tf.get_variable(name='b_out', shape=[self.tags_num], initializer=tf.zeros_initializer())
                predict = tf.nn.xw_plus_b(drop_hidden, weights=w_out, biases=b_out, name='predict')
            self.logits = tf.reshape(predict, [-1, self.steps_num, self.tags_num], name='logits')

    def __loss_layer(self, layer_input, length, name=None):
        with tf.variable_scope('loss' if not name else name):
            small = -1000.0
            start_logits = tf.concat([small*tf.ones(shape=[self.batch_size, 1, self.tags_num]),
                                      tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            pad_logits = tf.cast(small*tf.ones([self.batch_size, self.steps_num, 1]), tf.float32)
            logits = tf.concat([layer_input, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            label = tf.concat([tf.cast(self.tags_num * tf.ones([self.batch_size, 1]), tf.int32), self.label], axis=-1)
            self.trans = tf.get_variable('transitions', shape=[self.tags_num+1, self.tags_num+1],
                                         initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(inputs=logits, tag_indices=label,
                                                            transition_params=self.trans, sequence_lengths=length+1)
            return tf.reduce_mean(-log_likelihood)

    def __optimizer(self, name=None):
        with tf.variable_scope('optimizer' if not name else name):
            optimizer = self.config['optimizer']
            if optimizer == 'sgd':
                self.opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif optimizer == 'adam':
                self.opt = tf.train.AdamOptimizer(self.learning_rate)
            elif optimizer == 'adgrad':
                self.opt = tf.train.AdagradOptimizer(self.learning_rate)
            else:
                raise KeyError
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config['clip'], self.config['clip']), v] for g, v in
                                 grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)



