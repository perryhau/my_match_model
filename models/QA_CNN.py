import sys

import tensorflow as tf
import logging
import datetime
import time
import numpy as np
import os
import pandas as pd
import jieba



class CNN(object):
  def __init__(self, embeddings, args):

    self.logger = logging.getLogger('sentence_pairs')
    self.embedding_size = args.embedding_size
    self.num_filters = args.num_filters
    self.l2r = args.l2_reg_lambda
    self.trainable = args.trainable
    self.embeddings = embeddings
    self.filter_sizes = args.filter_sizes
    self.optim_type = args.optim
    self.learning_rate = args.learning_rate
    # length limit
    self.max_input_left = args.max_input_left
    self.max_input_right = args.max_input_right
    self.pooling = args.pooling
    self.args = args
    self.char_embed_size = args.char_embed_size
    self.num_classes = args.num_classes

    self.input_layer = args.input
    # sess config
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=sess_config)

    self._build_graph()

    self.saver = tf.train.Saver()
    self.sess.run(tf.global_variables_initializer())

  def _build_graph(self):
    start_t = time.time()
    self._create_placehoder()
    self._add_embedding()
    self._encode()
    self._pooling_graph()
    self.feed_neural_network()
    self._create_loss()
    self._summary()
    self._create_op()

    self.logger.info('Time to build graph: {}s'.format(time.time() - start_t))
    param_num = sum([np.prod(self.sess.run(tf.shape(v)))
                     for v in tf.trainable_variables()])
    self.logger.info('There are {} parameters in the model'.format(param_num))

  def _create_placehoder(self):

    self.q = tf.placeholder(tf.int32, [None, None], name='input_q')
    self.a = tf.placeholder(tf.int32, [None, None], name='input_a')
    self.y = tf.placeholder(tf.int32, [None], name='input_y')

    self.dropout_keep_prob = tf.placeholder(
        tf.float32, name='dropout_keep_prob')

  def _add_embedding(self):
    with tf.device('/cpu:0'), tf.variable_scope("word_embedding"):
      # one_hot embedding or pretrained embedding
      if self.embeddings:
        self.word_embeddings = tf.get_variable(
            'word_embeddings',
            shape=(len(self.embeddings), self.embedding_size),
            initializer=tf.constant_initializer(self.embeddings),
            trainable=self.trainable)
      else:
        self.word_embeddings = tf.concat([tf.zeros((1,self.args.alphabet_size)),
        tf.one_hot(np.arange(self.args.alphabet_size),self.args.alphabet_size,1.0,0.0)],0,name = 'onehot_embedding')
        self.embedding_size = self.args.alphabet_size
      self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)
      self.a_emb = tf.nn.embedding_lookup(self.word_embeddings, self.a)

  def _encode(self):
    self.q_emb = tf.expand_dims(self.q_emb, -1, name='q_emb')
    self.a_emb = tf.expand_dims(self.a_emb, -1, name='a_emb')

    self.kernels = []
    with tf.name_scope('convolution_encode'):
      cnn_outputs = []
      for i, filter_size in enumerate(self.filter_sizes):
        with tf.name_scope('conv-pool-%s' % filter_size):
          filter_shape = [filter_size,
                          self.embedding_size, 1, self.num_filters]
          W = tf.get_variable('Wc' + str(i), filter_shape, tf.float32,
                              tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
          b = tf.get_variable(
              'bc' + str(i), [self.num_filters], tf.float32, tf.constant_initializer(0.01))

          self.kernels.append((W, b))

          tf.summary.histogram('weights', W)
          tf.summary.histogram('bias', b)

      self.num_filters_total = self.num_filters * len(self.filter_sizes)
      self.q_conv = self.wide_convolution(self.q_emb)

      self.a_conv = self.wide_convolution(self.a_emb)

  def wide_convolution(self, embedding):
    cnn_outputs = []
    for i, filter_size in enumerate(self.filter_sizes):
      conv = tf.nn.conv2d(
          embedding,
          self.kernels[i][0],
          strides=[1, 1, self.embedding_size, 1],
          padding='SAME',
          name='conv-{}'.format(i)
      )

      h = tf.nn.relu(tf.nn.bias_add(
          conv, self.kernels[i][1]), name='relu-{}'.format(i))
      cnn_outputs.append(h)
    cnn_reshape = tf.concat(cnn_outputs, 3, name='concat')
    return cnn_reshape

  def max_pooling(self, conv, input_length):

    pooled = tf.nn.max_pool(
        conv,
        ksize=[1, int(input_length), 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name='pooling')

    return pooled

  def mean_pooling(self, conv):
    s = tf.squeeze(conv, 2)
    s_represent = tf.reduce_mean(s, 1)
    return s_represent

  def _pooling_graph(self):
    with tf.name_scope('pooling'):
      if self.pooling == 'max':

        self.q_pooling = tf.reshape(self.max_pooling(
            self.q_conv, self.max_input_left), [-1, self.num_filters_total])
        self.a_pooling = tf.reshape(self.max_pooling(
            self.a_conv, self.max_input_right), [-1, self.num_filters_total])

      elif self.pooling == 'mean':

        self.q_pooling = tf.reshape(self.mean_pooling(
            self.q_conv), [-1, self.num_filters_total])
        self.a_pooling = tf.reshape(self.mean_pooling(
            self.a_conv), [-1, self.num_filters_total])

      else:
        raise NotImplementedError(
            'unsupported optimizer:{}'.format(self.pooling))

  def feed_neural_network(self):
    with tf.name_scope('neural_network'):
      self.feature = tf.concat(
          [self.q_pooling, self.a_pooling], 1, name='feature')
      W = tf.get_variable(
          'w_hidden',
          shape=[self.num_filters_total * 2, self.num_classes],
          initializer=tf.contrib.layers.xavier_initializer()
      )

      b = tf.get_variable(
          'b_hidden',
          shape=[self.num_classes]
      )

      self.logits = tf.nn.xw_plus_b(self.feature, W, b, name='logits')
      self.scores = tf.nn.softmax(self.logits)
      self.predictions = tf.argmax(self.logits, 1, name='predictions')

  def _create_loss(self):

    with tf.name_scope('loss'):
      self.one_hot_labels = tf.one_hot(self.y, self.num_classes)
      losses = loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
          logits=self.logits, labels=self.one_hot_labels))
      l2_loss = tf.add_n([tf.nn.l2_loss(v)
                          for v in tf.trainable_variables()]) * self.l2r
      self.loss = losses + l2_loss
      correct_prediction = tf.equal(
          tf.cast(self.predictions, tf.int32), tf.cast(self.y, tf.int32))
      self.accuracy = tf.reduce_mean(
          tf.cast(correct_prediction, tf.float32), name='Accuracy')

  def _create_op(self):

    if self.optim_type == 'adagrad':
      self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
    elif self.optim_type == 'adam':
      self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    elif self.optim_type == 'rorop':
      self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
    elif self.optim_type == 'sgd':
      self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    else:
      raise NotImplementedError(
          'unsupported optimizer:{}'.format(self.optim_type))
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    grads_and_vars = self.optimizer.compute_gradients(self.loss)

    self.train_op = self.optimizer.apply_gradients(
        grads_and_vars, global_step=self.global_step)
    self.train_summary_op = tf.summary.merge_all()
    self.test_summary_op = tf.summary.merge(
        [self.loss_summary, self.acc_summary])

  def train(self, data, epochs, batch_size, save_dir, dropout_keep_prob=1.0, evaluate=True):
    acc_max = 0
    for epoch in range(1, epochs + 1):
      self.logger.info('Train the model for epoch {}'.format(epoch))
      train_batches = data.batch_iter(data.train_set, batch_size, shuffle=True)

      for batch in train_batches:
        question, answer, input_y = zip(*batch)

        feed_dict = {
            self.q: question,
            self.a: answer,
            self.y: input_y,
            self.dropout_keep_prob: dropout_keep_prob

        }
        _, summary, step, loss, accuracy, logits = self.sess.run([self.train_op,
                                                                  self.global_step, self.train_summary_op, self.loss, self.accuracy, self.logits], feed_dict)
        # print(logits)

        self.logger.info("loss {}, acc {}".format(loss, accuracy))

      if evaluate:
        self.logger.info('Evaluating the model after epoch {}'.format(epoch))
        eval_batches = data.all_iter(
            data.train_set, batch_size,)
        scores, predictions = self.evaluate(eval_batches)
        predict_overlap = self.predict_overlap(data=data.train_set)
        data.train_set['scores'] = scores
        data.train_set['overlap_score'] = predict_overlap
        data.train_set[['id', 'scores', 'overlap_score', 'flag']].to_csv(
            'atec_train.txt', index=None, sep='\t')
        '''if data.dev_set is not None:
                                  eval_batches = data.batch_iter(
                                      data.dev_set, batch_size, shuffle=False)

                                  acc, _ = self.evaluate(eval_batches)
                                  self.logger.info('acc test:{}'.format(acc))
                                  if acc > acc_max:
                                    acc_max = acc
                                    self.save(save_dir, self.args.pooling)'''

  def evaluate(self, eval_batches, result_dir=None, result_prefix=None):
    scores = []
    predictions = []
    qids = []

    for batch in eval_batches:
      question, answer, input_y= zip(*batch)

      feed_dict = {
          self.q: question,
          self.a: answer,
          self.y: input_y,
          self.dropout_keep_prob: 1.0
      }
      summary, step, acc, pred, score = self.sess.run(
          [self.test_summary_op, self.global_step, self.accuracy, self.predictions, self.scores], feed_dict)
      self.test_summary_writer.add_summary(summary, step)
      # scores.append(acc)
      scores.append(score)
      predictions.extend(pred)
      # qids.extend(qid)
      # print(pred)
    # return np.mean(scores), predictions
    print(len(scores))
    return scores, predictions

  def predict_overlap(self, data):
    overlap_scores = []
    for i in range(len(data)):
      question = [w for w in jieba.cut(data['s1'][i]) if w.strip()]
      answer = [w for w in jieba.cut(data['s2'][i]) if w.strip()]
      #question = cut(data['question'][i])
      #answer = cut(data['answer'][i])
      union = question + answer
      same_num = 0
      for w in union:
        if w in question and w in answer:
          same_num += 1
      overlap_score = same_num * 2 / len(union)
      # exit()
      overlap_scores.append(overlap_score)
    return overlap_scores

  def _summary(self):
    self.loss_summary = tf.summary.scalar("loss", self.loss)
    self.acc_summary = tf.summary.scalar("accuracy", self.accuracy)
    train_summary_dir = os.path.join(self.args.summary_dir, 'train')
    test_summary_dir = os.path.join(self.args.summary_dir, 'dev')
    self.train_summary_writer = tf.summary.FileWriter(train_summary_dir)
    self.test_summary_writer = tf.summary.FileWriter(test_summary_dir)

  def save(self, model_dir, model_prefix):
    self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
    self.logger.info(
        'model saved in {},with prefix {}.'.format(model_dir, model_prefix))

  def restore(self, model_dir, model_prefix):
    self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
    self.logger.info(
        'Model restored from {},with prefix {}'.format(model_dir, model_prefix))


if __name__ == '__main__':

  class config(object):
    embedding_size = 100
    num_filters = 64
    l2_reg_lambda = 0.001
    learning_rate = 0.01
    trainable = True
    max_input_left = 70
    max_input_right = 70
    filter_sizes = [3, 4, 5]
    optim = 'adam'
    pooling = 'max'
    char_alphabet_size = 70
    char_filter_size = [6, 7, 8]
    char_embed_size = 70
    char_num_filters = 128
    char_length = 70
    is_train = True
    num_classes = 2
    input = 'char'
    summary_dir = './'
  q = np.random.randint(30, size=(30, 70))
  a = np.random.randint(30, size=(30, 70))
  y = np.random.randint(2, size=(30))
  vocab = np.random.rand(1000, 100)
  model = CNN(vocab, config)

  score = model.sess.run(model.logits, feed_dict={model.q: q,
                                                  model.a: a,
                                                  model.y: y,
                                                  model.dropout_keep_prob: 1.0})
  print(score)

  # print(q)
  exit()
