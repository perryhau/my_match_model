
import logging
from collections import Counter
import os
import pandas as pd
import numpy as np
import jieba


class SentencePairDataset(object):

  def __init__(self, train_file=None, dev_file=None, test_file=None, args=None):
    # continue

    self.logger = logging.getLogger('sentence_pairs')
    self.train_set, self.dev_set, self.test_set = None, None, None

    if train_file:
      self.train_set = self._load_data(train_file)
      self.logger.info('Train set size: {}'.format(len(self.train_set)))
    if dev_file:
      self.dev_set = self._load_data(dev_file)
      self.logger.info('Dev set size: {}'.format(len(self.dev_set)))
    if test_file:
      self.test_set = self._load_data(test_file)
      self.logger.info('Test set size:{}'.format(len(self.test_set)))
    self.max_input_left = args.max_input_left
    self.max_input_right = args.max_input_right
    # max_s1 = max(map(lambda x:len(x),self.train_set['s1'].str.split()))
    # max_s2 = max(map(lambda x:len(x),self.train_set['s2'].str.split()))
    # self.logger.info('max_length_s1:{} --- max_length_s2:{}'.format(max_s1,max_s2))
    self.alphabet = None

  def cut(self, sentence):
    tokens = jieba.cut(str(sentence))
    return tokens

  def _load_data(self, data_path):
    data = pd.read_csv(data_path, sep='\t', names=[
                      's1', 's2', 'flag'], quoting=3).fillna('')
    return data

  def get_alphabet(self, corpuses):
    word_counter = Counter()
    for corpus in corpuses:
      for texts in [corpus['s1'].unique(), corpus['s2']]:

        for sentence in texts:

          tokens = self.cut(sentence)
          for token in set(tokens):
            word_counter[token] += 1

    word_dict = {w: index + 2 for (index, w) in enumerate(list(word_counter))}
    word_dict['NULL'] = 0
    word_dict['UNK'] = 1
    self.alphabet = word_dict
    return word_dict

  def get_embedding(self, fname, vocab, dim=50):

    embeddings = np.random.normal(0, 1, size=[len(vocab), dim])
    # word_vecs = {}
    # with open(fname) as f:
    #   i = 0
    #   for line in f:
    #     i += 1
    #     if i % 100000 == 0:
    #       print ('epch %d' % i)
    #     items = line.strip().split(' ')
    #     if len(items) == 2:
    #       vocab_size, embedding_size = items[0], items[1]
    #       print (vocab_size, embedding_size)
    #     else:
    #       word = items[0]
    #       if word in vocab:
    #         embeddings[vocab[word]] = items[1:]
    return embeddings


  def convert_to_word_ids(self, sentence, alphabet, max_len=40):
    indices = []
    tokens = self.cut(sentence)
    for word in tokens:
      if word in alphabet:
        indices.append(alphabet[word])
      else:
        continue
    results = indices + [alphabet['NULL']] * (max_len - len(indices))
    return results[:max_len]

  def convert_to_ids(self, sentence, alphabet, max_len):
    return self.convert_to_word_ids(sentence, alphabet, max_len)

  def gen_with_pair_single(self, df, alphabet, q_len, a_len):
    pairs = []
    for index, row in df.iterrows():
      question_indices = self.convert_to_ids(
          row['s1'], alphabet, max_len=q_len)
      answer_indices = self.convert_to_ids(row['s2'], alphabet, max_len=a_len)
      input_y = int(row['flag'])
      pairs.append((question_indices, answer_indices, input_y))
    return pairs

  def batch_iter(self, data, batch_size, shuffle=False):

    data = self.gen_with_pair_single(
        data, self.alphabet, self.max_input_left, self.max_input_right)

    data = np.array(data)
    data_size = len(data)

    if shuffle:
      shuffle_indice = np.random.permutation(np.arange(data_size))
      data = data[shuffle_indice]

    num_batch = int((data_size - 1) / float(batch_size)) + 1

    for i in range(num_batch):
      start_index = i * batch_size
      end_index = min((i + 1) * batch_size, data_size)

      yield data[start_index:end_index]

  def all_iter(self, data, batch_size, shuffle=False):

    data = self.gen_with_pair_single(
        data, self.alphabet, self.max_input_left, self.max_input_right)

    data = np.array(data)
    yield data


# data_path = 'data/tencent'
# class config(object):
#   max_input_left = 10
#   max_input_right = 10
# args = config()
# train_file = os.path.join(data_path,'train.txt')
# dev_file = os.path.join(data_path,'dev.txt')
# test_file = os.path.join(data_path,'test.txt')

# embeddings_file = 'embedding/wiki.ch.text100.vector'
# data_set = SentencePairDataset(train_file,dev_file,test_file,args)


# print(data_set.train_set.head())
# # print(train_file)
# alphabet = data_set.get_alphabet([data_set.train_set,data_set.test_set,data_set.dev_set])

# # embeddings = data_set.get_embedding(embeddings_file,alphabet,100)

# batch = data_set.batch_iter(data_set.train_set,60,True)


# for d in batch:
# 	question,answer,input_y = zip(*d)
# 	print(np.array(question))
