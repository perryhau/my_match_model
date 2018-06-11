from config import args
from models.QA_CNN import CNN
import models.evaluation as evaluation
from dataset import SentencePairDataset
import logging
import os
import pickle

import pandas as pd


def prepare(args):
  logger = logging.getLogger('sentence_pairs')
  logger.info('Checking the data files')
  for data_path in [args.train_files, args.dev_files, args.test_files]:
    assert os.path.exists(
        data_path), '{} file does not exist.'.format(data_path)

  for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)

  logger.info('Building vocabulary...')

  qa_data = SentencePairDataset(
      args.train_files, args.dev_files, args.test_files, args)

  alphabet = qa_data.get_alphabet(
      [qa_data.train_set, qa_data.test_set, qa_data.dev_set])
  logger.info("the final vocab size is {}".format(len(alphabet)))
  embeddings = qa_data.get_embedding(args.embedding_file, alphabet, dim=100)

  para = {'alphabet': alphabet, 'embeddings': embeddings}

  logger.info('save the embedding...')
  with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
    pickle.dump(para, fout)

  logger.info('Done with preparing')


def train(args):
  logger = logging.getLogger('sentence_pairs')
  logger.info('load data_set and vocab')
  with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
    vocab = pickle.load(fin)

  qa_data = SentencePairDataset(
      args.train_files, args.dev_files, args.test_files, args)
  qa_data.alphabet = vocab['alphabet']
  logger.info('initialize the model')

  args.alphabet_size = len(qa_data.alphabet)
  qa_model = CNN(vocab['embeddings'], args)
  logger.info('Training the model...')
  qa_model.train(qa_data, args.num_epochs, args.batch_size, save_dir=args.model_dir,
                 dropout_keep_prob=args.dropout_keep_prob)
  logger.info('Done with model training')


def evaluate(args):
  logger = logging.getLogger('sentence_pairs')
  logger.info('Load data_set and vocab...')
  with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
    vocab = pickle.load(fin)

  qa_data = SentencePairDataset(dev_file=args.dev_files, args=args)
  qa_data.alphabet = vocab['alphabet']
  args.max_input_left, args.max_input_right = args.max_input_left, args.max_input_right
  logger.info('restoring the model')
  qa_model = CNN(vocab['embeddings'], args)
  qa_model.restore(args.model_dir, args.pooling)

  dev_batches = qa_data.batch_iter(
      qa_data.dev_set, args.batch_size, shuffle=False)
  _, pred = qa_model.evaluate(dev_batches, args.result_dir, args.pooling)

  result_file = os.path.join(args.result_dir, args.pooling)
  df = pd.DataFrame({'pred': pred})
  df.to_csv(result_file, sep='\t', index=None, header=None)
  logger.info('result data is saved in {}'.format(args.result_dir))
  score = evaluation.my_f1_score(pred, qa_data.dev_set['flag'])
  logger.info('result f1-score is {}'.format(score))


def predict(args):
  logger = logging.getLogger('sentence_pairs')
  logger.info('load data_set and vocab')
  with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
    vocab = pickle.load(fin)

  qa_data = SentencePairDataset(test_file=args.test_files, args=args)
  qa_data.alphabet = vocab['alphabet']
  args.max_input_left, args.max_input_right = args.max_input_left, args.max_input_right
  logger.info('restoring the model')
  qa_model = CNN(vocab['embeddings'], args)
  qa_model.restore(args.model_dir, args.pooling)

  test_batches = qa_data.batch_iter(
      qa_data.test_set, args.batch_size, shuffle=False)
  _, pred = qa_model.evaluate(test_batches, args.result_dir, args.pooling)
  result_file = args.outpath

  df = pd.DataFrame({'aid': qa_data.test_set['qid'], 'pred': pred})
  df.to_csv(result_file, sep='\t', index=None, header=None)


def run():
  logger = logging.getLogger('sentence_pairs')
  logger.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s - %(message)s')

  if args.log_path:
    file_handler = logging.FileHandler(args.log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
  else:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

  logger.info('Running with args s:{}'.format(args))

  os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  if args.prepare:
    prepare(args)
  if args.train:
    train(args)
  if args.evaluate:
    evaluate(args)
  if args.predict:
    predict(args)


if __name__ == '__main__':
  run()
