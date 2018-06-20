# my_match_model

This is a tensorflow implementation of recent paper about question answering . It can also be treated as a sentence pair problem.

[https://github.com/shuishen112/tensorflow-deep-qa]

is a implementation of Learning to [Rank Short Text Pairs with Convolutional Deep Neural Networks](http://disi.unitn.it/~severyn/papers/sigir-2015-long.pdf)

https://github.com/shuishen112/pairwise-deep-qa
is a implementation of 
[Enhanced Embedding based Attentive Pooling Network for Answer Selection](https://rd.springer.com/chapter/10.1007/978-3-319-73618-1_59 ). It reimplements the [attentive pooling network](https://arxiv.org/abs/1602.03609).

[https://github.com/shuishen112/pairwise-rnn]
is a implemention of [Inner attention based recurrent neural networks for answer selection](http://www.aclweb.org/anthology/P16-1122)

## Dataset

The dataset for these model is TRECQA and WIKIQA dataset. We also utlize YahooCQA dataset. The pretrained embedding is [Glove embeddings](https://nlp.stanford.edu/projects/glove/)

## Requirements
	* Python >= 2.7
	* Numpy
	* Tensorflow==1.2
	* pandas

