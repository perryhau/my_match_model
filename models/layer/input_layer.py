import tensorflow as tf 

class word_based_layer(object):
	def __init__(self,embeddings,embedding_size,q,a,a_neg,trainable):
		with tf.device('/cpu:0'),tf.variable_scope("word_embedding"):
			self.word_embeddings = tf.get_variable(
				'word_embeddings',
				shape = (len(embeddings),embedding_size),
				initializer = tf.constant_initializer(embeddings),
				trainable = trainable)

			self.q_emb = tf.nn.embedding_lookup(self.word_embeddings,q)
			self.a_emb = tf.nn.embedding_lookup(self.word_embeddings,a)
			self.a_neg_emb = tf.nn.embedding_lookup(self.word_embeddings,a_neg)
# def word_based_layer(embeddings,embedding_size,q,a,a_neg,trainable):
# 	with tf.device('/cpu:0'),tf.variable_scope("word_embedding"):
# 		word_embeddings = tf.get_variable(
# 			'word_embeddings',
# 			shape = (len(embeddings),embedding_size),
# 			initializer = tf.constant_initializer(embeddings),
# 			trainable = trainable)

# 		q_emb = tf.nn.embedding_lookup(word_embeddings,q)
# 		a_emb = tf.nn.embedding_lookup(word_embeddings,a)
# 		a_neg_emb = tf.nn.embedding_lookup(word_embeddings,a_neg)
class char_based_layer(object):
	def __init__(self,q,a,a_neg,args,dropout_keep):

		with tf.device('/cpu:0'),tf.variable_scope('char_embedding'):
			char_embeddings = tf.concat([tf.zeros((1,args.char_alphabet_size)),
				tf.one_hot(range(args.char_alphabet_size),args.char_alphabet_size,1.0,0.0)],0,name = 'char_embedding')

			 ## Fucntion for embedding lookup and dropout at embedding layer
			def emb_drop(E, x):
				emb = tf.nn.embedding_lookup(E, x)
				emb_drop = tf.nn.dropout(emb, dropout_keep)
				return emb_drop
			self.q_char_emb = emb_drop(char_embeddings,q)
			self.a_char_emb = emb_drop(char_embeddings,a)
			self.a_char_neg_emb = emb_drop(char_embeddings,a_neg)

		self.char_filter_size = args.char_filter_size
		self.char_embed_size = args.char_embed_size
		self.char_num_filters = args.char_num_filters
		
		self._encode()

	def _encode(self):
		self.q_emb = tf.expand_dims(self.q_char_emb,-1,name = 'q_emb')
		self.a_emb = tf.expand_dims(self.a_char_emb,-1,name = 'a_emb')
		self.a_neg_emb = tf.expand_dims(self.a_char_neg_emb,-1,name = 'a_neg_emb')

		self.kernels = []
		with tf.name_scope('convolution_encode'):
			cnn_outputs = []
			for i, filter_size in enumerate(self.char_filter_size):
				with tf.name_scope('conv-pool-%s' % filter_size):
					filter_shape = [filter_size,self.char_embed_size,1,self.char_num_filters]
					W = tf.get_variable('W'+str(i),filter_shape,tf.float32,tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_IN',uniform=True))
					b = tf.get_variable('b'+str(i),[self.char_num_filters],tf.float32,tf.constant_initializer(0.01))

					self.kernels.append((W,b))
			
			self.char_num_filters_total = self.char_num_filters * len(self.char_filter_size)
			self.q_conv = self.wide_convolution(self.q_emb)		
			self.a_conv = self.wide_convolution(self.a_emb)
			self.a_neg_conv = self.wide_convolution(self.a_neg_emb)

			self.q_conv = tf.squeeze(self.q_conv,2)
			self.a_conv = tf.squeeze(self.a_conv,2)
			self.a_neg_conv = tf.squeeze(self.a_neg_conv,2)

	def wide_convolution(self,embedding):
		cnn_outputs = []
		for i,filter_size in enumerate(self.char_filter_size):
			conv = tf.nn.conv2d(
					embedding,
					self.kernels[i][0],
					strides = [1,1,self.char_embed_size,1],
					padding = 'SAME',
					name = 'conv-{}'.format(i)
				)

			h = tf.nn.relu(tf.nn.bias_add(conv,self.kernels[i][1]),name = 'relu-{}'.format(i))
			cnn_outputs.append(h)
		cnn_reshape = tf.concat(cnn_outputs,3,name = 'concat')
		return cnn_reshape

