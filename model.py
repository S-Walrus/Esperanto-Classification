'''
Imports
'''
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import re
import numpy as np
import random
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dot, Activation, Dense, Permute, Dropout, Lambda, Concatenate
from keras.optimizers import SGD
from keras.utils import plot_model
from keras import backend as K
from time import time
from math import ceil
from scipy.spatial.distance import cosine
from tqdm import tqdm
from sklearn.preprocessing import normalize


'''
Code below is by isaacsultan from 
https://stackoverflow.com/questions/51003027/computing-cosine-similarity-between-two-tensors-in-keras
'''
def cosine_distance(vests):
	x, y = vests
	x = K.l2_normalize(x, axis=-1)
	y = K.l2_normalize(y, axis=-1)
	return -K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0],1)
'''
Here code by isaacsultan ends.
'''


class MemNN:

	# [vocab] == np.array(string)
	# [aliases] == np.array(string)
	def __init__(self, vocab, aliases, d=90):
		self.vocab = vocab
		self.aliases = aliases

		self.ns = 1969831
		self.nv = 61383
		# self.nv = vocab.shape[0]

		# hyperparameters
		self.d = d
		# TODO (matbe) make gamma work
		# self.gamma = 0.2

		self.word_embedding = Dense(self.d, input_shape=(self.nv,), name='word_embedding', use_bias=False)
		self.entity_embedding = Dense(self.d, input_shape=(self.ns,), name='entity_embedding', use_bias=False)

		g_q = Input((self.nv,), name='g_q')
		w_q = self.word_embedding(g_q)

		f_y = Input((self.ns,), name='f_y')
		w_y = self.entity_embedding(f_y)

		f_y_s = Input((self.ns,), name='f_y-false')
		w_y_s = self.entity_embedding(f_y_s)

		# the following lines are by isaacsultan from stackoverflow
		cos_dist = Lambda(cosine_distance, output_shape=(1,), name='cosine_distance')
		s = cos_dist([w_q, w_y])
		s_s = cos_dist([w_q, w_y_s])

		out = Concatenate(axis=-1, name='out')([s, s_s])
		# y must be [1, 0]

		# build embedding trainer
		self.model = Model([g_q, f_y, f_y_s], out)
		self.model.compile('sgd', loss='categorical_hinge', metrics=[])

		# build cosine distance calculator
		self.cosine = Model([g_q, f_y], s)
		self.cosine.compile('sgd', loss='categorical_hinge', metrics=[])


	def generate_batch(self, q, y, batch_size, batch_start, n_samples):
		target = np.array([np.array([1, -1]) for _ in range(batch_size)])
		idx = range(batch_start, min(batch_start+batch_size, n_samples))
		mq = np.array([np.array(q[i].todense())[0] for i in idx])
		my = np.array([np.array(y[i].todense())[0] for i in idx])
		ys = np.array([np.array(y[i].todense())[0] for i in
			np.random.choice(n_samples, batch_size)])
		batch = [mq, my, ys]
		return batch, target


	def train_embeddings(self, q, y, epochs=6000, rep_epoch=1, batch_size=32):
		count = q.shape[0]
		print('Train data consists of ' + str(count) + ' samples')
		n_batches = ceil(count / batch_size)
		batch_start = 0
		for epoch in range(1, epochs+1):
			print('Epoch ' + str(epoch) + '/' + str(epochs))
			if epoch % 10 == 0:
				# look at loss
				batch, target = self.generate_batch(q, y, 1, random.randint(0, count-1), count)
				self.model.fit(x=batch, y=target, epochs=rep_epoch, verbose=True)
				self.normalize_layer(self.word_embedding)
				self.normalize_layer(self.entity_embedding)
			else:
				batch, target = self.generate_batch(q, y, batch_size, random.randint(0, count-batch_size), count)
				self.model.fit(x=batch, y=target, epochs=rep_epoch, verbose=False)
				self.normalize_layer(self.word_embedding)
				self.normalize_layer(self.entity_embedding)
			if epoch % 100 == 0:
				print('Saving model to embedding.h5')
				self.model.save('embedding.h5')
				!! mv embedding.h5 /content/drive/My\ Drive/
		print('Saving model to embedding.h5')
		self.model.save('embedding.h5')


	def normalize_layer(self, layer):
		W = layer.get_weights()
		W[0] = self.normalize_weights(W[0])
		layer.set_weights(W)


	def normalize_weights(self, W):
		return normalize(W)
		# return np.array([normalize(item) for item in W])


	def load(self, path_to_model):
		self.model = load_model(path_to_model)
		self.word_embedding = self.model.get_layer('word_embedding')
		self.entity_embedding = self.model.get_layer('entity_embedding')


	def is_candidate(self, y, bow):
		for word in bow:
			if word in y:
				return True
		return False


	def join_aliases(self, f_y):
		# TODO write aliases on init
		return ' '.join(self.aliases[f_y.where(f_y == 1)])


	def generate_cands(self, f_y, g_q):
		# TODO write vocab on init
		bow = self.vocab[g_q.where(g_q == 1)]
		return np.array([item for i, item in f_y
			if self.is_candidate(self.join_aliases(f_y), bow)])


	def predict(self, g_q, y):
		cosine = []
		x = np.array(g_q.todense())[0]
		for fact in tqdm(self.generate_cands(y, x)):
			cosine.append(self.cosine.predict([x, np.array(fact.todense())[0]])[0])
		return cosine.index(max(cosine))


	def score(self, X, y):
		pass