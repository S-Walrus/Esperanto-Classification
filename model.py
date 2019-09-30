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

	def __init__(self):
		self.ns = 1969831
		self.nv = 61383

		# hyperparameters
		self.d = 30
		self.gamma = 0.2

		self.word_embedding = Dense(self.d, input_shape=(self.nv,), name='word_embedding')
		self.entity_embedding = Dense(self.d, input_shape=(self.ns,), name='entity_embedding')

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
		self.model.compile('sgd', loss='categorical_hinge', metrics=['accuracy'])

		# build cosine distance calculator
		self.cosine = Model([g_q, f_y], s)
		self.cosine.compile('sgd', loss='categorical_hinge', metrics=['accuracy'])

	def __input(self, fact_list, question_list):
		pass

	def __output(self, X, y):
		pass

	def generate_batch(self, q, y, batch_size, batch_start, n_samples):
		target = np.array([np.array([1, -1]) for _ in range(batch_size)])
		idx = range(batch_start, min(batch_start+batch_size, n_samples))
		mq = np.array([np.array(q[i].todense())[0] for i in idx])
		my = np.array([np.array(y[i].todense())[0] for i in idx])
		ys = np.array([np.array(y[i].todense())[0] for i in
			np.random.choice(n_samples, batch_size)])
		batch = [mq, my, ys]
		return batch, target

	def train_embeddings(self, q, y, epochs=3, rep_epoch=3, batch_size=40):
		count = q.shape[0]
		print('Train data consists of ' + str(count) + ' samples')
		n_batches = ceil(count / batch_size)
		batch_start = 0
		for epoch in range(1, epochs+1):
			print('Global epoch ' + str(epoch))
			while batch_start < count:
				print('Batch ' + str(batch_start // batch_size + 1) + '/' + str(n_batches))
				batch, target = self.generate_batch(q, y, batch_size, batch_start, count)
				self.model.fit(x=batch, y=target, epochs=rep_epoch, verbose=True)
				batch_start += batch_size
			if batch_start % 500 == 0:
				print('Saving model to embedding.hd5')
				self.model.save('embeddind.hd5')
			print('Saving model to embedding.hd5')
			self.model.save('embeddind.hd5')

	def load(self, path_to_model):
		self.model = load_model(path_to_model)
		self.word_embedding = self.model.get_layer('word_embedding')
		self.entity_embedding = self.model.get_layer('entity_embedding')

	def predict(self, q, y):
		cosine = []
		x = np.array(q.todense())
		for fact in tqdm(y):
			cosine.append(self.cosine.predict([x, fact.todense()])[0])
		return cosine.index(max(cosine))


	def score(self, X, y):
		pass