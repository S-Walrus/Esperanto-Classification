'''
Imports
'''
import re
import numpy as np
import random
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dot, Activation, Dense, Permute, Dropout, Lambda, Concatenate
from keras.optimizers import SGD
from keras.utils import plot_model
from keras import backend as K


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

		self.model = Model([g_q, f_y, f_y_s], out)
		self.model.compile('sgd', loss='categorical_hinge', metrics=['accuracy'])

	def __input(self, fact_list, question_list):
		pass

	def __output(self, X, y):
		pass

	def train_embeddings(self, q, y, epochs=1000, batch_size=40):
		count = q.shape[0]
		target = np.array([np.array([1, 0]) for _ in range(batch_size)])
		for epoch in range(epochs):
			idx = np.random.choice(count, batch_size)
			mq = np.array([np.array(q[i].todense())[0] for i in idx])
			my = np.array([np.array(y[i].todense())[0] for i in idx])
			ys = np.array([np.array(y[i].todense())[0] for i in
				np.random.choice(count, batch_size)])
			batch = [mq, my, ys]
			# batch = np.array([[np.array(q[i].todense()).flatten(),
			# 		  np.array(y[i].todense()).flatten(),
			# 		  np.array(y[np.random.randint(count)].todense()).flatten()]
			# 		  for i in idx])
			self.model.fit(batch, target, verbose=True)

	def predict(self, X):
		pass

	def score(self, X, y):
		pass