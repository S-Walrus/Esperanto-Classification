'''
Imports
'''
import re
import numpy as np
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
		self.ns = 2000000
		self.nv = 600000

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
		plot_model(self.model)

	def __input(self, fact_list, question_list):
		pass

	def __output(self, X, y):
		pass

	def final_loss(self, x):
		y, y_s = x
		return self.gamma - y + y_s

	'''
	Translates FB data row into bag-of-symbol vector
	'''
	# def __parse_fact(self, s):
	# 	tokens = s.split('\t')
	# 	object = tokens[0]
	# 	relationship = tokens[1]
	# 	subjects = tokens[2:]
	# 	k = len(s)-2
	# 	fact = np.zeros(self.ns)
	# 	fact[self.entity_map[object]] = 1
	# 	fact[self.entity_map[relationship]] = 1
	# 	for item in subjects:
	# 		fact[self.entity_map[item]] = 1/k
	# 	return fact

	'''
	Translates question in natural language
	into bag-of-ngrams vector
	'''
	# def __parse_question(self, s):
	# 	s = s.lower()
	# 	tokens = re.split(r"[:punct:]", s)
	# 	q = np.zeros(self.nv)
	# 	# fill tokens
	# 	for token in s:
	# 		q[self.vocab_map[token]] = 1
	# 	# fill aliases
	# 	for alias in self.alias_map.keys():
	# 		if alias in q:
	# 			q[self.alias_map[alias]] = 1
	# 	return q

	def train_embeddings(self, X, y):
		pass

	def predict(self, X):
		pass

	def score(self, X, y):
		pass