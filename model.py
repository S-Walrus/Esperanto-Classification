'''
Imports
'''
import re
import numpy as np


class MemNN:

	def __init__(self):
		this.ns = 0
		this.nv = 0
		this.entity_map = map()
		this.vocab_map = map()
		pass

	def __input(self, fact_list, question_list):
		pass

	def __output(self, X, y):
		pass

	'''
	Translates FB data row into bag-of-symbol vector
	'''
	def __parse_fact(self, s):
		tokens = s.split('\t')
		object = tokens[0]
		relationship = tokens[1]
		subjects = tokens[2:]
		k = len(s)-2
		fact = np.zeros(this.ns)
		fact[this.entity_map[object]] = 1
		fact[this.entity_map[relationship]] = 1
		for item in subjects:
			fact[this.entity_map[item]] = 1/k
		return fact

	'''
	Translates question in natural language
	into bag-of-ngrams vector
	'''
	def __parse_question(self, s):
		s = s.lower()
		tokens = re.split(r"[:punct:]", s)
		q = np.zeros(this.nv)
		# fill tokens
		for token in s:
			q[this.vocab_map[token]] = 1
		# fill aliases
		for alias in this.alias_map.keys():
			if alias in q:
				q[this.alias_map[alias]] = 1
		return q

	def train_embeddings(self, X, y):
		pass

	def predict(self, X):
		pass

	def score(self, X, y):
		pass