from model import MemNN
import pickle
import numpy as np

with open('data/f_y_matrix', 'rb') as f:
    f_y = pickle.load(f)
with open('data/g_q_matrix', 'rb') as f:
    g_q = pickle.load(f)
with open('data/vocabulary.txt') as f:
    vocab = np.array(f.read.split())

model = MemNN()
# model.load('embedding.h5')
model.train_embeddings(g_q, f_y)
# print(model.predict(g_q[1], f_y))
