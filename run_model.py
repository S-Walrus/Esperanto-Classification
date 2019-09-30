from model import MemNN
import pickle

with open('data/f_y_matrix', 'rb') as f:
	f_y = pickle.load(f)
with open('data/g_q_matrix', 'rb') as f:
	g_q = pickle.load(f)

model = MemNN()
model.train_embeddings(g_q, f_y)
