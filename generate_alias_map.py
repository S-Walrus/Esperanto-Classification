import numpy as np
import pickle
import re
from tqdm import tqdm

with open('data/aliases') as f:
	d = {re.search(r'<http://rdf\.freebase\.com/ns/m\.([a-z0-9_]+)>.*', s).group(1):
			re.search(r'"(.*)"@en', s).group(1) for s in f}
with open('data/subjects.txt') as f:
	res = []
	for s in tqdm(f):
		if re.match(r'www\.freebase\.com/m/([a-z0-9_]+).*'):
			res.append(d[re.match(r'www\.freebase\.com/m/([a-z0-9_]+).*').group(1)])
		else:
			res.append('__@@__')

out = np.array(res)
print('Success!')
print(out)

with open('data/compiled_aliases', 'wb') as f:
	pickle.dump(out, f)