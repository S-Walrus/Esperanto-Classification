import numpy as np
import pickle
import re
from tqdm import tqdm

# rdf_id_pattern = \
#         re.compile(r'<http://rdf\.freebase\.com/ns/m\.([a-z0-9_]+)>.*')
rdf_alias_pattern = re.compile(r'"(.*)"@en')
rdf_id_pattern = re.compile(r'([0-9a-z_]+)')
fb_id_pattern = re.compile(r'www\.freebase\.com/m/([a-z0-9_]+).*')

with open('data/labels') as f:
    d = {re.search(rdf_id_pattern, s).group(1):
         re.search(rdf_alias_pattern, s).group(1) for s in tqdm(f)}
with open('data/subjects.txt') as f:
    res = []
    for s in tqdm(f):
        if re.match(fb_id_pattern, s):
            res.append(d[re.match(fb_id_pattern, s).group(1)])
        else:
            res.append('__@@__')

out = np.array(res)
print('Success!')
print(out)

with open('data/compiled_aliases', 'wb') as f:
    pickle.dump(out, f)
