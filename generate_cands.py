path = 'data/'
data = set()
tag_i = 0

for name in ['train.txt', 'test.txt', 'valid.txt']:
    with open(path + name) as f:
        for line in f:
            _, tag, _, text = line.split('\t', 3)
            tag = tag.split('/')[1:]
            data.add(tag[tag_i])
            print(tag[tag_i])

with open('cands.txt', 'w') as f:
    f.write('\n'.join(data))
