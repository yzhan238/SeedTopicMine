import numpy as np
import scipy.sparse as ss
from collections import defaultdict
from corextopic import corextopic as ct
from nltk.corpus import stopwords
sw = stopwords.words('english')

dataset = 'nyt'
dim = 'topics'

word2cnt = defaultdict(int)
with open(f'../datasets/{dataset}/corpus_train.txt') as fin:
	for line in fin:
		data = line.strip().split()
		for word in data:
			word2cnt[word] += 1

seeds = []
with open(f'../datasets/{dataset}/{dim}.txt') as fin:
	for line in fin:
		data = line.strip()
		seeds.append(data)

word2id = {}
vocab = []
texts = []
with open(f'../datasets/{dataset}/corpus_train.txt') as fin:
	for line in fin:
		data = line.strip().split()
		text = []
		for word in data:
			if word not in sw and word.replace('_', '').isalpha() and word2cnt[word] >= 10 \
			   or word in seeds:
				text.append(word)
				if word not in word2id:
					vocab.append(word)
					word2id[word] = len(word2id)
		texts.append(text)

X = np.zeros((len(texts), len(word2id)), dtype=int)
for idx, text in enumerate(texts):
	for word in text:
		X[idx][word2id[word]] += 1
print(X.shape)
X = ss.csr_matrix(X)

topic_model = ct.Corex(n_hidden=len(seeds))
topic_model.fit(X, words=vocab, anchors=[[x] for x in seeds], anchor_strength=10)

# topic_model = ct.Corex(n_hidden=100)
# topic_model.fit(X, words=vocab)

n_top_words = 20
topics = topic_model.get_topics(n_words=n_top_words)
with open(f'CorEx_{dim}.txt', 'w') as fout:
	for i, topic in enumerate(topics):
		words = [x[0] for x in topic]
		fout.write(seeds[i]+':'+','.join(words)+'\n')