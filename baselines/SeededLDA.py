import numpy as np
import guidedlda
from collections import defaultdict
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

model = guidedlda.GuidedLDA(n_topics=len(seeds), n_iter=2000, refresh=50)
seed_topics = {}
for t_id, word in enumerate(seeds):
	seed_topics[word2id[word]] = t_id
model.fit(X, seed_topics=seed_topics, seed_confidence=0.5)

n_top_words = 20
topic_word = model.topic_word_
with open(f'SeededLDA_{dim}.txt', 'w') as fout:
	for i, topic_dist in enumerate(topic_word):
		topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
		fout.write(seeds[i]+':'+','.join(topic_words)+'\n')