import json
from collections import defaultdict
import math
from nltk.corpus import stopwords
import numpy as np
import argparse
from gensim.models import KeyedVectors

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='nyt', type=str)
parser.add_argument('--topic', default='topic', type=str)
parser.add_argument('--topk', default=20, type=int)
parser.add_argument('--catesim', action='store_true', help='If enabled, include cate similarity')
parser.add_argument('--bertsim', action='store_true', help='If enabled, include bert similarity')
parser.add_argument('--alpha', default=0.2, type=float)
args = parser.parse_args()
print(args)
print("==================")
def BM25(df, maxdf, tf, dl, avgdl, k=1.2, b=0.5):
	score = tf * (k + 1) / (tf + k * (1 - b + b * (dl / avgdl)))
	df_factor = math.log(1 + df, 2) / math.log(1 + maxdf, 2)
	score *= df_factor
	return score

def Softmax(score_list):
	exp_sum = 1
	for score in score_list:
		exp_sum += math.exp(score)
	exp_list = [math.exp(x) / exp_sum for x in score_list]
	return exp_list

word2emb = {}
with open(f'cate_datasets/{args.dataset}/emb_{args.topic}_w.txt') as fin:
	for idx, line in enumerate(fin):
		if idx == 0:
			continue
		data = line.strip().split()
		if len(data) != 101:
			continue
		word = data[0]
		emb = np.array([float(x) for x in data[1:]])
		emb = emb / np.linalg.norm(emb)
		word2emb[word] = emb

bert_file = f'cate_datasets/{args.dataset}/{args.dataset}_bert'
word2bert_raw = KeyedVectors.load(bert_file)
word2bert = {}
for word in word2bert_raw.index_to_key:
	emb = word2bert_raw[word]
	emb = emb / np.linalg.norm(emb)
	word2bert[word] = emb


cate_res = []
seeds = []
with open(f'caseolap_datasets/{args.dataset}/seed.txt') as fin:
	for line in fin:
		data = line.strip().split(':')
		seeds.append(data[0])
		data = data[1].split(',')
		cate_res.append(data)
n = len(cate_res)

tf = [defaultdict(int) for _ in range(n)]
df = [defaultdict(int) for _ in range(n)]
with open(f'caseolap_datasets/{args.dataset}/top_sentences.json') as fin:
	for idx, line in enumerate(fin):
		data = json.loads(line)
		for sent in data['sentences']:
			words = sent.split()
			for word in words:
				tf[idx][word] += 1
			words = set(words)
			for word in words:
				df[idx][word] += 1

stop_words = set(stopwords.words('english'))
candidate = set()
for idx in range(n):
	for word in tf[idx]:
		if tf[idx][word] >= 5 and word not in stop_words: # and '_' in word:
			candidate.add(word)

maxdf = [max(df[x].values()) for x in range(n)]
dl = [sum(tf[x].values()) for x in range(n)]
avgdl = sum(dl) / len(dl)
bm25 = [defaultdict(float) for _ in range(n)]
for idx in range(n):
	for word in candidate:
		bm25[idx][word] = BM25(df[idx][word], maxdf[idx], tf[idx][word], dl[idx], avgdl)

dist = {}
for word in candidate:
	dist[word] = Softmax([bm25[x][word] for x in range(n)])

alpha = args.alpha
topk = args.topk
with open(f'caseolap_datasets/{args.dataset}/output.txt', 'w') as fout1:
	for idx in range(n):
		seed = seeds[idx]
		caseolap = {}
		for word in candidate:
			if word in word2emb and word in word2bert:
				sim1 = 1 if args.catesim == False else np.dot(word2emb[word], word2emb[seed])
				sim2 = 1 if args.bertsim == False else np.dot(word2bert[word], word2bert[seed])
				pop = math.log(1 + df[idx][word], 2)
				caseolap[word] = (pop ** alpha) * (dist[word][idx] ** (1-alpha)) * sim2 * sim1     
		caseolap_sorted = sorted(caseolap.items(), key=lambda x: x[1], reverse=True)
		
		top_terms = [x[0] for x in caseolap_sorted[:topk]]
		fout1.write(seed+':'+','.join(top_terms)+'\n')