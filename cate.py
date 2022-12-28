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
parser.add_argument('--curr_seeds', default='seed_topics.txt', type=str)
parser.add_argument('--cate', action='store_true', help='If enabled, include cate')
parser.add_argument('--bert', action='store_true', help='If enabled, include bert')
args = parser.parse_args()

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

cur_seeds = []
with open(f'cate_datasets/{args.dataset}/{args.curr_seeds}') as fin:
	for line in fin:
		data = line.strip().split(' ')
		cur_seeds.append(data)

seeds = []
with open(f'cate_datasets/{args.dataset}/seed_{args.topic}.txt') as fin:
	for line in fin:
		data = line.strip()
		seeds.append(data.split(' ')[0])
		

topK = args.topk
with open(f'cate_datasets/{args.dataset}/result_{args.topic}.txt', 'w') as fout:
	for seed, seeds in zip(seeds, cur_seeds):
		score = {}
		for word in word2emb:
			if word not in word2bert:
				continue
			cate_cate = 1 if args.cate == False else np.mean([np.dot(word2emb[word], word2emb[s]) for s in seeds])
			cate_bert = 1 if args.bert == False else np.mean([np.dot(word2bert[word], word2bert[s]) for s in seeds])
			score[word] =  cate_cate * cate_bert

		score_sorted = sorted(score.items(), key=lambda x: x[1], reverse=True)
		top_terms = [x[0] for x in score_sorted[:topK]]
		fout.write(seed+':'+','.join(top_terms)+'\n')