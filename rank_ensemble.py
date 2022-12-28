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
parser.add_argument('--thres', default=0.5, type=float)
parser.add_argument('--curr_seeds', default='seed_topics.txt', type=str)
parser.add_argument('--cate', action='store_true', help='If enabled, include cate')
parser.add_argument('--bert', action='store_true', help='If enabled, include bert')
parser.add_argument('--caseolap', action='store_true', help='If enabled, include caseolap')
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

caseolap_results = []
with open(f'caseolap_datasets/{args.dataset}/output.txt') as fin:
    for line in fin:
        data = line.strip()
        _, res = data.split(':')
        caseolap_results.append(res.split(','))

seeds = []
with open(f'cate_datasets/{args.dataset}/seed_{args.topic}.txt') as fin:
    for line in fin:
        data = line.strip().split(' ')
        seeds.append(data)
        
cur_seeds = []
with open(f'cate_datasets/{args.dataset}/{args.curr_seeds}') as fin:
    for line in fin:
        data = line.strip().split(' ')
        cur_seeds.append(data)


with open(f'cate_datasets/{args.dataset}/new_seeds.txt', 'w') as fout:
    for init_seed, seeds, caseolap_res in zip(seeds, cur_seeds, caseolap_results):
        word2mrr = defaultdict(float)
        if args.cate:
            word2cate_score = {word:np.mean([np.dot(word2emb[word], word2emb[s]) for s in seeds]) for word in word2emb}
            r = 1.
            for w in sorted(word2cate_score.keys(), key=lambda x: word2cate_score[x], reverse=True)[:args.topk]:
                if w not in word2bert: continue
                word2mrr[w] += 1./r
                r += 1
                
        if args.bert:
            word2bert_score = {word:np.mean([np.dot(word2bert[word], word2bert[s]) for s in seeds]) for word in word2bert}
            r = 1.
            for w in sorted(word2bert_score.keys(), key=lambda x: word2bert_score[x], reverse=True)[:args.topk]:
                if w not in word2emb: continue
                word2mrr[w] += 1./r
                r += 1
            
        if args.caseolap:
            r = 1.
            for w in caseolap_res[:args.topk]:
                word2mrr[w] += 1./r
                r += 1

        score_sorted = sorted(word2mrr.items(), key=lambda x: x[1], reverse=True)
        top_terms = [x[0].replace(' ', '') for x in score_sorted if x[1] > args.thres and x[0] != '']
        fout.write(' '.join(top_terms) + '\n')