import json
from collections import defaultdict
import math
from nltk.corpus import stopwords
import numpy as np
import argparse
from utils import *


def rank_ensemble(args, topk=20):

    word2emb = load_cate_emb(f'datasets/{args.dataset}/emb_{args.topic}_w.txt')
    word2bert = load_bert_emb(f'datasets/{args.dataset}/{args.dataset}_bert')

    caseolap_results = []
    with open(f'datasets/{args.dataset}/intermediate_2.txt') as fin:
        for line in fin:
            data = line.strip()
            _, res = data.split(':')
            caseolap_results.append(res.split(','))
            
    cur_seeds = []
    with open(f'datasets/{args.dataset}/{args.topic}_seeds.txt') as fin:
        for line in fin:
            data = line.strip().split(' ')
            cur_seeds.append(data)


    with open(f'datasets/{args.dataset}/{args.topic}_seeds.txt', 'w') as fout:
        for seeds, caseolap_res in zip(cur_seeds, caseolap_results):
            word2mrr = defaultdict(float)

            # cate mrr
            word2cate_score = {word:np.mean([np.dot(word2emb[word], word2emb[s]) for s in seeds]) for word in word2emb}
            r = 1.
            for w in sorted(word2cate_score.keys(), key=lambda x: word2cate_score[x], reverse=True)[:topk]:
                if w not in word2bert: continue
                word2mrr[w] += 1./r
                r += 1
                 
            # bert mrr
            word2bert_score = {word:np.mean([np.dot(word2bert[word], word2bert[s]) for s in seeds]) for word in word2bert}
            r = 1.
            for w in sorted(word2bert_score.keys(), key=lambda x: word2bert_score[x], reverse=True)[:topk]:
                if w not in word2emb: continue
                word2mrr[w] += 1./r
                r += 1
            
            # caseolap mrr
            r = 1.
            for w in caseolap_res[:topk]:
                word2mrr[w] += 1./r
                r += 1

            score_sorted = sorted(word2mrr.items(), key=lambda x: x[1], reverse=True)
            top_terms = [x[0].replace(' ', '') for x in score_sorted if x[1] > args.rank_ens and x[0] != '']
            fout.write(' '.join(top_terms) + '\n')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='nyt', type=str)
    parser.add_argument('--topic', default='topic', type=str)
    parser.add_argument('--topk', default=20, type=int)
    parser.add_argument('--rank_ens', default=0.3, type=float)
    args = parser.parse_args()

    rank_ensemble(args, args.topk)