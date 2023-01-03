import numpy as np
import argparse
from utils import *

def process_cate(args, topK=20):

    word2emb = load_cate_emb(f'datasets/{args.dataset}/emb_{args.topic}_w.txt')
    word2bert = load_bert_emb(f'datasets/{args.dataset}/{args.dataset}_bert')

    cur_seeds = []
    with open(f'datasets/{args.dataset}/{args.topic}_seeds.txt') as fin:
        for line in fin:
            data = line.strip().split(' ')
            cur_seeds.append(data)

    seeds = []
    with open(f'datasets/{args.dataset}/{args.topic}.txt') as fin:
        for line in fin:
            data = line.strip()
            seeds.append(data.split(' ')[0])
            

    with open(f'datasets/{args.dataset}/intermediate_1.txt', 'w') as fout:
        for seed, seeds in zip(seeds, cur_seeds):
            score = {}
            for word in word2emb:
                if word not in word2bert:
                    continue
                cate_cate = np.mean([np.dot(word2emb[word], word2emb[s]) for s in seeds])
                cate_bert = np.mean([np.dot(word2bert[word], word2bert[s]) for s in seeds])
                score[word] =  cate_cate * cate_bert

            score_sorted = sorted(score.items(), key=lambda x: x[1], reverse=True)
            top_terms = [x[0] for x in score_sorted[:topK]]
            fout.write(seed+':'+','.join(top_terms)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='nyt', type=str)
    parser.add_argument('--topic', default='topic', type=str)
    parser.add_argument('--topk', default=20, type=int)
    args = parser.parse_args()

    process_cate(args, args.topk)