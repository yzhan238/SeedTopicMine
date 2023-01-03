import json
from collections import defaultdict
import math
from nltk.corpus import stopwords
import numpy as np
import argparse
from utils import *


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


def sentence_retrieval(args, seeds, keywords):

    scores = defaultdict(dict)
    id2sent = {}
    id2start = {}
    id2end = {}
    with open(f'datasets/{args.dataset}/sentences.json') as fin:
        for idx, line in enumerate(fin):
            if idx % 10000 == 0:
                print(idx)
            data = json.loads(line)
            start = len(id2sent)
            end = start+len(data['sentences'])-1
            for sent in data['sentences']:
                sent_id = len(id2sent)
                id2sent[sent_id] = sent
                id2start[sent_id] = start
                id2end[sent_id] = end

                words = sent.split()
                word_cnt = defaultdict(int)
                for word in words:
                    word_cnt[word] += 1
                score = defaultdict(int)
                for seed in keywords:
                    for kw in keywords[seed]:
                        score[seed] += word_cnt[kw]
                pos_seeds = [x for x in score if score[x] > 0]
                if len(pos_seeds) == 1:
                    seed = pos_seeds[0]
                    scores[seed][sent_id] = score[seed]


    # print out top sentences
    topk = args.num_sent
    wd = args.sent_window
    top_sentences = []
    with open(f'datasets/{args.dataset}/top_sentences.json', 'w') as fout:
        for seed in seeds:
            out = {}
            out['seed'] = seed
            out['sentences'] = []
            scores_sorted = sorted(scores[seed].items(), key=lambda x: x[1], reverse=True)[:topk]
            # print(scores_sorted[-1])

            for k0, v in scores_sorted:
                out['sentences'].append(id2sent[k0])
                
                for k in range(k0-1, k0-wd-1, -1):
                    if k < id2start[k0]:
                        break
                    excl = 1
                    for seed_other in seeds:
                        if seed_other == seed:
                            continue
                        if k in scores[seed_other]:
                            excl = 0
                            break
                    if excl == 1:
                        out['sentences'].append(id2sent[k])
                    else:
                        break

                for k in range(k0+1, k0+wd+1):
                    if k > id2end[k0]:
                        break
                    excl = 1
                    for seed_other in seeds:
                        if seed_other == seed:
                            continue
                        if k in scores[seed_other]:
                            excl = 0
                            break
                    if excl == 1:
                        out['sentences'].append(id2sent[k])
                    else:
                        break
            fout.write(json.dumps(out)+'\n')
            top_sentences.append(out)
    return top_sentences

def caseolap(args, topk=20):
    seeds = []
    keywords = {}
    with open(f'datasets/{args.dataset}/intermediate_1.txt') as fin:
        for line in fin:
            data = line.strip().split(':')
            seed = data[0]
            seeds.append(seed)
            kws = [data[0]] + data[1].split(',')
            keywords[seed] = kws

    word2emb = load_cate_emb(f'datasets/{args.dataset}/emb_{args.topic}_w.txt')
    word2bert = load_bert_emb(f'datasets/{args.dataset}/{args.dataset}_bert')

    top_sentences = sentence_retrieval(args, seeds, keywords)

    n = len(seeds)
    tf = [defaultdict(int) for _ in range(n)]
    df = [defaultdict(int) for _ in range(n)]
    for idx, data in enumerate(top_sentences):
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
            if tf[idx][word] >= 5 and word not in stop_words:
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

    with open(f'datasets/{args.dataset}/intermediate_2.txt', 'w') as fout1:
        for idx in range(n):
            seed = seeds[idx]
            caseolap = {}
            for word in candidate:
                if word in word2emb and word in word2bert:
                    sim1 = np.dot(word2emb[word], word2emb[seed])
                    sim2 = np.dot(word2bert[word], word2bert[seed])
                    pop = math.log(1 + df[idx][word], 2)
                    caseolap[word] = (pop ** args.alpha) * (dist[word][idx] ** (1-args.alpha)) * sim2 * sim1     
            caseolap_sorted = sorted(caseolap.items(), key=lambda x: x[1], reverse=True)
            
            top_terms = [x[0] for x in caseolap_sorted[:topk]]
            fout1.write(seed+':'+','.join(top_terms)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='nyt', type=str)
    parser.add_argument('--topic', default='topic', type=str)
    parser.add_argument('--topk', default=20, type=int)
    parser.add_argument('--alpha', default=0.2, type=float)
    args = parser.parse_args()
    
    caseolap(args, args.topk)