import os
import argparse
from cate import process_cate
from caseolap import caseolap
from rank_ensemble import rank_ensemble
from utils import *

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='nyt', type=str, help='name of dataset folder')
parser.add_argument('--text_file', default='corpus_train.txt', type=str, help='training corpus')
parser.add_argument('--topic', default='topics', type=str, help='name of topics')
parser.add_argument('--pretrain_emb', default='word2vec_100.txt', type=str, help='pretrained word2vec embeddings for CatE')
parser.add_argument('--num_iter', default=4, type=int, help='number of iterations')
parser.add_argument('--num_sent', default=500, type=int, help='maximum number of retrieved sentences')
parser.add_argument('--sent_window', default=4, type=int, help='window size for retrieving context sentences')
parser.add_argument('--alpha', default=0.2, type=float, help='weight for calculating topic-indicative context scores')
parser.add_argument('--rank_ens', default=0.3, type=float, help='threshold for rank ensemble')
args = parser.parse_args()


if not os.path.exists(f'datasets/{args.dataset}/sentences.json'):
    process_sentences(args)

assert os.system(f"cp datasets/{args.dataset}/{args.topic}.txt datasets/{args.dataset}/{args.topic}_seeds.txt") == 0

for iteration in range(args.num_iter):

    print(f'start iteration {iteration+1}')

    # execuate cate c command
    cate_c = f"""./cate/cate -train ./datasets/{args.dataset}/{args.text_file} -topic-name ./datasets/{args.dataset}/{args.topic}_seeds.txt \
                -load-emb {args.pretrain_emb} \
                -res ./datasets/{args.dataset}/res_{args.topic}.txt -k 10 -expand 1 \
                -word-emb ./datasets/{args.dataset}/emb_{args.topic}_w.txt -topic-emb ./datasets/{args.dataset}/emb_{args.topic}_t.txt \
                -size 100 -window 5 -negative 5 -sample 1e-3 -min-count 3 \
                -threads 20 -binary 0 -iter 10 -pretrain 2"""
    assert os.system(cate_c) == 0, "cate c command execute failed"

    # initial term ranking
    print('initial term ranking')
    process_cate(args)

    if iteration == args.num_iter - 1:
        assert os.system(f"cp datasets/{args.dataset}/intermediate_1.txt datasets/{args.dataset}/{args.topic}_results.txt") == 0
        break

    # second term ranking
    print('second term ranking')
    caseolap(args)
    
    # rank ensemble
    print('rank ensemble')
    rank_ensemble(args)
    
