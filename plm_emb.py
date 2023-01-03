import json
from tqdm import tqdm
import numpy as np
import copy
from transformers import AutoTokenizer, AutoModel
import torch
import os
import argparse
from nltk import sent_tokenize
from functools import reduce
import mmap
from gensim.models.keyedvectors import KeyedVectors
import pickle

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='nyt', type=str)
parser.add_argument('--text_file', default='corpus_train.txt', type=str)
parser.add_argument('--plm', default='bert-base-uncased', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--gpu', default=0, type=int)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu}")
else:
    print('CUDA not available')
    exit()

tokenizer = AutoTokenizer.from_pretrained(args.plm)
model = AutoModel.from_pretrained(args.plm)
model.to(device)

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


file = f'datasets/{args.dataset}/{args.text_file}'
vocab = {}
inv_vocab = {}
sentences = []
with open(file) as f:
    for line in tqdm(f, total=get_num_lines(file)):
        for sent in sent_tokenize(line.strip().replace(' .', '.')):
            sent_toks = [tok for tok in sent.replace('.', ' .').split(' ') if tok != '']
            tok_enc = tokenizer([tok.replace('_', ' ') for tok in sent_toks], add_special_tokens=False)['input_ids']
            indices = [1] + (np.cumsum([len(ids) for ids in tok_enc]) + 1).tolist()
            flat_ids = [tokenizer.cls_token_id] + reduce(lambda x, y: x + y, tok_enc, []) + [tokenizer.sep_token_id]
            if len(flat_ids) > 512: continue
            for tok in sent_toks:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
                    inv_vocab[vocab[tok]] = tok
            sentences.append((flat_ids, [vocab[tok] for tok in sent_toks], indices))

batch_size = args.batch_size
iterations = int(len(sentences)/batch_size) + (0 if len(sentences) % batch_size == 0 else 1)
phrase_div = np.zeros((len(vocab), 1))
phrase_emb = np.zeros((len(vocab), 768))
for i in tqdm(range(iterations)):
    start = i * batch_size
    end = min((i+1)*batch_size, len(sentences))
    batch_ids = [ids for ids,_,_ in sentences[start:end]]
    batch_max_length = max(len(ids) for ids in batch_ids)
    ids = torch.tensor([ids + [0 for _ in range(batch_max_length - len(ids))] for ids in batch_ids]).long()
    masks = (ids != 0).long()
    ids = ids.to(device)
    masks = masks.to(device)
    with torch.no_grad():
        batch_final_layer = model(ids, masks)[0]
    for final_layer, (_,sent_ids,indices) in zip(batch_final_layer, sentences[start:end]):
        for idx in range(len(sent_ids)):
            tok_id = sent_ids[idx]
            phrase_emb[tok_id] += np.mean(final_layer[indices[idx]:indices[idx+1]].cpu().numpy(), axis=0)
            phrase_div[tok_id] += 1
            
ave_phrase_emb = phrase_emb / phrase_div

kv = KeyedVectors(768)
kv.add_vectors([inv_vocab[i] for i in range(len(vocab))], ave_phrase_emb)
kv.save(f'datasets/{args.dataset}/{args.dataset}_bert')