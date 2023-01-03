import numpy as np
from gensim.models import KeyedVectors
import json

def process_sentences(args):
    with open(f'datasets/{args.dataset}/{args.text_file}') as fin, \
         open(f'datasets/{args.dataset}/sentences.json', 'w') as fout:
        for idx, line in enumerate(fin):
            out = {'doc_id':idx}
            ss = []
            data = line.strip().replace('!', '.').replace('?', '.')
            sents = data.split(' .')
            for sent in sents:
                s = sent.strip()
                if len(s) >= 5:
                    ss.append(s)
            out['sentences'] = ss
            fout.write(json.dumps(out)+'\n')

def load_cate_emb(file):
    word2emb = {}
    with open(file) as fin:
        for idx, line in enumerate(fin):
            if idx == 0:
                continue
            data = line.strip().split()
            word = data[0]
            emb = np.array([float(x) for x in data[1:]])
            emb = emb / np.linalg.norm(emb)
            word2emb[word] = emb
    return word2emb

def load_bert_emb(file):
    word2bert_raw = KeyedVectors.load(file)
    word2bert = {}
    for word in word2bert_raw.index_to_key:
        emb = word2bert_raw[word]
        emb = emb / np.linalg.norm(emb)
        word2bert[word] = emb
    return word2bert