import json
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='nyt', type=str)
parser.add_argument('--window', type=int, default=3)
parser.add_argument('--sentences', type=int, default=500)
args = parser.parse_args()

keywords = {}
seeds = []
with open(f'caseolap_datasets/{args.dataset}/seed.txt') as fin:
	for line in fin:
		data = line.strip().split(':')
		seed = data[0]
		seeds.append(seed)
		kws = [data[0]] + data[1].split(',')
		keywords[seed] = kws

scores = defaultdict(dict)
id2sent = {}
id2start = {}
id2end = {}
with open(f'caseolap_datasets/{args.dataset}/sentences.json') as fin:
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


# print out top-500 sentences for caseolap
topk = args.sentences
with open(f'caseolap_datasets/{args.dataset}/top_sentences.json', 'w') as fout:
	for seed in seeds:
		out = {}
		out['seed'] = seed
		out['sentences'] = []
		scores_sorted = sorted(scores[seed].items(), key=lambda x: x[1], reverse=True)[:topk]
		print(scores_sorted[-1])

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