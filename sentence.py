import json
import argparse
import re
from nltk.tokenize import sent_tokenize

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='nyt', type=str)
parser.add_argument('--nltk', action='store_true', help='If enabled, use nltk toeknization')
args = parser.parse_args()

with open(f'caseolap_datasets/{args.dataset}/phrase_text.txt') as fin, open(f'caseolap_datasets/{args.dataset}/sentences.json', 'w') as fout:
	for idx, line in enumerate(fin):
		out = {}
		out['doc_id'] = idx
		ss = []
		test = []
		if args.nltk == False:
			data = line.strip().replace('!', '.').replace('?', '.')
			sents = data.split(' .')
			for sent in sents:
				s = sent.strip()
				if len(s) >= 5:
					ss.append(s)
			
			text = re.sub(r'http\S+', '', line)
			sent_token = sent_tokenize(text)
			for token in sent_token:
				token = re.sub(r'[^A-Za-z0-9\_\-]', ' ', token)
				test.append(token.strip())
			
		else:
			text = re.sub(r'http\S+', '', line)
			sent_token = sent_tokenize(text)
			for token in sent_token:
				token = re.sub(r'[^A-Za-z0-9\_\-]', ' ', token)
				ss.append(token.strip())
		out['sentences'] = ss
		fout.write(json.dumps(out)+'\n')