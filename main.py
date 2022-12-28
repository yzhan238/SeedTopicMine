import json
import os
from tqdm import tqdm 

# default configuration
command = {"cate_cate":True,\
           "cate_bert":True,\
           "sentences":500,\
           "alpha":0.2,\
           "window":4,\
           "bertsim":True,\
           "catesim":True,\
           "dataset":"yelp",\
           "topic":"senti",\
           "nltk":False,
           "rank_ens":0.3,
           "first_top_k":20,
           "second_top_k":20,
           "save_res_file":None,
           "num_iter":5}

with open("job_json/temp.json") as f:
    lines = f.readlines()
    for i in tqdm(range(len(lines))):
        # read command from json, update default setting
        input_command = json.loads(lines[i])
        cur_command = command.copy()
        for command_name, command_value  in input_command.items():
            cur_command[command_name] = command_value
        # define variables
        topic = cur_command["topic"]
        dataset = cur_command['dataset']
        topic_file=f"seed_{topic}.txt"
        pretrain_emb="word2vec_100.txt"
        text_file="corpus_train.txt"
        rank_ens_thres = cur_command['rank_ens']
        first_top_k = cur_command["first_top_k"]
        second_top_k = cur_command["second_top_k"]
        num_iter = cur_command["num_iter"]
        
        
        assert os.system(f"cp ./cate_datasets/{dataset}/{topic_file} ./cate_datasets/{dataset}/0_{topic}_seeds.txt") == 0
        
        for iteration in range(num_iter):

            # execuate cate c command
            cate_c = f"""./src/cate -train ./cate_datasets/{dataset}/{text_file} -topic-name ./cate_datasets/{dataset}/{iteration}_{topic}_seeds.txt \
                        -load-emb {pretrain_emb} \
                        -res ./cate_datasets/{dataset}/res_{topic}.txt -k 10 -expand 1 \
                        -word-emb ./cate_datasets/{dataset}/emb_{topic}_w.txt -topic-emb ./cate_datasets/{dataset}/emb_{topic}_t.txt \
                        -size 100 -window 5 -negative 5 -sample 1e-3 -min-count 3 \
                        -threads 20 -binary 0 -iter 10 -pretrain 2"""
            assert os.system(cate_c) == 0, "cate c command execute failed"

            #  execute cate python command
            cate_python = f"python cate.py --dataset {dataset} --topic {topic} --topk {first_top_k} --curr_seeds {iteration}_{topic}_seeds.txt"
            if cur_command['cate_cate'] == True:
                cate_python += " --cate"
            if cur_command['cate_bert'] == True:
                cate_python += " --bert"
            assert os.system(cate_python) == 0, "cate python command execute failed"

            # move files
            assert os.system(f"cp ./cate_datasets/{dataset}/{text_file} ./caseolap_datasets/{dataset}/phrase_text.txt") == 0, "move file failed"
            assert os.system(f"cp ./cate_datasets/{dataset}/result_{topic}.txt ./caseolap_datasets/{dataset}/seed.txt") == 0, "move file failed"
            assert os.system(f"cp ./cate_datasets/{dataset}/result_{topic}.txt ./cate_datasets/{dataset}/{topic}_{iteration}_first_results.txt") == 0, "move file failed"

            # execute sentence.py
            sentence_command = f"python sentence.py --dataset {dataset}"
            if cur_command['nltk']:
                sentence_command += " --nltk"
            assert os.system(sentence_command) == 0, "execute sentence.py failed"

            # execute vote.py
            assert os.system(f"python vote.py --dataset {dataset} --window {cur_command['window']} --sentences {cur_command['sentences']}") == 0, "execute vote.py failed"

            # execute caseolap.py
            caseolap_command = f"python caseolap.py --dataset {dataset} --topic {topic} --topk {second_top_k} --alpha {cur_command['alpha']}"
            if cur_command['bertsim'] == True:
                caseolap_command += " --bertsim"
            if cur_command['catesim'] == True:
                caseolap_command += " --catesim"
            assert os.system(caseolap_command) == 0, "execute caseolap.py failed"

            # move output file
            assert os.system(f"cp ./caseolap_datasets/{dataset}/output.txt ./caseolap_datasets/{dataset}/{topic}_{iteration}_second_results.txt") == 0
            
            # rank ensemble
            rank_ens_command = f'python rank_ensemble.py --dataset {dataset} --topic {topic} --topk 20  --thres {rank_ens_thres} --curr_seeds {iteration}_{topic}_seeds.txt --cate --bert --caseolap'
            assert os.system(rank_ens_command) == 0, "execute rank_ensemble.py failed"
            assert os.system(f"cp ./cate_datasets/{dataset}/new_seeds.txt ./cate_datasets/{dataset}/{iteration+1}_{topic}_seeds.txt") == 0, "move file failed"
            
            if iteration == num_iter - 1 and cur_command["save_res_file"] is not None:
                fname = cur_command["save_res_file"]
                assert os.system(f"cp ./cate_datasets/{dataset}/{topic}_{iteration}_first_results.txt ./cate_datasets/{dataset}/{fname}.txt") == 0, "move file failed"
            
