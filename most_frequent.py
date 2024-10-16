from plot.plot_evaluation import get_json
import pickle
import numpy as np
import spacy
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
from evaluation.utils import align_rationale_to_model, align_model_to_rationale
from os.path import join


import re
import numpy as np
import string

def _recover_control_sequences(s: str) -> str:
    s = s.replace(r"\}", "}")
    s = s.replace(r"\{", "{")
    s = s.replace(r"textbackslash ", "")
    return s

def clean_token(tokens):
    s = ''.join(tokens).replace('âĢĻ','').replace('Ġ', ' ').replace('▁', '').replace('.Ċ', '. ').replace('>','').replace('<','')
    if s and s[-1] in string.punctuation:
        return s[:-1]
    return s.strip()


models = ['llama3', 'mistral'] #'llama', 'llama3', 'mistral', 'mixtral']
models = ['llama', 'llama3', 'mistral', 'mixtral']


seed = 79
contrastive = False
task = 'sst'
#task = 'forced_labour'

sharey = True if task == 'sst' else False

sparsity = 'oracle' if task == 'sst' else 'full'
languages = ['', '_EN', '_DK', '_IT']
articles = ['_article_1', '_article_2', '_article_5', '_article_8']
loop_over = languages if task == 'sst' else articles


data_tokens = {m: {l: {c: None for c in ['human', 'model', 'lrp', 'all']} for l in loop_over} for m in models}


all_pos = []
for model in ['llama', 'llama3', 'mistral', 'mixtral']:

    correct_number = 0
    mismatched_number = 0
    for loop_idx, loop in enumerate(loop_over):
        # for loop_idx, loop in enumerate(['1']):

        filename = f"model_responses/eval_results/{task}/{model}/{sparsity}/{seed}/lrp/{task}_{model}{loop}_seed_{seed}_rationales_human_rationale.jsonl"
        filename = filename.replace('lrp', 'lrp_contrast') if contrastive else filename
        filename = filename.replace(f'seed_{seed}',
                                    f'seed_{seed}_sparsity_{sparsity}') if sparsity != 'full' else filename
        filename = filename.replace('sst', 'sst_multilingual') if loop in ['_EN', '_DK', '_IT'] else filename
        filename = filename.replace('seed', 'quant_seed') if model == 'mixtral' else filename
        filename = filename.replace('lrp', 'random') if model == 'mixtral' else filename
        rationales = get_json(filename)

        if loop == '_DK':
            nlp = spacy.load("da_core_news_sm")
        elif loop in ['', '_EN'] or task == 'forced_labour':
            nlp = spacy.load("en_core_web_sm")
        elif loop == '_IT':
            nlp = spacy.load("it_core_news_sm")

        human_rationales = []
        model_rationales = []
        lrp_rationales = []
        all_tokens = []

        for sample in rationales.values():
            if np.sum(sample['rationales_model_pre_alignment']) > np.sum(sample['rationales_human_pre_alignment']):
                mismatched_number += 1
            else:
                correct_number +=1

            # human annotations
            human_rationales.extend(
                [token for score, token in zip(sample['rationales_human_pre_alignment'], sample['tokens']) if
                 int(score) == 1])

            # self-explanations
            model_rationales.extend(
                [token for score, token in zip(sample['rationales_model_pre_alignment'], sample['tokens']) if
                 int(score) == 1])

            # lrp-explanations
            context_tokens = [token for token, mask in zip(sample['model_tokens'], sample['context_mask']) if
                              int(mask) == 1]
            lrp_binary = [relevance for relevance, mask in zip(sample['relevance_lrp_binary'], sample['context_mask'])
                          if int(mask) == 1]
            # import pdb;pdb.set_trace()
            aligned_tokens, aligned_rationales = align_model_to_rationale(sample['tokens_pre_alignment'],
                                                                          context_tokens,
                                                                          lrp_binary)
                                                                          # sample['relevance_lrp_binary'])

            lrp_rationales.extend([clean_token(token) for token, relevance in
                                   zip(aligned_tokens, aligned_rationales) if relevance > 0])


            all_tokens.extend([clean_token(token) for token, relevance in
                                   zip(aligned_tokens, aligned_rationales)])

        print(model, loop, 'correct_number vs. mismatched_number',
              np.around(mismatched_number / (mismatched_number + correct_number), decimals=2))


        def proc_docs(docs, filter_pos = ['PUNCT', 'SPACE'], filter_toks = ['âģľthe']):
            tokens = []
            for i, doc in enumerate(docs):
                for token in doc:
                    if token.pos_ in filter_pos:
                        continue
                    if token.text.lower() in filter_toks:
                        continue
                    if  not token.is_stop:
                        tokens.append(str(token).lower())
            return list(tokens)

        tokens_humans = proc_docs(nlp.pipe(human_rationales))
        tokens_model = proc_docs(nlp.pipe(model_rationales))
        tokens_lrp = proc_docs(nlp.pipe(lrp_rationales))
        tokens_all = proc_docs(nlp.pipe(all_tokens))


        topn=8
        data_tokens[model][loop]['human'] =  [t[0] for t in Counter(tokens_humans).most_common(topn)]
        data_tokens[model][loop]['model'] =  [t[0] for t in Counter(tokens_model).most_common(topn)]
        data_tokens[model][loop]['lrp'] =  [t[0] for t in Counter(tokens_lrp).most_common(topn)]
        data_tokens[model][loop]['all'] =  [t[0] for t in Counter(tokens_all).most_common(topn)]


if False:

    for loop_idx, loop in enumerate(loop_over):
        for m in ['llama3', 'mistral']:
            data_tokens[m][loop] = {k:', '.join(v) for k,v in data_tokens[m][loop].items()}
    
    df1 = pd.DataFrame.from_dict(data_tokens['llama3']).rename({'model': 'llama3', 'lrp': 'llama3 post-hoc', 'all': 'full corpus'}).T
    df2 = pd.DataFrame.from_dict(data_tokens['mistral']).rename({'model': 'mistral', 'lrp': 'mistral post-hoc', 'all': 'full corpus'}).T
    
    df = df1.join(df2.drop(['human', 'full corpus'], axis=1))
    df.index = df.index.map(lambda x: x.replace('_',' '))
    
    
    selected = [ ' article 1',' article 2']
    str_ = df.T[selected].to_latex(index=True)
    print(_recover_control_sequences(str_))

else:
    for loop_idx, loop in enumerate(loop_over):
        for m in ['llama', 'llama3', 'mistral', 'mixtral']:
            data_tokens[m][loop] = {k:', '.join(v) for k,v in data_tokens[m][loop].items()}
            
    df0 = pd.DataFrame.from_dict(data_tokens['llama']).rename({'model': 'llama2', 'lrp': 'llama2 post-hoc', 'all': 'full corpus'}).T
    df1 = pd.DataFrame.from_dict(data_tokens['llama3']).rename({'model': 'llama3', 'lrp': 'llama3 post-hoc', 'all': 'full corpus'}).T
    df2 = pd.DataFrame.from_dict(data_tokens['mistral']).rename({'model': 'mistral', 'lrp': 'mistral post-hoc', 'all': 'full corpus'}).T
    df3 = pd.DataFrame.from_dict(data_tokens['mixtral']).rename({'model': 'mixtral', 'lrp': 'NONE', 'all': 'full corpus'}).T
    
    
    df_ = df0.join(df1.drop(['human', 'full corpus'], axis=1))
    df_ = df_.join(df2.drop(['human', 'full corpus'], axis=1))
    df = df_.join(df3.drop(['human', 'full corpus'], axis=1))
    
    df.index = df.index.map(lambda x: x.replace('_',' '))

    if task == 'sst':
        a = ['', ' EN', ' DK', ' IT']

    else:
        a = [ ' article 1',' article 2',  ' article 5',' article 8' ]
    str_ = df.T[a].to_latex(index=True)
    print(_recover_control_sequences(str_))







    