from plot.plot_evaluation import get_json
import pickle
import numpy as np
import spacy
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
from evaluation.utils import align_rationale_to_model, align_model_to_rationale
from os.path import join

import string


def clean_token(tokens):
    s = ''.join(tokens).replace('âĢĻ', '').replace('Ġ', ' ').replace('▁', '').replace('.Ċ', '. ').replace('>',
                                                                                                          '').replace(
        '<', '')
    if s and s[-1] in string.punctuation:
        return s[:-1]
    return s.strip()


important_pos = sorted(
    {'ADV', 'ADJ', 'SCONJ', 'PART', 'VERB', 'AUX', 'CCONJ', 'SPACE', 'PROPN', 'NOUN', 'PUNCT', 'X', 'ADP', 'NUM',
     'PRON', 'INTJ', 'DET'})
dict_colors = {pos: plt.colormaps['tab20'](i) for i, pos in enumerate(important_pos)}

all_pos = []
for model in ['llama3', 'mistral', 'llama', 'mixtral']:

    task = 'sst'
    sharey = True if task == 'sst' else False
    contrastive = False
    cut_off = 6
    sparsity = 'oracle' if task == 'sst' else 'full'
    languages = ['', '_EN', '_DK', '_IT']
    languages = ['_EN']
    articles = ['_article_1', '_article_2', '_article_5', '_article_8']
    seed = 28
    loop_over = languages if task == 'sst' else articles
    fig, axs = plt.subplots(len(loop_over), 3, figsize=(10, 4 * len(loop_over)), sharey=sharey)
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

        for sample in rationales.values():
            if np.sum(sample['rationales_model_pre_alignment']) > np.sum(sample['rationales_human_pre_alignment']):
                mismatched_number += 1
            else:
                correct_number += 1

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

            # if 'potatoes' in sample['tokens']:
            #     print([clean_token(token) for token, relevance in zip(aligned_tokens, aligned_rationales) if relevance > 0])
            #     print([(clean_token(token), relevance) for token, relevance in zip(aligned_tokens, aligned_rationales) if
            #            relevance > 0])
            #     import pdb;pdb.set_trace()

            # if np.sum([1 if rel == 1 and mask == 0 else 0 for rel, mask in
            #            zip(sample['relevance_lrp_binary'], sample['context_mask'])]) > 0:
            #     print(np.sum([1 if rel == 1 and mask == 0 else 0 for rel, mask in
            #                   zip(sample['relevance_lrp_binary'], sample['context_mask'])]))
            # import pdb;pdb.set_trace()

            # assert np.sum([1 if rel == 1 and mask == 0 else 0 for rel, mask in
            #                zip(sample['relevance_lrp_binary'], sample['context_mask'])]) == 0

        print(model, loop, 'correct_number vs. mismatched_number',
              np.around(mismatched_number / (mismatched_number + correct_number), decimals=2))

        docs = nlp.pipe(human_rationales)
        tagging_pos = []
        tagging_tag = []
        for i, doc in enumerate(docs):
            for token in doc:
                if token.pos_ in ['PUNCT', 'SPACE']:
                    continue
                tagging_pos.append(token.pos_)
                tagging_tag.append(token.tag_)
        print('human', Counter(tagging_pos))
        df = pd.DataFrame.from_dict(Counter(tagging_pos), orient='index', columns=['POS_count'])
        title = 'Human' if loop_idx == 0 else ''
        pos_tags = df.sort_values(by='POS_count', ascending=False).iloc[:cut_off].index.tolist()
        pos_colors = [dict_colors[pos] for pos in pos_tags]
        # import pdb;pdb.set_trace()
        # df.sort_values(by='POS_count', ascending=False).iloc[:10].plot.bar(ax=ax0, legend=False, label='index',
        #                                                                      title=title, rot=45, color=pos_colors)
        ax0 = axs[loop_idx, 0] if len(loop_over) > 1 else axs[0]
        ax0.bar(range(cut_off), df.sort_values(by='POS_count', ascending=False).iloc[:cut_off]['POS_count'],
                color=pos_colors)
        ax0.set_xticks(range(cut_off))
        ax0.set_xticklabels(pos_tags, rotation=45)
        ax0.set_title(title)
        all_pos.extend(df.sort_values(by='POS_count', ascending=False).iloc[:cut_off].index.tolist())

        docs = nlp.pipe(model_rationales)
        tagging_pos = []
        tagging_tag = []
        for i, doc in enumerate(docs):
            for token in doc:
                if token.pos_ in ['PUNCT', 'SPACE']:
                    continue
                tagging_pos.append(token.pos_)
                tagging_tag.append(token.tag_)
        print('model', Counter(tagging_pos))
        df = pd.DataFrame.from_dict(Counter(tagging_pos), orient='index', columns=['POS_count'])
        title = 'Self-Explanations' if loop_idx == 0 else ''
        pos_tags = df.sort_values(by='POS_count', ascending=False).iloc[:cut_off].index.tolist()
        pos_colors = [dict_colors[pos] for pos in pos_tags]
        # df.sort_values(by='POS_count', ascending=False).iloc[:cut_off].plot.bar(ax=ax1, legend=False,
        #                                                                    title=title, rot=45, color=pos_colors)
        ax1 = axs[loop_idx, 1] if len(loop_over) > 1 else axs[1]
        ax1.set_title(title)
        ax1.bar(range(min(cut_off, len(df))),
                df.sort_values(by='POS_count', ascending=False).iloc[:cut_off]['POS_count'],
                color=pos_colors)
        ax1.set_xticks(range(min(cut_off, len(df))))
        ax1.set_xticklabels(pos_tags, rotation=45)
        all_pos.extend(df.sort_values(by='POS_count', ascending=False).iloc[:cut_off].index.tolist())

        docs = nlp.pipe(lrp_rationales)
        tagging_pos = []
        tagging_tag = []
        for i, doc in enumerate(docs):
            for token in doc:
                if token.pos_ in ['PUNCT', 'SPACE']:
                    continue
                tagging_pos.append(token.pos_)
                tagging_tag.append(token.tag_)
        print('lrp', Counter(tagging_pos))
        df = pd.DataFrame.from_dict(Counter(tagging_pos), orient='index', columns=['POS_count'])
        title = 'LRP (post-hoc)' if loop_idx == 0 else ''
        pos_tags = df.sort_values(by='POS_count', ascending=False).iloc[:cut_off].index.tolist()
        pos_colors = [dict_colors[pos] for pos in pos_tags]
        # df.sort_values(by='POS_count', ascending=False).iloc[:cut_off].plot.bar(ax=ax2, legend=False,
        #                                                                    title=title, rot=45, color=pos_colors)
        ax2 = axs[loop_idx, 2] if len(loop_over) > 1 else axs[2]
        ax2.bar(range(min(cut_off, len(df))),
                df.sort_values(by='POS_count', ascending=False).iloc[:cut_off]['POS_count'],
                color=pos_colors)
        ax2.set_title(title)
        ax2.set_xticks(range(min(cut_off, len(df))))
        ax2.set_xticklabels(pos_tags, rotation=45)
        all_pos.extend(df.sort_values(by='POS_count', ascending=False).iloc[:cut_off].index.tolist())

    plot_dir = 'model_responses/plots/'
    filename_out = f'pos_analysis_{model}_{task}_{seed}'
    filename_out = filename_out + '_contrastive' if contrastive else filename_out
    filename_out = filename_out + '.png'
    plt.subplots_adjust(wspace=0, hspace=0.25)
    plt.savefig(join(plot_dir, filename_out), dpi=300, bbox_inches='tight')
print(set(all_pos))
