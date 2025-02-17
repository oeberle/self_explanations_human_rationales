from plot.plot_evaluation import get_json
import numpy as np
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from evaluation.utils import align_model_to_rationale
from os.path import join
import string


def clean_token(tokens):
    s = ''.join(tokens).replace('âĢĻ', '').replace('Ġ', ' ').replace('▁', '').replace('.Ċ', '. ').replace('>',
                                                                                                          '').replace(
        '<', '')
    if s and s[-1] in string.punctuation:
        return s[:-1]
    return s.strip()


plot_dir = 'model_responses/plots/'
important_pos = sorted(
    {'ADV', 'ADJ', 'SCONJ', 'PART', 'VERB', 'AUX', 'CCONJ', 'SPACE', 'PROPN', 'NOUN', 'PUNCT', 'X', 'ADP', 'NUM',
     'PRON', 'INTJ', 'DET'})
dict_colors = {pos: plt.colormaps['tab20'](i) for i, pos in enumerate(important_pos)}

for model in ['mistral', 'llama3']:

    pos_tags_legend = []
    lines_all = []
    labels_all = []
    task = 'sst'
    # task = 'forced_labour'
    cut_off = 6
    sparsity = 'oracle' if task == 'sst' else 'full'
    languages = ['', '_EN', '_DK', '_IT']
    articles = ['_article_1', '_article_2', '_article_5', '_article_8']
    seed = 28
    loop_over = languages if task == 'sst' else articles
    fig, axs = plt.subplots(2, 2, figsize=(5, 4), sharey=True, sharex=True)
    correct_number = 0
    mismatched_number = 0
    width = 0.12  #width of the bars
    x = np.arange(-cut_off / 2, cut_off / 2)  #label locations
    for loop_idx, loop in enumerate(loop_over):
        filename = f"model_responses/eval_results/{task}/{model}/{sparsity}/{seed}/lrp/{task}_{model}{loop}_seed_{seed}_rationales_human_rationale.jsonl"
        filename = filename.replace(f'seed_{seed}',
                                    f'seed_{seed}_sparsity_{sparsity}') if sparsity != 'full' else filename
        filename = filename.replace('sst', 'sst_multilingual') if loop in ['_EN', '_DK', '_IT'] else filename
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

            aligned_tokens, aligned_rationales = align_model_to_rationale(sample['tokens_pre_alignment'],
                                                                          context_tokens,
                                                                          lrp_binary)
            # sample['relevance_lrp_binary'])

            lrp_rationales.extend([clean_token(token) for token, relevance in
                                   zip(aligned_tokens, aligned_rationales) if relevance > 0])

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
        df.POS_count = df.POS_count / df.POS_count.sum()
        df_human = df.copy()

        pos_tags_human = df.sort_values(by='POS_count', ascending=False).iloc[:cut_off].index.tolist()

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
        df.POS_count = df.POS_count / df.POS_count.sum() - df_human.POS_count

        pos_tags = pos_tags_human
        pos_colors = [dict_colors[pos] for pos in pos_tags]

        for pos_tag in pos_tags:
            if pos_tag not in df.index:
                df.loc[pos_tag] = 0
        ax_x = 0 if loop_idx in [0, 2] else 1
        ax_y = 0 if loop_idx < 2 else 1
        ax = axs[ax_y, ax_x]
        ax.bar(width * x, df.loc[pos_tags].POS_count, width, color=pos_colors)
        ax.set_xticks(range(min(cut_off, len(df))))
        ax.set_xticklabels(pos_tags, rotation=45)

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
        df.POS_count = df.POS_count / df.POS_count.sum() - df_human.POS_count
        title = 'post-hoc' if loop_idx == 0 else ''
        pos_tags = pos_tags_human
        pos_colors = [dict_colors[pos] for pos in pos_tags]
        for pos_tag in pos_tags:
            if pos_tag not in df.index:
                df.loc[pos_tag] = 0
        ax = axs[ax_y, ax_x]
        ax.bar(width * x + 1, df.loc[pos_tags].POS_count, width, color=pos_colors, label=pos_tags)

        if model == 'llama3' and task == 'sst':
            title_dict = {
                '': 'EN SST',
                '_EN': 'EN mSST',
                '_DK': 'DA mSST',
                '_IT': 'IT mSST',
            }
            ax.text(-0.1, 0.07, title_dict[loop])

        elif model == 'mistral' and task == 'sst':
            title_dict = {
                '': 'EN SST',
                '_EN': 'EN mSST',
                '_DK': 'DA mSST',
                '_IT': 'IT mSST',
            }
            ax.text(0.2, 0.055, title_dict[loop])

        elif model == 'mistral' and task == 'forced_labour':
            title_dict = {
                '_article_1': '#1',
                '_article_2': '#2',
                '_article_5': '#5',
                '_article_8': '#8',
            }
            ax.text(-0.4, 0.065, title_dict[loop])

        elif model == 'llama3' and task == 'forced_labour':
            title_dict = {
                '_article_1': '#1',
                '_article_2': '#2',
                '_article_5': '#5',
                '_article_8': '#8',
            }
            ax.text(-0.4, 0.11, title_dict[loop])

        lines_labels = ax.get_legend_handles_labels()
        lines, labels = lines_labels
        if len(lines_all) == 0:
            lines_all.extend(lines)
            labels_all.extend(labels)
        else:
            labels_new = [la for la in labels if la not in labels_all]
            lines_new = [li for li, la in zip(lines, labels) if la not in labels_all]
            labels_all.extend(labels_new)
            lines_all.extend(lines_new)
        assert len(lines_all) == len(labels_all)

        if loop_idx == len(loop_over) - 1:
            bbox_y = 1.5 if len(labels_all) > 8 else 1.4
            axs[0, 0].legend(lines_all, labels_all, loc='upper right', bbox_to_anchor=(2.1, bbox_y), ncols=4)
        if ax_y == 1:
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['self-explanations', 'post-hoc'], rotation=0, fontsize=8)
        if ax_x == 0 and ax_y == 0:
            ax.set_ylabel('$\\Delta$ POS distribution [model - human]')
            ax.yaxis.set_label_coords(-.28, 0)

    filename_out = f'pos_analysis_{model}_{task}_{seed}'
    filename_out = filename_out + '.png'
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(join(plot_dir, filename_out), dpi=300, bbox_inches='tight')