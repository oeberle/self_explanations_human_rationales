import matplotlib.pyplot as plt

cmap = plt.get_cmap("Set2")
import numpy as np
import pandas as pd
import json
import pickle
from os.path import isfile


def get_f1_data(comps, strategy):
    m, base_str, dict_key, eval_key, seeds = comps
    missing_files_f1 = []

    data = []
    for seed in seeds:
        base_str_seed = base_str.format(seed, seed)
        if not isfile(base_str_seed + "_rationales_{}.jsonl".format(strategy)):
            missing_files_f1.append(base_str_seed + "_rationales_{}.jsonl".format(strategy))
            continue
        eval_data = get_json(base_str_seed + "_rationales_{}.jsonl".format(strategy))
        data_tmp = []
        for j in range(len(eval_data)):
            data_tmp.append([eval_data[j][eval_key][x] for x in range(len(eval_data[j][eval_key])) if
                             eval_data[j]['context_mask'][x] == 1])
        assert len(data_tmp[0]) == np.sum(eval_data[0]['context_mask'])
        data.extend(data_tmp)
        # _ = data.extend([eval_data[j][eval_key] for j in range(len(eval_data))])

    return data, missing_files_f1


def get_flip_data(dict_key, strategy, eval_key, base_str, data_p, data_ratios, data_p_init):
    file_name = base_str + "_flipping_{}.p".format(strategy)
    if not isfile(file_name):
        print("File not found: ", file_name)
       # return None, None, None
        return data_p, data_ratios, data_p_init

    
    flip_data = pickle.load(open(file_name, 'rb'))
    #  print(file_name)
    eval_data = get_json(base_str + "_rationales_{}.jsonl".format(strategy))

    try:
        v = flip_data[dict_key]
    except:
        import pdb;
        pdb.set_trace()

    prob_init = [v_[0][0] for v_ in v]
    prob_flipped = [v_[0][-1] for v_ in v]

    ns = []
    toks = []
    for j in range(len(v)):
        rats = np.array(eval_data[j][eval_key])
        mask_ = np.array(eval_data[j]['context_mask'])
        assert len(rats) == len(mask_)
        rats[mask_ < 1] = 0.
        frac = np.array(sum(rats)) / np.array(sum(mask_))
        ns.append(frac)

        if frac > 1.:
            raise

    data_p.extend(prob_flipped)
    data_ratios.extend(ns)
    data_p_init.extend(prob_init)

    return data_p, data_ratios, data_p_init


def format_key(key: str, replace_rules: list[tuple[str, str]]) -> str:
    """
    Format the key by applying a list of replacement rules.

    Args:
        key (str): The original key to format.
        replace_rules (list of tuples): A list of (old, new) replacement rules.

    Returns:
        str: The formatted key.
    """
    for old, new in replace_rules:
        key = key.replace(old, new)
    return key


def get_json(filename):
    dict_out = {}
    with open(filename) as f:
        i = 0
        for idx, example in enumerate(f.readlines()):
            example = json.loads(example)
            dict_out[idx] = example
            i += 1
    return dict_out


def plot_hist(dat, keys, ylabel, fax=None, name_dict=None, model_color='#939393', hatch=None, j=1, width=0.5, dx=0,
              dx2=0,
              labelstr='', plot_xticks=True):
    if name_dict is not None:
        key_names = [name_dict[k] for k in keys]
    else:
        key_names = keys

    xs = [i * (width + dx) + j * width - j * dx2 for i in range(len(keys))]

    if fax is not None:
        f, ax = fax
    else:
        f, ax = plt.subplots(1, 1, figsize=(4, 5))

    df = pd.DataFrame.from_dict(dat)
    colors = [model_color for k in keys]

    ax.bar(xs, df.mean(), capsize=6, width=0.8 * width, color=colors, linewidth=1., edgecolor='#616161', label=labelstr,
           hatch=hatch)
    # '#939393')

    for i, k in enumerate(keys):
        dat = df[k]
        jitter = np.random.normal(0, 0.01, len(dat))
        ax.scatter([xs[i]] * len(dat) + jitter, dat, color='black', alpha=0.5, s=3)

    ax.set_ylabel(ylabel, fontsize=14)

    if plot_xticks:
        ax.set_xticks(xs)
        ax.set_xticklabels(key_names, rotation=45, ha="right", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)

    ax.spines[['right', 'top']].set_visible(False)


def plot_perturbation(d_prob, d_log, d_n):
    df = pd.DataFrame.from_dict(d_prob)

    f, axs = plt.subplots(1, 3, figsize=(13, 3))

    ax = axs[0]
    bars = ax.bar(np.arange(df.shape[1]), df.mean(), yerr=[df.mean() - df.min(), df.max() - df.mean()], capsize=6)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylabel('p after masking')

    df = pd.DataFrame.from_dict(d_log)
    ax = axs[1]
    bars = ax.bar(np.arange(df.shape[1]), df.mean(), yerr=[df.mean() - df.min(), df.max() - df.mean()], capsize=6)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylabel('logit after masking')

    df = pd.DataFrame.from_dict(d_n)
    ax = axs[2]
    bars = ax.bar(np.arange(df.shape[1]), df.mean(), yerr=[df.mean() - df.min(), df.max() - df.mean()], capsize=6)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylabel('#tokens masked')

    for ax in axs.flatten():
        ax.spines[['right', 'top']].set_visible(False)

    plt.show()


def main(dataset_name, models, overall_accuracy, overall_f1, overall_recall, overall_rationale_accuracy,
         overall_rationale_count, ratio_tokens_model, ratio_tokens_human, sparsity, seed):
    article_ids = ['1', '2', '5', '8']
    # languages = ['EN1', 'EN2', 'DK', 'IT']
    languages = ['EN1', 'EN2']
    loop = article_ids if dataset_name == 'rafola' else languages
    fig, axs = plt.subplots(figsize=(8, 7), ncols=2, nrows=2)
    fig_count, axs_count = plt.subplots(figsize=(8, 7), ncols=2, nrows=2, sharey=True)
    fig_f1, axs_f1 = plt.subplots(figsize=(8, 7), ncols=2, nrows=2, sharey=True)
    fig_recall, axs_recall = plt.subplots(figsize=(8, 7), ncols=2, nrows=2, sharey=True)
    fig_ratio, axs_ratio = plt.subplots(figsize=(8, 7), ncols=2, nrows=2, sharey=True)

    if dataset_name == 'rafola':
        instances = [172, 255, 116, 123]
    else:
        instances = [263, 250, 250, 250]
    for idx, article in enumerate(loop):
        axs[idx // 2, idx % 2].scatter(models, overall_accuracy[article], color=cmap(0))
        axs[idx // 2, idx % 2].scatter(models, overall_f1[article], color=cmap(1))
        axs[idx // 2, idx % 2].scatter(models, overall_recall[article], color=cmap(3))
        # axs[idx // 2, idx % 2].scatter(models, overall_precision[article], color=cmap(4))
        axs[idx // 2, idx % 2].scatter(models, overall_rationale_accuracy[article], color=cmap(2))
        axs[idx // 2, idx % 2].set_ylim([0, 1])

        p = axs_count[idx // 2, idx % 2].bar(models, overall_rationale_count[article], color=cmap(0))
        axs_count[idx // 2, idx % 2].axhline(y=instances[idx], color='k', linestyle='-')

        axs_f1[idx // 2, idx % 2].scatter(models, overall_accuracy[article], color=cmap(0))
        axs_f1[idx // 2, idx % 2].bar(models, overall_f1[article], color=cmap(1))
        fig_f1.suptitle('F1 Score')

        axs_recall[idx // 2, idx % 2].scatter(models, overall_accuracy[article], color=cmap(0))
        axs_recall[idx // 2, idx % 2].bar(models, overall_recall[article], color=cmap(3))
        fig_recall.suptitle('Recall Score')

        x = np.arange(len(models))
        axs_ratio[idx // 2, idx % 2].bar(x - 0.125, ratio_tokens_model[article], 0.25, label='models', color=cmap(3))
        axs_ratio[idx // 2, idx % 2].bar(x + 0.125, ratio_tokens_human[article], 0.25, label='humans', color=cmap(2))
        axs_ratio[idx // 2, idx % 2].set_xticks(x)
        axs_ratio[idx // 2, idx % 2].set_xticklabels(models)
        axs_ratio[idx // 2, idx % 2].set_ylim([0, 1])
        if idx == 0:
            axs_ratio[idx // 2, idx % 2].legend(loc='upper left', ncols=2)
        fig_ratio.suptitle('Ratio of tokens in rationales')

        if dataset_name == 'rafola':
            axs_count[idx // 2, idx % 2].bar_label(p, padding=3)
            axs_count[idx // 2, idx % 2].set_title(f'Article {article} N={instances[idx]}')
            axs_f1[idx // 2, idx % 2].set_title(f'Article {article}')
            axs_recall[idx // 2, idx % 2].set_title(f'Article {article}')
            axs_ratio[idx // 2, idx % 2].set_title(f'Article {article}')
        else:
            axs_count[idx // 2, idx % 2].bar_label(p, label_type='center')
            title = f'SST_multilingual -{article}' if idx > 0 else f'SST -{article}'
            axs[idx // 2, idx % 2].set_title(title)
            axs_count[idx // 2, idx % 2].set_title(title + ' N= ' + str(instances[idx]))
            axs_f1[idx // 2, idx % 2].set_title(title)
            axs_recall[idx // 2, idx % 2].set_title(title)
            axs_ratio[idx // 2, idx % 2].set_title(title)

    fig.legend(['accuracy', 'f1', 'recall', 'rationale match'])
    fig_f1.legend(['accuracy', 'f1'])
    fig_recall.legend(['accuracy', 'recall'])

    if sparsity in ['words', 'phrases']:
        fig_count.savefig(f'figs/model_comparison_{dataset_name}_rationale_count_seed_{seed}_sparsity_{sparsity}.png')
        fig.savefig(f'figs/model_comparison_{dataset_name}_seed_{seed}_sparsity_{sparsity}.png')
        fig_f1.savefig(f'figs/model_comparison_{dataset_name}_f1_seed_{seed}_sparsity_{sparsity}.png')
        fig_recall.savefig(f'figs/model_comparison_{dataset_name}_recall_seed_{seed}_sparsity_{sparsity}.png')
        fig_ratio.savefig(f'figs/model_comparison_{dataset_name}_ratio_seed_{seed}_sparsity_{sparsity}.png')


    else:
        fig_count.savefig(f'figs/model_comparison_{dataset_name}_rationale_count_seed_{seed}.png')
        fig.savefig(f'figs/model_comparison_{dataset_name}_seed_{seed}.png')
        fig_f1.savefig(f'figs/model_comparison_{dataset_name}_f1_seed_{seed}.png')
        fig_recall.savefig(f'figs/model_comparison_{dataset_name}_recall_seed_{seed}.png')
        fig_ratio.savefig(f'figs/model_comparison_{dataset_name}_ratio_seed_{seed}.png')


if __name__ == '__main__':
    main()