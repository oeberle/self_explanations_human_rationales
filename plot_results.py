import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import f1_score
import os
from os.path import join
from plot.plot_evaluation import format_key, get_json, plot_hist

method_keys = [("baseline$one_over_n", 'context_mask'),
               ("human_rationales$one_over_n", 'rationales_human'),
               ("model_rationales$one_over_n", 'rationales_model'),
               ("relevance_lrp$one_over_n", "relevance_lrp_binary"),
               # ("relevance_lrp$model_rationale", "relevance_lrp_binary"),
               # ("relevance_lrp$human_rationale", "relevance_lrp_binary")
               ]

name_dict = {"baseline$one_over_n": "baseline",
             "human_rationales$one_over_n": "human",
             "model_rationales$one_over_n": "generated",
             "relevance_lrp$one_over_n": "post-hoc (1/n)",
             "relevance_lrp$model_rationale": "post-hoc (#model tokens)",
             "relevance_lrp$human_rationale": "post-hoc (#human tokens)",

             }

plot_dict = {'llama': ('#87adf1',),
             'llama3': ('#1755c4',),
             'mistral': ('#5adada',),
             'mixtral': ('#0a7b8a',),
             }

comparisons = [('rationales_human', 'rationales_model$one_over_n'),
               ('rationales_human', 'relevance_lrp_binary$one_over_n'),
               ('rationales_model', 'relevance_lrp_binary$one_over_n'),
               # ('rationales_human', 'relevance_lrp_binary$model_rationale'),
               # ('rationales_model', 'relevance_lrp_binary$model_rationale'),
               # ('rationales_human', 'relevance_lrp_binary$human_rationale'),
               # ('rationales_model', 'relevance_lrp_binary$human_rationale'),
               # ('rationales_human', 'relevance_lrp_binary')
               ]

# plot F1
keys_f1 = ['rationales_humanXrationales_model',
           'rationales_humanXrelevance_lrp_binary',
           'rationales_modelXrelevance_lrp_binary',
           ]

keys_f1 = ["rationales_humanXrationales_model$one_over_n",
           "rationales_humanXrelevance_lrp_binary$one_over_n",
           "rationales_modelXrelevance_lrp_binary$one_over_n",
           # "rationales_humanXrelevance_lrp_binary$model_rationale",
           # "rationales_modelXrelevance_lrp_binary$model_rationale",
           # "rationales_humanXrelevance_lrp_binary$human_rationale",
           # "rationales_modelXrelevance_lrp_binary$human_rationale"
           ]

# List of replacement rules (old, new)
replace_rules = [
    ('rationales_', ''),
    ('relevance_lrp_binary', 'post-hoc'),
    ('model$one_over_n', 'model'),
    ('$one_over_n', ' 1/n'),
    ('$model_rationale', ' #model'),
    ('$human_rationale', ' #human'),
]


def main():
    plot_dir = 'model_responses/plots/'
    task = 'forced_labour'  # 'sst'
    sparsity = 'full'

    if task == 'sst':
        model_keys = ['llama', 'llama3', 'mistral', 'mixtral']
        loops = ['', '_EN', '_DK', '_IT']  # , '_IT']
        loops = ['', '_EN']  # , '_IT']
        seeds = [28]

    elif task == 'forced_labour':
        loops = ['_article_1', '_article_2', '_article_5', '_article_8']
        model_keys = ['llama', 'llama3', 'mistral', 'mixtral']
        # model_keys = ['mistral']
        seeds = [28]

    keys = [k[0] for k in method_keys]

    eval_stats_cols = ['model accuracy', 'f1', 'recall', 'ratio_tokens_model', 'ratio_tokens_human']

    model_loop_keys = [m + l_ for l_ in loops for m in model_keys]
    print(model_loop_keys)
    stats_data = {m: {k_: [] for k_ in eval_stats_cols} for m in model_loop_keys}
    f1_data = {m: {c1 + 'X' + c2: [] for (c1, c2) in comparisons} for m in model_loop_keys}
    tokens_all = {l: {} for l in loops}
    xai_strategy = 'lrp'

    for lidx, lang in enumerate(loops):
        print(f'----------------------{lidx}{lang}----------------------')

        f1, ax1 = plt.subplots(1, 2, figsize=(9, 5))
        f2, ax2 = plt.subplots(1, 1, figsize=(5, 5))

        for midx, m in enumerate(model_keys):
            print(f'----------------------{m}----------------------')

            #  data = []
            data_p_init = {k[0]: [] for k in method_keys}
            data_p = {k[0]: [] for k in method_keys}
            data_log = {k[0]: [] for k in method_keys}
            numbers = {k[0]: [] for k in method_keys}
            data_tokens = {k[0]: [] for k in method_keys if 'baseline' not in k[0]}
            skip=False

            #  f1 = {k[0]:{c1+'X'+c2:[] for (c1,c2) in comparisons} for k in method_keys}

            #   print(data_log)
            for seed in seeds:

                DATA_DIR_DATASET = join('model_responses', 'eval_results', task, m, sparsity, str(seed), xai_strategy)
                DATA_DIR_DATASET = DATA_DIR_DATASET.replace('sst', 'sst_multilingual' if lang in ['_EN', '_IT',
                                                                                                  '_DK'] else 'sst')

                if m == 'mixtral':
                    base_str = f'./{DATA_DIR_DATASET}/{task}_{m + lang}_quant_seed_{seed}_sparsity_{sparsity}' if sparsity != 'full' \
                        else f'./{DATA_DIR_DATASET}/{task}_{m + lang}_quant_seed_{seed}'

                    base_str_old = f'./model_responses/eval_results/{task}_{m + lang}_quant_seed_{seed}_sparsity_{sparsity}' if sparsity != 'full' \
                        else f'./model_responses/eval_results/{task}_{m + lang}_quant_seed_{seed}'
                else:
                    base_str = f'./{DATA_DIR_DATASET}/{task}_{m + lang}_seed_{seed}_sparsity_{sparsity}' if sparsity != 'full' \
                        else f'./{DATA_DIR_DATASET}/{task}_{m + lang}_seed_{seed}'

                    base_str_old = f'./model_responses/eval_results/{task}_{m + lang}_seed_{seed}_sparsity_{sparsity}' if sparsity != 'full' \
                        else f'./model_responses/eval_results/{task}_{m + lang}_seed_{seed}'

                # they should all be the same across strategies
                strategy = 'one_over_n'
                if os.path.isfile(base_str + "_scores_{}.p".format(strategy)):
                    scores_data = pickle.load(open(base_str + "_scores_{}.p".format(strategy), 'rb'))
                # elif os.path.isfile(base_str_old + "_scores_{}.p".format(strategy)):
                #     scores_data = pickle.load(open(base_str_old + "_scores_{}.p".format(strategy), 'rb'))
                #     base_str = base_str_old
                #     print('files in old structure found')
                else:
                    print(base_str + f"_scores_{strategy}.p not found")
                    skip = True
                    continue

                # try:
                #     scores_data = pickle.load(open(base_str + "_scores_{}.p".format(strategy), 'rb'))
                # except FileNotFoundError:
                #     scores_data = pickle.load(open(base_str_old + "_scores_{}.p".format(strategy), 'rb'))
                #     print(base_str + f"_scores_{strategy}.p not found")
                #     continue
                [stats_data[m + lang][k_].append(scores_data[k_]) for k_ in eval_stats_cols]

                # F1 scores
                for c1, c2 in comparisons:
                    c2_, strategy = c2.split('$')
                    eval_data = get_json(base_str + "_rationales_{}.jsonl".format(strategy))
                    f1_scores = [f1_score(eval_data[j][c1], eval_data[j][c2_]) for j in range(len(eval_data))]
                    f1_data[m + lang][c1 + 'X' + c2].extend(f1_scores)

                for j in range(len(eval_data)):
                    print(j, np.sum(eval_data[j]['rationales_human']))
                    print(j, np.sum(eval_data[j]['rationales_human_pre_alignment']))

                for k, eval_keys in method_keys:
                    k_, strategy = k.split('$')
                    print(base_str + "_flipping_{}.p".format(strategy))
                    if os.path.isfile(base_str + "_flipping_{}.p".format(strategy)):
                        flip_data = pickle.load(open(base_str + "_flipping_{}.p".format(strategy), 'rb'))
                        eval_data = get_json(base_str + "_rationales_{}.jsonl".format(strategy))
                    else:
                        continue

                    v = flip_data[k_]

                    prob_init = [v_[0][0] for v_ in v]
                    prob_flipped = [v_[0][-1] for v_ in v]
                    log_flipped = [v_[1][-1] for v_ in v]

                    ns = []
                    toks = []
                    if len(v) != len(eval_data):
                        import pdb;
                        pdb.set_trace()
                    for j in range(len(v)):
                        rats = np.array(eval_data[j][eval_keys])
                        mask_ = np.array(eval_data[j]['context_mask'])
                        if len(rats) != len(mask_):
                            import pdb;
                            pdb.set_trace()

                        rats[mask_ < 1] = 0.
                        ns.append(int(sum(rats)) / int(sum(mask_)))

                        # get tokens
                        if 'baseline' not in k:
                            toks.append(np.array(eval_data[j]['model_tokens'])[rats == 1])

                        if int(sum(rats)) / int(sum(mask_)) > 1.:
                            raise ValueError('sum bigger than 1')

                    data_log[k].extend(log_flipped)
                    data_p[k].extend(prob_flipped)
                    numbers[k].extend(ns)
                    data_p_init[k].extend(prob_init)

                    if 'baseline' not in k:
                        data_tokens[k].extend(toks)

            if skip:
                continue
            mean_p = np.mean(data_p_init[k])  # prob at the beginning are the same across_method_keys

            ax1[0].hlines(mean_p, xmin=-0.2, xmax=2.9, color=plot_dict[m][0], ls='--')

            ystr = 'probability after masking' if lidx == 0 else None

            plot_hist(data_p, keys, ystr, (f1, ax1[0]), name_dict, plot_dict[m][0],
                      midx, width=0.25, dx=0.6, dx2=0.06, labelstr=m, plot_xticks=True if midx == 1 else False)

            plot_hist(numbers, keys, 'fraction tokens masked', (f1, ax1[1]), name_dict, plot_dict[m][0], midx,
                      width=0.25, dx=0.6, dx2=0.06, labelstr=m, plot_xticks=True if midx == 1 else False)

            print("\n\n*******\n\n")

            name_map_f1 = {k: format_key(k, replace_rules) for k in keys_f1}

            cmap = plt.get_cmap("tab20b_r")
            ystr = 'f1'
            plot_hist(f1_data[m + lang], keys_f1, ystr, (f2, ax2), name_map_f1, cmap(midx), midx,
                      width=0.25, dx=0.6, dx2=0.06, labelstr=m, plot_xticks=True if midx == 1 else False)

            tokens_all[lang][m] = data_tokens

        for ax_ in [ax1[0], ax1[1], ax2]:
            h1, l1 = ax_.get_legend_handles_labels()
            ax_.legend(h1, l1,
                       bbox_to_anchor=(.5, 1.12, 0, 0),
                       loc=9, ncol=3, fontsize=11)
            # if lidx ==0:
            #     pass
            # else:
            #     import matplotlib.patches as mpatches
            #     empty_patch = mpatches.Patch(color='none', label='')
            #     h1, l1 = ax_.get_legend_handles_labels()
            #     ax_.legend([''], [empty_patch], bbox_to_anchor=(.5,1.12, 0,0),
            #                     loc=9, ncol=3, fontsize=11, frameon=False)

        f1.tight_layout()
        f1.suptitle(lang[1:], x=0.03, y=0.95, horizontalalignment='left', fontsize=16, fontweight='bold')
        if len(seeds)==1:
            f1.savefig(join(plot_dir, 'faithfulness_{}{}_{}_{}_{}.pdf'.format(task, lang, strategy, xai_strategy, seed)),
                       dpi=300)
        else:
            f1.savefig(join(plot_dir, 'faithfulness_{}{}_{}_{}.pdf'.format(task, lang, strategy, xai_strategy)), dpi=300)  # ,  transparent=True)
        f1.show()

        f2.tight_layout()
        f2.suptitle(lang[1:], x=0.03, y=0.95, horizontalalignment='left', fontsize=16, fontweight='bold')
        if len(seeds) == 1:
            f2.savefig(join(plot_dir, 'plausibility_{}{}_{}_{}_{}.pdf'.format(task, lang, strategy, xai_strategy, seed)),
                       dpi=300)  # ,  transparent=True)
        else:
            f2.savefig(join(plot_dir, 'plausibility_{}{}_{}_{}.pdf'.format(task, lang, strategy, xai_strategy)),
                       dpi=300)  # ,  transparent=True)
        f2.show()


if __name__ == '__main__':
    main()
