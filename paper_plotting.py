import pickle
import json
import numpy as np
import pandas as pd
import seaborn as sns
import os
from plot.plot_evaluation import format_key, get_json, plot_hist, get_flip_data, get_f1_data
import matplotlib.pyplot as plt
from os import makedirs
from os.path import join, exists, isfile
from plot.plot_specs import plot_dict, hatch_dict, replace_rules


def get_base_str(task, m, lang, sparsity, seed, xai_method):
    root = "model_responses/eval_results/{}/{}/{}/{}/{}/".format(task, m, sparsity, seed, xai_method)
    if m == 'mixtral':
        base_str = root + "{}_{}{}_quant_seed_{}_sparsity_{}".format(task, m, lang, seed,
                                                                     sparsity) if sparsity != 'full' \
            else root + "{}_{}{}_quant_seed_{}".format(task, m, lang, seed)
    else:
        base_str = root + "{}_{}{}_seed_{}_sparsity_{}".format(task, m, lang, seed, sparsity) if sparsity != 'full' \
            else root + "{}_{}{}_seed_{}".format(task, m, lang, seed)
    return base_str


def join_groups(group1, group2):
    group_joined = []
    for a, b in zip(group1, group2):
        group_joined.extend([a, b])
    return group_joined


SAVE = True

models = ['llama', 'llama3', 'mistral', 'mixtral']
# models = ['mixtral', 'mistral', 'llama3', 'llama']
missing_files = []

task = 'forced_labour'
sparsity = 'full'
seeds = [28, 79, 96]
strategies = ['human_rationale']
plausibility_metric = 'kappa'

if task == 'sst':
    loops = ['', ]  # , '_IT']

elif task == 'sst_multilingual':
    loops = ['_EN', '_DK', '_IT']  # , '_IT']

elif task == 'forced_labour':
    loops = ['_article_1', '_article_2', '_article_5', '_article_8']
    # loops = ['_article_1', '_article_2', '_article_5']
print(f'----------------------Article {task}----------------------')
for lidx, lang in enumerate(loops):

    ### group masked
    # contains tuple of (modelname, file_base_str, dict_key, eval_key)
    # group_masked = [(m, get_base_str(task, m, lang, sparsity, '{}', 'random'), 'baseline', 'context_mask', seeds) for m
    #                 in models]

    group_masked = [
        (m, get_base_str(task, m, lang, sparsity, '{}', 'random'), 'relevance_lrp', 'relevance_lrp_binary', seeds) for m
        in models]

    ### group human
    group_human = [
        (m, get_base_str(task, m, lang, sparsity, '{}', 'random'), 'human_rationales', 'rationales_human', seeds) for m
        in models]

    ### group generated
    group_generated = [
        (m, get_base_str(task, m, lang, sparsity, '{}', 'random'), 'model_rationales', 'rationales_model', seeds)
        for m in models]

    ### group post-hoc
    group_lrp = [(m, get_base_str(task, m, lang, sparsity, '{}', 'lrp'), 'relevance_lrp', 'relevance_lrp_binary', seeds)
                 for m in
                 models if m not in ['mixtral']]
    group_lrp_contrast = [
        (m, get_base_str(task, m, lang, sparsity, '{}', 'lrp_contrast'), 'relevance_lrp', 'relevance_lrp_binary', seeds)
        for m in
        models if m not in ['mixtral']]

    group_lrp_joined = join_groups(group_lrp, group_lrp_contrast)

    # group_masked = join_groups(group_masked, group_random)

    bar_groups = [('random', group_masked),
                  ('human', group_human),
                  ('generated', group_generated),
                  ('post hoc', group_lrp_joined)
                  ]

    plot_dir = 'model_responses/plots/'
    if not exists(plot_dir):
        makedirs(plot_dir)

    for strategy in strategies:
        print(f'***************Strategy {strategy}***************')
        plot_ratios = True

        if plot_ratios:
            f1, ax1 = plt.subplots(1, 1, figsize=(6, 5))

        f0, ax0 = plt.subplots(1, 1, figsize=(6, 5))

        bar_width = 0.05
        x_extra = 0.15

        x0 = 0.

        xs = []
        xticks = []
        for i, (group_name, group) in enumerate(bar_groups):
            print(group_name)

            xs_group = [x0 + j * bar_width for j in range(len(group))]

            for ind, (m, base_str, dict_key, eval_key, seeds) in enumerate(group):

                xai_method = base_str.split('/')[-2]
                data_p, data_ratios, data_p_init = [], [], []

                # collect data over seeds
                for seed in seeds:
                    base_str_seed = base_str.format(seed, seed)
                    data_p, data_ratios, data_p_init = get_flip_data(dict_key, strategy, eval_key, base_str_seed,
                                                                     data_p,
                                                                     data_ratios, data_p_init)
                    if data_p is None:
                        missing_files.append(base_str_seed + "_flipping_{}.p".format(strategy))

                # plot dotted line of performance before removal of rationale tokans
                if i == 0:
                    if data_p_init is None:
                        continue
                    mean_p = np.mean(data_p_init)  # prob at the beginning are the same across_method_keys
                    # if plot_ratios:
                    #     ax.hlines(mean_p, xmin=-0.05, xmax=1.2, color=plot_dict[m][0], ls='--')
                    # else:
                    ax0.hlines(mean_p, xmin=-0.05, xmax=1.2, color=plot_dict[m][0], ls='--')
                labelstr = m if i == 0 else None

                # mark random baseline with separate hatch
                if group_name == 'random' and dict_key == 'relevance_lrp':
                    hatch = hatch_dict['baseline']
                else:
                    hatch = hatch_dict[xai_method]

                # plot probability after token removal
                if data_p is None:
                    continue

                ax0.bar(xs_group[ind], np.mean(data_p), capsize=6, width=bar_width, color=plot_dict[m],
                        linewidth=1., edgecolor='#616161', label=labelstr, hatch=hatch)
                jitter = np.random.normal(0, 0.001, len(data_p))
                ax0.scatter([xs_group[ind]] * len(data_p) + jitter, data_p, color='black', alpha=0.5, s=3)
                if plot_ratios:
                    # plot ratios of tokens removed
                    ax1.bar(xs_group[ind], np.mean(data_ratios), capsize=6, width=bar_width, color=plot_dict[m],
                            linewidth=1., edgecolor='#616161', label=labelstr, hatch=hatch)
                    jitter = np.random.normal(0, 0.001, len(data_ratios))
                    ax1.scatter([xs_group[ind]] * len(data_ratios) + jitter, data_ratios, color='black', alpha=0.5,
                                s=3)
                # else:
                #     ax.bar(xs_group[ind], np.mean(data_p), capsize=6, width=bar_width, color=plot_dict[m], linewidth=1.,
                #            edgecolor='#616161', label=labelstr, hatch=hatch)
                #     jitter = np.random.normal(0, 0.001, len(data_p))
                #     ax.scatter([xs_group[ind]] * len(data_p) + jitter, data_p, color='black', alpha=0.5, s=3)

            xticks.append(np.mean(xs_group))
            x0 = xs_group[-1] + x_extra

        if plot_ratios:
            ax1.set_xticks(xticks)
            ax1.set_xticklabels([x[0] for x in bar_groups], fontsize=13)  # rotation=45, ha="right",
            ax1.tick_params(axis='both', which='major', labelsize=13)
            ax1.tick_params(axis='both', which='minor', labelsize=13)
            ax1.spines[['right', 'top']].set_visible(False)

        ax0.set_xticks(xticks)
        ax0.set_xticklabels([x[0] for x in bar_groups], fontsize=13)  # rotation=45, ha="right",
        ax0.tick_params(axis='both', which='major', labelsize=13)
        ax0.tick_params(axis='both', which='minor', labelsize=13)

        ax0.set_ylabel('probability after masking', fontsize=13)
        ax0.spines[['right', 'top']].set_visible(False)

        lidx = 0

        axes = [ax0, ax1] if plot_ratios else [ax0]

        for ax_ in axes:
            if lidx == 0:
                h1, l1 = ax_.get_legend_handles_labels()
                import matplotlib.patches as mpatches

                extra_patches = [
                    # mpatches.Patch(color='none', label=''),
                    mpatches.Patch(facecolor='white', edgecolor='grey', hatch='///', label='abc'),
                    mpatches.Patch(facecolor='white', edgecolor='grey', hatch='...', label='def')]

                hs_, ls_ = h1 + extra_patches, l1 + ['contrastive', 'random'],

                ax_.legend(hs_, ls_,
                           bbox_to_anchor=(.5, 1.17, 0, 0),
                           loc=9, ncol=3, fontsize=10)

            else:
                import matplotlib.patches as mpatches

                empty_patch = mpatches.Patch(color='none', label='')
                h1, l1 = ax_.get_legend_handles_labels()
                ax_.legend([''], [empty_patch], bbox_to_anchor=(.5, 1.12, 0, 0),
                           loc=9, ncol=3, fontsize=11, frameon=False)

        if SAVE:
            f0.savefig(join(plot_dir, 'faithfulness_{}{}_{}.png'.format(task, lang, strategy)), dpi=300,
                       bbox_inches='tight')
            if plot_ratios:
                f1.savefig(join(plot_dir, 'ratios_{}{}_{}.png'.format(task, lang, strategy)), dpi=300,
                           bbox_inches='tight')
        plt.close()

        # comparison human to model

        ### group random
        group_human_random = [
            ((m, get_base_str(task, m, lang, sparsity, '{}', 'random'), 'human_rationales', 'rationales_human', seeds),
             (m, get_base_str(task, m, lang, sparsity, '{}', 'random'), 'relevance_lrp', 'relevance_lrp_binary', seeds))
            for m in models]

        ### group human
        group_human_model = [
            ((m, get_base_str(task, m, lang, sparsity, '{}', 'random'), 'human_rationales', 'rationales_human', seeds),
             (m, get_base_str(task, m, lang, sparsity, '{}', 'random'), 'model_rationales', 'rationales_model', seeds))
            for m in models]

        ### group human
        group_human_lrp = [
            ((m, get_base_str(task, m, lang, sparsity, '{}', 'random'), 'human_rationales', 'rationales_human', seeds),
             (m, get_base_str(task, m, lang, sparsity, '{}', 'lrp'), 'relevance_lrp', 'relevance_lrp_binary', seeds))
            for m in models if m not in ['mixtral']]

        ### group human
        group_human_lrp_contrast = [
            ((m, get_base_str(task, m, lang, sparsity, '{}', 'random'), 'human_rationales', 'rationales_human', seeds),
             (m, get_base_str(task, m, lang, sparsity, '{}', 'lrp_contrast'), 'relevance_lrp', 'relevance_lrp_binary',
              seeds))
            for m in models if m not in ['mixtral']]

        group_human_lrp_joined = join_groups(group_human_lrp, group_human_lrp_contrast)

        ### group model
        group_model_lrp = [
            ((m, get_base_str(task, m, lang, sparsity, '{}', 'random'), 'model_rationales', 'rationales_model', seeds),
             (m, get_base_str(task, m, lang, sparsity, '{}', 'lrp'), 'relevance_lrp', 'relevance_lrp_binary', seeds))
            for m in models if m not in ['mixtral']]

        ### group model
        group_model_lrp_contrast = [
            ((m, get_base_str(task, m, lang, sparsity, '{}', 'random'), 'model_rationales', 'rationales_model', seeds),
             (m, get_base_str(task, m, lang, sparsity, '{}', 'lrp_contrast'), 'relevance_lrp', 'relevance_lrp_binary',
              seeds))
            for m in models if m not in ['mixtral']]

        group_model_lrp_joined = join_groups(group_model_lrp, group_model_lrp_contrast)

        bar_groups_f1 = [('human X random', group_human_random),
                         ('human X model', group_human_model),
                         ('human X post-hoc', group_human_lrp_joined),
                         ('model X post-hoc', group_model_lrp_joined),
                         ]

        from sklearn.metrics import f1_score, recall_score, cohen_kappa_score

        f, ax = plt.subplots(1, 1, figsize=(3, 3))
        f_contrast, ax_contrast = plt.subplots(1, 1, figsize=(3, 3))

        xs = []
        xticks = []
        models_contrast = [model + '_contrast' for model in models if model not in ['mixtral']]
        model_labels = [None] * (len(models) - 1 + len(models_contrast))
        model_labels[::2] = [model for model in models if model not in ['mixtral']]
        model_labels[1::2] = models_contrast
        df_plausibility = pd.DataFrame(columns=models, index=[g[0] for g in bar_groups_f1[1:]])
        df_plausibility_contrastive = pd.DataFrame(
            columns=model_labels, index=[g[0] for g in bar_groups_f1[1:] if g[0] != 'human X model'])

        for i, (group_name, group) in enumerate(bar_groups_f1):
            print(group_name)

            xs_group = [x0 + j * bar_width for j in range(len(group))]
            f1_scores_random = []
            cmap = plt.get_cmap("tab20b_r")

            for ind, (comp1, comp2) in enumerate(group):
                f1_scores = []
                # print(comp1, comp2)
                m = comp1[0]
                xai_method = comp2[1].split('/')[-2]

                data1, missing = get_f1_data(comp1, strategy)
                missing_files.extend(missing)
                data2, missing = get_f1_data(comp2, strategy)
                missing_files.extend(missing)
                if np.sum([1 for d1, d2 in zip(data1, data2) if len(d1) != len(d2)]) > 1:
                    print('not same length:', np.sum([1 for d1, d2 in zip(data1, data2) if len(d1) != len(d2)]))
                    # for (i1, d1), d2 in zip(enumerate(data1), data2):
                    #     if len(d1) != len(d2):
                    #         print(i1, len(d1), len(d2))
                    if comp1[0] == comp2[0] == 'llama' and seed == 96 and strategy == 'human_rationale' \
                            and group_name in ['human X post-hoc', 'model X post-hoc'] and lang == '_article_8' \
                            and comp1[1].split('/')[6] == 'random' and comp2[1].split('/')[6] == 'lrp_contrast':
                        data1.insert(24, [0] * len(data1[0]))
                        data2.insert(104, [0] * len(data1[0]))
                    elif comp1[0] == comp2[0] == 'llama3' and seed == 96 and strategy == 'human_rationale' \
                            and group_name in ['human X post-hoc', 'model X post-hoc'] and lang == '_article_2' \
                            and comp1[1].split('/')[6] == 'random' and comp2[1].split('/')[6] == 'lrp_contrast':
                        data1.insert(618, [0] * len(data1[0]))
                        data2.insert(305, [0] * len(data1[0]))
                if plausibility_metric == 'recall':
                    f1_scores.extend([recall_score(d1, d2) for d1, d2 in zip(data1, data2) if len(d1) == len(d2)])
                elif plausibility_metric == 'f1':
                    f1_scores.extend([f1_score(d1, d2) for d1, d2 in zip(data1, data2) if len(d1) == len(d2)])
                elif plausibility_metric == 'kappa':
                    f1_scores.extend([cohen_kappa_score(d1, d2, labels=[0, 1])
                                      if len(d1) == len(d2) and np.sum(d1) > 0 and np.sum(d2) > 0 else 0
                                      for d1, d2 in zip(data1, data2)])
                    # tmp = [cohen_kappa_score(d1, d2, labels=[0, 1])
                    #                   if len(d1) == len(d2) and np.sum(d1) > 0 and np.sum(d2) > 0 else 0
                    #                   for d1, d2 in zip(data1, data2)]
                    # if np.mean(tmp) != np.nanmean(tmp):
                    #     import pdb;pdb.set_trace()
                else:
                    raise NotImplementedError(f"plausibility_metric {plausibility_metric} not in [recall,f1,kappa]")
                if np.sum([1 for d1, d2 in zip(data1, data2) if len(d1) != len(d2)]) > 1:
                    print('not same length:', np.sum([1 for d1, d2 in zip(data1, data2) if len(d1) != len(d2)]))

                if group == group_human_random:
                    f1_scores_random.extend(f1_scores)
                    # if ind == len(group) - 1:
                    #     ax.axline((xs_group[0], np.mean(f1_scores_random)), slope=0, color='black', linestyle='--',
                    #               linewidth=1.5)

                else:
                    labelstr = m if i == 1 else None
                    # if group_name == 'human X model':
                    #     print(lang, m, np.around(np.nanmean(f1_scores), decimals=2))
                    # ax.bar(xs_group[ind], np.nanmean(f1_scores), capsize=6, width=bar_width, color=cmap(models.index(m)),
                    #        linewidth=1., edgecolor='#616161', label=labelstr, hatch=hatch_dict[xai_method])
                    if 'contrast' in xai_method:
                        df_plausibility_contrastive.loc[group_name, m + '_contrast'] = np.nanmean(f1_scores)
                    else:
                        df_plausibility.loc[group_name, m] = np.nanmean(f1_scores)
                        if group_name not in ['human X model']:
                            df_plausibility_contrastive.loc[group_name, m] = np.nanmean(f1_scores)

                    # jitter = np.random.normal(0, 0.001, len(f1_scores))
                    # ax.scatter([xs_group[ind]] * len(f1_scores) + jitter, f1_scores, color='black', alpha=0.5, s=3)

            xticks.append(np.mean(xs_group))
            x0 = xs_group[-1] + x_extra

        # import pdb;
        # pdb.set_trace()
        # cbar = True if lang in ['_IT', '_article_8'] else False
        # if cbar:
        #     cbar_ax = f.add_axes([.91, .3, .01, .4])
        # else:
        #     cbar_ax = None
        sns.heatmap(df_plausibility.astype('float').values, ax=ax, mask=df_plausibility.isnull().values, fmt=".2f",
                    cmap='rocket', annot=df_plausibility.values, square=True, cbar=False, vmin=0, vmax=0.6)

        sns.heatmap(df_plausibility_contrastive.astype('float').values, ax=ax_contrast,
                    mask=df_plausibility_contrastive.isnull().values, fmt=".2f",
                    cmap='rocket', annot=df_plausibility_contrastive.values, square=True, cbar=False, vmin=0, vmax=0.6)
        # cbar=cbar, cbar_ax=cbar_ax)

        for ax_i, df_i in zip([ax, ax_contrast], [df_plausibility, df_plausibility_contrastive]):
            ax_i.set_yticks(np.arange(0, len(df_i)) + 0.5)
            if lang in ['', '_article_1']:
                ax_i.set_yticklabels(df_i.index, rotation=0)
            else:
                ax_i.set_yticklabels([])
            # if iroc == len(categories) - 1:
            ax_i.set_xticks(np.arange(0, len(df_i.columns)) + 0.5)

        ax.set_xticklabels(df_plausibility.columns, rotation=0, ha='center')
        ax_contrast.set_xticklabels(df_plausibility_contrastive.columns, rotation=90, ha='right')

        # ax.set_xticks(xticks)
        # ax.set_xlim([xticks[1] - x_extra, xticks[-1] + x_extra])
        # ax.set_xticklabels([x[0] for x in bar_groups_f1], fontsize=13, rotation=45, ha="right")
        # ax.tick_params(axis='both', which='major', labelsize=13)
        # ax.tick_params(axis='both', which='minor', labelsize=13)
        #
        # ax.set_ylabel(plausibility_metric, fontsize=13)
        # ax.spines[['right', 'top']].set_visible(False)

        # for ax_ in [ax]:
        #     if lidx == 0:
        #         h1, l1 = ax_.get_legend_handles_labels()
        #         import matplotlib.patches as mpatches
        #
        #         extra_patches = [  # mpatches.Patch(color='none', label=''),
        #             mpatches.Patch(facecolor='white', edgecolor='grey', hatch='///', label='abc'),
        #             mpatches.Patch(facecolor='white', edgecolor='grey', hatch='...', label='def')]
        #
        #         ax_.legend(h1 + extra_patches, l1 + ['contrastive', 'random'],
        #                    bbox_to_anchor=(.5, 1.17, 0, 0),
        #                    loc=9, ncol=3, fontsize=10)
        #
        #     else:
        #         import matplotlib.patches as mpatches
        #
        #         empty_patch = mpatches.Patch(color='none', label='')
        #         h1, l1 = ax_.get_legend_handles_labels()
        #         ax_.legend([''], [empty_patch], bbox_to_anchor=(.5, 1.12, 0, 0),
        #                    loc=9, ncol=3, fontsize=11, frameon=False)

        if SAVE:
            f.savefig(join(plot_dir, 'plausibility_{}{}_{}.png'.format(task, lang, plausibility_metric)), dpi=300,
                      bbox_inches='tight')
            f_contrast.savefig(
                join(plot_dir, 'plausibility_{}{}_{}_contrastive.png'.format(task, lang, plausibility_metric)), dpi=300,
                bbox_inches='tight')
        plt.close()

    print('Missing files')
    for m in missing_files:
        print(m)