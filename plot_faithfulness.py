import pickle
import json
import numpy as np
import os
from plot.plot_evaluation import format_key, get_json, plot_hist, get_flip_data, get_f1_data
import matplotlib.pyplot as plt
from os import makedirs
from os.path import join, exists, isfile
from plot.plot_specs import plot_dict,plot_dict2, hatch_dict, replace_rules
from utils import get_base_str


def join_groups(group1, group2):
    group_joined = []
    for a, b in zip(group1, group2):
        group_joined.extend([a, b])
    return group_joined

def jitter(ax, data, xs):
    jitter_ = np.random.normal(0, 0.001, len(data))
    ax.scatter([xs] * len(data) + jitter_, data, color='black', alpha=0.4, s=0.5)


SAVE = True
jitter_flag = True

models = ['llama', 'llama3', 'mistral',  'mixtral']
seeds = [28, 79, 96]
strategies = ['human_rationale']
plausibility_metric = 'f1'


for task in ['forced_labour', 'sst', 'sst_multilingual']:

    missing_files = []

    if task == 'sst':
        sparsity = 'oracle'
        loops = ['', ]  # , '_IT']
    
    elif task == 'sst_multilingual':
        sparsity = 'oracle'
        loops = ['_EN', '_DK', '_IT']  # , '_IT']
    
    elif task == 'forced_labour':
        sparsity = 'full'
        loops = ['_article_1', '_article_2', '_article_5', '_article_8']
        # loops = ['_article_2', '_article_8']
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
    
        plot_dir = 'model_responses/plots/{}'.format(task)
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
                        data_p, data_ratios, data_p_init = get_flip_data(dict_key, strategy, eval_key, base_str_seed, data_p,
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
    
    
                    if True:
    
                        def get_clipped_std(data):
                            mean_acc = np.mean(data)
                            std_acc = np.std(data)
                            
                            # Clip the upper bound to 1 (and lower bound to 0 if needed)
                            upper_bound = min(mean_acc + std_acc, 1)
                            lower_bound = max(mean_acc - std_acc, 0)
                            yerr = [[mean_acc - lower_bound], [upper_bound - mean_acc]]
                            #return yerr
                            return None
                        
                        ax0.bar(xs_group[ind], np.mean(data_p), yerr=get_clipped_std(data_p), width=bar_width, color=plot_dict[m], linewidth=1.,
                                   edgecolor='#616161', label=labelstr, hatch=hatch, error_kw=dict(lw=1, capsize=3, capthick=1))
                    else:
                        # Plot the violin plot
                        violin_parts = ax0.violinplot(
                            dataset=data_p,
                            positions=[xs_group[ind]],    # Position of the violin on the x-axis
                            widths=bar_width,             # Width of the violin plot
                            showmeans=True,               # Show the mean value
                            showextrema=False,             # Show the extrema
                            showmedians=False             # Optionally show the median
                        )
    
                        # Customize the violin plot appearance
                        for pc in violin_parts['bodies']:
                            pc.set_facecolor(plot_dict[m])      # Set the face color to match your color mapping
                            pc.set_edgecolor('#616161')         # Set the edge color
                            pc.set_alpha(1)                     # Set the transparency (1 is opaque)
                        
                        # Customize the mean line
                        if 'cmeans' in violin_parts:
                            violin_parts['cmeans'].set_color('black')
                            violin_parts['cmeans'].set_linewidth(1.0)
                        
                        # Customize the lines for extrema (whiskers)
                        if 'cmaxes' in violin_parts and 'cmins' in violin_parts:
                            violin_parts['cmaxes'].set_color('#616161')
                            violin_parts['cmaxes'].set_linewidth(1.0)
                            violin_parts['cmins'].set_color('#616161')
                            violin_parts['cmins'].set_linewidth(1.0)
                        
                                                                
                    if jitter_flag:
                        jitter(ax0, data_p, xs_group[ind])
    
                        
                    if plot_ratios:
                        # plot ratios of tokens removed
                        ax1.bar(xs_group[ind], np.mean(data_ratios), capsize=6, width=bar_width, color=plot_dict[m],
                                   linewidth=1., edgecolor='#616161', label=labelstr, hatch=hatch)
    
                        if jitter_flag:
    
                            jitter(ax1, data_ratios, xs_group[ind])
    
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
        
            axes = [ax0, ax1] if plot_ratios else [ax0]
    
            for ax_ in axes:
                if task in ['sst'] or lang in ['_article_1']:
                    h1, l1 = ax_.get_legend_handles_labels()
                    import matplotlib.patches as mpatches
                    from matplotlib.lines import Line2D

    
                    extra_patches = [
                        # mpatches.Patch(color='none', label=''),
                        mpatches.Patch(facecolor='white', edgecolor='grey', hatch='///', label='abc'),
                        Line2D([0], [0], color='grey', linestyle='--'),

                    ]
    
                    hs_, ls_ = h1 + extra_patches, l1 + ['contrastive',  r'probability $p_0$']
    
                    ax_.legend(hs_, ls_,
                               bbox_to_anchor=(.5, 1.17, 0, 0),
                               loc=9, ncol=3, fontsize=10)
    
                else:
                    import matplotlib.patches as mpatches
    
                    empty_patch = mpatches.Patch(color='none', label='')
                    h1, l1 = ax_.get_legend_handles_labels()
                    ax_.legend([''], [empty_patch], bbox_to_anchor=(.5, 1.17, 0, 0),
                               loc=9, ncol=3, fontsize=10, frameon=False)
    
            if SAVE:
                f0.savefig(join(plot_dir, 'faithfulness_{}{}_{}.png'.format(task, lang, strategy)), dpi=300,
                           bbox_inches='tight')
                if plot_ratios:
                    f1.savefig(join(plot_dir, 'ratios_{}{}_{}.png'.format(task, lang, strategy)), dpi=300,
                               bbox_inches='tight')
            plt.close()
    
    
        print('Missing files')
        for m in missing_files:
            print(m)
