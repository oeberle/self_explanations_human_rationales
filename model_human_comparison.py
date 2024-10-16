import json
import numpy as np
import sys
import os
from sklearn.metrics import f1_score, recall_score, cohen_kappa_score
import spacy
from evaluation.utils import normalize_responses, normalized_text, binarize_overlap, \
    check_fuzzy_substring, create_rationale_list, fix_syntax
from plot.plot_evaluation import main as plot_main
import traceback
from model_utils import get_model
from evaluation.utils import align_model_to_rationale, align_rationale_to_model
import torch
from os.path import join, exists, isfile
from os import makedirs
import pickle
import argparse
from xai.flipping import flip_simple, flip, threshold_relevances


def get_aligned_rationales(tokens_raw, tokens_normalized, scores_normalized):
    '''
    tokens_raw = ['The', "movie's", '--', 'plot']
    tokens_normalized = ['the', "movie", "s", 'plot']

    output: ['the', 'movie*s', '*', 'plot']
    '''
    tokens_raw = [t.lower() for t in tokens_raw]

    # Get the map
    inds_map = {}
    token_map = {}
    skip_ids = []
    i_ = 0
    for i, t in enumerate(tokens_raw):
        text_norm = normalized_text(t).split(' ')
        if text_norm == ['']:
            skip_ids.append(i)
        else:
            inds_new = list(range(i_, i_ + len(text_norm)))
            inds_map[i_] = inds_new
            token_map[t] = text_norm
            i_ += len(inds_new)
    tokens_reconstructed = []
    scores_normalized_reconstructed = []

    for k, v in inds_map.items():
        if v is None:
            continue
        else:
            k_ = list(v)
            if len(k_) == 1:
                tokens_reconstructed.extend(np.array(tokens_normalized)[k_].tolist())
                scores_normalized_reconstructed.extend(np.array(scores_normalized)[k_].tolist())
            else:
                tokens_reconstructed.append('*'.join(np.array(tokens_normalized)[k_].tolist()))
                scores_normalized_reconstructed.append(sum(np.array(scores_normalized)[k_].tolist()))

    for ind in skip_ids:
        _ = tokens_reconstructed.insert(ind, '*')
        _ = scores_normalized_reconstructed.insert(ind, 0.)

    assert len(tokens_reconstructed) == len(scores_normalized_reconstructed) == len(tokens_raw)

    assert sum(scores_normalized_reconstructed) == sum(scores_normalized)
    return tokens_reconstructed, np.array(scores_normalized_reconstructed)


def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument('--src_dir', default='model_responses')
    parser.add_argument('--dataset_name', default='forced_labour', help='Dataset name')
    parser.add_argument('--model_name_short', default='all', help='Model name (short)')
    parser.add_argument('--xai_strategy', default='all', help='contrastive, lrp, random')
    parser.add_argument('--loops', default='all', help='languages (SST) or article_ids (forced_labour)')
    parser.add_argument('--select_strategy', default='one_over_n', help='selection strategy for xai')
    parser.add_argument('--sparsity', default='full', help='How to prompt for rationales {full, phrases, words}')
    parser.add_argument('--seed', default=28, help='Set a seed for reproducibility')
    parser.add_argument('--flipping', action=argparse.BooleanOptionalAction, help='Whether to plot heatmaps')

    config = parser.parse_args()
    dataset_name = config.dataset_name
    select_strategy = config.select_strategy
    sparsity = config.sparsity
    FLIPPING = config.flipping

    DATA_DIR = config.src_dir
    RES_DIR = join(DATA_DIR, 'eval_results')

    if not exists(RES_DIR):
        makedirs(RES_DIR)

    if config.model_name_short == 'all':
        models_short = ['llama', 'llama3', 'mistral', 'mixtral']
    else:
        models_short = [config.model_name_short]

    if config.xai_strategy == 'all':
        xai_strategies = ['contrastive', 'lrp', 'random']
    else:
        xai_strategies = [config.xai_strategy]

    models = [model + '_' + xai for model in models_short for xai in xai_strategies if model != 'mixtral']

    # mixtral random 
    if 'mixtral' in models_short:
        models.append('mixtral_random')

    
    # # allow for different XAI methods to be used can also be none (e.g. for mixtral)
    # models = ['llama3_lrp', 'llama3_lrp_contrast', 'llama3_random']
    # models = ['mistral_lrp', 'mistral_lrp_contrast', 'mistral_random']

    if config.loops == 'all':
        languages = ['EN1', 'EN2', 'DK', 'IT']
        article_ids = ['1', '2', '5', '8']
        loop_over = article_ids if dataset_name == 'forced_labour' else languages
    else:
        loop_over = [config.loops]


    print('loop', loop_over) 
    
    SHUFFLE = False
    PLOT_EXAMPLES = False
    overall_accuracy = {key: [] for key in loop_over}
    overall_f1 = {key: [] for key in loop_over}
    overall_recall = {key: [] for key in loop_over}
    overall_rationale_accuracy = {key: [] for key in loop_over}
    overall_rationale_count = {key: [] for key in loop_over}
    overall_perturbation_score = {key: {} for key in loop_over}
    ratio_tokens_human = {key: [] for key in loop_over}
    ratio_tokens_model = {key: [] for key in loop_over}

    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # plot_idx = torch.randint(0, 250, (10,))
    plot_idx = [10, 4, 11, 166, 130, 248, 175, 185, 232, 220]
    print(plot_idx)

    for ii, loop in enumerate(loop_over):

        if dataset_name == 'forced_labour':
            print(f'----------------------Article {loop}----------------------')
            multilingual = None
        else:
            print(f'----------------------Language {loop}----------------------')
      #      multilingual = True if loop in ['EN2', 'DK', 'IT'] else False
            multilingual = True if dataset_name == 'sst_multilingual' else False
            
            if loop == 'DK':
                nlp = spacy.load("da_core_news_sm")
            elif 'EN' in loop:
                nlp = spacy.load("en_core_web_sm")
            elif loop == 'IT':
                nlp = spacy.load("it_core_news_sm")
            else:
                raise
                
        for model_name in models:

            model_name_xai = model_name
            model_name = model_name.split('_')[0]
            xai_method = model_name_xai.replace(model_name + '_', '')

            
            if len(xai_method) > 0:
                DATA_DIR_DATASET = join(RES_DIR, config.dataset_name, model_name, config.sparsity, config.seed,
                                        xai_method)
            else:
                DATA_DIR_DATASET = join(RES_DIR, config.dataset_name, model_name, config.sparsity, config.seed)

           # DATA_DIR_DATASET = DATA_DIR_DATASET.replace('sst', 'sst_multilingual' if multilingual else 'sst')
            
            if not exists(DATA_DIR_DATASET):
                makedirs(DATA_DIR_DATASET)

            if FLIPPING:
                model, tokenizer = get_model(model_name)

            results = []
            misses = []
            match = []
            skip = 0
            true_answer = []
            extracted_answer = []
            rationale_match = []
            f1_scores = []
            cohen_kappa_scores = []
            recall_scores = []
            rationale_count = 0
            n_tokens_humans = []
            n_tokens_models = []

            perturbation_scores = {'relevance_lrp': [],
                                   'human_rationales': [],
                                   'model_rationales': [],
                                   'baseline': []}

            syntax_error = 0
            different_length = 0
            index_error = 0
            rationale_align_error = 0
            rationale_align_error2 = 0

            try:

                # join(DATA_DIR, config.dataset_name, config.model_name_short, config.sparsity, config.seed)  

                filename_root = f'./{DATA_DIR}/{dataset_name}/{model_name_xai}/{config.sparsity}/{config.seed}'
                
                filename = f'{filename_root}/{model_name}_rationales_seed_{config.seed}.jsonl' if dataset_name in ['sst', 'sst_multilingual']  \
                    else f'{filename_root}/{model_name}_rationales_article_{loop}_seed_{config.seed}.jsonl'


                if dataset_name == 'sst':
                    filename = f'{filename_root}/{model_name}_rationales_seed_{config.seed}.jsonl'
                elif  dataset_name == 'sst_multilingual':
                    filename = f'{filename_root}/{model_name}_rationales_{loop}_seed_{config.seed}.jsonl'
                else:
                    filename = f'{filename_root}/{model_name}_rationales_article_{loop}_seed_{config.seed}.jsonl'

                filename = filename.replace('_seed', '_quant_seed') if model_name == 'mixtral' else filename

                
                #if dataset_name == 'sst':
                #    filename = filename.replace('sst', 'sst_multilingual') if multilingual else filename
                #    filename = filename.replace('rationales', f'rationales_{loop[:2]}') if multilingual else filename

                
                filename = filename.replace('.jsonl', '_shuffle.jsonl') if SHUFFLE else filename

                if sparsity != '':
                    filename = filename.replace('.jsonl',
                                                f'_sparsity_{sparsity}.jsonl') if sparsity != 'full' else filename


                if 'mixtral' != model_name and xai_method=='random':
                    # load lrp as the reference file and overwrite the lrp scores
                    print('Creating random', filename)
                    filename_to_load = filename.replace('random', 'lrp')
                    print('\n', filename_to_load)

                else:
                    filename_to_load = filename
                    print('\n', filename)

                    
        
                examples_out = []
                scores_out = {}

                with open(filename_to_load) as f:
                    for idx, example in enumerate(f.readlines()):
                        example_out = {}
                        example = json.loads(example)
                        example = normalize_responses(example, idx, shuffle=SHUFFLE)
                        res = 1 if example["true_label"] == example[f"normalized_response_{idx}"] else 0
                        results.append(res)
                        miss = 1 if example[f"normalized_response_{idx}"] not in [0, 1] else 0
                        misses.append(miss)
                        extracted_rationales = []
                       # if dataset_name == 'sst':
                       #  if dataset_name in ['sst', 'sst_multilingual']:
                        rationales_key = 'rationales_checked' if 'rationales_checked' in example else 'rationales'
                        # else:
                        #     rationales_key = 'rationales'

                        if rationales_key in example:
                            if dataset_name == 'forced_labour':
                                gold_label_rationale = [el for gold_label in example['gold_label_rationales'] for el in
                                                        gold_label if gold_label[0].startswith(f"{int(loop):02}")]
                                mask_gold = np.array(
                                    [0 for _ in range(len(normalized_text(example['content']).split(" ")))])
                                mask_predicted = np.array(
                                    [0 for _ in range(len(normalized_text(example['content']).split(" ")))])
                                for gold_rationale in gold_label_rationale[1:]:
                                    if gold_rationale.startswith(f"{int(loop):02}."):
                                        continue
                                    mask_gold = mask_gold + np.array(
                                        binarize_overlap(example['content'], gold_rationale))

                            else:
                                gold_rationale_idx = example['gold_label_rationales']
                                mask_gold = np.array(
                                    [0 for _ in range(len(normalized_text(example['content']).split(" ")))])
                                mask_predicted = np.array(
                                    [0 for _ in range(len(normalized_text(example['content']).split(" ")))])
                                if (len(example['content'].split(" ")) != len(gold_rationale_idx)) and multilingual:
                                    different_length += 1
                                    example['content'] = " ".join([word.text for word in nlp(example['content'])])

                            rationale_tokens = example['content'].split(" ")
                            rationale_tokens_normalized = normalized_text(example['content']).split(" ")

                            try:
                                eval(example[rationales_key])
                            except (NameError, SyntaxError):
                                syntax_error += 1
                                example = fix_syntax(example, traceback, rationales_key)

                            try:
                                if isinstance(eval(example[rationales_key]), dict):
                                    example[rationales_key] = '[' + example[rationales_key] + ']'
                            except (NameError, SyntaxError):
                                rationales_key = 'rationales'
                                print(rationales_key, 'rationales')

                            try:
                                for rationale in eval(example[rationales_key]):
                                    if isinstance(rationale, str):
                                        rationale = eval(rationale)
                                    if 'rationals' in rationale:
                                        rationale['rationales'] = rationale['rationals']
                                    elif 'rationale' in rationale:
                                        rationale['rationales'] = rationale['rationale']
                                    elif '?rationale' in rationale:
                                        rationale['rationales'] = rationale['?rationale']
                                    if 'rationales' not in rationale:
                                        if isinstance(rationale, tuple):
                                            extracted_rationales.extend(rationale[0])
                                        elif isinstance(rationale, str):
                                            extracted_rationales.append(rationale)
                                        match.append(check_fuzzy_substring(example['content'], extracted))
                                        if check_fuzzy_substring(example['content'], extracted):
                                            mask_predicted = mask_predicted + np.array(
                                                binarize_overlap(example['content'], extracted))
                                    else:
                                        extracted = rationale['rationales']
                                        if isinstance(extracted, list):
                                            extracted_rationales.extend(extracted)
                                            for ii_extracted in extracted:
                                                if not isinstance(ii_extracted, str):
                                                    ii_extracted = ii_extracted['text']
                                                match.append(check_fuzzy_substring(example['content'], ii_extracted))
                                                if check_fuzzy_substring(example['content'], ii_extracted):
                                                    mask_predicted = mask_predicted + np.array(
                                                        binarize_overlap(example['content'], ii_extracted))
                                        elif isinstance(extracted, str):
                                            extracted_rationales.append(extracted)
                                            match.append(check_fuzzy_substring(example['content'], extracted))
                                            if check_fuzzy_substring(example['content'], extracted):
                                                mask_predicted = mask_predicted + np.array(
                                                    binarize_overlap(example['content'], extracted))
                                        else:
                                            skip += 1
                                            continue
                            #    if dataset_name == 'sst':
                                if dataset_name in ['sst', 'sst_multilingual']:
                                    gold_label_rationale = create_rationale_list(gold_rationale_idx,
                                                                                 example['content'].split(" "))
                                for gold_rationale in gold_label_rationale:
                                    if dataset_name == 'forced_labour' and gold_rationale.startswith(
                                            f"{int(loop):02}."):
                                        continue

                                    mask_gold = mask_gold + np.array(
                                        binarize_overlap(example['content'], gold_rationale))
                                mask_gold = np.clip(mask_gold, 0, 1)
                                mask_predicted = np.clip(mask_predicted, 0, 1)
                                # f1_scores.append(f1_score(np.clip(mask_gold, 0, 1), np.clip(mask_predicted, 0, 1)))
                                # recall_scores.append(recall_score(np.clip(mask_gold, 0, 1), np.clip(mask_predicted, 0, 1)))
                                # rationale_count += 1
                                # n_tokens_models.append(np.sum(mask_predicted) / len(mask_predicted))
                                # n_tokens_humans.append(np.sum(mask_gold) / len(mask_gold))
                                #### RE_ALIGN
                                # print(len(mask_gold), len(rationale_tokens_normalized), len(rationale_tokens))

                                if len(rationale_tokens) != rationale_tokens_normalized:

                                    #  print(rationale_tokens)
                                    #  print(rationale_tokens_normalized)
                                    #  print(mask_predicted)

                                    try:
                                        _, mask_predicted = get_aligned_rationales(rationale_tokens,
                                                                                   rationale_tokens_normalized,
                                                                                   mask_predicted)
                                        _, mask_gold = get_aligned_rationales(rationale_tokens,
                                                                              rationale_tokens_normalized, mask_gold)
                                    except IndexError:
                                        rationale_align_error2 += 1
                                        continue

                                example_out['content'] = example['content']
                                example_out['tokens_pre_alignment'] = rationale_tokens
                                example_out['rationales_human_pre_alignment'] = mask_gold.tolist()
                                example_out['rationales_model_pre_alignment'] = mask_predicted.tolist()
                                #### RE_ALIGN

                                f1_scores.append(f1_score(np.clip(mask_gold, 0, 1), np.clip(mask_predicted, 0, 1)))
                                cohen_kappa_scores.append(
                                    cohen_kappa_score(np.clip(mask_gold, 0, 1), np.clip(mask_predicted, 0, 1),
                                                      labels=[0, 1]))
                                recall_scores.append(
                                    recall_score(np.clip(mask_gold, 0, 1), np.clip(mask_predicted, 0, 1)))
                                rationale_count += 1
                                n_tokens_models.append(np.sum(mask_predicted) / len(mask_predicted))
                                n_tokens_humans.append(np.sum(mask_gold) / len(mask_gold))

                                # if PLOT_EXAMPLES and idx in plot_idx:
                                #     if len(example['content'].split(" ")) == len(mask_gold):
                                #         fig, axs = plt.subplots(2, 1, figsize=(20, 3))
                                #         sns.heatmap(np.array(np.clip(mask_gold, 0, 1))[None, :],
                                #                     annot=np.array(example['content'].split(" "))[None, :],
                                #                     fmt='', ax=axs[0], cbar=False, vmin=0, vmax=1)
                                #         axs[0].set_title('Human annotations')
                                #         axs[0].set_xticks([])
                                #         axs[0].set_yticks([])
                                #
                                #         cbar_ax = fig.add_axes([0.92, .3, .01, .4])
                                #
                                #         # if loop == 'DK' and idx == 4 and model == 'mixtral':
                                #         #     import pdb; pdb.set_trace()
                                #
                                #         sns.heatmap(np.clip(mask_predicted, 0, 1)[None, :],
                                #                     annot=np.array(example['content'].split(" "))[None, :],
                                #                     fmt='', ax=axs[1], cbar=True, cbar_ax=cbar_ax, vmin=0, vmax=1)
                                #         axs[1].set_title(f'Generated annotations by {model}')
                                #         axs[1].set_xticks([])
                                #         axs[1].set_yticks([])
                                #         plt.savefig(f'figs/{idx}_{loop}_{model}.png')
                                #         plt.close()

                            except (SyntaxError, KeyError, NameError, TypeError, IndexError):
                                skip += 1
                                exc_type, exc_obj, exc_tb = sys.exc_info()
                                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                # print(fname, exc_tb.tb_lineno, exc_type, exc_obj)
                                continue

                            if FLIPPING and 'perturbation' in example:
                                # Flipping (I think if we make it until here we have both human and model rationales, right?
                                # model tokenization
                                answer_id = example['perturbation']['answer']

                                if 'llama' in model_name:
                                    replace_token_id = int(tokenizer("_", add_special_tokens=False).input_ids[0])
                                elif 'mistral' in model_name:
                                    replace_token_id = tokenizer.unk_token_id
                                elif 'mixtral' in model_name:
                                    replace_token_id = tokenizer.unk_token_id
                                else:
                                    raise

                                #  replace_token_id = int(tokenizer("_", add_special_tokens=False).input_ids[0])

                                input_ids = example['perturbation']['tokenized_chat_until_answer']
                                context_mask = np.array(example['perturbation']['context_mask'])
                                model_tokens = np.array(tokenizer.convert_ids_to_tokens(input_ids[0]))

                                example_out['context_mask'] = context_mask.tolist()
                                example_out['model_tokens'] = model_tokens.tolist()

                                if len(mask_gold) != len(rationale_tokens) or len(mask_predicted) != len(
                                        rationale_tokens):
                                    # something went wrong before with the alignment
                                    rationale_align_error += 1

                                    print(len(mask_gold), len(mask_predicted), len(rationale_tokens))
                                    continue

                                # gold_tokens_selected = np.array(rationale_tokens)[mask_gold == 1]
                                # model_tokens_selected = np.array(rationale_tokens)[mask_predicted >= 1]

                                # Aligning attribution scores to the rationales

                                if 'mixtral' != model_name and xai_method=='random':
                                    # use lrp as the ref case
                                    model_relevance = np.random.normal(0, 1, len(example['relevance_lrp'])).tolist()

                                    
                                else:
                                    model_relevance = example['relevance_{}'.format(xai_method)]

                                

                                # tokens_model_merged, relevance_model_merged = align_model_to_rationale(
                                # rationale_tokens, model_tokens, model_relevance)

                                # Aligning rationale scores to attribution vectors
                                _, gold_rationales = align_rationale_to_model(rationale_tokens, model_tokens, mask_gold)

                                _, model_rationales = align_rationale_to_model(rationale_tokens, model_tokens,
                                                                               mask_predicted)

                                example_out['tokens'] = rationale_tokens
                                example_out['rationales_human'] = gold_rationales.tolist()
                                example_out['rationales_model'] = model_rationales.tolist()
                                example_out['relevance_lrp'] = model_relevance  #maybe rename

                                for case, scores, mask_ in [('relevance_lrp', model_relevance, context_mask),
                                                            ('human_rationales', gold_rationales, None),
                                                            ('model_rationales', model_rationales, None),
                                                            ('baseline', context_mask, None)
                                                            ]:

                                    if 'relevance_lrp' in case:
                                        #    model_relevance = example['relevance_lrp']
                                        model_relevance = np.array(scores)
                                        model_relevance[mask_ != 1] = -10e6

                                        if select_strategy == 'percentile':
                                            relevance_binarized = threshold_relevances(np.array(model_relevance), p=80,
                                                                                       mask=mask_)

                                        elif select_strategy == 'max_rationale':
                                            n_most = int(max([gold_rationales.sum(), model_rationales.sum()]))
                                            relevance_binarized = np.zeros_like(model_relevance)
                                            relevance_binarized[np.argsort(model_relevance)[::-1][:n_most]] = 1.
                                            assert n_most == int(relevance_binarized.sum())

                                        elif select_strategy == 'human_rationale':
                                            n_most = int(gold_rationales.sum())
                                            relevance_binarized = np.zeros_like(model_relevance)
                                            relevance_binarized[np.argsort(model_relevance)[::-1][:n_most]] = 1.
                                            assert n_most == int(relevance_binarized.sum())

                                        elif select_strategy == 'model_rationale':
                                            n_most = int(model_rationales.sum())
                                            relevance_binarized = np.zeros_like(model_relevance)
                                            relevance_binarized[np.argsort(model_relevance)[::-1][:n_most]] = 1.
                                            assert n_most == int(relevance_binarized.sum())

                                        elif select_strategy == "one_over_n":
                                            relevance_binarized = np.zeros_like(model_relevance)
                                            # only consider positive scores
                                            r_ = np.clip(model_relevance, a_min=0., a_max=None)
                                            r_ = model_relevance / r_.sum()
                                            n = int(mask_.sum())  # len of context
                                            mask_threshold = r_ >= 1. / n
                                            mask_threshold = np.logical_and(mask_threshold, mask_ == 1)
                                            relevance_binarized[mask_threshold == 1] = 1.

                                        scores = relevance_binarized
                                        example_out['relevance_lrp_binary'] = relevance_binarized.tolist()

                                        examples_out.append(example_out)

                                    inputs = torch.tensor(input_ids).to('cuda')
                                    probs, logs = flip_simple(model, tokenizer, inputs, scores, answer_id,
                                                              replace_token_id)
                                    perturbation_scores[case].append((probs, logs, int(scores.sum())))

                rationales_filename = filename.split('.')[-2].split('/')[-1] + '_rationales'
                rationales_filename = rationales_filename.replace(f'{model_name}_rationales', f'{model_name}')
                flipping_filename = filename.split('.')[-2].split('/')[-1] + '_flipping'
                flipping_filename = flipping_filename.replace(f'{model_name}_rationales', f'{model_name}')

                if FLIPPING:
                    with open(join(DATA_DIR_DATASET, f'{dataset_name}_{rationales_filename}_{select_strategy}.jsonl'),
                              "w") as f:
                        for example in examples_out:
                            f.write(json.dumps(example) + "\n")
                    pickle.dump(perturbation_scores,
                                open(join(DATA_DIR_DATASET, f'{dataset_name}_{flipping_filename}_{select_strategy}.p'),
                                     'wb'))
                    overall_perturbation_score[loop][model_name] = perturbation_scores

                overall_accuracy[loop].append(np.around((sum(results) / len(results)), decimals=2))
                overall_rationale_count[loop].append(rationale_count)
                overall_f1[loop].append(np.around(np.mean(f1_scores), decimals=2))
                overall_recall[loop].append(np.around(np.mean(recall_scores), decimals=2))
                ratio_tokens_model[loop].append(np.around(np.mean(n_tokens_models), decimals=2))
                ratio_tokens_human[loop].append(np.around(np.mean(n_tokens_humans), decimals=2))

                print(f'{model_name}', 'acc:', np.around(100 * (sum(results) / len(results)), decimals=2))

                if len(match) == 0:
                    print(f'{model_name}', 'matches:', 0)
                    overall_rationale_accuracy[loop].append(0)
                else:
                    print(f'{model_name}', 'matches:', np.around(sum(match) / len(match), decimals=2))
                    overall_rationale_accuracy[loop].append(np.around(sum(match) / len(match), decimals=2))

                scores_out['model accuracy'] = np.around(100 * (sum(results) / len(results)), decimals=2)
                scores_out['f1'] = np.around(np.mean(f1_scores), decimals=2)
                scores_out['cappa'] = np.around(np.nanmean(cohen_kappa_scores), decimals=2)
                scores_out['recall'] = np.around(np.mean(recall_scores), decimals=2)
                scores_out['rationale_count'] = rationale_count
                scores_out['skip'] = skip
                scores_out['rationale_align_error'] = rationale_align_error
                scores_out['rationale_align_error_2'] = rationale_align_error2
                scores_out['ratio_tokens_model'] = np.around(np.mean(n_tokens_models), decimals=2)
                scores_out['ratio_tokens_human'] = np.around(np.mean(n_tokens_humans), decimals=2)

                score_filename = flipping_filename.replace("flipping", "scores")
                score_dir = join(DATA_DIR_DATASET,f'{dataset_name}_{score_filename}_{select_strategy}.p')
                print('Output file',score_dir)
                pickle.dump(scores_out, open(score_dir,'wb'))

                print(f'{model_name}', 'f1:', np.around(np.mean(f1_scores), decimals=2))
                print(f'{model_name}', 'kappa:', np.around(np.nanmean(cohen_kappa_scores), decimals=2))
                print(f'{model_name}', 'mispredicted:', len(results) - sum(results))
                print(f'{model_name}', 'rationale count:', rationale_count)
                print(f'{model_name}', 'skipped:', skip)
                print(f'{model_name}', 'different lengths:', different_length)
                print(f'{model_name}', 'syntax errors:', syntax_error)
                print(f'{model_name}', 'index errors:', index_error)
                print(f'{model_name}', 'rationale align errors:', rationale_align_error)
                print(f'{model_name}', 'ratio_tokens_model:', np.around(np.mean(n_tokens_models), decimals=2))
                print(f'{model_name}', 'ratio_tokens_human:', np.around(np.mean(n_tokens_humans), decimals=2))

            except FileNotFoundError:
                print(filename_to_load, 'FileNotFoundError')
                continue

    # plot_main(dataset_name, models, overall_accuracy, overall_f1, overall_recall, overall_rationale_accuracy,
    #           overall_rationale_count, ratio_tokens_model, ratio_tokens_human)
    # pickle.dump(overall_perturbation_score, open(join(RES_DIR, 'flipping_{}.p'.format(select_strategy)), 'wb'))


if __name__ == '__main__':
    main()
