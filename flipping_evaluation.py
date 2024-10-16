import pandas as pd
import numpy as np
import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
from os.path import join, exists, isfile
from os import makedirs
from datasets import load_dataset
import re
from xai.xai_mistral import  override_mistral_xai_layers
from xai.xai_llama import override_llama_xai_layers
from transformers.generation import StoppingCriteria, StoppingCriteriaList
import itertools
from evaluation.utils import normalize_responses, normalized_text

from plot.plot_utils import plot_heatmap
from xai.flipping import flip

import matplotlib.pyplot as plt


def merge_hyphens(tokens, importance):
    adjusted_tokens = []
    adjusted_importance = []
    initial_symbols = ["L", "l", "Dall", "dall", "all", "dell", "nell", "d", "C", "c", "sull", "un", "nient", "quest", "Un", "po"]
    indices = [i for i, x in enumerate(tokens) if x in initial_symbols and tokens[i+1] == "'"]

    if len(indices) > 0:
        i = 0
        while i <= len(tokens)-1:
            if i in indices and i + 2 < len(tokens):
                combined_token = tokens[i] + tokens[i + 1] + tokens[i + 2]
                combined_heat = importance[i] + importance[i + 1] + importance[i + 2]
                i += 3
                adjusted_tokens.append(combined_token)
                adjusted_importance.append(combined_heat)
            else:
                adjusted_tokens.append(tokens[i])
                adjusted_importance.append(importance[i])
                i += 1
        return adjusted_tokens, adjusted_importance

    else:
        return tokens, importance


def rat_num(len_sen, rationale):
    try:
        rationale = [el - 1 for el in list(map(int, rationale.split(",")))] if len(rationale.split(",")) > 1 else [
            int(float(rationale)) - 1]
        return [1 if idx in rationale else 0 for idx in range(len_sen)]
    except ValueError:
        import pdb;
        pdb.set_trace()


def prepare_sst_dataset():
    dataset = load_dataset("coastalcph/fair-rationales", 'sst2', trust_remote_code=True)
    df = pd.DataFrame(dataset['train'])
    df_ = df.groupby('originaldata_id')

    df_out = pd.DataFrame(columns=['id', 'sentence', 'label', 'rationale_numeric', 'rationale_binary'])

    labels = {0: 'negative', 1: 'positive'}
    labels_inv = {v: k for k, v in labels.items()}

    rationales = {}
    ii = 0
    for id, subdf in df_:
        rationales[id] = {}
        sentence = list(set(subdf['sentence']))
        assert len(sentence) == 1
        sentence = sentence[0]

        subdf = subdf.copy()
        subdf = subdf.query("original_label==label")

        subdf['len_sen'] = None
        subdf['len_sen'] = subdf['sentence'].apply(lambda x: len(x.split(" ")))

        rationales[id]['rationale_numeric'] = subdf.apply(lambda x: rat_num(x.len_sen, x.rationale_index), axis=1)
        rationales[id]['rationale_numeric'] = np.mean(rationales[id]['rationale_numeric'].tolist(), axis=0)
        rationales[id]['rationale_binary'] = [1 if rat >= .5 else 0 for rat in rationales[id]['rationale_numeric']]

        df_out.loc[ii] = [id, sentence, subdf['label'].iloc[0], rationales[id]['rationale_numeric'],
                          rationales[id]['rationale_binary']]
        ii += 1

    return df_out, labels


def prepare_sst_multilingual_dataset(language):
    if language == 'EN':
        labels = {0: 'negative', 1: 'positive'}
    elif language == 'DK':
        labels = {0: 'negativ', 1: 'positiv'}
    elif language == 'IT':
        labels = {0: 'negativa', 1: 'positiva'}

    dataset = pd.read_excel(f"./data/Multilingual_Interpretability_{language}.xlsx")
    # dataset['rationales'] = dataset['Span'].apply(
    #     lambda x: [word[:-3] for word in x.split("|") if word.endswith("(1)")])
    if language == 'EN':
        dataset['normalized'] = dataset['Sentence'].apply(lambda x: normalized_text(x))
    else:
        dataset['normalized'] = dataset['Original'].apply(lambda x: normalized_text(x))

    splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet',
              'test': 'data/test-00000-of-00001.parquet'}
    df_original = pd.read_parquet("hf://datasets/stanfordnlp/sst2/" + splits["validation"])
    df_original['normalized'] = df_original['sentence'].apply(lambda x: normalized_text(x))

    dataset['label'] = None
    dataset['rationale_binary'] = [[] for _ in range(len(dataset))]
    dataset['rationales'] = [[] for _ in range(len(dataset))]

    for idx, row in dataset.iterrows():
        normalized_sentence = row['normalized']
        if language == 'EN':
            dataset.loc[idx, 'rationale_binary'].extend([int(word[-2]) for word in row['Span'].split('|')])
            dataset.loc[idx, 'rationales'].extend([word[:-3] for word in row['Span'].split("|") if word.endswith("(1)")])
        elif language == 'DK':
            punctuation_dk = ['.', ';', ',', '?', '(', ')', ':', '!']
            span = [word[:-3] for word in row['Span'].split('|')
                    if word[:-3] not in punctuation_dk]
            span_rationale = [int(word[-2]) for word in row['Span'].split('|') if
                              word[:-3] not in punctuation_dk]
            assert len(span) == len(span_rationale)
            dataset.loc[idx, 'rationale_binary'].extend(span_rationale)
            dataset.loc[idx, 'rationales'].extend(span)
        elif language == 'IT':
            punctuation_it = ['(', ')', '.', ',', '...', '?', ':', '!', ';']
            span = " ".join([word[:-3] for word in dataset.loc[idx, 'Span'].split('|')
                             if word[:-3] not in punctuation_it])
            span_rationale = [int(word[-2]) for word in dataset.loc[idx, 'Span'].split('|') if
                              word[:-3] not in punctuation_it]

            tokens, importance = merge_hyphens(span.split(" "), span_rationale)
            assert len(tokens) == len(importance)
            dataset.loc[idx, 'Sentence'] = dataset.loc[idx, 'Sentence'].replace(' .', '.').replace("po' ", "po'")
            dataset.loc[idx, 'rationale_binary'].extend(importance)
            dataset.loc[idx, 'rationales'].extend(tokens)
            dataset.loc[idx, 'normalized'] = normalized_text(row['Sentence'])
        for idx_y, row_y in df_original.iterrows():
            normalized_sentence_y = row_y['normalized']
            if normalized_sentence in normalized_sentence_y:
                dataset.loc[idx, 'label'] = row_y['label']
                # assert dataset.loc[idx, 'normalized'] == normalized_sentence

    dataset = dataset.rename(columns={"Sentence": "sentence"})

    return dataset, labels


def prepare_labour_dataset(article_id):
    # True label: 1 (contains article_id)
    # True label: 0 (does not contain article_id)

    
    # Creating Python Object
    with open('./data/rationales_forced_labour_dataset.json', 'r') as json_file:
        dataset = json.load(json_file)

    # Creating DataFrame
    dataset_df = pd.DataFrame(dataset)
    df_labels = pd.DataFrame(dataset_df.labels.tolist(), index=dataset_df.index,
                             columns=[str(i + 1) for i in range(11)])

    dataset_df['labels_subset'] = df_labels[str(article_id)]

    with open('./data/definitions.json', 'r') as json_file:
        definitions = json.load(json_file)

    article_definition = definitions[article_id - 1]['definition']
    article_label = definitions[article_id - 1]['label']

    return dataset_df, article_label, article_definition


# Define the custom stopping criteria
class StopOnAnyTokenSequence(StoppingCriteria):
    def __init__(self, stop_sequences_ids):
        self.stop_sequences_ids = stop_sequences_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for seq in self.stop_sequences_ids:
            if input_ids.shape[1] >= len(seq):
                if torch.equal(input_ids[0, -len(seq):], torch.tensor(seq, device=input_ids.device)):
                    print('Stop generation', seq)
                    return True
        return False


# Function to find the start and end indices of the specific sentence in input_ids
def find_subsequence(input_ids, sentence_ids):
    for i in range(len(input_ids) - len(sentence_ids) + 1):
        if input_ids[i:i + len(sentence_ids)] == sentence_ids:
            return i, i + len(sentence_ids) - 1
    return None, None


ONLY_CONTEXT=True

def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    DATA_DIR = './xai_experiments_context_grad_mistral_unktoken'
    if not exists(DATA_DIR):
        makedirs(DATA_DIR)

    # Required arguments
    parser.add_argument('--repetition_penalty', default=1.0, type=float, help='Repetition penalty')
    parser.add_argument('--max_length', default=128, type=int, help='Maximum length of the generated text')
    parser.add_argument('--article_id', default=2, type=int, help='Article that might or might not be violated')
    parser.add_argument('--model_name', default='meta-llama/Llama-2-13b-chat-hf', help='Model name in HF Hub')
    parser.add_argument('--dataset_name', default='meta-llama/Llama-2-13b-chat-hf', help='Dataset name')
    parser.add_argument('--language', default='EN', help='language for multilingual SST dataset')
    parser.add_argument('--model_name_short', default='llama', help='Model name in HF Hub')
    parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction,
                        help='Whether to reverse order of options in prompt')
    parser.add_argument('--quant', action=argparse.BooleanOptionalAction, help='Whether to quantize the model')
    parser.add_argument('--xai', default='none',
                        help='Use lrp, gi, lrp_min, gi_min, lrp_contrast, gi_contrast to compute explanations.')
    parser.add_argument('--seed', default=0, help='Set a seed for reproducibility')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, help='Whether to plot heatmaps')
    parser.add_argument('--max_samples', default=None, type=int, help='Maximum length of the generated text')
    parser.add_argument('--extra_flag', default='', type=str, help='Maximum length of the generated text')


    config = parser.parse_args()

    DATA_DIR_DATASET = join(DATA_DIR, config.dataset_name, config.model_name_short + '_' + config.extra_flag + '_' + config.xai, str(config.seed))
    if not exists(DATA_DIR_DATASET):
        makedirs(DATA_DIR_DATASET)

    print(config)

    override_dict = {'llama': override_llama_xai_layers, 
                     'llama3': override_llama_xai_layers,
                      'mistral': override_mistral_xai_layers}

    
    # Set the seed for reproducibility
    torch.manual_seed(config.seed)

    # Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    

    if config.quant:
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            # use_flash_attention=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
        )
    else:
        bnb_config = None

    model_config = AutoConfig.from_pretrained(
        config.model_name
    )

    print(f"load model: {config.model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.float16
    )

    _ = model.eval()

    
    if config.xai.split('_')[0] in ['lrp', 'gi']:

        test_ids = tokenizer("Paris is the capital of", return_tensors='pt').to(model.device)
        test1 = model(**test_ids).logits

        if config.model_name_short in override_dict:

            override_func = override_dict[config.model_name_short]

            # Does not change the forward predictions, only the gradient computation for lrp
            if 'lrp' in config.xai:
                model = override_func(model, model.config) 
               # model = model.to(torch.float16)
                test2 = model(**test_ids).logits
                #assert torch.isclose(test1, test2)
                print(torch.isclose(test1, test2))
                print( ((test1-test2)**2).sum())
                
                embedding_module = model.model.embed_tokens  # (input_ids)

        else:
            raise ValueError(f"Model {config.model_name} not supported yet in XAI mode")

    else:
        embedding_module = model.model.embed_tokens 


    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,

    if config.xai in ['lrp', 'gi', 'lrp_min', 'gi_min', 'lrp_contrast', 'gi_contrast', 'random']:
        # Define a subtoken indicative when the model selects an answer
        if config.model_name_short in ["llama3", "llama"]:
            # answer_tokens = ['(a)', 'a)', 'yes', '(b)', 'b)', 'no']
            answer_tokens = ['(a', 'yes', '(b', 'no']
            stop_token_ids = [tokenizer(token, add_special_tokens=False).input_ids for token in answer_tokens]

    if config.xai in ['lrp', 'gi', 'lrp_min', 'gi_min', 'lrp_contrast', 'gi_contrast', 'random']:
        # Define a siubtoken indicative when the model selects an answer
        if config.model_name_short in ["llama3", "llama", "mistral"]:
            #answer_tokens = ['(a)', 'a)', 'yes', '(b)', 'b)', 'no']
            answer_tokens = ['(a', 'yes', '(b', 'no']
            stop_token_ids = [tokenizer(token, add_special_tokens=False).input_ids for token in answer_tokens]
            if config.model_name_short == "llama":
                # add tokens without "_", e.g. ["_(", "a"] -> ["(", "a"]
                answer_tokens_additional = ['\n(a', '\nyes', '\n(b', '\nno']
                stop_token_ids += [tokenizer(token, add_special_tokens=False).input_ids[2:] for token in
                                   answer_tokens_additional]

        else:
            raise

        stop_token_ids_decoded = [tokenizer.decode(token) for token in stop_token_ids]
        stop_token_ids_decoded_split = [tokenizer.convert_ids_to_tokens(token) for token in stop_token_ids]

    # ['(a', 'a', 'yes', '(b', 'b', 'no']

    if config.dataset_name == 'forced_labour':
        data, article_label, article_definition = prepare_labour_dataset(config.article_id)

        general_prompt = [f'Consider the following definition of "{article_label}": {article_definition} \n'
                          f'Does the following text show any evidence for "{article_label}": '
                          '{} \n Choose one of the following options and start your answer with the respective letter: \n'
                          f'(a) Yes, this text contains evidence for "{article_label}"\n'
                          f'(b) No, this text does not contain any evidence for "{article_label}"\n'
                          'Answer:']


    elif config.dataset_name == 'sst':
        data, label_dict = prepare_sst_dataset()

        if config.shuffle:
            general_prompt = ['Consider the following movie review: {} \n'
                              'Choose one of the following options and start your answer with the respective letter: \n'
                              '(a) This review is overall rather negative \n'
                              '(b) This review is overall rather positive \n'
                              'Answer:']

        else:
            general_prompt = ['Consider the following movie review: {} \n'
                              'Choose one of the following options and start your answer with the respective letter: \n'
                              '(a) This review is overall rather positive \n'
                              '(b) This review is overall rather negative \n'
                              'Answer:']

            options = {
                1: '(a) This review is overall rather positive',
                0: '(b) This review is overall rather negative'}

    elif config.dataset_name == 'sst_multilingual':
        data, label_dict = prepare_sst_multilingual_dataset(config.language)
        language_dict = {'EN': 'English', 'IT': 'Italian', 'DK': 'Danish'}
        if config.language == 'EN':
            if config.shuffle:
                general_prompt = ['Consider the following movie review: {} \n'
                                  'Choose one of the following options and start your answer with the respective letter: \n'
                                  '(a) This review is overall rather negative \n'
                                  '(b) This review is overall rather positive \n'
                                  'Answer:']
            else:
                general_prompt = ['Consider the following movie review: {} \n'
                                  'Choose one of the following options and start your answer with the respective letter: \n'
                                  '(a) This review is overall rather positive \n'
                                  '(b) This review is overall rather negative \n'
                                  'Answer:']
        elif config.language == 'IT':
            if config.shuffle:
                general_prompt = ['Considera la seguente recensione di un film: {} \n'
                                  'Scegli una delle seguenti opzioni e inizia la tua risposta con la rispettiva lettera: \n'
                                  '(a) Questa recensione è nel complesso piuttosto negativa \n'
                                  '(b) Questa recensione è nel complesso piuttosto positiva \n'
                                  'Risposta:']
            else:
                general_prompt = ['Considera la seguente recensione di un film: {} \n'
                                  'Scegli una delle seguenti opzioni e inizia la tua risposta con la rispettiva lettera: \n'
                                  '(a) Questa recensione è nel complesso piuttosto positiva \n'
                                  '(b) Questa recensione è nel complesso piuttosto negativa \n'
                                  'Risposta:']
        elif config.language == 'DK':
            if config.shuffle:
                general_prompt = ['Overvej følgende filmanmeldelse: {} \n'
                                  'Vælg en af følgende muligheder, og start dit svar med det pågældende bogstav: \n'
                                  '(a) Denne anmeldelse er generelt ret negativ \n'
                                  '(b) Denne anmeldelse er generelt ret positiv \n'
                                  'Svar:']
            else:
                general_prompt = ['Overvej følgende filmanmeldelse: {} \n'
                                  'Vælg en af følgende muligheder, og start dit svar med det pågældende bogstav: \n'
                                  '(a) Denne anmeldelse er generelt ret positiv \n'
                                  '(b) Denne anmeldelse er generelt ret negativ \n'
                                  'Svar:']
        else:
            raise NotImplementedError(f"Language {config.language} not supported")

    else:
        raise ValueError(f"Dataset {config.dataset_name} not supported")

    examples = []
    num_rationales = 0



    for idx, subdf in data.iterrows():

        if config.xai in ['none']:
            compute_xai = False
        else:
            compute_xai = True

        
       # if idx <=0:
        #    continue
        
        #    if idx <=66:
        #        continue

        # if num_rationales == 5:
        #     break

        example = {}
        if config.dataset_name == 'forced_labour':
            annotation_request = general_prompt[0].format(subdf['content'])
            example['content'] = subdf['content']
            example['true_label'] = subdf['labels_subset']
            example['gold_label_rationales'] = subdf['rationales']
        elif config.dataset_name == 'sst':
            annotation_request = general_prompt[0].format(subdf['sentence'])
            example['content'] = subdf['sentence']
            example['true_label'] = subdf['label']
            example['gold_label_rationales'] = subdf['rationale_binary']
        elif config.dataset_name == 'sst_multilingual':
            if config.language == 'EN':
                annotation_request = general_prompt[0].format(subdf['sentence'])
            else:
                annotation_request = general_prompt[0].format(subdf['sentence'])
            example['content'] = subdf['sentence']
            example['true_label'] = subdf['label']
            example['gold_label_rationales'] = subdf['rationale_binary']

        messages = [
            {
                "role": "user",
                "content": annotation_request,
            },
        ]
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_tensors="pt").to(model.device)

        output = model.generate(tokenized_chat, max_new_tokens=config.max_length, eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.eos_token_id, repetition_penalty=config.repetition_penalty,
                                do_sample=True, num_return_sequences=1)

        prompt_length = tokenized_chat.shape[1]
        responses = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

        if config.xai in ['lrp', 'gi', 'lrp_min', 'gi_min', 'lrp_contrast', 'gi_contrast', 'random']:
            # Create a stopping criteria list
            stopping_criteria = StoppingCriteriaList([StopOnAnyTokenSequence(stop_token_ids)])
            ### Now Explain ####

            seq_len = len(tokenized_chat.squeeze())
            print(idx, seq_len)
            if seq_len > 10000:
                example['relevance_{}'.format(config.xai)] = 'N/A'
                compute_xai = False
                print('Skipping xai for', idx, seq_len)

        print(
            f'RESPONSE GIVEN PROMPT [{idx}]:\n True label: {example["true_label"]} \n {responses.strip()}')
        print("-" * 50)

        example[f'response_{idx}'] = responses.strip()
        example['annotation_request'] = annotation_request
        example = normalize_responses(example, idx, config.shuffle)

        print(example[f'normalized_response_{idx}'], example['true_label'])
        
        condition = (example[f'normalized_response_{idx}'] == 1 and example['true_label'] == 1) \
            if config.dataset_name == 'forced_labour' else (example[f'normalized_response_{idx}'] == example['true_label'])

        print('condition',  idx, condition)
        
        ### Now Explain ####

        seq_len = len(tokenized_chat.squeeze())
        print(idx, seq_len)
        #if seq_len > 10000:
        if seq_len > 3000:
            example['relevance_{}'.format(config.xai)] = 'N/A'
            compute_xai = False
            print('Skipping xai for', idx, seq_len)

        while compute_xai == True:

            # Create a stopping criteria list    
            stopping_criteria = StoppingCriteriaList([StopOnAnyTokenSequence(stop_token_ids)])

            # Regenerate for explainability but with a stopping criterion, requires seed to be set again
            torch.manual_seed(config.seed) 
            
            # Generate with a stopping criteria and from input_ids (to find prompt length until answer)
            output_until_answer = model.generate(tokenized_chat,    
                    max_new_tokens=config.max_length, 
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id, 
                    repetition_penalty=config.repetition_penalty,
                    do_sample=True, 
                    num_return_sequences=1,
                    stopping_criteria=stopping_criteria)  

            # Prepare embeddings
            tokenized_chat_until_answer = output_until_answer[:,:-1]
            embeddings_out  = embedding_module(tokenized_chat_until_answer)#.detach().cpu().numpy()




            # find context range
            context_ids = tokenizer(f' {example["content"].strip()}', add_special_tokens=False).input_ids
            xai_input_ids = tokenized_chat_until_answer[0].tolist()
            output_words = tokenizer.convert_ids_to_tokens(xai_input_ids, skip_special_tokens=False)
            output_words = [token.replace('Ġ','') for token in output_words]
            start_idx, end_idx = find_subsequence(xai_input_ids, context_ids)

            if start_idx is None and end_idx is None:
                import pdb;pdb.set_trace()

            context_mask = torch.zeros_like(embeddings_out)
            context_mask[:, start_idx:end_idx+1,: ] = 1


            embeddings_ = embeddings_out.data.requires_grad_(True)            
            embeddings_in = embeddings_*context_mask + embeddings_.detach()*(1-context_mask)

            
            #embeddings_ = embeddings_out


            # Regenerate for explainability but with a stopping criterion, requires seed to be set again            
            # model.generate does not allow gradient computation, instead use wrapped generate function without @torch.no_grad() decorator

            # Generate with a stopping criteria and from embeddings to explain the answer

            torch.manual_seed(config.seed)
            output_xai = model.generate.__wrapped__(model,
                                                    inputs_embeds=embeddings_in,
                                                    max_new_tokens=1,  # just generate one next token
                                                    eos_token_id=tokenizer.eos_token_id,
                                                    pad_token_id=tokenizer.eos_token_id,
                                                    repetition_penalty=config.repetition_penalty,
                                                    do_sample=True,
                                                    num_return_sequences=1,
                                                    stopping_criteria=stopping_criteria,
                                                    return_dict_in_generate=True,
                                                    output_scores=True,
                                                    output_logits=True
                                                    )
            # odict_keys(['sequences', 'scores', 'logits',+ 'past_key_values'])

            # Take the logits of the last generated token (which should be the answer token)
            xai_scores = output_xai.scores[-1]
            xai_logits = output_xai.logits[-1]

            xai_generated_ids = output_xai.sequences[0]
            xai_output_ids = torch.cat([tokenized_chat_until_answer, xai_generated_ids[None,:]], dim=-1)

            try:
                assert xai_scores.argmax() ==  xai_logits.argmax()
            except:
                print('assert failed')
    
            next_token = xai_logits.argmax()
            token_explained = tokenizer.decode([next_token])

            orig_logit = xai_logits[:,next_token]

            if False:

                if token_explained not in list(itertools.chain(*stop_token_ids_decoded_split)):
                    print('not in list', token_explained)
                    example['relevance_{}'.format(config.xai)] = 'N/T'
                    compute_xai = False
                    break
                    # import pdb;pdb.set_trace()



            if '_contrast' in config.xai:
                if 'yes' in token_explained:
                    contrast_token = token_explained.replace('yes', 'no')
                elif 'no' in token_explained:
                    contrast_token = token_explained.replace('no', 'yes')
                elif 'a' in token_explained:
                    contrast_token = token_explained.replace('a', 'b')
                elif 'b' in token_explained:
                    contrast_token = token_explained.replace('b', 'a')
                else:
                    import pdb;
                    pdb.set_trace()

                contrast_id = tokenizer(contrast_token, add_special_tokens=False).input_ids
                selected_logit = xai_logits[:, next_token] - xai_logits[:, contrast_id]

            elif '_min' in config.xai:
                mask = torch.zeros((len(tokenizer)))
                mask[int(next_token)] = 1
                contrastive_token_idx = torch.argmax(xai_logits[:, mask != 1])
                selected_logit = torch.min(xai_logits[:, next_token] - xai_logits[:, mask != 1])
                contrast_token = tokenizer.decode([contrastive_token_idx])

            else:
                selected_logit = xai_logits[:, next_token]
                contrast_token = ''

            print(selected_logit)

            
            context_ids = tokenizer(f' {example["content"].strip()}', add_special_tokens=False).input_ids

            xai_output_ids = xai_output_ids[0].tolist()
            output_words = tokenizer.convert_ids_to_tokens(xai_output_ids, skip_special_tokens=False)
            output_words = [token.replace('Ġ','') for token in output_words]
            start_idx, end_idx = find_subsequence(xai_output_ids, context_ids)

            if start_idx is None and end_idx is None:
                print('example[content]', example['content'])
                print('example[content]_tokenized', tokenizer(example['content'], add_special_tokens=False).input_ids)
                # print('context_str', context_str)
                print('context_ids', context_ids)
                print('xai_output_ids', xai_output_ids)
                print('start_idx', start_idx)
                print('end_idx', end_idx)
            
            
            if config.xai == 'random':
                example['relevance_{}'.format(config.xai)] = np.random.normal(0,1, int(tokenized_chat_until_answer.squeeze().shape[0])).tolist()
                example['selected_logit'] = selected_logit.detach().cpu().numpy().squeeze().tolist()
                compute_xai = False
                break
            

            responses_xai = tokenizer.decode(xai_output_ids[prompt_length:], skip_special_tokens=False)

            #  tokenizer.convert_ids_to_tokens(xai_generated_ids, skip_special_tokens=False)

            #  tokenizer.convert_ids_to_tokens(tokenized_chat_until_answer, skip_special_tokens=False)

            # Compute explanation
            selected_logit.backward()

            gradient = embeddings_.grad
            relevance_ = gradient * embeddings_
            relevance = relevance_.sum(2).detach().cpu().numpy().squeeze()




           # relevance = (((relevance_**2).sum(axis=2))**0.5).detach().cpu().numpy().squeeze()
            relevance = relevance_.sum(2).detach().cpu().numpy().squeeze()

            example['relevance_{}'.format(config.xai)] = relevance.tolist()

            # Filter relevancies to focus on the context only
            # if config.dataset_name == 'forced_labour':
            #     context_str = ' {}\n'.format(example['content'])
            # else:
            #     context_str = ' {} \n'.format(example['content'])

            # print('context_str', context_str)

            example['selected_logit'] = selected_logit.detach().cpu().numpy().squeeze().tolist()


            # output_words_context = output_words[start_idx:end_idx + 1]
            # relevance_context = relevance_with_next[start_idx:end_idx + 1]

            relevance_with_next = np.hstack([relevance, np.array(0)])


            if config.plot:
                PLOT_DIR_DATASET = join(DATA_DIR_DATASET, config.model_name_short)
                if not exists(PLOT_DIR_DATASET):
                    makedirs(PLOT_DIR_DATASET)

                print(PLOT_DIR_DATASET)
                # Plot heatmap
                logit_score = selected_logit.detach().cpu().numpy().squeeze()
                title = r'token explained {}, {}, logit={:0.2f}, heatmap sum={:0.2f}'.format(token_explained,
                                                                                             contrast_token,
                                                                                             logit_score,
                                                                                             relevance.sum())

                if len(output_words) != len(relevance_with_next):
                    import pdb;
                    pdb.set_trace()


                relevance_with_next_normalized = relevance_with_next / np.max(np.abs(relevance_with_next))
                html_heatmap = plot_heatmap(output_words, relevance_with_next_normalized, title, transparency=95)

                output_name = f'heatmap_article_{config.article_id}' # if config.dataset_name == 'forced_labour' else output_name
                output_name = output_name + f'_{config.language}' if config.dataset_name == 'sst_multilingual' else output_name
                output_name = output_name + f'_idx_{idx}' + f'_{config.xai}'

                html_file = join(PLOT_DIR_DATASET, f"{output_name}.html")

                # Save the HTML content to a temporary file
                with open(html_file, 'w') as file:
                    file.write(html_heatmap)

                # Plot context heatmap
                if start_idx is not None and end_idx is not None:
                    output_words_context = output_words[start_idx:end_idx + 1]
                    relevance_context = relevance_with_next[start_idx:end_idx + 1]
                    relevance_context_normalized = relevance_context / np.max(np.abs(relevance_context))

                    html_file_context = html_file.replace('.html', '_context.html')

                    title_context = r'token explained {}, {}, logit={:0.2f}, heatmap sum={:0.2f}'.format(
                        token_explained, contrast_token, logit_score, relevance_context.sum())

                    html_heatmap_context = plot_heatmap(output_words_context, relevance_context_normalized,
                                                        title_context, transparency=95)

                    # Save the HTML content to a temporary file
                    with open(html_file_context, 'w') as file:
                        file.write(html_heatmap_context)

            compute_xai = False
            break


        # Flippig analysis / Prepare flipping analysis
        if 'relevance_{}'.format(config.xai) in example:
            if example['relevance_{}'.format(config.xai)] not in  ['N/T', 'N/A']:

                if condition:
                    output_words[start_idx:end_idx+1]
                    context_mask = np.zeros(tokenized_chat_until_answer.shape[-1])
                    context_mask[start_idx:end_idx+1] = 1.
                    answer = next_token
                    test_logit = model(tokenized_chat_until_answer).logits
                    assert orig_logit == test_logit[:,-1,next_token]

                    # prep perturbation
                    example['perturbation'] = {'E':None, 
                                               'tokenized_chat_until_answer': tokenized_chat_until_answer.detach().cpu().numpy().tolist(),
                                               'context_mask': context_mask.tolist(),
                                               'test_logit': test_logit[:,-1,next_token].detach().cpu().numpy().tolist(),
                                               'answer': int(next_token),
                                               'seed': int(config.seed)}
                                              

                    if True:
                        # Dont flip in here for now, do it in eval
                        R =  np.array(example['relevance_{}'.format(config.xai)])

                        if ONLY_CONTEXT:
                            # Only keep relevance for context
                            R[context_mask.squeeze()!=1] = -10e6
                            
                        fracs = np.linspace(0,1,21)


                        if 'llama' in config.model_name_short:
                            replace_token_id = int(tokenizer("_", add_special_tokens=False).input_ids[0])
                        else:
                           replace_token_id = tokenizer.unk_token_id
                        

                        E, E_log, M = flip(model, tokenizer, tokenized_chat_until_answer, R, answer, replace_token_id, int(context_mask.sum()), fracs =  fracs)
    
                        example['perturbation']['E'] =  (E, E_log)
    
                        # plotting flipping
                        
                        PLOT_DIR_DATASET_FLIP = join(DATA_DIR_DATASET, config.model_name_short, 'flip')
                        if not exists(PLOT_DIR_DATASET_FLIP):
                            makedirs(PLOT_DIR_DATASET_FLIP)
        
                        flip_evolution = M['flip_evolution'] 

                        if False:
                            for frac in [0.0, 0.5, 1.0]:
                            
                                t_ = flip_evolution[frac][0].squeeze().tolist()
                
                                output_words = tokenizer.convert_ids_to_tokens(t_, skip_special_tokens=False)
                                output_words = [token.replace('Ġ','') for token in output_words]
                                        
                                title = r''
                        
                                # Plot full input prompt heatmap
                                R_flip = R/np.max(np.abs(R))
                                html_heatmap = plot_heatmap(output_words, R_flip, title, transparency = 95)

                                output_name = f'heatmap_article_{config.article_id}' #if config.dataset_name == 'forced_labour' else output_name
                                output_name = output_name + f'_{config.language}' if config.dataset_name == 'sst_multilingual' else output_name
                                output_name = output_name + f'_idx_{idx}' + f'_{config.xai}' + "_frac_{}".format(frac)         
                                
                                html_file = join(PLOT_DIR_DATASET_FLIP, f"{output_name}.html")
                                
                                # Save the HTML content to a temporary file
                                with open(html_file, 'w') as file:
                                    file.write(html_heatmap)
        
        examples.append(example)

        torch.cuda.empty_cache()

        if config.max_samples is not None:
            if idx == config.max_samples:
                break
        

    print('-' * 50)
    print('MADE IT UNTIL HERE')
    print('-' * 50)

    output_name = f'{config.model_name_short}_rationales'
    output_name = output_name + f'_article_{config.article_id}' if config.dataset_name == 'forced_labour' else output_name
    output_name = output_name + f'_{config.language}' if config.dataset_name == 'sst_multilingual' else output_name
    output_name = output_name + '_shuffle' if config.shuffle else output_name
    output_name = output_name + '_quant' if config.quant else output_name


    output_name = output_name + f'_seed_{config.seed}' if config.seed is not None else output_name

    print(join(DATA_DIR_DATASET, f"{output_name}.jsonl"))

    with open(join(DATA_DIR_DATASET, f"{output_name}.jsonl"), "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
        
    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    Es = [e['perturbation']['E'][0] for e in examples if 'perturbation' in e]
    ax.plot(fracs, np.nanmean(Es, axis=0), label=config.xai)
    plt.legend()
    f.savefig(join(DATA_DIR_DATASET,f"{config.model_name_short}_{config.xai}_flipping.png"), dpi=200)
    plt.close()


    # conservations

    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    Rs = [np.sum(e['relevance_{}'.format(config.xai)]) for e in examples if 'perturbation' in e]
    Ls = [e['selected_logit'] for e in examples if 'perturbation' in e]
    ax.plot(Ls,Ls, color='black', linestyle='-', linewidth=1)
    ax.scatter(Ls, Rs, label=config.xai)
    plt.legend()
    f.savefig(join(DATA_DIR_DATASET,f"{config.model_name_short}_{config.xai}_conservation.png"), dpi=200)
    plt.close()


if __name__ == '__main__':
    main()
