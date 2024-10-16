from utils import AUTH_TOKEN
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from xai.xai_llama import LlamaForCausalLMXAI
from xai.xai_mistral import MistralForCausalLMXAI


def get_xai_model( model_name,
                  state_dict,
                  xai_flag, 
                  model_config):

    if 'llama' in model_name:    
        model_xai_base = LlamaForCausalLMXAI
    elif 'mistral' in model_name:
        model_xai_base = MistralForCausalLMXAI
    else:
        raise

    model_xai = model_xai_base(
                config=model_config,
                lrp=xai_flag)
    import pdb;pdb.set_trace()
    
    _ = model_xai.load_state_dict(state_dict)
    
    return model_xai





def get_llama_explanation(model_xai, tokenizer, input_ids, config, device, generation_strategy='max'):


    model_inputs = {'input_ids': None, 
                'position_ids': None, 
                 'past_key_values': None, 
                 'use_cache': False, 
                 'attention_mask': torch.ones_like(input_ids).to(device),
               }

    inputs_embeds = model_xai.model.embed_tokens(input_ids) 
    _ = model_inputs.update({'inputs_embeds': inputs_embeds})


    # explanations
    model_inputs_detached = model_inputs
    input_embeds_detached = model_inputs_detached['inputs_embeds'].detach().requires_grad_(True)
    model_inputs_detached['inputs_embeds'] =  input_embeds_detached
    outputs = model_xai(**model_inputs_detached,
                        output_hidden_states=False,)

    logits = outputs.logits[:,-1, : ]
    next_token_scores = logits
    probs = nn.functional.softmax(next_token_scores, dim=-1)
    
    if generation_strategy == 'max':
        # take the maximally likely next token        
        next_tokens = torch.argmax(probs).unsqueeze(0) #
    else:
        # sample a token from a multinomial distribution defined by the vocabularity probabilities
        next_tokens_many = torch.multinomial(probs, num_samples=10).squeeze(1)
        next_tokens = next_tokens_many[:,0]
        print(tokenizer.convert_ids_to_tokens(next_tokens_many.detach().cpu().numpy().squeeze()))
    
    logits = outputs.logits[:,-1, : ]
    selected_logit = logits[:,next_tokens]
    selected_logit = torch.min(logits[:,next_tokens] - logits[:, torch.eye(config.vocab_size)[int(next_tokens)] != 1])
    selected_logit.sum().backward()
    gradient = input_embeds_detached.grad

    relevance = gradient * model_inputs_detached['inputs_embeds']
    relevance = relevance[0,:].sum(1).detach().cpu().numpy()

    return relevance, selected_logit.detach().cpu().numpy(), next_tokens,  tokenizer.convert_ids_to_tokens(next_tokens.detach().cpu().numpy().tolist())


def plot_conservation(Ls, Rs, filename=None, fax=None):
    if fax is not None:
        f,ax = fax
    else:
        f, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(Ls, Rs, s=20, label='R', c='blue')
    ax.plot(Ls, Ls, color='black', linestyle='-', linewidth=1)

    # ax.set_ylabel('$\sum_i R_i$', fontsize=30, usetex=True)
    # ax.set_xlabel('output $f$', fontsize=30,  usetex=True)
    ax.set_ylabel('$\sum_i R_i$', fontsize=30, usetex=False)
    ax.set_xlabel('output $f$', fontsize=30, usetex=False)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.tick_params(axis='both', which='minor', labelsize=22)
    f.tight_layout()
    if filename:
        f.savefig(filename, dpi=100)
        plt.close()

        
    
    
        