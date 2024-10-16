import math
from typing import Union, Optional, Tuple, Dict
import copy
import torch
from torch import nn, Tensor
from transformers.models.mistral.modeling_mistral import MistralForCausalLM, MistralAttention, MistralMLP, apply_rotary_pos_emb, repeat_kv
import types

#from xai.layer_utils import  SiLUXAI 

from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.generation.utils import *

#from transformers.generation.configuration_utils import GenerationConfig
#from transformers.deepspeed import is_deepspeed_zero3_enabled
import numpy as np


CASE = 'VDETACH' 


from transformers.models.mistral.modeling_mistral import MistralForCausalLM, MistralRMSNorm, MistralSdpaAttention, MistralMLP, apply_rotary_pos_emb, repeat_kv



class SiLUXAI(nn.Module):
    r"""Applies the Sigmoid Linear Unit (SiLU) function, element-wise.

    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/SiLU.png

    Examples::

        >>> m = nn.SiLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['inplace']


    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
      #  return F.silu(input, inplace=self.inplace)

        input_dtype = input.dtype
        input = input.to(torch.float32)
        
        func = F.silu(input, inplace=False)
    
        const = 1e-8
        out = input*((func/(input+const)).detach())     

        # sometimes isntabilities for llama3
        if torch.isinf(out).any():
            import pdb;pdb.set_trace()
        elif  torch.isnan(out).any():
            out = torch.nan_to_num(out)
            import pdb;pdb.set_trace()
            return out.to(input_dtype)
        else:
            return out.to(input_dtype)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def check_and_return_new_forward(new_forward_func, module, test_input=None):

    if test_input:
        before = module.forward(**test_input)
        after = new_forward_func(**test_input)
        assert torch.isclose(before, after)

    return types.MethodType(new_forward_func, module)


def override_mistral_xai_layers(model, config):
    
    device = model.device
    print(f"Overriding XAI layers for model: {config._name_or_path}")

    for i in range(config.num_hidden_layers):

        input_shape = (1,1,config.hidden_size)
        test_inputs =  torch.rand(input_shape).to(device)
        test_dict = {'hidden_states': test_inputs}

        test_dict = None
        #import pdb;pdb.set_trace()

        if config._attn_implementation == 'sdpa':
            model.model.layers[i].self_attn.forward = check_and_return_new_forward(MistralSdpaAttentionXAI.forward, model.model.layers[i].self_attn, test_dict) 
        else:
            raise

        model.model.layers[i].mlp.forward = check_and_return_new_forward(MistralMLPXAI.forward, model.model.layers[i].mlp, test_dict)

        if CASE == 'HALF': 
            model.model.layers[i].mlp.act_fn = SiLUXAI() # this created the inf problem

        model.model.layers[i].input_layernorm.forward = check_and_return_new_forward(rms_norm_forward, model.model.layers[i].input_layernorm, test_dict)
        
        model.model.layers[i].post_attention_layernorm.forward = check_and_return_new_forward(rms_norm_forward, model.model.layers[i].post_attention_layernorm, test_dict)
        
    model.model.norm.forward = check_and_return_new_forward(rms_norm_forward, model.model.norm, test_dict)

    return model




# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Mistral
class MistralRMSNormXAI(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance.detach() + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def rms_norm_forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance.detach() + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
   

# copied from transformers.models.llama.modeling_llama.LlamaSdpaAttention with Llama->Mistral
# TODO(joao): add me back asap :)
class MistralSdpaAttentionXAI(MistralAttention):
    """
    Mistral attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MistralAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from MistralAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "MistralModel is using MistralSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False


        # XAI part
        query_states = query_states.detach()
        key_states = key_states.detach()
        
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value



class MistralMLPXAI(MistralMLP):
    def __init__(self, config):  
        super().__init__(config) 
        self.config = config

    def forward(self, x):

        if CASE == 'VDETACH':
            #half-rule approach
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)).detach() * self.up_proj(x))        

        elif CASE == 'HALF':
            # half-rule approach
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))        
            down_proj = 0.5*down_proj + 0.5*down_proj.detach()
        
        return down_proj



