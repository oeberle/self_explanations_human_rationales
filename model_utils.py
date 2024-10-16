import transformers



model_configs = {'llama3': "meta-llama/Meta-Llama-3.1-8B-Instruct",
                 'llama': 'meta-llama/Llama-2-13b-chat-hf',
                 'mistral': 'mistralai/Mistral-7B-Instruct-v0.3',
                 'mixtral': "mistralai/Mixtral-8x7B-Instruct-v0.1",
                 }


def get_model(model_name_short, bnb_config=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    model_name = model_configs[model_name_short]
    model_config = AutoConfig.from_pretrained(
        model_name)

    if 'mixtral' == model_name_short:
      bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
            )
    else:
        bnb_config = None
        
      #  raise ValueError('Mixtral not supported yet, quant config missing')

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.float16
    )

    _ = model.eval()

    return model, tokenizer
