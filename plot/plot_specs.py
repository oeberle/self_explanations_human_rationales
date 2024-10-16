plot_dict = {'llama': ('#87adf1',),
             'llama3': ('#1755c4',),
             'mistral': ('#5adada',),
             'mixtral': ('#0a7b8a',),
             }

plot_dict2 = {'llama': ('#DC9AB7',),
             'llama3': ('#b64c7a',),
             'mistral': ('#CDBDE2',),
             'mixtral': ('#8a87d6',),
             }


hatch_dict = {'lrp': (None,),
              'lrp_contrast': ('///',),
              'random': (None,),
              'baseline': (None,), #('...',)
              }

replace_rules = [
    ('rationales_', ''),
    ('relevance_lrp_binary', 'post-hoc'),
    ('model$one_over_n', 'model'),
    ('$one_over_n', ' 1/n'),
    ('$model_rationale', ' model'),
    ('$human_rationale', ' human'),
]
