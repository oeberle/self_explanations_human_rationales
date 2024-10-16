import numpy as np
import torch
import torch.nn as nn




def threshold_relevances(scores, p, mask=None):

    if mask is not None:
        scores_for_thresholding = scores[mask==1]
    else:
        scores_for_thresholding = scores

    # Calculate the p-th percentile
    threshold = np.percentile(scores_for_thresholding, p)
    
    # Binarize the scores using the percentile as threshold
    binarized_scores = np.array([1 if score >= threshold else 0 for score in scores])

    return binarized_scores


def flip(model, tokenizer, inputs, relevance, answer_id, replace_token, N,  fracs =  np.linspace(0,1.,11)):

    input_ids0 = inputs.detach()#.clone()
    outputs = model(input_ids0).logits

    try:
        p0_ = outputs[:,-1,answer_id].detach().cpu().numpy().squeeze()
        p0 = nn.functional.softmax(outputs[:,-1,:], dim=-1)[:,answer_id]

    except:
        import pdb;pdb.set_trace()

  #  print(p0_)

  #  inds_sorted =  np.argsort(relevance) #very small to large
    inds_sorted =  np.argsort(relevance)[::-1] # large to small
    vals = relevance[inds_sorted]

    mse = []
    evidence = []
    evidence_log = []
    model_outs = {'y_true' : answer_id, 'relevance':relevance} 
    evolution = {}

    for frac in fracs:

        inds_generator = iter(inds_sorted)
        n_flip=int(np.ceil(frac*N))
        inds_flip = [next(inds_generator) for i in range(n_flip)]

        input_ids_ = input_ids0.detach()# .clone()
        
        input_ids_[:,inds_flip] = replace_token
     #   input_ids_flip = input_ids0[:,inds_flip].squeeze().tolist()

        outputs = model(input_ids_).logits

        try:
            p_ = outputs[:,-1,answer_id].detach().cpu().numpy().squeeze()
            p = nn.functional.softmax(outputs[:,-1,:], dim=-1)[:,answer_id]

        except:
            import pdb;pdb.set_trace()

        input_ids_save = np.copy(input_ids_.detach().cpu().numpy())

        evidence.append(float(p))
        evidence_log.append(float(p_))
  #      evolution[frac] = (input_ids_save, None, input_ids_flip, None) #outputs.detach().cpu().numpy())
        evolution[frac] = (input_ids_save, None, None, None) #outputs.detach().cpu().numpy())

        torch.cuda.empty_cache()
    #   print(p)

    model_outs['flip_evolution']  = evolution
    return evidence, evidence_log, model_outs



def flip_simple(model, tokenizer, inputs, relevance, answer_id, replace_token): #,  fracs =  np.linspace(0,1.,11)):

    input_ids0 = inputs.detach()#.clone()
    outputs = model(input_ids0).logits

    try:
        p0_ = outputs[:,-1,answer_id].detach().cpu().numpy().squeeze()
        p0 = nn.functional.softmax(outputs[:,-1,:], dim=-1)[:,answer_id]
    except:
        import pdb;pdb.set_trace()

    evidence = [float(p0)]
    evidence_log = [float(p0_)]
    evolution = {}

    inds_flip = relevance==1
    input_ids_ = input_ids0.detach()
    
    input_ids_[:,inds_flip] = replace_token
    outputs = model(input_ids_).logits

    try:
        p_ = outputs[:,-1,answer_id].detach().cpu().numpy().squeeze()
        p = nn.functional.softmax(outputs[:,-1,:], dim=-1)[:,answer_id]
    except:
        import pdb;pdb.set_trace()

    input_ids_save = np.copy(input_ids_.detach().cpu().numpy())

    evidence.append(float(p))
    evidence_log.append(float(p_))

    torch.cuda.empty_cache()
#    print(p)

    return evidence, evidence_log