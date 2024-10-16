#!/bin/bash

DATASET="sst_multilingual"
LANGUAGES=(IT DK EN)
ARTICLE_ID=1

# First run mixtral (no xai needed)

SEEDS=(28 79 96)
MODELS=(
  'meta-llama/Llama-2-13b-chat-hf llama'
  'meta-llama/Meta-Llama-3.1-8B-Instruct llama3'
  'mistralai/Mistral-7B-Instruct-v0.3 mistral'
)

for LANGUAGE in "${LANGUAGES[@]}"
do
    for SEED in "${SEEDS[@]}"
    do
        # Define the list of model paths and corresponding short names
        for MODEL_INFO in "${MODELS[@]}"
        do
            # Split the model path and short name
            MODEL=$(echo $MODEL_INFO | awk '{print $1}')
            MODEL_SHORT=$(echo $MODEL_INFO | awk '{print $2}')
            
            # Loop over each XAI strategy
            for XAI in lrp lrp_contrast random
            do
                echo $XAI $MODEL $MODEL_SHORT $SEED
                
                python rationale_prompting.py \
                  --max_length 100 \
                  --model_name ${MODEL} \
                  --model_name_short ${MODEL_SHORT} \
                  --article_id ${ARTICLE_ID} \
                  --dataset_name ${DATASET} \
                  --seed ${SEED} \
                  --xai ${XAI} \
                  --language ${LANGUAGE} \
                  --sparsity oracle 
            done
        done
    done
done


for LANGUAGE in "${LANGUAGES[@]}"
do
    for SEED in "${SEEDS[@]}"
    do
        # Define the list of model paths and corresponding short names
        for MODEL_INFO in "${MODELS[@]}"
        do
            # Split the model path and short name
            MODEL=$(echo $MODEL_INFO | awk '{print $1}')
            MODEL_SHORT=$(echo $MODEL_INFO | awk '{print $2}')
            
            for XAI in lrp  random lrp_contrast
            do
                python model_human_comparison.py \
                  --dataset_name ${DATASET}  \
                  --seed ${SEED} \
                  --select_strategy human_rationale \
                  --loops ${LANGUAGE} \
                  --xai_strategy ${XAI} \
                  --model_name_short ${MODEL_SHORT} \
                  --sparsity oracle \
                  --flipping
            done
        done
    done
done