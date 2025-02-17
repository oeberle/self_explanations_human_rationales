# Do Transformer Models Show Similar Attention Patterns to Task-Specific Human Gaze?
Code for "Comparing zero-shot self-explanations with human rationales in multilingual text classification" paper (arxiv 2024)
https://arxiv.org/pdf/2410.03296


## First Steps
Set-up a suitable environment, see `requirements.txt`. We used `python=3.11.8'.


## Scripts
* `rationale_prompting.py`: runs the prompting procedure, see `run_eval.sh' for usage. 
* `model_human_comparison.py`: runs the evaluation procedure, see `run_eval.sh' for usage.  

## Plots
* `pos_rel_analysis.py`: runs and plots the POS analysis where we show difference in relative POS distribution with human rationales as baseline.

## Files
* `model_responses/`: already contains the model responses for SST.


## Cite
    @article{brandl2024comparing,
      title={Comparing zero-shot self-explanations with human rationales in multilingual text classification},
      author={Brandl, Stephanie and Eberle, Oliver},
      journal={arXiv preprint arXiv:2410.03296},
      year={2024}
    }

