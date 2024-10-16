import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import numpy as np
import datetime
import logging

AUTH_TOKEN='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'

def get_base_str(task, m, lang, sparsity, seed, xai_method):
    root = "model_responses/eval_results/{}/{}/{}/{}/{}/".format(task, m, sparsity, seed, xai_method)
    if m == 'mixtral':
        base_str = root + "{}_{}{}_quant_seed_{}_sparsity_{}".format(task, m, lang, seed,
                                                                     sparsity) if sparsity != 'full' \
            else root + "{}_{}{}_quant_seed_{}".format(task, m, lang, seed)
    else:
        base_str = root + "{}_{}{}_seed_{}_sparsity_{}".format(task, m, lang, seed, sparsity) if sparsity != 'full' \
            else root + "{}_{}{}_seed_{}".format(task, m, lang, seed)
    return base_str



def init_logger(file_name, experiment_name):
    # logging stuff
    logging_datetime_format = '%Y%m%d__%H%M%S'
    logging_time_format = '%H:%M:%S'
    exp_start_time = datetime.datetime.now().strftime(logging_datetime_format)
    log_level = logging.DEBUG
    log_format = '%(asctime)10s,%(msecs)-3d %(module)-30s %(levelname)s %(message)s'
    
    logging.basicConfig(datefmt=logging_time_format)

    logger = logging.getLogger(name=experiment_name)
    logger.setLevel(log_level)
    fh = logging.FileHandler(file_name, mode="w")
    fh.setFormatter(logging.Formatter(log_format))

    logger.addHandler(fh)
    return logger

def set_up_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def plot_generation(R, tokens, column_idx = 0, fax=None, fontsize=11):

    if len(tokens) == 2:
    
        tokens_x = tokens_y = tokens
        
    elif len(tokens) == 1:
        tokens_x = tokens_y = tokens[0]

    if fax is None:
        f,ax = plt.subplots(1,1, figsize=(8,6))
    else:
        f,ax = fax
    h = sns.heatmap(R[:, column_idx:], annot=R[:, column_idx:], vmin=-1, vmax=1,
                cmap='bwr',
                ax=ax, 
                fmt="0.1f",
                annot_kws={"size": fontsize})

    # Adjust colorbar (legend) ticks and font size
    cbar = h.collections[0].colorbar
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])  # Set the ticks you want on the colorbar
    cbar.set_ticklabels([-1.0, -0.5, 0.0, 0.5, 1.0])  # Set the tick labels accordingly

    # Set the width of the colorbar
    cbar.ax.figure.canvas.draw_idle()  # Ensure the canvas is drawn before setting the width
    cbar.ax.set_aspect(22) 
    
    # Set font size for colorbar ticks
    cbar.ax.tick_params(labelsize=10)
    
        

    ax.xaxis.tick_top()
    ax.set_xticks(np.array(range(len(tokens_x)-column_idx))+0.5)
    ax.set_yticks(np.array(range(len(tokens_y)))+0.5)

    ax.set_xticklabels(tokens_x[column_idx:], rotation=90, fontsize=14)#, va='center')
    ax.set_yticklabels(tokens_y, rotation=0, fontsize=14) #, va='center')

    if fax is None:
        plt.show()
    