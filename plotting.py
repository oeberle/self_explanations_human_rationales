import numpy as np
import seaborn as sns
import itertools 
import matplotlib.pyplot as plt

from spacy import displacy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np


def heat2hex(heat, transparency=95):
    # Define your range of values from -1 to 1
    min_value = -1
    max_value = 1
    
    # Create a colormap using 'bwr'
    cmap = plt.get_cmap('bwr')
    
    # Normalize your values to the [0, 1] range
    norm = Normalize(vmin=min_value, vmax=max_value)
    
    # Create a ScalarMappable to map values to colors
    sm = ScalarMappable(cmap=cmap, norm=norm)
    
    # Get the RGBA color for your value
    rgba_color = sm.to_rgba(heat)
    
    # Convert the RGBA color to HEX
    hex_color = "#{:02x}{:02x}{:02x}{:02x}".format(int(rgba_color[0] * 255), int(rgba_color[1] * 255), int(rgba_color[2] * 255), transparency)
    return hex_color


def plot_heatmap(tokens, relevance, title, transparency):

    TPL_TOK  = """
    <!DOCTYPE html>
    <html>
    <body>
        <mark class="entity" style="background: {bg}; padding: 0.5em 0.3em; margin: 0 0.1em; line-height: 0.8; border-radius: 0.0em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
            {text}
        </mark>
    </body>
    </html>
    """

    
    heat_bins = relevance 
    ents = []
    ii = 0
    for itok in range(len(tokens)):
        ff = ii + len(tokens[itok]) 
        ent = {
            'start': ii,
            'end': ff,
            'label': str(itok),
        }
        ents.append(ent)
        ii = ff
        
    to_render = {
        'text': ''.join(tokens),
        'ents': ents,
        "title": title}

    
    html = displacy.render(
        to_render, 
        style='ent', 
        page=True,
        manual=True, 
        jupyter=False, 
        options={'colors':  {str(i):heat2hex(val, transparency) for i,val in enumerate(relevance)} ,  'template': TPL_TOK})

    html = html.replace('<h2 style="margin: 0"', '<h5 style="text-align: center; margin: 0;margin-bottom: 5px; font-size: 14px;"')
    html = html.replace('h2>', "h5>")

    html = html.replace('<figure style="margin-bottom: 6rem">', '<figure style="margin-bottom: 0rem; margin-top: 0rem">')
    html = html.replace('padding: 4rem 2rem', 'padding: 2rem 2rem')
    html = html.replace('unpleasant\n        </mark>', 'unpleasant\n        </mark><br>')
    html = html.replace('product.\n        </mark>', 'product.\n        </mark><br>')
    html = html.replace('text\n        </mark>', 'text\n        </mark><br>')

    html = html.replace('<body style="font-size: 16px;', '<body style="font-size: 16px; width: 110%;>')

    #html = html.replace('padding: 2rem 2rem;', 'padding: 0rem 0rem; direction: ltr"')
    
   # display(HTML(html))

    return html


def get_canvas(words, x, H=200, W=50):
    ntoks = len(''.join(words))
    W_all = W*ntoks
    fracs = [len(w_)/ntoks for w_ in words]
    delta_even = int(W_all/ntoks)
    X = np.zeros((H, W_all))    
    x0=0
    
    x_centers = []
    for i, (w_,b) in enumerate(zip(words, x)):
        delta = int((len(w_)/ntoks)*W_all)
        
        delta = int((0.85*delta_even + 0.15*delta))
        X[:, x0:x0+delta] = b
        
        x_centers.append(x0+int(delta/2))
        x0 = x0+delta

    X = X[:, :x0]
    return X, x_centers

def plot_sentence(words, x, H0=100, W0=52, fax=None):
    if fax is None:
        f,ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 2), 
                                gridspec_kw={'height_ratios':[1]})
    else:
        f,ax = fax
    x_, x_centers = get_canvas(words, x, H=H0, W=W0)
    h = ax.imshow(x_, cmap='bwr', vmin=-1, vmax=1., alpha=1.)

    for k, word in zip(x_centers, words):
        ax.text(k, H0/2, word, ha='center', va='center')
    plt.colorbar(h)
    plt.axis('off')
    if fax is None:
        plt.show()
    

def plot_conservation(Ls, Rattns):
    f, ax = plt.subplots(1,1, figsize=(5, 5))
    from pylab import cm
    cmap = cm.get_cmap('plasma') #, 5) 
    for i in range(12):
        ax.scatter(Ls, Rattns[i], s=20, label=str(i), c=np.array([cmap(0.1+i/12)])) 
    ax.plot(Ls,Ls, color='black', linestyle='-', linewidth=1)

    ax.set_ylabel('$\sum_i R_i$', fontsize=30, usetex=True)
    ax.set_xlabel('output $f$', fontsize=30,  usetex=True)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.tick_params(axis='both', which='minor', labelsize=22)

    plt.legend()
    plt.ylim([-22,22])

    f.tight_layout()
    plt.show()
    
def plot_conservation_simple(Ls, Rs):
    f, ax = plt.subplots(1,1, figsize=(5, 5))
    from pylab import cm
    cmap = cm.get_cmap('plasma') #, 5) 
    
    ax.scatter(Ls, Rs, s=20, c='cyan', alpha=0.8) 
    ax.plot(Ls,Ls, color='black', linestyle='-', linewidth=1)

    ax.set_ylabel('$\sum_i R_i$', fontsize=30, usetex=True)
    ax.set_xlabel('output $f$', fontsize=30,  usetex=True)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.tick_params(axis='both', which='minor', labelsize=22)

    plt.legend()
    f.tight_layout()
    plt.show()



import seaborn as sns

def plot_long(R, words, threshold = 10):

    factor = int(np.ceil(len(words)/threshold))*0.5
   # factor = 1 * factor if factor == 3 else factor
    fig, ax = plt.subplots(1, 1, figsize=(10, factor))

    append = [0 for _ in range(threshold - len(words) % threshold)]
    tok_append = ['PAD' for _ in range(len(append))]
    reshape = (-1, threshold)

    
    r_normalization = np.max(np.abs(R))

    R = np.array(R)
    R_plot = np.array(R.tolist() + append)

    sns.heatmap(np.array(R_plot).reshape(reshape),
                annot=np.array(words + tok_append)[np.newaxis, :].reshape(reshape),
                fmt='', ax=ax, cmap='bwr', vmin=-r_normalization, vmax=r_normalization,
                annot_kws={"size": 10},
                cbar=False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
