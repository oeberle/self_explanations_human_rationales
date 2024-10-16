from spacy import displacy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

def heat2hex(heat, transparency=90):
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
