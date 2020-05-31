import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from help_functions import *
import json
import pickle as pkl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import time
import copy
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pylab as pl
import plotly.io as pio
pio.renderers.default = "browser"
#
import mrcfile 
import pathlib
import wget



def download_structure(list_ids):
    # make sure we have a local directory to download the data in:
    data_directory = pathlib.Path('.') / 'data'  # directory where the data is

    map_dir = data_directory / 'EMPIAR' / '10077' 
    map_dir.mkdir(parents=True, exist_ok=True)

    url_emdb='ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/'


    for id in list_ids:
        fname = f'emd_{id}.map.gz'
        if pathlib.Path(map_dir / fname).is_file():
            print(f'>>> {fname} already downloaded')
            
        else:
            url = f'{url_emdb}EMD-{id}/map/{fname}'
            print(f'>>> downloading {url}...')
            try : 
                wget.download(url, out=str(map_dir))
            except : 
                print(f'{id} is an unexisting structure')
                err_count += 1
            
    print(f'Done! with ',err_count,'error(s)')
        
print(f'Done!')