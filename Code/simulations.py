import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from help_functions import *
import pylab as pl
import mrcfile 
import pathlib
import wget
#sys.stdout = open('output.log', 'a')

def main() :
    

    # SETTING PARAMETERS : 
    # -------------------
    # Tuples of set of structure you want to get barycenters from, please include file format
    #list_structures = [('emd_4121.map.gz','emd_4122.map.gz','emd_4123.map.gz'),('2jd8_01.mrc','2jd8_012.mrc','2jd8_020.mrc')]
    #list_structures = [('2jd8_01.mrc','2jd8_012.mrc','2jd8_020.mrc')]
    #list_structures = [('emd_4121.map.gz','emd_4122.map.gz','emd_4123.map.gz')]
    list_structures = [('migrating_t0.jpg','migrating_t4.jpg'),
                        ('migrating_t4.jpg','migrating_t9.jpg'),
                        ('migrating_t9.jpg','migrating_t15.jpg'),
                        ]


    # -------------------
    # Set of weights for each tuple of structures. Type : list of list of tuples. Please have as many sub list as there are sub-simulations
    #list_weights = [[(1,0,0),(0,1,0),(0,0,1),(1,1,1),(2,1,0),(0,1,2),(1,0,2),(2,0,1),(0,2,1),(1,2,0)],
    #                [(1,0,0),(0,1,0),(0,0,1),(1,1,1),(2,1,0),(0,1,2),(1,0,2),(2,0,1),(0,2,1),(1,2,0)]]
    #list_weights = [[(1,0,0),(0,1,0),(0,0,1),(1,1,1),(2,1,0),(0,1,2),(1,0,2),(2,0,1),(0,2,1),(1,2,0)]]
    list_weights = [[(10,0),(9,1),(8,2),(7,3),(6,4),(5,5),(4,6),(3,7),(2,8),(1,9)],
                    [(10,0),(9,1),(8,2),(7,3),(6,4),(5,5),(4,6),(3,7),(2,8),(1,9)],
                    [(10,0),(9,1),(8,2),(7,3),(6,4),(5,5),(4,6),(3,7),(2,8),(1,9)],
                    ]
    W = [
    [0, 0, 1], 
    [1, 0, 3] ,[0, 1, 3] ,
    [1,0,1], [1,1,2] ,[0,1,1] ,
    [3,0,1], [2,1,1] ,[1,2,1], [0,3,1] ,  
    [1,0,0] ,[3,1,0], [1,1,0], [1,3,0], [0,1,0] 
    ]

    W6 = [
        [1,0,0],
        [4,0,1],[4,1,0],
        [3,0,2],[3,1,1],[3,2,0],
        [2,0,3],[2,1,2],[2,2,1],[2,3,0],
        [1,0,4], [1,1,3], [1,2,2], [1,3,1] , [1,4,0],
        [0,0,1],[0,1,4],[0,2,3],[0,3,2],[0,4,1],[0,1,0]
        ]

    W15_only_outside = [(15,0,0),(14,1,0),(13,2,0),(12,3,0),(11,4,0),(10,5,0),
                    (9,6,0),(8,7,0),(7,8,0),(6,9,0),(5,10,0),(4,11,0),
                    (3,12,0),(2,13,0),(1,14,0),(0,15,0),
                    
                    (0,15,0),(0,14,1),(0,13,2),(0,12,3),(0,11,4),(0,10,5),
                    (0,9,6),(0,8,7),(0,7,8),(0,6,9),(0,5,10),(0,4,11),
                    (0,3,12),(0,2,13),(0,1,14),(0,0,15),
                    
                    (0,0,15),(1,0,14),(2,0,13),(3,0,12),(4,0,11),(5,0,10),
                    (6,0,9),(7,0,8),(8,0,7),(9,0,6),(10,0,5),(11,0,4),
                    (12,0,3),(13,0,2),(14,0,1),(15,0,0),]


    #list_weights = [W6]

    reg = 22 # if reg is not specified here (ie reg = 0), the regularization term will be chosen based on the space size
    niter = 30 # Maximum number of iterations, usually 100 is enough
    tol = 1e-9  # convergence threshold, if the change if below this number between two iteration, it will override niter and stop the computation
    sharpening = False # Whether or not entropic sharpening is used : beware it increases computation time a lot
    verbose = True # Whether or not the computation prints a lot of things. It will anyway only be written in 'output.log' file

    downsample_ratio = 1

    thresholded = False
    ResDir = 'Results_migrating'
    if thresholded : 
        ResDir = ResDir+'thresholded'
    SubResDir = str(reg)
    all_structures = np.array(list_structures).flatten()
    for structure in all_structures : 
        download_structure(structure)

    counter = 0 #here to name the structure in 'chronological' order

    for i,tuple_structures in enumerate(list_structures) : 
        ### getting structures  in Hv 
        Hv = []
        data_directory = pathlib.Path('.') / 'data'
        map_dir = data_directory 

        main_dict_key = ''
        for structure in tuple_structures : 
            main_dict_key += '-' + structure.split('.')[0]
            fname = structure 
            try : 
                with mrcfile.open(f'{str(map_dir)}/{fname}') as mrc:
                    save_as_im = False
                    volume = mrc.data
                    volume = volume[::downsample_ratio, ::downsample_ratio, :: downsample_ratio]
                    volume = volume - np.min(volume)
                    volume = volume/volume.sum()
                    Hv.append(volume)
                    print(f'> Loaded map of shape: {volume.shape}')
            
            except : 
                save_as_im = True
                volume = pl.imread(f'{str(map_dir)}/{fname}')[:,:,1]
                if thresholded : 
                    volume = (volume>30).astype('float32')
                    
                volume = volume[::downsample_ratio, ::downsample_ratio]
                volume = volume - np.min(volume)
                volume = volume/volume.sum()
                Hv.append(volume.astype('float32'))
                print(f'> Loaded image of shape: {volume.shape}')
        

        for tuple_weights in list_weights[i]:
            weights_key = str(tuple_weights)
            results_directory = pathlib.Path('.') / 'Results' 
            results_directory.mkdir(parents=True, exist_ok=True)

            if not save_as_im : 
                if  counter<10 : 
                    result_file = pathlib.Path('.') / ('%s'%ResDir)/('r00%i_%s_%s_%s_%s_sharpen%s.mrc'%(counter,main_dict_key,weights_key,str(Hv[0].shape),str(tol),str(sharpening)))
                else : 
                    result_file = pathlib.Path('.') / ('%s'%ResDir)/('r0%i_%s_%s_%s_%s_sharpen%s.mrc'%(counter,main_dict_key,weights_key,str(Hv[0].shape),str(tol),str(sharpening)))

            else : 
                if counter<10 : 
                    result_file = pathlib.Path('.') / ('%s'%ResDir)/('%s'%SubResDir)/('r00%i_%s_%s_%s_%s_sharpen%s.jpg'%(counter,main_dict_key,weights_key,str(Hv[0].shape),str(tol),str(sharpening)))
                else : 
                    result_file = pathlib.Path('.') / ('%s'%ResDir)/('%s'%SubResDir)/('r0%i_%s_%s_%s_%s_sharpen%s.jpg'%(counter,main_dict_key,weights_key,str(Hv[0].shape),str(tol),str(sharpening)))

            counter +=1
            if not reg : 
                reg = max(Hv[0].shape)/40

            if pathlib.Path(result_file).is_file():
                continue
            else: 
                shapes = list(set([elm.shape for elm in Hv]))
                if len(shapes)>1 : 
                    print('the maps do not have the same shape, break')
                    break
                
                barycenter = convolutional_barycenter(Hv, reg, tuple_weights, niter = niter, tol = tol, sharpening = sharpening, verbose = verbose) 

                if not save_as_im : 

                    with mrcfile.new(result_file) as mrc:
                    
                        print(barycenter.dtype)
                        mrc.set_data(barycenter)
                else : 

                    pl.imsave(result_file,barycenter,cmap='Greys_r')

    # ------------------------------------------------------------------------------------------------------------------------
    #

def download_structure(structure):
    # make sure we have a local directory to download the data in:
    data_directory = pathlib.Path('.') / 'data'  # directory where the data is

    map_dir = data_directory 
    map_dir.mkdir(parents=True, exist_ok=True)

    url_emdb='ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/'

    id = structure.split('_')[1].split('.')[0]
    err_count = 0
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
        
    #print(f'Done!')




if __name__ == '__main__' : 
    main()



