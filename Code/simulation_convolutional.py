import numpy as np
import mrcfile
import pathlib
import time
from utils import *
import sys, os



with mrcfile.open('./data/3los.mrc') as mrc : 
    _3los = mrc.data

with mrcfile.open('./data/3iyf.mrc') as mrc : 
    _3iyf = mrc.data


frames = 25 
weights = [s/frames for s in range(frames)]
weights = [(elm,1-elm) for elm in weights]
print(weights)
for cgpu in ['cpu','gpu'] : 
    for niter in [20, 50, 100, 200] : 
        for downsample in [1,2,3,4] : 
        

            t0 = time.time()
            tmp1 = _3los[::downsample,::downsample,::downsample]
            tmp2 = _3iyf[::downsample,::downsample,::downsample]
            file_name = './time_results/%s_%s_niter=%i.txt'%(cgpu,tmp1.shape,niter)
            isFile = os.path.isfile(file_name) 
            if isFile : 
                print('stopping')
                continue
            
            for i in range(10) : 

                for weight in weights :
                    if cgpu == 'cpu' :
                        convolutional_barycenter_cpu([tmp1,tmp2],max(tmp1.shape)/60,weight, niter = niter)
                    else : 
                        convolutional_barycenter_gpu([tmp1,tmp2],max(tmp1.shape)/60,weight, niter = niter)

            
            time_for_10 = time.time()-t0

            file_name = './time_results/%s_%s_niter=%i.txt'%(cgpu,tmp1.shape,niter)
            with open(file_name,'w') as f : 
                f.write(str(time_for_10) + 'seconds for 10 times 25 frames')    

