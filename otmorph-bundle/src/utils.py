import numpy as np
from scipy.optimize import newton 
from scipy.ndimage import gaussian_filter 
import mrcfile
import time
from pkg_resources import packaging
try : 
    import cupy as cp
    if packaging.version.parse(cupy.__version__) < packaging.version.parse('8.2.0'):
        print('Update cupy to at least 8.2.0 for improved GPU performances')
    else :
        print('all is good wrt to cupy')
except : 
    pass
try : 
    import cupyx.scipy.ndimage
except :
    pass



###################################################################################################################################
###################################################################################################################################
############################################### TOY SIMULATION FUNCTION ###########################################################
###################################################################################################################################
###################################################################################################################################



# ----------------------------------------------------------------------------
#

def compute_from_shape_names(list_shape_names,N, weights):
    """From a list of shapes in 'sphere', 'cube', 'cone', 'spheres', 'cubes','cone_rotated','torus','bar' and a set of weights, return 
    the wasserstein barycenter distribution

    Arguments:
        list_shape_names {list of strings in 'sphere', 'cube', 'cone', 'spheres', 'cubes','cone_rotated','torus','bar'} -- 
        N {integer} -- Size of the meshgrid : NxNxN
        weights {list of float} -- weights for weighted barycenter
        result_file -- file in which the barycenter is written, if None, written locally

    Returns:
        [ndarray] -- Wasserstein barycenter of input distributions
    """
    Hv = []

    for i in list_shape_names:
        Hv.append(normalize(load_volume(N,i)[1]))
    Hv = np.array(Hv)
    barycenter = convolutional_barycenter(Hv,N/40.,weights,tol=2e-4).astype('float32')
    

    return barycenter

    
    
###################################################################################################################################
###################################################################################################################################
################################## MAIN TRANSPORT MAPS FUNCTIONS ##################################################################
###################################################################################################################################
###################################################################################################################################

def sinkhorn(mu0, mu1, reg, stabThresh = 1e-30, niter = 100) : 
    
    def K(x):
        
        return gaussian_filter(x,sigma=reg)
    
    
    mu0 = mu0 / mu0.sum()
    mu1 = mu1 / mu1.sum()
    
    if mu0.shape != mu1.shape :
        raise ValueError('Shapes are not the same')
    
    v = np.ones(mu0.shape)
    w = np.ones(mu1.shape)
    
    for i in range(niter):
        v = mu0 / np.maximum(stabThresh,K(w) )
        w = mu1 / np.maximum(stabThresh,K(v) ) 
        
        
    
    return v,w


    

def wasserstein_distance(mu0, mu1, reg, stabThresh = 1e-30, niter = 100) : 
    
    def K(x):
        
        return gaussian_filter(x,sigma=reg)
    
    
    mu0 = mu0 / mu0.sum()
    mu1 = mu1 / mu1.sum()
    area_weights = np.ones(mu0.shape)/mu0.size
    if mu0.shape != mu1.shape :
        raise ValueError('Shapes are not the same')
    
    v = np.ones(mu0.shape)
    w = np.ones(mu1.shape)
    
    for i in range(niter):
        v = mu0 / np.maximum(stabThresh,K(w) )
        w = mu1 / np.maximum(stabThresh,K(v) ) 
        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    
    # return np.sqrt(reg) * (mu0*np.log(np.maximum(v,1e-30)) + mu1*np.log(np.maximum(w,1e-30))).mean()
    # Not using the above formula because the regularization term is a function of the 
    # parameter in gaussian blurr, we don't precisely know what it is so we use a 
    # formula only based on the gaussian blurr
    return (v*np.log(np.maximum(v,1e-60))*K(w) + v * K(w*np.log(np.maximum(w,1e-60)))).mean()
###################################################################################################################################
###################################################################################################################################
################################## MAIN BARYCENTER FUNCTIONS ######################################################################
###################################################################################################################################
###################################################################################################################################

# ----------------------------------------------------------------------------
#
def convolutional_barycenter(Hv,reg,alpha,stabThresh=1e-30,niter=1500,tol=1e-9,sharpening=False, verbose = False, force_cpu =False, cupy_var = None, filter_func = None):
    """Calls convolutional barycenter function depending on whether or not cuda gpus are available

    Arguments:
        Hv {Set of distributions (ndarray)} -- 
        reg {regularization term "gamma"} -- float superior to 0, generally equals size of space/40
        alpha {list} -- set of weights

    Keyword Arguments:
        stabThresh {float} -- Stabilization threshold to prevent division by 0 (default: {1e-30})
        niter {int} -- Maximum number of loop iteration (default: {1500})
        tol {float} -- convergence tolerance at which point iterations stop (default: {1e-9})
        sharpening {bool} -- Whether or not entropic sharpening is used (default: {False})
        verbose {bool} --  verbose option

    Returns:
        ndarray/cparray -- solution of weighted wassertein barycenter problem
    """
    try : 
        #print(cupy_var.__version__)
        #print('cupy is here, running on gpu')
        is_gpu = True
    except :
        print('cupy not installed / no cuda GPU on computer, running on cpu')
        is_gpu = False 
        
    if is_gpu :
        return convolutional_barycenter_gpu(Hv,reg,alpha,stabThresh,niter,tol,sharpening, verbose,cupy_var,filter_func)
    else : 
        return convolutional_barycenter_cpu(Hv,reg,alpha,stabThresh,niter,tol,sharpening, verbose)

# ----------------------------------------------------------------------------
#    
def convolutional_barycenter_cpu(Hv, reg, alpha, stabThresh = 1e-30, niter = 1500, tol = 1e-9, sharpening = False, verbose = False):
    """Main function solving wasserstein barycenter problem using cpu

    Arguments:
        Hv {Set of distributions (ndarray)} -- 
        reg {regularization term "gamma"} -- float superior to 0, generally equals size of space/40
        alpha {list} -- set of weights

    Keyword Arguments:
        stabThresh {float} -- Stabilization threshold to prevent division by 0 (default: {1e-30})
        niter {int} -- Maximum number of loop iteration (default: {1500})
        tol {float} -- convergence tolerance at which point iterations stop (default: {1e-9})
        sharpening {bool} -- Whether or not entropic sharpening is used (default: {False})
        verbose {bool} --  verbose option

    Returns:
        ndarray -- solution of weighted wassertein barycenter problem
    """

    def K(x):
        
        return gaussian_filter(x,sigma=reg)
    
    def to_find_root(barycenter, H0, beta):
        return entropy(barycenter**beta) - H0
    
    alpha = np.array(alpha)/np.array(alpha).sum()
    Hv = np.array(Hv)
    mean_weights = (Hv[0].sum()*alpha[0]+Hv[1].sum()*alpha[1])

    #print('mean weights', mean_weights)
    for i in range(len(Hv)):

        Hv[i] = (Hv[i]-Hv[i].min())/Hv[i].sum()
        
    entropy_max = max_entropy(Hv)
    
    v = np.ones(Hv.shape)
    Kw = np.ones(Hv.shape)
    barycenter = np.zeros(Hv[0].shape)
    
    change = 1
    
    for j in range(niter):
        t0 = time.time()
        barycenterOld = barycenter
        
        barycenter = np.zeros_like(Hv[0,:,:])
        for i in range(Hv.shape[0]):

            
            Kw[i,:,:] = K(Hv[i,:,:] / np.maximum(stabThresh,K(v[i, :, :])) )
            
            
            barycenter += alpha[i] * np.log(np.maximum(stabThresh, v[i, :, :]*Kw[i, :, :]))
            

        barycenter = np.exp(barycenter)  
        change = np.sum(np.abs(barycenter-barycenterOld))
        
        if sharpening :
            if (entropy(barycenter)) > (entropy_max): 
                beta = newton(lambda beta : to_find_root(barycenter,entropy_max,beta), 1, tol=1e-6)
                if beta < 0 : 
                    beta = 1
            else :
                beta = 1
                
            barycenter = barycenter**beta
        
        for i in range(Hv.shape[0]):
            v[i, :, :] =  barycenter / np.maximum(stabThresh, Kw[i, : ,: ])
            
        if verbose : 
            print("iter : ",j , "change : ", change, "time :" , time.time()-t0)
            print("\n")
            print("\n")
        if change<tol :
            break

    return barycenter

# ----------------------------------------------------------------------------
#
def convolutional_barycenter_gpu(Hv,reg,alpha,stabThresh = 1e-30,niter = 1500, tol = 1e-9,sharpening = False,verbose = False, cp = None, filter_func = None):
    """Main function solving wasserstein barycenter problem using gpu

    """Main function solving wasserstein barycenter problem using gpu
    Arguments:
        Hv {Set of distributions (cparray)} -- 
        reg {regularization term "gamma"} -- float superior to 0, generally equals size of space/40
        alpha {list} -- set of weights

    Keyword Arguments:
        stabThresh {float} -- Stabilization threshold to prevent division by 0 (default: {1e-30})
        niter {int} -- Maximum number of loop iteration (default: {1500})
        tol {float} -- convergence tolerance at which point iterations stop (default: {1e-9})
        sharpening {bool} -- Whether or not entropic sharpening is used (default: {False})
        verbose {bool} --  verbose option
    Returns:
        cparray -- solution of weighted wassertein barycenter problem
    """
    #cp = cupy_var
    def K(x):
        try :
            
            lv = filter_func(x,sigma=reg)
        except : 
            lv = cp.array(gaussian_filter(cp.asnumpy(x),sigma=reg))
        return lv
        #return cp.array(gaussian_filter(cp.asnumpy(x),sigma=reg))
    
    def to_find_root(barycenter, H0, beta):
        return entropy(barycenter**beta) - H0
    
    alpha = cp.array(alpha)
    alpha = alpha/alpha.sum()
    Hv = cp.array(Hv)
    mean_weights = (Hv[0].sum()+Hv[1].sum())/2.
    #print('mean weights', mean_weights)
    for i in range(len(Hv)):
        Hv[i] = Hv[i]/Hv[i].sum()
    v = cp.ones(Hv.shape)
    Kw = cp.ones(Hv.shape)

    entropy_max = max_entropy(Hv)
    barycenter = cp.zeros(Hv[0].shape)
    
    change = 1
    for j in range(niter):
        t0 = time.time()
        barycenterOld = barycenter
        
        barycenter = cp.zeros_like(Hv[0, :, :])
        for i in range(Hv.shape[0]):

            
            Kw[i, :, :] = K(Hv[i, :, :] / cp.maximum(stabThresh,K(v[i, :, :])) )
            barycenter += alpha[i] * cp.log(cp.maximum(stabThresh, v[i, :, :]*Kw[i, :, :]))
            
        barycenter = cp.exp(barycenter)  
        change = cp.sum(cp.abs(barycenter-barycenterOld))
        if sharpening :
            if (entropy(barycenter)) > (entropy_max): 
                
                beta = newton(lambda beta : to_find_root(barycenter,entropy_max,beta), 1, tol=1e-6)
                if beta < 0 : 
                    beta = 1
                
            else :
                beta = 1
            barycenter = barycenter**beta
        
        for i in range(Hv.shape[0]):
            v[i, :, :] =  barycenter / cp.maximum(stabThresh, Kw[i, : ,: ])
        

        if verbose :
            #sys.stdout('output.log','a')
            print("iter : ",j , "change : ", change, 'time :', time.time()-t0)
        if change<tol :
            break

    return cp.asnumpy(barycenter)




###################################################################################################################################
###################################################################################################################################
################################# ENTROPY FUNCTIONS ###############################################################################
###################################################################################################################################
###################################################################################################################################



# ----------------------------------------------------------------------------
#
def entropy(distrib):
    """Returns the entropy of a distribution using numpy (cpu)

    Arguments:
        distrib {ndarray} -- input distribution

    Returns:
        float -- entropy of input distribution
    """
    return -np.sum(distrib[distrib>0]*np.log(distrib[distrib>0]))

# ----------------------------------------------------------------------------
#
def entropy_gpu(distrib):
    """Returns the entropy of a distribution using cupy (gpu)

    Arguments:
        distrib {cparray} -- input distribution

    Returns:
        float -- entropy of input distribution
    """
    return -cp.sum(distrib[distrib>0]*cp.log(distrib[distrib>0]))
# ----------------------------------------------------------------------------
#
def max_entropy_gpu(Hv):
    """Returns the maximum entropy of a set of distributions using cupy (gpu)

    Arguments:
        distrib {set of cp_array distributions} -- input distributions

    Returns:
        float -- maximum entropy of input distributions
    """
    entropy_max = entropy_gpu(Hv[0])
    for i in range(1,len(Hv)):
        entropy_i = entropy_gpu(Hv[i])
        if entropy_i>entropy_max:
            
            entropy_max = entropy_i
    return entropy_i
# ----------------------------------------------------------------------------
#
def max_entropy(Hv):
    """Returns the maximum entropy of a set of distributions using numpy (cpu)

    Arguments:
        distrib {set of nd_array distributions} -- input distributions

    Returns:
        float -- maximum entropy of input distributions
    """
    entropy_max = entropy(Hv[0])
    for i in range(1,len(Hv)):
        entropy_i = entropy(Hv[i])
        if entropy_i>entropy_max:
            
            entropy_max = entropy_i
    return entropy_i






####################################################################################################################################################
####################################################################################################################################################
################################# HELPING FUNCTIONS FOR TOY SIMULATIONS ############################################################################
####################################################################################################################################################
####################################################################################################################################################


# ----------------------------------------------------------------------------
#
def trim_zeros(arr, margin=0):
    '''
    Trim the leading and trailing zeros from a N-D array.

    :param arr: numpy array
    :param margin: how many zeros to leave as a margin
    :returns: trimmed array
    :returns: slice object
    '''
    s = []
    for dim in range(arr.ndim):
        start = 0
        end = -1
        slice_ = [slice(None)]*arr.ndim

        go = True
        while go:
            slice_[dim] = start
            go = not np.any(arr[tuple(slice_)])
            start += 1
        start = max(start-1-margin, 0)

        go = True
        while go:
            slice_[dim] = end
            go = not np.any(arr[tuple(slice_)])
            end -= 1
        end = arr.shape[dim] + min(-1, end+1+margin) + 1

        s.append(slice(start,end))
    return arr[tuple(s)], tuple(s)

# ----------------------------------------------------------------------------
#
    
def load_volume(N,shape):
    
    available_shapes = ['sphere', 'cube', 'cone', 'spheres', 'cubes','cone_rotated','torus','bar']
    N+=1
    x, y, z = np.indices((N, N, N))/(N-1)
    xc = midpoints(x)
    yc = midpoints(y)
    zc = midpoints(z)
    if shape not in available_shapes:
        print('shape not available')
        return None
    else :
        if shape =='sphere':
            r = 0.35
            c = np.array([1,1,1])/2
            res_shape = (xc - c[0])**2 + (yc - c[1])**2 + (zc - c[2])**2 < r**2
            
        
        elif shape == 'cube':
            r = .35
            c = np.array([1, 1, 1])/2
            res_shape = (abs(xc-c[0]) < r) & (abs(yc-c[1]) < r) & (abs(zc-c[2]) < r)
            
            
        elif shape == 'cubes':
            r = .225
            c1 = np.array([.5, .5, .225])
            c2 = np.array([.5, .5, .775])
            cube1 = (abs(xc-c1[0]) < r) & (abs(yc-c1[1]) < r) & (abs(zc-c1[2]) < r)
            cube2 = (abs(xc-c2[0]) < r) & (abs(yc-c2[1]) < r) & (abs(zc-c2[2]) < r)
            res_shape = cube1 + cube2
            
        
        elif shape == 'spheres' : 
            r = .15
            c1 = np.array([.25, .5, .5])
            c2 = np.array([.75, .5, .5])
            sphere1 = (xc - c1[0])**2 + (yc - c1[1])**2 + (zc - c1[2])**2 < r**2
            sphere2 = (xc - c2[0])**2 + (yc - c2[1])**2 + (zc - c2[2])**2 < r**2
            res_shape = sphere1 + sphere2 
            
        elif shape == 'cone' :  
            
            cone = ((xc-.5)**2 + (yc-.5)**2 <= (zc*.36)**2 ) & (zc<.8)
            res_shape = cone
            
        elif shape == 'cone_rotated':
            radius = .8
            center = [1, 1, 1]
            [X,Y,Z] = np.mgrid[-1:1:60j, -1:1:60j, -1:1:60j]
            D = np.sqrt(Y**2+Z**2)
            eta = .9
            res_shape = (X>=0)*(D<=radius*(X/eta))
            zeros = np.zeros((30,60,60))
            new=np.vstack((res_shape,zeros))
            res_shape = new[30:,:,:]
            
        elif shape == 'torus':
            Radius = .3
            radius = .1
            res_shape = (np.sqrt((xc-.5)**2+(yc)**2) - Radius)**2 + (zc-.5)**2 <= radius**2
            
        elif shape == 'bar':
            r = .1
            c = np.array([1, 1, 1])/2
            res_shape = (abs(xc-.3) < 1) & (((yc-.3)**2+(zc-c[2])**2) < r**2)
            
            
        
        
        return res_shape
    
# ----------------------------------------------------------------------------
#
        
def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

# ----------------------------------------------------------------------------
#
    
def normalize(array):
    return array/array.sum()

