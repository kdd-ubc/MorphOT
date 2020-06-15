import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import newton
import multiprocessing as mp
import time
try : 
    import cupy as cp
except : 
    pass

#import mrcfile


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
################################## MAIN BARYCENTER FUNCTIONS ######################################################################
###################################################################################################################################
###################################################################################################################################

# ----------------------------------------------------------------------------
#
def convolutional_barycenter(Hv,reg,alpha,stabThresh=1e-30,niter=1500,tol=1e-9,sharpening=False, verbose = False):
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
        import cupy
        #print('cupy is here, running on gpu')
        is_gpu = True
    except :
        #print('cupy not installed / no cuda GPU on computer, running on cpu')
        is_gpu = False 
        
    #print(alpha)
    if is_gpu :
        return convolutional_barycenter_gpu(Hv,reg,alpha,stabThresh,niter,tol,sharpening, verbose)
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

            
            Kw[i,:,:] = K(Hv[i,:,:] / np.maximum(stabThresh,K(v[i,:,:])) )
            
            
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

    return barycenter*mean_weights

# ----------------------------------------------------------------------------
#
def convolutional_barycenter_gpu(Hv,reg,alpha,stabThresh = 1e-30,niter = 1500, tol = 1e-9,sharpening = False,verbose = False):
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
    
    def K(x):
        return cp.array(gaussian_filter(cp.asnumpy(x),sigma=reg))
    
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
            print("iter : ",j , "change : ", change, 'time :', time.time()-t0)
        if change<tol :
            break
    
    
    return cp.asnumpy(barycenter*mean_weights)




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
            r = .15
            c1 = np.array([.5, .5, .25])
            c2 = np.array([.5, .5, .75])
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
            
            
        
        
        return (x,y,z),res_shape
    
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

####################################################################################################################################################
####################################################################################################################################################
################################# GAUSSIAN FILTER WITHOUT SCIPY (BASED ON VOLUME FILTER EXTENXION CHIMERA)##########################################
####################################################################################################################################################
####################################################################################################################################################


# ----------------------------------------------------------------------------
#
def ceil_power_of_2(n):
    """[summary]

    Arguments:
        n {integer} -- power of 2

    Returns:
        {integer} -- 2 to the power of n
    """
    p = 1
    while n > p:
        p *= 2
    return p

# ----------------------------------------------------------------------------
#
def gaussian(sdev, size, value_type):

    from math import exp
    from numpy import empty, add, divide

    g = empty((size,), value_type)
    for i in range(size):
        u = min(i,size-i) / sdev
        p = min(u*u/2, 100)               # avoid OverflowError with exp()
        g[i] = exp(-p)
    area = add.reduce(g)
    divide(g, area, g)
    return g

# ----------------------------------------------------------------------------
#
fast_fft_sizes = [2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 20, 21, 25, 32, 33, 36, 40, 48, 50, 54, 55, 64, 65, 72, 80, 81, 90, 96, 100, 108, 120, 128, 144, 150, 160, 162, 180, 192, 200, 216, 256, 270, 300, 320, 324, 325, 336, 360, 400, 432, 450, 480, 500, 512, 540, 576, 600, 640, 648, 720, 750, 800, 810, 864, 900, 960, 1024, 1080, 1152, 1200, 1280, 1296, 1350, 1440, 1500, 1536, 1600, 1620, 1728, 1800, 1920, 2048, 2160, 2250, 2304, 2400, 2560, 2700, 2880, 3000, 3072, 3200, 3240, 3600, 3840, 4096, 4160, 4176, 4180, 4186, 4187, 4192]
def efficient_fft_size(n):

    if n < fast_fft_sizes[-1]:
        from bisect import bisect
        b = bisect(fast_fft_sizes, n)
        s = fast_fft_sizes[b]
    else:
        s = ceil_power_of_2(n)
    return s

# ----------------------------------------------------------------------------
# Compute with zero padding in real-space to avoid cyclic-convolution.
#
def gaussian_convolution(data, ijk_sdev, value_type = None,
                         cyclic = False, cutoff = 5, invert = False, task = None):

    if value_type is None:
        value_type = data.dtype
      
    from numpy import array, float32, float64, multiply, divide, swapaxes
    vt = value_type if value_type == float32 or value_type == float64 else float32
    c = array(data, vt)
      
    from numpy.fft import rfft, irfft
    for axis in range(3):           # Transform one axis at a time.
        size = c.shape[axis]
        if size == 1:
            continue          # For a plane don't try to filter normal to plane.
        sdev = ijk_sdev[2-axis]       # Axes i,j,k are 2,1,0.
        hw = min(size/2, int(cutoff*sdev+1)) if cutoff else size/2
        nzeros = 0 if cyclic else hw  # Zero-fill for non-cyclic convolution.
        if nzeros > 0:
            # FFT performance is much better (up to 10x faster in numpy 1.2.1)
            # than other sizes.
            nzeros = efficient_fft_size(size + nzeros) - size
        g = gaussian(sdev, size + nzeros, vt)
        g[hw:-hw] = 0
        fg = rfft(g)                  # Fourier transform of 1-d gaussian.
        cs = swapaxes(c, axis, 2)     # Make axis 2 the FT axis.
        s0 = cs.shape[0]
        for p in range(s0):  # Transform one plane at a time.
            cp = cs[p,...]
            try:
                ft = rfft(cp, n=len(g))   # Complex128 result, size n/2+1
            except ValueError:
                raise MemoryError     # Array dimensions too large.
            if invert:
                divide(ft, fg, ft)
            else:
                multiply(ft, fg, ft)
            cp[:,:] = irfft(ft)[:,:size] # Float64 result
            if task:
                pct = 100.0 * (axis + float(p)/s0) / 3.0
                task.updateStatus('%.0f%%' % pct)
      
    if value_type != vt:
        return c.astype(value_type)
      
    return c