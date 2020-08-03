# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===




#### Scale_Factors and addMode should be deleted to my mind 


# -----------------------------------------------------------------------------
#
class Interpolated_Map:

  def __init__(self, volumes, scale_factors = None, adjust_thresholds = False,
               add_mode = False, interpolate_colors = True,
               subregion = 'all', step = 1, model_id = None, niter = 20, reg = None, rate = 'linear'):

    self.volumes = volumes
    v0 = volumes[0]
    self.session = session = v0.session

    r = None
    if model_id:
      vlist = session.models.list(model_id = model_id)
      if len(vlist) == 1:
        v = vlist[0]
        if (v.matrix_size(step = (1,1,1), subregion = 'all')
            == v0.matrix_size(step = step, subregion = subregion)):
          r = v
    if r is None:
      r = v0.writable_copy(require_copy = True, copy_colors = False,
                           unshow_original = False,
                           subregion = subregion, step = step,
                           model_id = model_id, name = 'morph')
    self.result = r
    self.subregion = subregion
    self.step = step

    if scale_factors is None:
      scale_factors = [1] * len(volumes)
    self.scale_factors = scale_factors
    self.add_mode = add_mode

    self.interpolate_colors = interpolate_colors

    self.f = self.fmin = self.fmax = self.fstep = 0
    self.steps = None
    self.f_changed_cb = None
    self.play_handler = None
    self.step_direction = 1     # 1 or -1, current direction when looping
    self.recording = False

    self.niter = niter
    self.reg = reg
    self.rate = rate
    self.adjust_thresholds = adjust_thresholds
    self.surface_level_ranks = []       # For avoiding creep during threshold
    self.image_level_ranks = []         #   normalization.

  # ---------------------------------------------------------------------------

  ######### AJOUT PERSO 
  def interpolate_ot(self, f):

    v = self.result
    if self.adjust_thresholds:
      self.record_threshold_ranks(v)

    changed_f = get_weights(self.rate,f)
    #f1, f2, sf1, sf2, v1, v2 = self.coefficients(f)
    f1, f2, sf1, sf2, v1, v2 = self.coefficients(changed_f)
    if v.data is None or v1.data is None or v2.data is None:
      return False

    

    ot_combination(sf1, v1, sf2, v2, v, self.subregion, self.step, self.niter, self.reg)
    self.f = f

    if self.adjust_thresholds:
      self.set_threshold_ranks(v)

    if self.interpolate_colors:
      interpolate_colors(f1, v1, f2, v2, v)

    return True

  # ---------------------------------------------------------------------------
  #
  def coefficients(self, f):

    vlist = self.volumes
    n = len(vlist)
    i0 = min(n-2,max(0,int(f*(n-1))))
    f0 = float(i0)/(n-1)
    if self.add_mode:
      f1 = 1.0
      f2 = f
    else:
      f2 = (f-f0)*(n-1)
      f1 = 1-f2
    sf = self.scale_factors
    sf1 = f1 * sf[i0]
    sf2 = f2 * sf[i0+1]

    return f1, f2, sf1, sf2, vlist[i0], vlist[i0+1]
  
  


  # ---------------------------------------------------------------------------
  #AJOUT PERSO ATTENTION
  def play_ot(self, f, fmin, fmax, fstep, f_changed_cb, fdir = None, steps = None):

    self.f = f
    self.fmin = fmin
    self.fmax = fmax
    self.fstep = fstep
    self.f_changed_cb = f_changed_cb
    self.steps = steps
    if not fdir is None:
      if fdir >= 0: self.step_direction = 1
      else:         self.step_direction = -1
    if self.play_handler is None:
      h = self.session.triggers.add_handler('new frame', self.next_frame_cb_ot)
      self.play_handler = h
  # ---------------------------------------------------------------------------
  #
  def stop_playing(self):

    self.session.triggers.remove_handler(self.play_handler)
    self.play_handler = None
    if self.recording:
      self.finish_recording()

  # ---------------------------------------------------------------------------
  #
  def playing(self):

    return self.play_handler != None


  # ---------------------------------------------------------------------------
  #
  def next_frame_cb_ot(self, *_):

    if not self.steps is None:
      if self.steps <= 0:
        self.stop_playing()
        return
      else:
        self.steps -= 1

    fmin, fmax = self.fmin, self.fmax
    next_f = self.f + self.fstep * self.step_direction

    if next_f >= fmax:
      next_f = fmax
      self.step_direction = -1
    elif next_f <= fmin:
      next_f = fmin
      self.step_direction = 1
    if not self.interpolate_ot(next_f):
      self.stop_playing()       # Volume closed.
    fccb = self.f_changed_cb
    if fccb:
      fccb(self.f)
  """
  # ---------------------------------------------------------------------------
  #
  def record(self, fmin, fmax, fstep, f_changed_cb, roundtrip, record_args, save_movie_cb):

    if self.recording:
      return

    self.f = self.fmin = fmin
    self.fmax = fmax
    self.fstep = fstep
    self.f_changed_cb = f_changed_cb
    self.save_movie_cb = save_movie_cb
    
    self.recording = True

    from math import ceil
    steps = int(ceil((fmax - fmin) / fstep))
    if roundtrip:
      steps *= 2

    from chimerax.core.commands import run
    run(self.session, 'movie record ' + record_args)

    self.play(fmin, fmin, fmax, fstep, f_changed_cb, 1, steps)

  # ---------------------------------------------------------------------------
  #
  def finish_recording(self):

    self.recording = False

    if self.play_handler:
      self.stop_playing()

    from chimerax.core.commands import run
    runCommand(self.session, 'movie stop')

    self.save_movie_cb()
"""

  # ---------------------------------------------------------------------------
  #
  def record_threshold_ranks(self, v):

    ms = v.matrix_value_statistics()

    rlev = [ms.rank_data_value(r) for r in self.surface_level_ranks]
    slev = [s.level for s in v.surfaces]
    if slev != rlev:
      self.surface_level_ranks = [ms.data_value_rank(lev) for lev in slev]

    rlev = [ms.rank_data_value(r) for r in self.image_level_ranks]
    slev = [lev for lev,b in v.image_levels]
    if slev != rlev:
      self.image_level_ranks = [ms.data_value_rank(lev) for lev in slev]

  # ---------------------------------------------------------------------------
  #
  def set_threshold_ranks(self, v):

    ms = v.matrix_value_statistics()

    sflev = [ms.rank_data_value(r) for r in self.surface_level_ranks]
    imlev = list(zip([ms.rank_data_value(r) for r in self.image_level_ranks],
                     [b for lev,b in v.image_levels]))
    v.set_parameters(surface_levels = sflev, image_levels = imlev)


# -----------------------------------------------------------------------------
#
def interpolate_colors(f1, v1, f2, v2, v):

  nc = len(v.surfaces)
  if len(v1.surfaces) == nc and len(v2.surfaces) == nc:
    from chimerax.geometry import linear_combination
    for s, s1, s2 in zip(v.surfaces, v1.surfaces, v2.surfaces):
      s.rgba = linear_combination(f1, s1.rgba, f2, s2.rgba)

  nc = len(v.image_colors)
  if len(v1.image_colors) == nc and len(v2.image_colors) == nc:
    from chimerax.geometry import linear_combination
    scolors = [linear_combination(f1, v1.image_colors[c], f2, v2.image_colors[c])
               for c in range(nc)]
    v.set_parameters(image_colors = scolors)


### AJOUT PERSONNEL 

def morph_map_linear_ot(volumes, play_steps, play_start, play_step, play_direction,
               play_range, add_mode, adjust_thresholds, scale_factors,
               hide_maps, interpolate_colors, subregion, step, model_id, niter, reg, rate):

  if hide_maps:
    for v in volumes:
      v.display = False
  
  """if step > 1 and gaussfilt: 
    for v in volumes :
      from scipy.ndimage import gaussian_filter
      m = v.full_matrix() 
      m = gaussian_filter(m,sigma = 2*step/6.)
      v.data.values_changed()"""


  im = Interpolated_Map(volumes, scale_factors, adjust_thresholds, add_mode,
                        interpolate_colors, subregion, step, model_id, niter, reg, rate)
  if play_steps > 0:
    fmin, fmax = play_range
    im.play_ot(play_start, fmin, fmax, play_step, None, play_direction, play_steps)
  else:
    im.interpolate_ot(play_start)

  return im
  


def morph_maps_ot(volumes, play_steps, play_start, play_step, play_direction,
               play_range, add_mode, adjust_thresholds, scale_factors,
               hide_maps, interpolate_colors, subregion, step, model_id, niter, reg, rate):

  if hide_maps:
    for v in volumes:
      v.display = False
  
  """if step > 1 and gaussfilt: 
    for v in volumes :
      from scipy.ndimage import gaussian_filter
      m = v.full_matrix() 
      m = gaussian_filter(m,sigma = 2*step/6.)
      v.data.values_changed()"""


  im = Interpolated_Map(volumes, scale_factors, adjust_thresholds, add_mode,
                        interpolate_colors, subregion, step, model_id, niter, reg, rate)
  if play_steps > 0:
    fmin, fmax = play_range
    im.play_ot(play_start, fmin, fmax, play_step, None, play_direction, play_steps)
  else:
    im.interpolate_ot(play_start)

  return im


def ot_combination(f1, v1, f2, v2, v, subregion, step, niter, reg):
  
  m = v.full_matrix()


  m1 = v1.matrix(step = step, subregion = subregion)
  m2 = v2.matrix(step = step, subregion = subregion)
  from .utils import convolutional_barycenter, convolutional_barycenter_cpu, convolutional_barycenter_gpu
  import time 
  t0 = time.time()
  m[:,:,:] = convolutional_barycenter_gpu([m1,m2],reg,(f1,f2),niter=niter,verbose=False)
  print(time.time()-t0)
  v.data.values_changed()


def ot_barycenter(volumes, weights, niter, reg, subregion = 'all', step = 1, model_id = None) : 
  v1 = volumes[0]
  v2 = volumes[1]

  m1 = v1.matrix(step = step, subregion = subregion)
  m2 = v2.matrix(step = step, subregion = subregion)

  
  alpha = (weights[0],weights[1])

  
  r = v1.writable_copy(require_copy = True, copy_colors = False,
                           unshow_original = False,
                           subregion = subregion, step = step,
                           model_id = model_id, name = '%s %s weights %s'%(v1.name,v2.name, str(alpha)))

  m = r.full_matrix()

  from .utils import convolutional_barycenter
  m[:,:,:] = convolutional_barycenter([m1,m2],reg, weights, niter = niter, verbose = False)

  r.data.values_changed()
  return r


def ot_save(volumes, dir_name, frames, niter, reg, rate, subregion='all', step = 1, model_id = None, name1 = None, name2 = None) : 
  v1 = volumes[0]
  v2 = volumes[1]

  m1 = v1.matrix(step = step, subregion = subregion)
  m2 = v2.matrix(step = step, subregion = subregion)

  from .utils import convolutional_barycenter, convolutional_barycenter_cpu, convolutional_barycenter_gpu
  import mrcfile 
  import progressbar

  rate_function = RateMap[rate.lower()]

  all_weights = rate_function(frames)
  all_weights = [(round(i,6),round(1-i,6)) for i in all_weights]
  for i,weights in enumerate(all_weights): 
    if i%5 == 0 : 
      print('(%i of %i)'%(i,frames))
    padding = len(str(frames))+1
    str_i = str(i)
    padded_i = str_i.zfill(padding)
    if name1 == None : 
      name1 = v1.name
    if name2 == None : 
      name2 = v2.name
    #result_file = dir_name + '/' +padded_i +'ot_%s_%s_weights%s.mrc'%(v1.name,v2.name, str(weights))
    result_file = dir_name + '/' +padded_i +'ot_%s_%s_weights%s.mrc'%(name1,name2, str(weights))
    import time
    t0 = time.time()
    m = convolutional_barycenter([m1,m2],reg, weights, niter = niter, verbose = False)
    print(time.time()-t0)
    from chimerax.map.data import ArrayGridData
    from chimerax.map.data.mrc import save

    m_grid = ArrayGridData(m)

    save(m_grid, result_file)


def linear_save(volumes, dir_name, frames, niter, reg, rate, subregion='all', step = 1, model_id = None) : 
  v1 = volumes[0]
  v2 = volumes[1]

  m1 = v1.matrix(step = step, subregion = subregion)
  m2 = v2.matrix(step = step, subregion = subregion)

  from .utils import convolutional_barycenter, convolutional_barycenter_cpu, convolutional_barycenter_gpu
  import mrcfile 
  import progressbar

  rate_function = RateMap[rate.lower()]

  all_weights = rate_function(frames)
  all_weights = [(round(i,6),round(1-i,6)) for i in all_weights]
  for i,weights in enumerate(all_weights): 
    if i%5 == 0 : 
      print('(%i of %i)'%(i,frames))


    padding = len(str(frames))+1
    str_i = str(i)
    padded_i = str_i.zfill(padding)

    result_file = dir_name + '/' + padded_i + 'linear_%s_%s_weights%s.mrc'%(v1.name,v2.name, str(weights))
    import time
    t0 = time.time()
    m = weights[0]*m1+weights[1]*m2
    print(time.time()-t0)
    from chimerax.map.data import ArrayGridData
    from chimerax.map.data.mrc import save

    m_grid = ArrayGridData(m)

    save(m_grid, result_file)

  
def get_weights(rate, f):
  import math
  s = f
  if rate.lower() == 'linear' : 
    return f

  elif (rate.lower() == 'sinus') or (rate.lower() == 'sinusoidal') : 
    return (math.cos(math.pi +f*math.pi)+1)/2.

  elif (rate.lower() == 'rampup') or (rate.lower() == 'ramp up') : 
    return (math.cos( math.pi + f * math.pi / 2. ) + 1)
  
  elif (rate.lower() == 'rampdown') or (rate.lower() == 'ramp down') :
    return (math.sin( f * math.pi/2))

  else : 
    raise ValueError('rate not recognized, expected a string in ["linear","sinus","ramp up","ramp down"]') 
 


def rateLinear(frames):
        "Generate fractions from 0 to 1 linearly (excluding start/end)"
        return [ float(s) / frames for s in range(0, frames+1) ]

def rateSinusoidal(frames):
        """Generate fractions from 0 to 1 sinusoidally
        (slow at beginning, fast in middle, slow at end)"""
        import math
        piOverTwo = math.pi / 2
        rate = []
        for s in rateLinear(frames):
                a = math.pi + s * math.pi
                v = math.cos(a)
                r = (v + 1) / 2
                rate.append(r)
        return rate

def rateRampUp(frames):
        """Generate fractions from 0 to 1 sinusoidally
        (slow at beginning, fast at end)"""
        import math
        piOverTwo = math.pi / 2
        rate = []
        for s in rateLinear(frames):
                a = math.pi + s * piOverTwo
                v = math.cos(a)
                r = v + 1
                rate.append(r)
        return rate

def rateRampDown(frames):
        """Generate fractions from 0 to 1 sinusoidally
        (fast at beginning, slow at end)"""
        import math
        piOverTwo = math.pi / 2
        rate = []
        for s in rateLinear(frames):
                a = s * piOverTwo
                r = math.sin(a)
                rate.append(r)
        return rate

RateMap = {
        "linear": rateLinear,
        "sinusoidal": rateSinusoidal,
        "ramp up": rateRampUp,
        "ramp down": rateRampDown,
}

#######################################################################
#######################################################################
####################### NOUVEEEEAAAUTEEE ##############################
#######################################################################
#######################################################################





class Semi_Interpolated_Map:

  def __init__(self, volumes, scale_factors = None, adjust_thresholds = False,
               add_mode = False, interpolate_colors = True,
               subregion = 'all', step = 1, model_id = None, niter = 20, reg = None, rate = 'linear'):

    self.volumes = volumes
    v0 = volumes[0]
    self.session = session = v0.session

    r = None
    if model_id:
      vlist = session.models.list(model_id = model_id)
      if len(vlist) == 1:
        v = vlist[0]
        if (v.matrix_size(step = (1,1,1), subregion = 'all')
            == v0.matrix_size(step = step, subregion = subregion)):
          r = v
    if r is None:
      r = v0.writable_copy(require_copy = True, copy_colors = False,
                           unshow_original = False,
                           subregion = subregion, step = step,
                           model_id = model_id, name = 'morph')
    self.result = r
    self.subregion = subregion
    self.step = step

    if scale_factors is None:
      scale_factors = [1] * len(volumes)
    self.scale_factors = scale_factors
    self.add_mode = add_mode

    self.interpolate_colors = interpolate_colors

    self.f = self.fmin = self.fmax = self.fstep = 0
    self.total_steps = None
    self.ot_steps = None
    self.last_ot_weight = 0
    self.tmp_v1 = None
    self.tmp_v2 = None
    self.precomputed = None
    self.tmp_index = 0 
    self.precompute = False

    self.f_changed_cb = None
    self.play_handler = None
    self.step_direction = 1     # 1 or -1, current direction when looping
    self.recording = False

    self.niter = niter
    self.reg = reg
    self.rate = rate
    self.adjust_thresholds = adjust_thresholds
    self.surface_level_ranks = []       # For avoiding creep during threshold
    self.image_level_ranks = []         #   normalization.

    print('using new class')

  # ---------------------------------------------------------------------------

  ######### AJOUT PERSO 
  def interpolate_ot(self, f):

    v = self.result
    if self.adjust_thresholds:
      self.record_threshold_ranks(v)

    changed_f = get_weights(self.rate,f)
    #f1, f2, sf1, sf2, v1, v2 = self.coefficients(f)
    f1, f2, sf1, sf2, v1, v2 , vol1, vol2 = self.coefficients(changed_f)
    if v.data is None or v1.data is None or v2.data is None:
      return False

    

    linear_combination(sf1, vol1, sf2, vol2, v, self.subregion, self.step)
    self.f = f

    if self.adjust_thresholds:
      self.set_threshold_ranks(v)

    if self.interpolate_colors:
      interpolate_colors(f1, v1, f2, v2, v)

    return True

  # ---------------------------------------------------------------------------
  #
  def coefficients(self, f):
    import numpy as np
    vlist = self.volumes
    n = len(vlist)
    ot_weights = self.ot_weights 




    i0 = min(n-2,max(0,int(f*(n-1))))

    if f == 0 : 
      indices = (0,1)
    elif f == 1 : 
      indices = (-2,-1)
    else : 
      tmp = np.max(np.where(ot_weights <= f ))
      indices = (tmp, tmp+1)

    from .utils import convolutional_barycenter_gpu, convolutional_barycenter_cpu, convolutional_barycenter

    v1 = vlist[i0].matrix(step = self.step, subregion = self.subregion)
    v2 = vlist[i0+1].matrix(step = self.step, subregion = self.subregion)

    

    if f > self.last_ot_weight :  
      #print('changing volumes ')

      if not self.precompute : 
        import time
        t0 = time.time()
        weights0 = (round(ot_weights[indices[0]],3),round(1-ot_weights[indices[0]],3))
        weights1 = (round(ot_weights[indices[1]],3),round(1-ot_weights[indices[1]],3))
        reg = self.reg
        niter = self.niter

        print('weights change 0 :' , weights0 , 'weights change 1', weights1,  'reg and iter ' , reg, niter)
        #self.tmp_v1 = v1
        #self.tmp_v2 = v2
        if self.last_ot_weight == 0 :
          self.tmp_v1 = v1
          self.tmp_v2 = convolutional_barycenter_gpu([v1,v2],reg, weights1, niter = niter)

        
        #self.tmp_v1 = convolutional_barycenter_cpu([v1,v2],reg, weights0 ,niter = niter)
        self.tmp_v1 = self.tmp_v2
        self.tmp_v2 = convolutional_barycenter_cpu([v1,v2],reg, weights1, niter = niter)
        print('changing volume took', time.time()-t0)


      else :
        #print('getting precomputed volumes')
        tmp_index = self.tmp_index
        self.tmp_v1 = self.precomputed[tmp_index]
        self.tmp_v2 = self.precomputed[tmp_index+1]
        self.tmp_index = tmp_index+1
      



      self.last_ot_weight = ot_weights[indices[-1]]
      
      


    f0 = float(i0)/(n-1)
    if self.add_mode:
      f1 = 1.0
      f2 = f
    else:
      f2 = (f-f0)*(n-1)
      f1 = 1-f2
    sf = self.scale_factors
    sf1 = f1 * sf[i0]
    sf2 = f2 * sf[i0+1]
    
    sf2 = (sf2-ot_weights[indices[0]]) / (ot_weights[indices[1]] - ot_weights[indices[0]])
    sf1 = 1 - sf2


    vol1 = self.tmp_v1 
    vol2 = self.tmp_v2 

    

    
    return f1, f2, sf1, sf2, vlist[i0], vlist[i0+1], vol1, vol2
  


  # ---------------------------------------------------------------------------
  #AJOUT PERSO ATTENTION
  def play_ot(self, f, fmin, fmax, fstep, f_changed_cb, fdir = None, steps = None):

    self.f = f
    self.fmin = fmin
    self.fmax = fmax
    self.fstep = fstep
    self.f_changed_cb = f_changed_cb
    self.steps = steps
    if not fdir is None:
      if fdir >= 0: self.step_direction = 1
      else:         self.step_direction = -1
    if self.play_handler is None:
      h = self.session.triggers.add_handler('new frame', self.next_frame_cb_ot)
      self.play_handler = h
  # ---------------------------------------------------------------------------
  #
  def stop_playing(self):

    self.session.triggers.remove_handler(self.play_handler)
    self.play_handler = None
    if self.recording:
      self.finish_recording()

  # ---------------------------------------------------------------------------
  #
  def playing(self):

    return self.play_handler != None


  # ---------------------------------------------------------------------------
  #
  def next_frame_cb_ot(self, *_):

    if not self.steps is None:
      if self.steps <= 0:
        self.stop_playing()
        return
      else:
        self.steps -= 1

    fmin, fmax = self.fmin, self.fmax
    next_f = self.f + self.fstep * self.step_direction

    if next_f >= fmax:
      next_f = fmax
      self.step_direction = -1
    elif next_f <= fmin:
      next_f = fmin
      self.step_direction = 1
    if not self.interpolate_ot(next_f):
      self.stop_playing()       # Volume closed.
    fccb = self.f_changed_cb
    if fccb:
      fccb(self.f)
  """
  # ---------------------------------------------------------------------------
  #
  def record(self, fmin, fmax, fstep, f_changed_cb, roundtrip, record_args, save_movie_cb):

    if self.recording:
      return

    self.f = self.fmin = fmin
    self.fmax = fmax
    self.fstep = fstep
    self.f_changed_cb = f_changed_cb
    self.save_movie_cb = save_movie_cb
    
    self.recording = True

    from math import ceil
    steps = int(ceil((fmax - fmin) / fstep))
    if roundtrip:
      steps *= 2

    from chimerax.core.commands import run
    run(self.session, 'movie record ' + record_args)

    self.play(fmin, fmin, fmax, fstep, f_changed_cb, 1, steps)

  # ---------------------------------------------------------------------------
  #
  def finish_recording(self):

    self.recording = False

    if self.play_handler:
      self.stop_playing()

    from chimerax.core.commands import run
    runCommand(self.session, 'movie stop')

    self.save_movie_cb()
"""

  # ---------------------------------------------------------------------------
  #
  def record_threshold_ranks(self, v):

    ms = v.matrix_value_statistics()

    rlev = [ms.rank_data_value(r) for r in self.surface_level_ranks]
    slev = [s.level for s in v.surfaces]
    if slev != rlev:
      self.surface_level_ranks = [ms.data_value_rank(lev) for lev in slev]

    rlev = [ms.rank_data_value(r) for r in self.image_level_ranks]
    slev = [lev for lev,b in v.image_levels]
    if slev != rlev:
      self.image_level_ranks = [ms.data_value_rank(lev) for lev in slev]

  # ---------------------------------------------------------------------------
  #
  def set_threshold_ranks(self, v):

    ms = v.matrix_value_statistics()

    sflev = [ms.rank_data_value(r) for r in self.surface_level_ranks]
    imlev = list(zip([ms.rank_data_value(r) for r in self.image_level_ranks],
                     [b for lev,b in v.image_levels]))
    v.set_parameters(surface_levels = sflev, image_levels = imlev)

  # ----------------------------------------------------------------------------
  # 
  def precompute_volumes(self, volumes): 
    print('precomputing')
    ot_weights = self.ot_weights 
    weights = [(round(elm,3), round(1-elm,3)) for elm in ot_weights]

    res = []

    v1 = volumes[0].matrix(step = self.step, subregion = self.subregion)
    v2 = volumes[1].matrix(step = self.step, subregion = self.subregion)

    from .utils import convolutional_barycenter
    reg = self.reg
    niter = self.niter 
    for i,weight in enumerate(weights) :
      print('%i of %i precomputed maps'%(i,len(weights)))
      tmp = convolutional_barycenter([v1,v2], reg, weight , niter = niter)
      res.append(tmp)

    self.precomputed = res
    print('finished precomputing')
    #print(len(res))



# -----------------------------------------------------------------------------
#

def linear_combination(f1, v1, f2, v2, v, subregion, step):
  
  
  m = v.full_matrix()
  #m1 = v1.matrix(step = step, subregion = subregion)
  #m2 = v2.matrix(step = step, subregion = subregion)
  m1 = v1 
  m2 = v2
  if (m.flags.contiguous and m1.flags.contiguous and m2.flags.contiguous and
      m1.dtype == m.dtype and m2.dtype == m.dtype):
    # Optimize calculation of linear combination of matrices.
    # C++ routine is 7x faster (.1 vs .7 sec) than numpy on 256^3 matrix.
    from chimerax.map  import linear_combination
    linear_combination(f1, m1, f2, m2, m)
  else:
    m[:,:,:] = f1*m1[:,:,:] + f2*m2[:,:,:]

  v.data.values_changed()


def semi_morph_maps_ot(volumes, total_play_steps, ot_play_steps, play_start, play_step, play_direction,
               play_range, add_mode, adjust_thresholds, scale_factors,
               hide_maps, interpolate_colors, subregion, step, model_id, niter, reg, rate, precompute):

  import numpy as np

  if hide_maps:
    for v in volumes:
      v.display = False
  
  """if step > 1 and gaussfilt: 
    for v in volumes :
      from scipy.ndimage import gaussian_filter
      m = v.full_matrix() 
      m = gaussian_filter(m,sigma = 2*step/6.)
      #print('on gaussian filtre')
      #m = m[::step,::step,::step]
      v.data.values_changed()"""


  im = Semi_Interpolated_Map(volumes, scale_factors, adjust_thresholds, add_mode,
                        interpolate_colors, subregion, step, model_id, niter, reg, rate)

  im.total_steps = total_play_steps
  im.ot_steps = ot_play_steps 
  
  im.ot_weights = np.linspace(play_range[0],play_range[1],ot_play_steps+2)
  
  im.precompute = precompute 
  if precompute : 
    im.precompute_volumes(volumes)

  if total_play_steps > 0:
    fmin, fmax = play_range
    im.play_ot(play_start, fmin, fmax, play_step, None, play_direction, total_play_steps)
  else:
    im.interpolate_ot(play_start)

  return im