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

# -----------------------------------------------------------------------------
#
class Interpolated_Map:

  def __init__(self, volumes, scale_factors = None, adjust_thresholds = False,
               add_mode = False, interpolate_colors = True,
               subregion = 'all', step = 1, model_id = None, niter = 20, reg = None):

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
    self.adjust_thresholds = adjust_thresholds
    self.surface_level_ranks = []       # For avoiding creep during threshold
    self.image_level_ranks = []         #   normalization.

  # ---------------------------------------------------------------------------

  ######### AJOUT PERSO 
  def interpolate_ot(self, f):

    v = self.result
    if self.adjust_thresholds:
      self.record_threshold_ranks(v)

    f1, f2, sf1, sf2, v1, v2 = self.coefficients(f)
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

def morph_maps_ot(volumes, play_steps, play_start, play_step, play_direction,
               play_range, add_mode, adjust_thresholds, scale_factors,
               hide_maps, interpolate_colors, subregion, step, model_id, niter, reg):

  if hide_maps:
    for v in volumes:
      v.display = False

  im = Interpolated_Map(volumes, scale_factors, adjust_thresholds, add_mode,
                        interpolate_colors, subregion, step, model_id, niter, reg)
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
  from .help_functions import convolutional_barycenter
  #m[:,:,:] = f1*m1[:,:,:] + f2*m2[:,:,:]
  m[:,:,:] = convolutional_barycenter([m1,m2],reg,(f1,f2),niter=niter,verbose=False)
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

  from .help_functions import convolutional_barycenter
  m[:,:,:] = convolutional_barycenter([m1,m2],reg, weights, niter = niter, verbose = False)

  #from chimerax.map.data import ArrayGridData
  #d = v1.data

  #name = '%s %s barycenter, weights %s'%(v1.name, v2.name, str(weights))

  #gg = ArrayGridData(barycenter_m, origin, step, d.cell_angles, d.rotation,
                    # name = name)
  r.data.values_changed()
  return r


def ot_save(volumes, dir_name, frames, niter, reg, subregion='all', step = 1, model_id = None) : 
  v1 = volumes[0]
  v2 = volumes[1]

  m1 = v1.matrix(step = step, subregion = subregion)
  m2 = v2.matrix(step = step, subregion = subregion)

  from .help_functions import convolutional_barycenter
  import mrcfile 
  import progressbar

  all_weights = [(float(s)/frames,1-float(s)/frames) for s in range(frames)]

  #for weights in progressbar.progressbar(all_weights, redirect_stdout=True) : 
  for i,weights in enumerate(all_weights): 
    print('(%i of %i)'%(i,frames))

    result_file = dir_name + '/%s_%s_weights%s.mrc'%(v1.name,v2.name, str(weights))

    m = convolutional_barycenter([m1,m2],reg, weights, niter = niter, verbose = False)
    from chimerax.map.data import ArrayGridData
    from chimerax.map.data.mrc import save

    m_grid = ArrayGridData(m)

    save(m_grid, result_file)

  



def rateLinear(frames):
        "Generate fractions from 0 to 1 linearly (excluding start/end)"
        return [ float(s) / frames for s in range(1, frames) ]

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