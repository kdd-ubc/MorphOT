from chimerax.core.commands import CmdDesc, register, BoolArg, StringArg, EnumOf, IntArg, Int3Arg, ModelIdArg
from chimerax.map import MapsArg, MapStepArg, Float1or3Arg, ValueTypeArg
from chimerax.map.mapargs import Float2Arg, MapRegionArg
from chimerax.core.errors import UserError as CommandError

def volume_morphOT(session, volumes, frames = 25, start = 0, play_step = 0.04,
            play_direction = 1, niter = 20,  play_range = None, add_mode = False,
            constant_volume = False, scale_factors = None,
            hide_original_maps = True, interpolate_colors = True,
            subregion = 'all', step = 1, model_id = None):
    '''Linearly interpolate pointwise between maps.'''
    if len(volumes) < 2:
        raise CommandError('volume morph requires 2 or more volumes, got %d' % len(volumes))
    if play_range is None:
        if add_mode:
            prange = (-1.0,1.0)
        else:
            prange = (0.0,1.0)
    else:
        prange = play_range

    if not scale_factors is None and len(volumes) != len(scale_factors):
        raise CommandError('Number of scale factors (%d) doesn not match number of volumes (%d)'
                            % (len(scale_factors), len(volumes)))
    vs = [tuple(v.matrix_size(step = step, subregion = subregion))
            for v in volumes]
    if len(set(vs)) > 1:
        sizes = ' and '.join([str(s) for s in vs])
        raise CommandError("Volume grid sizes don't match: %s" % sizes)
    from .morph import morph_maps_ot
    im = morph_maps_ot(volumes, frames, start, play_step, play_direction, prange,
                    add_mode, constant_volume, scale_factors,
                    hide_original_maps, interpolate_colors, subregion, step, model_id)
    return im

def volume_barycenterOT(session, volumes, weights, niter = 20, reg = 0.8, interpolate_colors = True,
            subregion = 'all', step = 1, model_id = None):
    '''Linearly interpolate pointwise between maps.'''
    if len(volumes) < 2:
        raise CommandError('volume morph requires 2 or more volumes, got %d' % len(volumes))

    vs = [tuple(v.matrix_size(step = step, subregion = subregion))
            for v in volumes]
    if len(set(vs)) > 1:
        sizes = ' and '.join([str(s) for s in vs])
        raise CommandError("Volume grid sizes don't match: %s" % sizes)

    from .morph import ot_barycenter
    #grid_barycenter = ot_barycenter(volumes,weights, niter, reg, subregion, step, model_id) 

    #from chimerax.map import volume_from_grid_data
    #gv = volume_from_grid_data(grid_barycenter, session, model_id = modelId)
    #gv.copy_settings_from(volumes[0], copy_region = False, copy_colors = False, copy_thresholds = False)
  
    #volume.display = False  
    im = ot_barycenter(volumes, weights, niter, reg, subregion, step, model_id) 

    return im

varg = [('volumes', MapsArg)]
ssm_kw = [
    ('subregion', MapRegionArg),
    ('step', MapStepArg),
    ('model_id', ModelIdArg),
]