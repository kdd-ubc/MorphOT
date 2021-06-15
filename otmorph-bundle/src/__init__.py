# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI
from chimerax.core.commands import FloatArg, FloatsArg, CmdDesc, register, BoolArg, StringArg, EnumOf, IntArg, Int3Arg, ModelIdArg, SaveFolderNameArg
from chimerax.map import MapsArg, MapStepArg, Float1or3Arg, ValueTypeArg
from chimerax.map.mapargs import Float2Arg, MapRegionArg
# Subclass from chimerax.core.toolshed.BundleAPI and
# override the method for registering commands,
# inheriting all other methods from the base class.
class _MyAPI(BundleAPI):

    api_version = 1     # register_command called with BundleInfo and
                        # CommandInfo instance instead of command name
                        # (when api_version==0)
    
    # Override method
    @staticmethod
    def register_command(bi, ci, logger):
        # bi is an instance of chimerax.core.toolshed.BundleInfo
        # ci is an instance of chimerax.core.toolshed.CommandInfo
        # logger is an instance of chimerax.core.logger.Logger

        # This method is called once for each command listed
        # in bundle_info.xml.  Since we list two commands,
        # we expect two calls to this method.

        # We check the name of the command, which should match
        # one of the ones listed in bundle_info.xml
        # (without the leading and trailing whitespace),
        # and import the function to call and its argument
        # description from the ``cmd`` module.
        # If the description does not contain a synopsis, we
        # add the one in ``ci``, which comes from bundle_info.xml.
        # We then register the function as the command callback
        # with the chimerax.core.commands module.
        from chimerax.core.commands import register
        if ci.name == 'MorphOT morphOT':
            from . import morph
            from . import cmd
            func = cmd.volume_morphOT
            varg = cmd.varg
            ssm_kw = cmd.ssm_kw
            morphot_desc = CmdDesc(required = varg,
                            keyword = [('frames', IntArg),
                                        ('start', FloatArg),
                                        ('play_step', FloatArg),
                                        ('play_direction', IntArg),
                                        ('niter', IntArg),
                                        ('reg', FloatArg),
                                        ('play_range', Float2Arg),
                                        #('add_mode', BoolArg),
                                        ('constant_volume', BoolArg),
                                        #('scale_factors', FloatsArg),
                                        ('hide_original_maps', BoolArg),
                                        ('rate', StringArg),
                                        ('interpolate_colors', BoolArg),
                                        ('maxsize',IntArg)] + ssm_kw,
                            synopsis = 'OT interpolate between two or more maps')
            register(ci.name, morphot_desc, func)

        if ci.name == 'MorphOT semiMorphOT':
            from . import morph
            from . import cmd
            func = cmd.volume_semi_morphOT
            varg = cmd.varg
            ssm_kw = cmd.ssm_kw
            morphot_desc = CmdDesc(required = varg,
                            keyword = [('frames', IntArg),
                                        ('ot_frames',IntArg),
                                        ('start', FloatArg),
                                        ('play_step', FloatArg),
                                        ('play_direction', IntArg),
                                        ('niter', IntArg),
                                        ('reg', FloatArg),
                                        ('play_range', Float2Arg),
                                        ('constant_volume', BoolArg),
                                        ('hide_original_maps', BoolArg),
                                        ('rate', StringArg),
                                        ('interpolate_colors', BoolArg),
                                        ('maxsize',IntArg),
                                        ('precompute',BoolArg)] + ssm_kw,
                            synopsis = 'mix of OT and linear interpolate two or more maps')
            register(ci.name, morphot_desc, func)

        if ci.name == 'MorphOT oneBarycenter' : 
            from . import morph
            from . import cmd 
            func = cmd.volume_barycenterOT
            varg = cmd.varg 
            ssm_kw = cmd.ssm_kw 
            onebarycenter_desc = CmdDesc(required = varg + [('weights', FloatsArg)],
                                keyword = [
                                            ('niter', IntArg),
                                            ('reg', FloatArg),
                                            ('interpolate_colors', BoolArg),
                                            ('maxsize',IntArg)] + ssm_kw,
                                synopsis = 'Produce one weighted OT barycenter between two or more maps')
            register(ci.name, onebarycenter_desc, func)

        if ci.name == 'MorphOT barycenterSave' : 
            from . import morph
            from . import cmd 
            func = cmd.volume_barycenterSave
            varg = cmd.varg 
            ssm_kw = cmd.ssm_kw 
            savebarycenter_desc = CmdDesc(required = varg + [('folder', SaveFolderNameArg)],
                                keyword = [('frames', IntArg),
                                            ('niter', IntArg),
                                            ('reg', FloatArg),
                                            ('rate',StringArg),
                                            ('interpolate_colors', BoolArg),
                                            ('maxsize', IntArg),
                                            ('name1', StringArg),
                                            ('name2', StringArg)] + ssm_kw,
                                synopsis = 'save an OT iterpolation between two maps')
            register(ci.name, savebarycenter_desc, func)

        if ci.name == 'MorphOT linearBarycenterSave' : 
            from . import morph
            from . import cmd 
            func = cmd.volume_linearBarycenterSave
            varg = cmd.varg 
            ssm_kw = cmd.ssm_kw 
            savebarycenter_desc = CmdDesc(required = varg + [('folder', SaveFolderNameArg)],
                                keyword = [('frames', IntArg),
                                            ('niter', IntArg),
                                            ('reg', FloatArg),
                                            ('rate',StringArg),
                                            ('interpolate_colors', BoolArg),
                                            ('maxsize', IntArg)] + ssm_kw,
                                synopsis = 'save a linear interpolation between two maps')
            register(ci.name, savebarycenter_desc, func)

        
    
# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _MyAPI()

