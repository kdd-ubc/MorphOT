First implementation of OT interpolation as a ChimeraX bundle : 
To install this bundle please run in chimeraX command line 
devel build PATH_TO_otmorph-bundle

then 

devel install PATH_TO_SOURCE_CODE_FOLDER


finally  to use the method run in command line : 

volumeperso morphOT    ( followed by the same kind of arguments as those of volume::morph https://www.rbvi.ucsf.edu/chimerax/docs/user/commands/volume.html#morph ) 

example : 

volumeperso morphOT #1 #2 frame 15 hideOriginalMaps  true 


please note that for the moment it computes every step iteratively, thus for big maps it takes seems very laggy. I will either allow downsampling or try to precompute 
then display



