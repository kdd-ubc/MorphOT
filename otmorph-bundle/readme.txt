First implementation of OT interpolation as a ChimeraX bundle : 
To install this bundle please run in chimeraX command line 

devel build PATH_TO_otmorph-bundle

then 

devel install PATH_TO_otmorph-bundle


TO USE CUPY (GPU computing) 

If you have cuda installed on your computer, first you must know which version of cuda is installed ('nvcc --version' in the command prompt),
then in bundle_info.xml in dependency, uncomment (remove <! -- - and  <--!> around) the line '<Dependency name="cupy-cuda102" version=">=0.1"/>' you must make sure that
the number after cuda is the same as your cuda version number (in my case v10.2 and 'cupy-cuda102') 

When done, restart chimeraX and re-run both "devel" commands 



THEN RUN IN THE COMMAND LINE : 

volumeperso morphOT    ( followed by the same kind of arguments as those of volume::morph https://www.rbvi.ucsf.edu/chimerax/docs/user/commands/volume.html#morph ) 

example : 

volumeperso morphOT #1 #2 hideOriginalMaps true constantvolume true (#1 #2 are the ids of the volumes)


please note that for the moment it computes every step iteratively, thus for big maps it takes seems very laggy. I will either allow downsampling or try to precompute 
then display

For big maps such as ribosome, you can run display only one barycenter using command : 

volumeperso onebarycenter #1 #2 0.5,0.5   (models #1 and #2 with weights .5,.5 on each for this command) 



