import healpy as hp
import numpy as np
import pylab as py
import utils as ut
import os
import time
from functools import wraps
py.ion()

import pysm
from pysm.nominal import models

def timeit(nu):
    def decorator(func):
        @wraps(func)
        def wrapper(nu):
            t1 = time.time()
            x = func(nu)
            t2 = time.time()
            print('Time for one set of maps: ' + str(t2-t1) + '\n')
            return x
        return wrapper
    return decorator

############################################################################

#Define pixel size.
nside = 64 #Will give 
istart = 850
istop = 50000
npix = 128 #Flat-sky map will have npix**2 pixels.
map_size = [25.,25.]
sense_P = 70.0
outdir = '/media/jason/SSD2/deep_data/deep_foregrounds'
seed = 1111

#Define effective frequencies for output maps (in GHz).
#CMB-S4 definitions
nu = np.array([20., 30., 40., 85., 95., 145., 155., 220., 270.])
#nu = np.array([20., 40., 145., 220., 270.])
#nu = np.array([20., 155., 270.])

############################################################################


#Load in PySM foreground and CMB models
print('Loading CMB model...')
cmb_config = models('c1', nside=nside)

print('Loading foreground models...')
dust_config = models("d6", nside)
sync_config = models("s2", nside)
ff_config = models("f1", nside)
ame_config = models("a2", nside)
sky_config = {'dust': dust_config,
              'synchrotron': sync_config,
              'freefree':ff_config,
              'ame':ame_config,
              'cmb':cmb_config}
sky_config_cmb = {'cmb':cmb_config}

#Find a list of healpix indices to grab for a flat map.
#Defaults in obtain_healpix_list() give a 25x25 deg map, 50x50 pix.
print('Determining healpix indices for flat-sky maps...')
hp_pixels = ut.obtain_healpix_list(map_center=np.array([0.0,-57.5]),
                                   reso_arcmin=map_size[0]*60./npix,
                                   nside=nside,
                                   map_size = map_size,
                                   proj = 5)

#Map filling functions
#@timeit(nu)
def make_tot_maps(nu):
    return sky.signal()(nu)

#@timeit(nu)
def make_cmb_maps(nu):
    return sky.cmb(nu)

#@timeit(sky_config)
def sky_instance(sky_config):
    return pysm.Sky(sky_config)


for i in range(istart,istop):
    print('Realization ', i)

    if i==0:
        if os.path.exists(os.path.join(outdir, 'total_signal_SV_'+str(int(npix))+'x'+str(int(npix))+'_senseP'+str(sense_P)+'_seed'+str(int(seed))+'_nu'+str(len(nu)))):
            pass
        else:
            os.mkdir(os.path.join(outdir, 'total_signal_SV_'+str(int(npix))+'x'+str(int(npix))+'_senseP'+str(sense_P)+'_seed'+str(int(seed))+'_nu'+str(len(nu))))

        if os.path.exists(os.path.join(outdir, 'cmb_SV_'+str(int(npix))+'x'+str(int(npix))+'_senseP'+str(sense_P)+'_seed'+str(int(seed))+'_nu'+str(len(nu)))):
            pass
        else:
            os.mkdir(os.path.join(outdir, 'cmb_SV_'+str(int(npix))+'x'+str(int(npix))+'_senseP'+str(sense_P)+'_seed'+str(int(seed))+'_nu'+str(len(nu))))
            
    cmb_config[0]['cmb_seed'] = seed+i

    sky_config['cmb'] = cmb_config
    
    ##Instantiate a sky object.
    print('Instantiating sky object...')
    sky = sky_instance(sky_config)
    sky_cmb = sky_instance(sky_config_cmb)

    #Instrument config
    instr_config = {'nside':nside,
                    'frequencies':nu,
                    'use_smoothing':True,
                    'beams':np.ones(len(nu)) * 30., # arcmin
                    'add_noise':True,
                    'sens_I': np.ones(len(nu)) * sense_P/np.sqrt(2),
                    'sens_P': np.ones(len(nu)) * sense_P,
                    'noise_seed': seed + i,
                    'use_bandpass': False,
                    'output_units': 'uK_RJ',
                    'output_directory':outdir,
                    'output_prefix': 'test',
                    }
    instr = pysm.Instrument(instr_config)

    x = instr.observe(sky, write_outputs=False)
    
    tot_map_flat = x[0][:,:,hp_pixels]
    noise = x[1][:,:,hp_pixels]
    #tot_map_flat = instr.observe(sky, write_outputs=False)[0][:,:,hp_pixels]
    tot_map_flat = np.array(tot_map_flat, dtype=np.float32)
    
    np.save(os.path.join(outdir, 'total_signal_SV_'+str(int(npix))+'x'+str(int(npix))+'_senseP'+str(sense_P)+'_seed'+str(int(seed))+'_nu'+str(len(nu)),
                         'tot_'+str(i).zfill(6)+'.npy'), tot_map_flat)

    cmb_map_flat = instr.observe(sky_cmb, write_outputs=False)[0][:,:,hp_pixels]
    cmb_map_flat = cmb_map_flat[2,:,:] # Keep the 155 GHz channel as CMB Truth.
    cmb_map_flat = np.array(cmb_map_flat, dtype=np.float32) #Drop to 32-bit
    np.save(os.path.join(outdir, 'cmb_SV_'+str(int(npix))+'x'+str(int(npix))+'_senseP'+str(sense_P)+'_seed'+str(int(seed))+'_nu'+str(len(nu)),
                         'cmb_'+str(i).zfill(6)+'.npy'), cmb_map_flat)
