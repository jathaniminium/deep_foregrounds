import numpy as np
import pylab as py
import pickle as pk
import copy
import time
import glob
from scipy import ndimage

###################################################################
def read_spectra(filename, lensing=False, line_start=0):
    '''
    Read in a Camb spectrum and return C_ells.
    '''
    d = open(filename, 'r').read().split('\n')[line_start:-1]

    ell = []
    TT = []
    EE = []
    BB = []
    TE = []

    if lensing:
        dd = []
        dT = []
        dE = []
    for i in range(len(d)):
        this_line = []
        for j in range(len(d[i].split(' '))):
            if len(d[i].split(' ')[j]) != 0:
                this_line.append(d[i].split(' ')[j])
        
        ell.append(np.float(this_line[0]))
        TT.append(np.float(this_line[1]))
        EE.append(np.float(this_line[2]))
        BB.append(np.float(this_line[3]))
        TE.append(np.float(this_line[4]))

        if lensing:
            dd.append(np.float(this_line[5]))
            dT.append(np.float(this_line[6]))
            dE.append(np.float(this_line[7]))

    ell = np.array(ell)
    TT = np.array(TT)
    EE = np.array(EE)
    BB = np.array(BB)
    TE = np.array(TE)

    if lensing:
        dd = np.array(dd)
        dT = np.array(dT)
        dE - np.array(dE)


    if lensing:
         return ell, TT, EE, BB, TE, dd, dT, dE
    else:
        return ell, TT, EE, BB, TE

    
def plot_C_ells(sky_cls, save_dir='', theory_ell=None, theory_TT=None, theory_EE=None, theory_BB=None,
                cross=False, plot_TT=True, plot_EE=True, plot_BB=False, summed=True,
                map_depth=False, map_index=0, ell_min=1000., ell_max=3000., 
                xlim=[1e2,1e4],ylim=[1e-1,1e5]):
    py.clf()
    py.figure(1, figsize=(4,4))
    py.rc('axes', linewidth=1)
    py.rc('font', family='serif')
    ax = py.gca()
    if theory_ell != None and theory_TT != None:
        py.loglog(theory_ell,theory_TT, 'k-', linewidth=2)
    if theory_ell != None and theory_EE != None:
        py.loglog(theory_ell,theory_EE, 'k-', linewidth=2)
    if theory_ell != None and theory_BB != None:
        py.loglog(thoery_ell,theory_BB, 'k-', linewidth=2)

    if plot_TT:
        if not map_depth:
            labelT = 'TT'
        else:
            map_depth_T = get_noise_depth(sky_cls['ell']['TT'], sky_cls['TT'], 
                                          ell_min=ell_min, ell_max=ell_max)
            labelT = 'TT: %.1f $\mu$K-arcmin' % map_depth_T
        py.loglog(sky_cls['ell']['TT'], sky_cls['TT'], 'b.', label=labelT)
            
    if plot_EE:
        if not map_depth:
            labelE = 'EE'
        else:
            map_depth_E = get_noise_depth(sky_cls['ell']['EE'], sky_cls['EE'],
                                          ell_min=ell_min, ell_max=ell_max)
            labelE = 'EE: %.1f $\mu$K-arcmin' % map_depth_E
        py.loglog(sky_cls['ell']['EE'], sky_cls['EE'], 'r.', label=labelE)
    if plot_BB:
        if not map_depth:
            labelB = 'BB'
        else:
            map_depth_B = get_noise_depth(sky_cls['ell']['BB'], sky_cls['BB'],
                                          ell_min=ell_min, ell_max=ell_max)
            labelB = 'BB: %.1f $\mu$K-arcmin' % map_depth_B
        py.loglog(sky_cls['ell']['BB'], sky_cls['BB'], 'g.', label=labelB)

    py.xlim((xlim[0],xlim[1]))
    py.ylim((ylim[0],ylim[1]))
    py.xlabel('Multipole')
    py.ylabel('$l(l+1)C_l/2\pi$ [$\mu$K$^2$]')

    if not cross:
        if summed:
            py.title('Summed Auto-Spectra')
        else:
            py.title('Differenced Auto-Spectra')
    else:
        if summed:
            py.title('Summed Cross-Spectra')
        else:
            py.title('Differenced Cross-Spectra')
    py.legend(loc='upper left')
    if not cross:
        if summed:
            if not map_depth:
                py.savefig(save_dir+'/Spectra/summed_auto_spectra_coadd'+
                           str(map_index).zfill(5)+'.png', dpi=200)
            else:
                py.savefig(save_dir+'/Depth/summed_auto_spectra_coadd'+
                           str(map_index).zfill(5)+'.png', dpi=200)
        else:
            if not map_depth:
                py.savefig(save_dir+'/Spectra/diff_auto_spectra_coadd'+
                           str(map_index).zfill(5)+'.png', dpi=200)
            else:
                py.savefig(save_dir+'/Depth/diff_auto_spectra_coadd'+
                           str(map_index).zfill(5)+'.png', dpi=200)
    else:
        if summed:
            if not map_depth:
                py.savefig(save_dir+'/Spectra/summed_cross_spectra_coadd'+
                           str(map_index).zfill(5)+'.png', dpi=200)
            else:
                py.savefig(save_dir+'/Depth/summed_cross_spectra_coadd'+
                           str(map_index).zfill(5)+'.png', dpi=200)
        else:
            if not map_depth:
                py.savefig(save_dir+'/Spectra/diff_cross_spectra_coadd'+
                           str(map_index).zfill(5)+'.png', dpi=200)
            else:
                py.savefig(save_dir+'/Depth/diff_cross_spectra_coadd'+
                           str(map_index).zfill(5)+'.png', dpi=200)


                
def get_noise_depth(ell,D_ell, ell_min, ell_max):
    counter = 0.
    summed_C_ell = 0.
    for i in range(len(ell)):
        if ell[i] >= ell_min and ell[i] <= ell_max:
            if D_ell[i] == D_ell[i]: 
                counter += 1
                summed_C_ell += D_ell[i]*2.*np.pi/ell[i]/(ell[i]+1.)

    mean_C_ell = summed_C_ell/counter
    mean_depth = np.sqrt(mean_C_ell)/(np.pi/180/60.) # Convert from uK^2 to uK-arcmin
    return mean_depth


def calculate_sample_variance(ell,D_ell, ell_min, ell_max, sky_area):
    counter = 0.
    summed_C_ell = 0.
    fsky = sky_area/(4.*np.pi*(180./np.pi)**2.)
    for i in range(len(ell)):
        if ell[i] >= ell_min and ell[i] <= ell_max:
            if D_ell[i] == D_ell[i]: 
                counter += 1
                summed_C_ell += (2./((2.*ell[i] + 1.)*fsky)) * \
                                (D_ell[i]*2.*np.pi/ell[i]/(ell[i]+1.))**2.

    mean_C_ell = summed_C_ell/counter

    sample_variance = mean_C_ell*ell*(ell + 1.)/(2.*np.pi)
    return sample_variance

def get_instrument_noise(map_depth, ell):
    counter = 0.
    summed_C_ell = 0.
    for i in range(len(ell)):
        if ell[i] >= ell_min and ell[i] <= ell_max:
            if D_ell[i] == D_ell[i]: 
                counter += 1
                summed_C_ell += np.sqrt(2./((2.*ell[i] + 1.)*fsky)) * \
                                D_ell[i]*2.*np.pi/ell[i]/(ell[i]+1.)

    mean_C_ell = summed_C_ell/counter
    mean_depth = np.sqrt(mean_C_ell)/(np.pi/180/60.) # Convert from uK^2 to uK-arcmin
    return noise_variance


def RADecSexagesimalToDegrees(ra,dec):
    #Split ra into hours, minutes, seconds of RA.
    ra_hr = np.float(ra.split(':')[0])
    ra_min = np.float(ra.split(':')[1])
    ra_sec = np.float(ra.split(':')[2])

    #Add into degrees of RA.
    ra_deg = 15.*(ra_hr + ra_min/60. + ra_sec/3600.)

    #Split dec in degrees, arcminutes, arcseconds.
    dec_deg = np.float(dec.split(':')[0])
    dec_min = np.float(dec.split(':')[1])
    dec_sec = np.float(dec.split(':')[2])

    #Add into degrees of dec.
    if dec_deg > 0.:
        dec_deg += dec_min/60. + dec_sec/3600.
    else:
        dec_deg -= dec_min/60. + dec_sec/3600.

    return (ra_deg, dec_deg)


def degreesToSexagesimal(ra,dec):
    ra_resid = ra/15.
    dec_resid = dec

    ra_hr = np.int(ra_resid)
    ra_resid -= ra_hr
    ra_min = np.int(ra_resid*60.)
    ra_resid -= ra_min/60.
    ra_sec = np.round(ra_resid*3600.,2)

    dec_deg = np.int(dec_resid)
    dec_resid = np.abs(dec_resid-dec_deg)
    dec_min = np.int(dec_resid*60.)
    dec_resid -= dec_min/60.
    dec_sec = np.round(dec_resid*3600.,2)

    ra_out = str(int(ra_hr))+':'+str(int(ra_min))+':'+'%.2f' % ra_sec
    dec_out = str(int(dec_deg))+':'+str(int(dec_min))+':'+'%.2f' % dec_sec

    return ra_out, dec_out


def rotate(vector, theta):
    r = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    v_r = np.dot(r, vector)

    return v_r


def fl(my_input, decimals=2):
    '''
    Round floats to specified decimal places and print with exactly those decimal places.
    '''
    str_format = '%.'+str(int(decimals))+'f'
    this = lambda x: str_format % x
    return this(my_input)


def AzElSexagesimalToDegrees(az,el):
    #Split ra into hours, minutes, seconds of RA.                                                                               
    az_deg = az.split(':')[0]
    az_min = np.float(az.split(':')[1])
    az_sec = np.float(az.split(':')[2])

    if az_deg[0] == '-':
        az_sign = -1.
        az_deg = np.float(az_deg[1:])
    else:
        az_sign = +1.
        az_deg = np.float(az_deg)

    #Add into degrees of RA.                                                                                                    
    az_deg = az_sign*(np.abs(az_deg) + az_min/60. + az_sec/3600.)

    #Split dec in degrees, arcminutes, arcseconds.                                                                              
    el_deg = el.split(':')[0]

    if el_deg[0] == '-':
        el_sign = -1.
        el_deg = np.float(el_deg[1:])
    else:
        el_sign = +1.
        el_deg = np.float(el_deg)

    el_min = np.float(el.split(':')[1])
    el_sec = np.float(el.split(':')[2])

    #Add into degrees of dec.                                                                                                   
    el_deg = el_sign*(el_deg + el_min/60. + el_sec/3600.)

    return (az_deg, el_deg)


def AddSexagesimalOffset(az_off,el_off,az,el, sexagesimal=True):

    if sexagesimal:
        az_off, el_off = AzElSexagesimalToDegrees(az_off, el_off)
        az, el = AzElSexagesimalToDegrees(az, el)
        

    tot_az = az + az_off
    tot_el = el + el_off

    return str(np.round(tot_az,4)), str(np.round(tot_el,4)), \
           str(np.round(az,4)), str(np.round(el,4))


def read_ascii_cov(cov_file):
    data = open(cov_file, 'r').read().split('\n')[:-1]
    cov = []
    for i in range(len(data)):
        cov.append(float(filter(None, data[i].split('\t'))[-1]))

    nbins = int(np.sqrt(len(data)))
    cov = np.array(cov).reshape((nbins,nbins))

    return cov
    
