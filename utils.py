import numpy as np
import pylab as py
import healpy as hp
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
    


#################################################################
#The following function was shamelessly stolen and modifite from Stephen Hoover.
#Didn't want to re-invent the wheel.

def pix2Ang(pixel_coords, ra_dec_center, reso_arcmin, map_pixel_shape, proj=0, wrap=True):
    """
    Supported projections are:

     0:  Sanson-Flamsteed projection (x = ra*cos(dec), y = dec)
     1:  CAR projection (x = ra, y = dec)
     2:  SIN projection
     3:  Healpix (not a projection at all, but pixels on the sphere) [NOT IMPLEMENTED]
     4:  stereographic projection
     5:  Oblique Lambert azimuthal equal-area projection
         (ref p. 185, Snyder, J. P. 1987, Map Projections-A Working Manual 
         (Washington, DC: U.S. Geological Survey))
    INPUTS
        pixel_coords : (2-element tuple of arrays) A tuple or list of arrays, [y_coord, x_coord].
            Note the order! The first element is the "y", the mostly-dec coordinate, and the
            second element is the "x", the mostly-RA coordinate.

        ra_dec_center : (2-element array) The [RA, declination] of the center of the map.

        reso_arcmin : (float) The width of each pixel in arcminutes, assumed to be the same for
            both x and y directions.

        map_pixel_shape : (2-element array) The height and width of the map, in pixels.

        proj [0]: (int or string) Which map projection should I use to turn pixel coordinates
            on a flat map into angles on the curved sky? May be an integer index, or string
            name of the projection.

    OUTPUT
        (ra, dec): A 2-tuple of arrays. 'ra' and 'dec' are each arrays with the same number of elements as
        the arrays in the 'pixel_coords' input. 'ra' is the right ascension in degrees
        (wrapped to the range [0, 360) if wrap==True), and 'dec' is the declination in degrees.
        returns the ra,dec of the CENTER of the pixels
    """

    DTOR = np.pi/180.
    RTOD = 180/np.pi
    
    # Convert the "proj" input to an index.
    proj = int(proj)

    # Break out the x and y coordinates, and cast them as floats.
    pixel_coords = (y_coord, x_coord) = pixel_coords[0].astype(np.float), pixel_coords[1].astype(np.float)
    n_pixels = map_pixel_shape.astype(np.float)

    # shift to the center of the pixel, subtract off npix/2 to center around 0, then convert to degrees
    y_coord = (y_coord + 0.5 - 0.5 * n_pixels[0]) * reso_arcmin / 60
    x_coord = (x_coord + 0.5 - 0.5 * n_pixels[1]) * reso_arcmin / 60

    ra_dec_center_rad = ra_dec_center * DTOR  # Convert to radians

    if proj == 0:
        dec = ra_dec_center[1] - y_coord
        ra = x_coord / np.cos(dec * DTOR) + ra_dec_center[0]
    elif proj == 1:
        dec = ra_dec_center[1] - y_coord
        ra = x_coord + ra_dec_center[0]
    elif proj == 2:
        rho = np.sqrt(x_coord**2 + y_coord**2) * DTOR
        c = np.arcsin(rho)
        phi_temp = np.arcsin(np.cos(c) * np.sin(ra_dec_center_rad[1]) - DTOR * y_coord * np.cos(ra_dec_center_rad[1]))
        lambda_temp = RTOD * np.arctan2(x_coord * DTOR * np.sin(c),
                                        rho * np.cos(ra_dec_center_rad[1]) * np.cos(c) + y_coord * DTOR * np.sin(ra_dec_center_rad[1]) * np.sin(c))
        bad_rho = rho < 1e-8
        phi_temp[bad_rho] = ra_dec_center_rad[1]
        lambda_temp[bad_rho] = 0.

        dec = phi_temp * RTOD
        ra = ra_dec_center[0] + lambda_temp
    elif proj == 5 or proj == 4:
        rho = np.sqrt(x_coord**2 + y_coord**2) * DTOR
        if proj == 5:
            c = 2 * np.arcsin(rho / 2)
        if proj == 4:
            c = 2 * np.arctan(rho / 2)
        phi_temp = np.arcsin(np.cos(c) * np.sin(ra_dec_center_rad[1]) -
                             DTOR * y_coord * np.sin(c) / rho * np.cos(ra_dec_center_rad[1]))
        lambda_temp = RTOD * np.arctan2(x_coord * DTOR * np.sin(c),
                                        rho * np.cos(ra_dec_center_rad[1]) * np.cos(c) + y_coord * DTOR * np.sin(ra_dec_center_rad[1]) * np.sin(c))
        bad_rho = rho < 1e-8
        phi_temp[bad_rho] = ra_dec_center_rad[1]
        lambda_temp[bad_rho] = 0.

        dec = phi_temp * RTOD
        ra = ra_dec_center[0] + lambda_temp
    else:
        raise ValueError("I don't know what to do with proj " + str(proj) + ".")

    if wrap:
        # Wrap RA values to the [0, 360) range.
        while ra.min() < 0:
            ra[ra < 0 ] += 360.
        while ra.max() >= 360.:
            ra[ra >= 360.] -= 360.
    else:
        # make sure branch cut for ra is opposite map center
        too_low = np.where(ra - ra_dec_center[0] < -180.)
        ra[too_low] += 360.
        too_high = np.where(ra - ra_dec_center[0] >= 180.)
        ra[too_high] -= 360.

    return ra, dec
#################################################################


def obtain_healpix_list(map_center=np.array([0.0,-57.5]),
                        reso_arcmin=7.5,
                        map_size = [28.75,28.75],
                        proj = 5,
                        mask_file = None,
                        nside=1024,
                        galactic=True,
                        apply_mask=False,
                        save_mesh=False,
                        nest=False):
    '''
    Return a list of mesh of healpix pixel indices corresponding to the SPTpol 500d region.  Simply
    index a Planck map with the output array to produce a SPTpol-like map array with Planck data.
    '''
    
    #Generate pixel coordinate arrays
    x_coord = np.arange(int(map_size[1]*60./reso_arcmin))
    y_coord = np.arange(int(map_size[0]*60./reso_arcmin))

    coord_mesh = np.meshgrid(y_coord,x_coord)

    ra, dec = pix2Ang(coord_mesh,
                      ra_dec_center=np.array(map_center),
                      reso_arcmin=reso_arcmin,
                      map_pixel_shape=np.array([len(y_coord),len(x_coord)]),
                      proj=proj)

    ra = ra.T
    dec = dec.T

    #Load in mask to determine which pixels we need to keep.
    if mask_file != None:
        mask = files.read(mask_file)
        mask[mask > 0.] = 1.
        mask = np.array(mask, dtype=bool)
    else:
        mask = np.ones((int(map_size[0]*60/reso_arcmin), int(map_size[1]*60/reso_arcmin)))

    #Make a mesh grid of (ra,dec), and flatten into two lists while applying the mask.
    if apply_mask:
        ra = ra[mask].flatten()
        dec = dec[mask].flatten()
    else:
        ra = ra.flatten()
        dec = dec.flatten()


    if not galactic:
        phi = ra*np.pi/180.
        theta = np.abs(dec*np.pi/180. - np.pi/2.)

    if galactic:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        #Convert from (ra, dec) to (l,b) and then to (phi, theta)
        coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
        phi = coords.galactic.l.radian
        theta = np.abs(coords.galactic.b.radian - np.pi/2.)

    #Now obtain a list of relevant healpix pixels.  This is an healpix index mesh we can use to 
    #extract whatever portion of the sky map we want.
    all_pixels = hp.pixelfunc.ang2pix(nside=nside,theta=theta,phi=phi,nest=nest).reshape(len(y_coord),len(x_coord))

    if save_mesh:
        np.save('healpix_mesh.npy',all_pixels, allow_pickle=False)
    return all_pixels
