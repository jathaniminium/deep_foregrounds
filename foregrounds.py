import numpy as np
import pickle as pk
import healpy as hp
import pylab as py
import utils as ut
import copy



def gaussian_poisson_component(dl3000_poisson=19.3, pol_frac_poisson=0.10431, lmax=15000):
    '''
    Define a Gaussian realization of a Poisson foreground component.

    INPUTS:
        dl3000_poisson = D_\ell value of the foreground component at \ell = 3000 in uK^2.
        lmax = maximum \ell at which to generate component.
        pol_frac_poisson = Effective polarization fraction of Poisson sources.  This will be SQUARED
                           in the EE and BB Poisson terms.  This is chosen such that the EE poisson
                           component is the 68% confidence upper limit from the 100dEE (Crites, et al. 2015)
                           analysis (0.21 uK^2).

    OUTPUTS:
        List of 3 arrays of \Delta\ell = 1 C_\ells up to lmax. [TT, EE, BB]. 
    '''

    #Convert dl3000 into a C_\ell number.
    cl_amplitude = dl3000_poisson *2.*np.pi/3000./3001.

    clTT = np.zeros(lmax+1) + cl_amplitude
    clEE = np.zeros(lmax+1) + cl_amplitude*pol_frac_poisson**2.
    clBB = np.zeros(lmax+1) + cl_amplitude*pol_frac_poisson**2.

    #Assume zero correlation between T and E/B.
    clTE = np.zeros(lmax+1)

    return np.array([clTT, clEE, clBB, clTE])


def gaussian_clustered_poisson_component(dlTT_clust=5.0, pol_frac_clust=0.0, lmax=15000):
    '''
    Define a Gaussian realization of the clustered Poisson foreground. The model is assumed to be a 
    simple power law: D_\ell \propto \ell^0.8.

    INPUTS:
        dlTT_clust: D_\ell value of TT dust spectrum in uK^2 at pivot ell scale.
        pivot_ell: \ell at which dlEE is defined.
        power_law: Power law index.
        EEtoBB: Ratio of dust EE power to dust BB power.
        lmax: maximum \ell at which to generate component.
    '''

    clTT = dlTT_clust * (np.arange(lmax+1)/3000.)**0.8 * 2.*np.pi/np.arange(lmax+1)/(np.arange(lmax+1)+1.)
    clEE = clTT*pol_frac_clust**2.
    clBB = clTT*pol_frac_clust**2.

    #Remove NaNs.
    clTT[clTT != clTT] = 0.0
    clEE[clEE != clEE] = 0.0
    clBB[clBB != clBB] = 0.0

    #For now, assume clTE is zero.
    clTE = np.zeros(lmax+1)

    return np.array([clTT, clEE, clBB, clTE])


def gaussian_cirrus_component(dlTT3000=0.21, lmax=15000.):
    '''
    Define a Gaussian realization of galactic cirrus foreground. The model is assumed to be a 
    simple power law normalized at \ell=3000.

    INPUTS:
        dlTT3000: D_\ell value of TT in uK^2 at 150 GHz at pivot ell scale.  I'm pulling this
                  number from George 15, which has an effective band center of 154.1 GHz for
                  this component.
        pivot_ell: \ell at which dlEE is defined.
        power_law: Power law index.
        EEtoBB: Ratio of dust EE power to dust BB power.
        lmax: maximum \ell at which to generate component.
    '''

    clTT = dlTT3000 * (np.arange(lmax+1)/3000.)**-1.2 * 2.*np.pi/np.arange(lmax+1)/(np.arange(lmax+1)+1.)
    clEE = np.zeros(lmax+1)
    clBB = np.zeros(lmax+1)
    clTE = np.zeros(lmax+1)

    return np.array([clTT, clEE, clBB, clTE])


def gaussian_thermal_dust_component(dlTT_dust=1.15, pivot_ell_TT=80., power_law_TT=-0.42, 
                                    dlBB_dust=0.0118, pivot_ell_BB=80., power_law_BB=-0.42, 
                                    BBtoEE=2., lmax=15000.):
    '''
    Define a Gaussian realization of polarized dust foreground. The model is assumed to be a 
    simple power law with a characteristic pivot scale.

    INPUTS:
        dlBB: D_\ell value of BB dust spectrum in uK^2 at pivot ell scale.
        pivot_ell_BB: \ell at which dlEE is defined.
        power_law_BB: Power law index.
        EEtoBB: Ratio of dust EE power to dust BB power.
        lmax: maximum \ell at which to generate component.
        
    '''

    clBB = dlBB_dust * (np.arange(lmax+1)/pivot_ell_BB)**power_law_BB * 2.*np.pi/np.arange(lmax+1)/(np.arange(lmax+1)+1.)
    clEE = clBB*BBtoEE

    #Define clTT.  This is obtained my crossing Planck 353 GHz maps with the SPTpol 500d field and color-correcting
    #to SPTpol 150 GHz band.  From Tom, this is 1.04 uK^2 in D_\ell, assuming the same power law form as the EE
    #dust (i.e., TT and EE are 100% correlated).
    clTT = dlTT_dust * (np.arange(lmax+1)/pivot_ell_TT)**power_law_TT * 2.*np.pi/np.arange(lmax+1)/(np.arange(lmax+1)+1.)

    #Define clTE as the geometric mean of clTT and clEE.
    clTE = np.sqrt(clTT*clEE)

    #Get rid of NaNs and infs
    clTT[clTT != clTT] = 0.0
    clEE[clEE != clEE] = 0.0
    clBB[clBB != clBB] = 0.0
    clTE[clTE != clTE] = 0.0

    clTT[np.abs(clTT) == np.inf] = 0.0
    clEE[np.abs(clEE) == np.inf] = 0.0
    clBB[np.abs(clBB) == np.inf] = 0.0
    clTE[np.abs(clTE) == np.inf] = 0.0

    return np.array([clTT, clEE, clBB, clTE])


def gaussian_sz_component(dl3000_sz=5.5, sz_template_file='dl_sz.txt', lmax=15000.):
    '''
    Define a Gaussian realization of tSZ+kSZ foreground. We use the model from Shaw, et al. 2010,
    as in most previous SPT analyses.  We simply assume the polarized component is zero.

    INPUTS:
        dl3000: D_\ell value of TT in uK^2 at \ell=3000 defined at 153 GHz.  I'm pulling this
                  number from Story 12, which has an effective band center of 153 GHz for
                  this component.
        template_file: Filing containing model template (in D_\ell uK^2) normalized to 1 at \ell=3000.
                       Note, the template will be set to zero at all ells 
                       max_template_ell < ell < lmax.
        lmax: maximum \ell at which to generate component.
    '''
    
    #Read in SZ model template.
    d = open(sz_template_file, 'r').read().split('\n')[:-1]
    min_ell = int(np.min([lmax, len(d)]))

    template = np.zeros(lmax+1)
    for i in range(min_ell+1):
        template[i] += np.float(d[i].split(' ')[1])
        
    #Multiply by dl3000 scaling, and convert to C_\ell.
    ells = np.arange(lmax+1)
    clTT = dl3000_sz*template * 2.*np.pi/ells/(ells+1.)
    clEE = np.zeros(lmax+1)
    clBB = np.zeros(lmax+1)
    clTE = np.zeros(lmax+1)

    #Get rid of nans.
    clTT[clTT != clTT] = 0.0

    return np.array([clTT, clEE, clBB, clTE])


def gaussian_foreground_cls(poisson=True,
                            dl3000_poisson=19.3,
                            pol_frac_poisson=0.10431,
                            clustered_poisson=True,
                            dlTT_clust=5.0,
                            pol_frac_clust=0.0,
                            cirrus=False,
                            dlTT3000=0.21,
                            thermal_dust=True,
                            dlTT_dust=1.15,
                            pivot_ell_TT=80.,
                            power_law_TT=-0.42,
                            dlBB_dust=0.0118,
                            pivot_ell_BB=80.,
                            power_law_BB=-0.42,
                            BBtoEE=2.0,
                            sz=True,
                            dl3000_sz=5.5,
                            sz_template_file='dl_sz.txt',
                            return_dl = False,
                            return_components=True,
                            lmax=15000,
                            plot_components=False):
    '''
    Returns tuple of foreground cls: (clTT, clEE, clBB, clTE).
    This can be passed as-is to healpy.sphtfunc.symalm() to calculate
    tlm, elm, and blm for the foregrounds.
    '''
    
    ell = np.arange(lmax+1)
    
    clTT = np.zeros(lmax+1)
    clEE = np.zeros(lmax+1)
    clBB = np.zeros(lmax+1)
    clTE = np.zeros(lmax+1)

    if poisson:
        cl_poisson = gaussian_poisson_component(lmax=lmax)
        clTT += cl_poisson[0]
        clEE += cl_poisson[1]
        clBB += cl_poisson[2]
        clTE += cl_poisson[3]
    else:
        cl_poisson = np.array([np.zeros(lmax+1),np.zeros(lmax+1),np.zeros(lmax+1),np.zeros(lmax+1)])

    if clustered_poisson:
        cl_clustered_poisson = gaussian_clustered_poisson_component(lmax=lmax)
        clTT += cl_clustered_poisson[0]
        clEE += cl_clustered_poisson[1]
        clBB += cl_clustered_poisson[2]
        clTE += cl_clustered_poisson[3]
    else:
        cl_clustered_poisson = np.array([np.zeros(lmax+1),np.zeros(lmax+1),np.zeros(lmax+1),np.zeros(lmax+1)])

    if cirrus:
        cl_cirrus = gaussian_cirrus_component(lmax=lmax)
        clTT += cl_cirrus[0]
        clEE += cl_cirrus[1]
        clBB += cl_cirrus[2]
        clTE += cl_cirrus[3]
    else:
        cl_cirrus = np.array([np.zeros(lmax+1),np.zeros(lmax+1),np.zeros(lmax+1),np.zeros(lmax+1)])

    if thermal_dust:
        cl_dust = gaussian_thermal_dust_component(lmax=lmax)
        clTT += cl_dust[0]
        clEE += cl_dust[1]
        clBB += cl_dust[2]
        clTE += cl_dust[3]
    else:
        cl_dust = np.array([np.zeros(lmax+1),np.zeros(lmax+1),np.zeros(lmax+1),np.zeros(lmax+1)])

    if sz:
        cl_sz = gaussian_sz_component(lmax=lmax, sz_template_file=sz_template_file)
        clTT += cl_sz[0]
        clEE += cl_sz[1]
        clBB += cl_sz[2]
        clTE += cl_sz[3]
    else:
        cl_sz = np.array([np.zeros(lmax+1),np.zeros(lmax+1),np.zeros(lmax+1),np.zeros(lmax+1)])

    cl_tot = np.array([clTT, clEE, clBB, clTE])
    
    if return_dl:
        cl_tot *= ell*(ell+1.)/2./np.pi
        cl_poisson *= ell*(ell+1.)/2./np.pi
        cl_clustered_poisson *= ell*(ell+1.)/2./np.pi
        cl_cirrus *= ell*(ell+1.)/2./np.pi
        cl_dust *= ell*(ell+1.)/2./np.pi
        cl_sz *= ell*(ell+1.)/2./np.pi

    if plot_components:
        #Grab an primary CMB
        tell, tTT, tEE, tBB, tTE = ut.read_spectra('/home/jhenning/cambfits/planck_lensing_wp_highL_bestFit_20130627_massless0p046_massive3_lensedtotCls.dat')

        ells = np.arange(lmax+1)

        all_terms = copy.copy(cl_tot)
        all_terms[0][2:6000] += tTT[0:5998]
        all_terms[1][2:6000] += tEE[0:5998]
        all_terms[2][2:6000] += tBB[0:5998]
        all_terms[3][2:6000] += tTE[0:5998]

        py.figure(1)
        py.clf()
        py.loglog(ells, all_terms[0], 'k-', label='Total')
        py.loglog(tell[2:5998],tTT[2:5998], 'r-', label='Lensed CMB')
        py.loglog(ells, cl_poisson[0], 'b--', label='Poisson')
        py.loglog(ells, cl_clustered_poisson[0], 'orange', linestyle='--', label='Clustered Poisson')
        py.loglog(ells, cl_sz[0], 'g--', label='tSZ + kSZ')
        py.loglog(ells, cl_dust[0], 'k--', label='Thermal Dust')
        py.xlabel('Multipole $\ell$', fontsize=20)
        py.ylabel('$D_\ell^{TT}$ [$\mu$K$^2$]', fontsize=20)
        py.ylim((1e-3,1e4))
        py.xlim((2,11000))
        py.legend(loc='best', fontsize=10)
        py.savefig('TT_combined_spectrum.png')

        py.figure(2)
        py.clf()
        py.loglog(ells, all_terms[1], 'k-', label='Total')
        py.loglog(tell[2:5998],tEE[2:5998], 'r-', label='Lensed CMB')
        py.loglog(ells, cl_poisson[1], 'b--', label='Poisson')
        py.loglog(ells, cl_dust[1], 'k--', label='Thermal Dust')
        py.xlim((2,11000))
        py.ylim((1e-3,1e2))
        py.xlabel('Multipole $\ell$', fontsize=20)
        py.ylabel('$D_\ell^{EE}$ [$\mu$K$^2$]', fontsize=20)
        py.legend(loc='best', fontsize=10)
        py.savefig('EE_combined_spectrum.png')

        py.figure(3)
        py.clf()
        py.plot(ells, all_terms[3], 'k-', label='Total')
        py.plot(tell[2:5998],tTE[2:5998], 'r-', label='Lensed CMB')
        py.plot(ells, cl_dust[3], 'k--', label='Thermal Dust')
        py.xlim((2,70))
        py.ylim((-6.5,3))
        py.xlabel('Multipole $\ell$', fontsize=20)
        py.ylabel('$D_\ell^{TE}$ [$\mu$K$^2$]', fontsize=20)
        py.legend(loc='best', fontsize=10)
        py.savefig('TE_combined_spectrum.png')

        py.figure(4)
        py.clf()
        py.loglog(ells, all_terms[2], 'k-', label='Total')
        py.loglog(tell[2:5998],tBB[2:5998], 'r-', label='Lensed CMB')
        py.loglog(ells, cl_poisson[2], 'b--', label='Poisson')
        py.loglog(ells, cl_dust[2], 'k--', label='Thermal Dust')
        py.xlim((2,11000))
        py.ylim((1e-6,4))
        py.xlabel('Multipole $\ell$', fontsize=20)
        py.ylabel('$D_\ell^{BB}$ [$\mu$K$^2$]', fontsize=20)
        py.legend(loc='best', fontsize=10)
        py.savefig('BB_combined_spectrum.png')


    if return_components:
        return cl_tot, cl_poisson, cl_clustered_poisson, cl_cirrus, cl_dust, cl_sz
    else:
        clTT = cl_tot[0]
        clEE = cl_tot[1]
        clBB = cl_tot[2]
        clTE = cl_tot[3]
        
        #Assume EB and TB from foregrounds are zero.
        clEB = np.zeros(len(clTT))
        clTB = np.zeros(len(clTT))

        cls = (clTT, clEE, clBB, clTE, clEB, clTB)

        return cls


    
    
    



