#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 02:06:37 2022

@author: pierre
"""

import cloudy_parser as cp
import mappings_parser as mp
import numpy as np
import fitter
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy.ndimage import maximum_filter, minimum_filter, median_filter, zoom

from astropy.constants import pc, m_p, M_sun
pc = float(pc.value)
m_proton = float(m_p.value)
m_sun = float(M_sun.value)

dist = 14.4*1e6*pc
pir2 = 4*np.pi*dist**2

#%%

hlam = np.loadtxt('./PRODUITS/lambdas_h.dat')
hnod0 = np.loadtxt('./PRODUITS/hnod0.dat')
hnod1 = np.loadtxt('./PRODUITS/hnod1.dat')
hnod2 = np.loadtxt('./PRODUITS/hnod2.dat')
herr0 = np.loadtxt('./PRODUITS/herr0.dat')
herr1 = np.loadtxt('./PRODUITS/herr1.dat')
herr2 = np.loadtxt('./PRODUITS/herr2.dat')

jlam = np.loadtxt('./PRODUITS/lambdas_j.dat')
jnod0 = np.loadtxt('./PRODUITS/jnod0.dat')
jnod1 = np.loadtxt('./PRODUITS/jnod1.dat')
jnod2 = np.loadtxt('./PRODUITS/jnod2.dat')
jerr0 = np.loadtxt('./PRODUITS/jerr0.dat')
jerr1 = np.loadtxt('./PRODUITS/jerr1.dat')
jerr2 = np.loadtxt('./PRODUITS/jerr2.dat')

klam = np.loadtxt('./PRODUITS/lambdas_k.dat')
knod0 = np.loadtxt('./PRODUITS/knod0.dat')
knod1 = np.loadtxt('./PRODUITS/knod1.dat')
knod2 = np.loadtxt('./PRODUITS/knod2.dat')
kerr0 = np.loadtxt('./PRODUITS/kerr0.dat')
kerr1 = np.loadtxt('./PRODUITS/kerr1.dat')
kerr2 = np.loadtxt('./PRODUITS/kerr2.dat')

lambdas = np.concatenate([jlam, hlam, klam])
nod0 = np.concatenate([jnod0, hnod0, knod0])
err0 = np.concatenate([jerr0, herr0, kerr0])
nod1 = np.concatenate([jnod1, hnod1, knod1])
err1 = np.concatenate([jerr1, herr1, kerr1])
nod2 = np.concatenate([jnod2, hnod2, knod2])
err2 = np.concatenate([jerr2, herr2, kerr2])

def prop(x, a):
    return a*x


lambdas_0 = lambdas*1083/1084*1e-9
lambdas_1 = lambdas*1.0009242*1e-9
lambdas_2 = lambdas*231965/232115*1.0009242*1e-9

nod0b = nod0.copy()
for k in range(5):
    rawconts = []
    for k in range(150):
        rawcont_t = minimum_filter(median_filter(nod0b, 10+np.random.randint(100)), 10+np.random.randint(100))
        rawconts.append(rawcont_t)
    rawcont0 = np.median(rawconts, 0)
    spec0 = (nod0b-rawcont0)
    spec_for_fit_0 = spec0
    mask0 = spec_for_fit_0 > -1e6
    spec_for_fit_0 = spec_for_fit_0[mask0]
    nod0b = spec_for_fit_0.copy()
    
    
# spec_for_fit_0 = spec0/np.mean(spec0)
mask_sph = lambdas_0 < 1780
mask_sin = lambdas_0 > 1780
l_sph = lambdas_0[mask_sph]
l_sin = lambdas_0[mask_sin]
nod_sph = spec_for_fit_0[mask_sph]
nod_sin = spec_for_fit_0[mask_sin]

zoo = 3

lambdas_0 = np.concatenate([l_sph, zoom(l_sin, 1/zoo, order=0, mode='nearest')])
spec_for_fit_0 = np.concatenate([nod_sph, median_filter(zoom(nod_sin, 1/zoo, order=1), 5)])

rawconts = []
for k in range(150):
    rawcont_t = minimum_filter(median_filter(nod1, 10+np.random.randint(100)), 10+np.random.randint(100))
    rawconts.append(rawcont_t)
rawcont1 = np.median(rawconts, 0)
spec1 = (nod1-rawcont1)
spec_for_fit_1 = spec1.copy()
mask1 = spec_for_fit_1 > -1e6
mask1b = ((lambdas_1 > 1.78e-6)*(lambdas_1 < 1.95e-6))# + ((lambdas_1 > 1.36e-6)*(lambdas_1 < 1.5e-6)) 
# mask1b =  ~mask1
spec_for_fit_1 = spec_for_fit_1

rawconts = []
for k in range(150):
    rawcont_t = minimum_filter(median_filter(nod2, 10+np.random.randint(100)), 10+np.random.randint(100))
    rawconts.append(rawcont_t)
rawcont2 = np.median(rawconts, 0)
spec2 = (nod2-rawcont2)
spec2[12] = np.mean(spec2[11:14:2])
spec2[25] = np.mean(spec2[24:27:2])
spec_for_fit_2 = spec2
mask2 = spec_for_fit_2 > -1e6
mask2b = ((lambdas_2 > 1.78e-6)*(lambdas_2 < 1.95e-6))# + ((lambdas_2 > 1.36e-6)*(lambdas_2 < 1.5e-6)) 
# mask2b =  ~mask2
# mask2b = spec_for_fit_2 < 1e-16
spec_for_fit_2 = spec_for_fit_2[mask2]



#%%
from tqdm import tqdm

lam = lambdas_1[mask1]
spec = spec_for_fit_1[mask1]
err = err1[mask1]

lam_no_mask = lambdas_1[mask1]
spec_no_mask = spec1[mask1]


#%%


from fitter import fitter, convert_to_fluxes
lam = lambdas_2[mask1]
spec = spec_for_fit_2[mask1]*1e6
      
           
lines = [
    # [[np.mean(spec), 9.9e-7, 3.04e-9, 0],
    # [None, None, None,   0],
    # [9.81e-7, 9.98e-7]],
    
    [[np.mean(spec), 0.98248e-6, np.mean(spec), 0.99e-6, 3.04e-9, 0],
    [None, 0.98248e-6, None, 0.99e-6, None, 0],
    [9.81e-7, 9.98e-7]],
    
    [[np.mean(spec), 1.00494e-6, np.mean(spec), 1.01237e-6, 3.04e-9, 0],
    [None, 1.00494e-6, None, 1.01237e-6, None, 0],
    [9.95e-7, 10.22e-7]],
    
    [[np.mean(spec), 1.032e-6, np.mean(spec), 1.039e-6, 3.04e-9, 0],
    [None, 1.032e-6, None, 1.039e-6, 3.04e-9, 0],
    [1.024e-6, 1.05e-6]],
    
    # [[np.mean(spec), 1.0744e-6, np.mean(spec), 1.083e-6, np.mean(spec), 1.0938e-6, 3.04e-9, 0],
    # [None, 1.0744e-6, None, 1.083e-6, None, 1.0938e-6, None, 0],
    # [1.06e-6, 1.11e-6]],
    
    [[np.mean(spec), 1.083e-6, np.mean(spec), 1.0938e-6, 3.04e-9, 0],
    [None, 1.083e-6, None, 1.0938e-6, None, 0],
    [1.06e-6, 1.11e-6]],
    
    [[np.mean(spec), 1.146e-6, np.mean(spec), 1.163e-6, np.mean(spec), 1.189e-6, 3.04e-9, 0],
    [None, 1.146e-6, None, 1.163e-6, None, 1.189e-6, None, None],
    [1.135e-6, 1.235e-6]],
    
    [[np.mean(spec), 1.252e-6, np.mean(spec), 1.281e-6, 3.04e-9, 0],
    [None, None, None, None, None, None],
    [1.21e-6, 1.33e-6]],
    
    [[np.mean(spec), 1.43e-6, 3.04e-9, 0],
    [None, None, None, None],
    [1.4e-6, 1.46e-6]],
    
    [[np.mean(spec), 1.64e-6, np.mean(spec), 1.68e-6, np.mean(spec), 1.74e-6, 3.04e-9, 0],
    [None, None, None, None, None, None, 3.04e-9, None],
    [1.6e-6, 1.77e-6]],

    [[np.mean(spec), 1.875e-6, np.mean(spec), 1.963e-6, 3.04e-9, 0],
    [None, None, None, None, None, None],
    [1.75e-6, 2.0e-6]],
    
    [[np.mean(spec), 2.033e-6, np.mean(spec), 2.045e-6, np.mean(spec), 2.0582e-6, 2.36e-9, 0],
    [None, None, None, None, None, None, 2.36e-9, None],
    [2.02e-6, 2.07e-6]],

    [[np.mean(spec), 2.12e-6, 2.36e-9, 0],
    [None, None, None, None],
    [2.117e-6, 2.14e-6]],
    
    [[np.mean(spec), 2.166e-6, 2.36e-9, 0],
    [None, None, 2.36e-9, None],
    [2.146e-6, 2.186e-6]],
    
    [[np.mean(spec), 2.32e-6, 2.36e-9, 0],
    [None, None, None, None],
    [2.28e-6, 2.34e-6]],

    [[np.mean(spec), 2.406e-6, np.mean(spec), 2.423e-6, 2.36e-9, 0],
    [None, None, None, None, 2.36e-9, 0],
    [2.38e-6, 2.43e-6]]
    
    ]

infos_lines = [
# ["[C I]", 982.5, 11.26],
["[C I]", 985.1, 11.26],
["[S VII]", 986.9, 281.0],
# ["He II", 1004.5, 54.4],
["H I", 1004.9, 13.6],
["He II", 1012.4, 54.4],
# ["[S II]", 1028.7, 23.3],
["[S II]*", 1032.0, 23.3],
# ["[S II]", 1033.6, 23.3],
["[N I]*", 1039.8, 14.5],
["He I", 1083.3, 24.6],
["H I", 1093.8, 13.6],
["[P II]", 1146.0, 19.8],
["He II", 1167.3, 54.4],
["[P II]", 1188.6, 19.8],
["[S IX]", 1251.7, 379.8],
["H I", 1281.8, 13.6],
["[Si X]", 1430.2, 401.4],
["[Fe II]", 1643.6, 16.2],
["H I", 1680.7, 13.6],
["H I", 1736.2, 13.6],
["H I", 1875.6, 13.6],
["[Si VI]", 1960.23, 205.3],
["[V I]/[Cs II]?", 2033.0, 24.6],
["[Cs II]/[V I]?", 2045.0, 24.6],
["[V I]/He I?", 2058.0, 24.6],
["$H_2$", 2121.2, 0],
["H I", 2165.5, 13.6],
["[Ca VIII]", 2322.2, 147.3],
["[Mg II]?", 2404.8, 15.0],
["[Mg II]?", 2419.5, 15.0],
] 
    
    
#%%


import pyneb as pn


print('############')
print('# Nodule 0 #')
print('############')

params_0, covs_0 = fitter(lambdas_0, spec_for_fit_0*1e6, lines, plot=True)
lams, fls, sts = convert_to_fluxes(lines, params_0, covs_0)
lams, fls_0, sts_0 = convert_to_fluxes(lines, params_0, covs_0)
for i in range(len(fls)):
    print(str(int(lams[i]*1e9))+' : '+str(fls[i])+' +/- '+str(sts[i])+'  ('+str(np.around(100*sts[i]/fls[i]))+'%)')


H1 = pn.RecAtom('H', 1)
pn.atomicData.getAllAvailableFiles('H1')

lams_obs = np.array([lams[7], lams[12]])
H_obs = np.array([fls[7], fls[12]])
st_obs = np.array([sts[7], sts[12]])
lams_obs = np.array([lams[7], lams[12]])
H_obs = np.array([fls[7], fls[12]])
st_obs = np.array([sts[7], sts[12]])
st_obs /= H_obs[1]
H_obs /= H_obs[1]

AL_AK = np.loadtxt('./AL_AK.csv', delimiter=';')
AL_AK[:,0] = 1/AL_AK[::-1,0]
AL_AK[:,1] = AL_AK[::-1,1]

def extinct(lam, AK):
    alak = np.interp(lam, AL_AK[:, 0]*1e-6, AL_AK[:, 1])
    ext = 10**(0.4*alak*AK)
    return ext

def func(lams, AK):
    AK = abs(AK)
    T = 1e4
    N = 1e3
    H_ths = np.array([H1.getEmissivity(tem=T, den=N, lev_i=6, lev_j=3), H1.getEmissivity(tem=T, den=N, lev_i=5, lev_j=3)])
    H_ths /= H_ths[1]
    H_ths *= extinct(lams, AK)
    return H_ths

p0, c0 = curve_fit(func, lams_obs, H_obs, sigma=st_obs, p0 = [0.1])
print('A_k = '+str(p0)+' +/- '+str(np.sqrt(c0)))

extinction_0 = extinct(lambdas_0, p0[0])

print('   ')
print('   ')
print('   ')
    
#%%


import pyneb as pn


print('############')
print('# Nodule 1 #')
print('############')

params_1, covs_1 = fitter(lambdas_1, spec_for_fit_1*1e6, lines, plot=True)
lams, fls, sts = convert_to_fluxes(lines, params_1, covs_1)
lams, fls_1, sts_1 = convert_to_fluxes(lines, params_1, covs_1)
for i in range(len(fls)):
    print(str(int(lams[i]*1e9))+' : '+str(fls[i])+' +/- '+str(sts[i])+'  ('+str(np.around(100*sts[i]/fls[i]))+'%)')


H1 = pn.RecAtom('H', 1)
pn.atomicData.getAllAvailableFiles('H1')

lams_obs = np.array([lams[7], lams[12], lams[15], lams[16], lams[17], lams[23]])
H_obs = np.array([fls[7], fls[12], fls[15], fls[16], fls[17], fls[23]])
st_obs = np.array([sts[7], sts[12], sts[15], sts[16], sts[17], sts[23]])
lams_obs = np.array([lams[7], lams[12], lams[15], lams[16], lams[23]])
H_obs = np.array([fls[7], fls[12], fls[15], fls[16], fls[23]])
st_obs = np.array([sts[7], sts[12], sts[15], sts[16], sts[23]])
st_obs /= H_obs[1]
H_obs /= H_obs[1]

AL_AK = np.loadtxt('./AL_AK.csv', delimiter=';')
AL_AK[:,0] = 1/AL_AK[::-1,0]
AL_AK[:,1] = AL_AK[::-1,1]

def extinct(lam, AK):
    alak = np.interp(lam, AL_AK[:, 0]*1e-6, AL_AK[:, 1])
    ext = 10**(0.4*alak*AK)
    return ext

def func(lams, AK):
    AK = abs(AK)
    T = 1e4
    N = 1e3
    H_ths = np.array([H1.getEmissivity(tem=T, den=N, lev_i=6, lev_j=3), H1.getEmissivity(tem=T, den=N, lev_i=5, lev_j=3), H1.getEmissivity(tem=T, den=N, lev_i=11, lev_j=4), H1.getEmissivity(tem=T, den=N, lev_i=10, lev_j=4), H1.getEmissivity(tem=T, den=N, lev_i=7, lev_j=4)])
    H_ths /= H_ths[1]
    H_ths *= extinct(lams, AK)
    return H_ths

p1, c1 = curve_fit(func, lams_obs, H_obs, sigma=st_obs, p0 = [0.1])
print('A_k = '+str(p1)+' +/- '+str(np.sqrt(c1)))

extinction_1 = extinct(lambdas_1, p1[0])

print('   ')
print('   ')
print('   ')
#%%

import pyneb as pn


print('############')
print('# Nodule 2 #')
print('############')

params_2, covs_2 = fitter(lambdas_2, spec_for_fit_2*1e6, lines, plot=True)
lams, fls, sts = convert_to_fluxes(lines, params_2, covs_2)
lams, fls_2, sts_2 = convert_to_fluxes(lines, params_2, covs_2)
for i in range(len(fls)):
    print(i, str(int(lams[i]*1e9))+' : '+str(fls[i])+' +/- '+str(sts[i])+'  ('+str(np.around(100*sts[i]/fls[i]))+'%)')

H1 = pn.RecAtom('H', 1)
pn.atomicData.getAllAvailableFiles('H1')

lams_obs = np.array([lams[7], lams[12], lams[15], lams[16], lams[17], lams[23]])
H_obs = np.array([fls[7], fls[12], fls[15], fls[16], fls[17], fls[23]])
st_obs = np.array([sts[7], sts[12], sts[15], sts[16], sts[17], sts[23]])
lams_obs = np.array([lams[7], lams[12], lams[15], lams[16], lams[23]])
H_obs = np.array([fls[7], fls[12], fls[15], fls[16], fls[23]])
st_obs = np.array([sts[7], sts[12], sts[15], sts[16], sts[23]])
st_obs /= H_obs[1]
H_obs /= H_obs[1]


AL_AK = np.loadtxt('./AL_AK.csv', delimiter=';')
AL_AK[:,0] = 1/AL_AK[::-1,0]
AL_AK[:,1] = AL_AK[::-1,1]

def extinct(lam, AK):
    alak = np.interp(lam, AL_AK[:, 0]*1e-6, AL_AK[:, 1])
    ext = 10**(0.4*alak*AK)
    return ext

def func(lams, AK):
    AK = abs(AK)
    T = 1e4
    N = 1e3
    H_ths = np.array([H1.getEmissivity(tem=T, den=N, lev_i=6, lev_j=3), H1.getEmissivity(tem=T, den=N, lev_i=5, lev_j=3), H1.getEmissivity(tem=T, den=N, lev_i=11, lev_j=4), H1.getEmissivity(tem=T, den=N, lev_i=10, lev_j=4), H1.getEmissivity(tem=T, den=N, lev_i=7, lev_j=4)])
    H_ths /= H_ths[1]
    H_ths *= extinct(lams, AK)
    return H_ths

p2, c2 = curve_fit(func, lams_obs, H_obs, sigma=st_obs, p0 = [0.1])
print('A_k = '+str(p2)+' +/- '+str(np.sqrt(c2)))

extinction_2 = extinct(lambdas_2, p2[0])

#%%

print(r'\begin{table*}[ht]')
print(r'\centering')
print(r'\caption{\label{table_em} Emission lines summary}')
print(r'\begin{tabular}{c|c|c|c|c|c}')
print(r'Line & Rest & Ionization & Measured flux & Measured flux & Measured flux \\')
print(r'Identification & Wavelength & Energy & Cloud 0 & Cloud 1 & Cloud 2 \\')
print(r'  & (nm)  & (eV) & $(10^{-18}\ W.m^{-2}$) & $(10^{-18}\ W.m^{-2}$) & $(10^{-18}\ W.m^{-2}$)\\')
print(r'\hline')
for i in range(len(fls)):
    info = infos_lines[i]
    fl0 = fls_0[i]*1e18
    fl1 = fls_1[i]*1e18
    fl2 = fls_2[i]*1e18
    st0 = sts_0[i]*1e18*3
    st1 = sts_1[i]*1e18*3
    st2 = sts_2[i]*1e18*3
    b0 = r'{'+str(info[0])+r'}'
    b1 = '{:.1f}'.format(info[1])
    b2 = '{:.1f}'.format(info[2])
    b3, b4, b5 = '', '', ''
    if st0<(3*fl0) and (info[1]<1440):
        b3 = '{:.1f}'.format(fl0)+' $\pm$ '+'{:.1f}'.format(st0)
    if st1<(3*fl1):
        b4 = '{:.1f}'.format(fl1)+' $\pm$ '+'{:.1f}'.format(st1)
    if st2<(3*fl2):
        b5 = '{:.1f}'.format(fl2)+' $\pm$ '+'{:.1f}'.format(st2)
    print(b0+' & '+b1+' & '+b2+' & '+b3+' & '+b4+' & '+b5+r' \\')
print(r'\end{tabular}')
print(r'\end{table*}')


#%%


lam0 = lambdas_0[~mask1b]
spec0 = spec_for_fit_0[~mask1b]
std0 = err0[~mask1b]
lam1 = lambdas_1[~mask1b]
spec1 = spec_for_fit_1[~mask1b]*extinction_1[~mask1b]
std1 = err1[~mask1b]
lam2 = lambdas_2[~mask2b]
spec2 = spec_for_fit_2[~mask2b]*extinction_1[~mask2b]
std2 = err2[~mask2b]
spec_glob = spec.copy()

#%%

def is_cloudy(info):
    return len(info)>6

def lines_modificiations(infos_to_clean, lines_to_clean):

    clean_infos_t = []
    clean_lines_t = []
    
    for n in range(len(infos_to_clean)):
        lines = lines_to_clean[n]
        infos = infos_to_clean[n]
        lines_t = []
        infos_t = []
        # if ('n100_' in infos[0]) or is_cloudy(infos):
        for n in range(len(lines)):
            line = lines[n]
            if 'Fe' not in line[0]:
                if 'Si' in line[0] and (abs(1-line[2]/1.960e-6) < 0.002):
                    line[2] = 1.963e-6
                lines_t.append(line)
        clean_lines_t.append(lines_t)
        clean_infos_t.append(infos)
            
    clean_lines = clean_lines_t
    clean_infos = clean_infos_t
    
    return clean_infos, clean_lines

infos_cloudy, lines_cloudy = cp.parse_file('model.out', './MODEL_LOW_L/', emergent=True, lam0=np.min(lam), lam1=np.max(lam))
# infos_cloudy, lines_cloudy = cp.parse_file('model.out', './MODEL_SEMI_COARSE_erg_NOT_TOTAL/', emergent=True, lam0=np.min(lam), lam1=np.max(lam))
infos_cloudy, lines_cloudy = lines_modificiations(infos_cloudy, lines_cloudy)
infos_mappings, lines_mappings = mp.parse_directory('./MAPPINGS/emission_line_ratios/', lam0=np.min(lam), lam1=np.max(lam))
infos_mappings, lines_mappings = lines_modificiations(infos_mappings, lines_mappings)



#%%

specs_cloudy_1 = []
specs_cloudy_2 = []
for line_cloudy in lines_cloudy:
    spec_cloudy_1 = cp.make_spec(line_cloudy, lam1, 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)*4*np.pi*(10**20)**2
    specs_cloudy_1.append(spec_cloudy_1*1e-7/pir2)
    spec_cloudy_2 = cp.make_spec(line_cloudy, lam2, 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)*(0.7/0.3)**2*4*np.pi*(10**20)**2
    specs_cloudy_2.append(spec_cloudy_2*1e-7/pir2)

specs_mappings_1 = []
specs_mappings_2 = []
for line_mappings in lines_mappings:
    spec_mappings_1 = mp.make_spec(line_mappings, lam1, 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)
    specs_mappings_1.append(spec_mappings_1*4*np.pi*(1e20)**2*1e-7/pir2)
    spec_mappings_2 = mp.make_spec(line_mappings, lam2, 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)*(0.7/0.3)**2
    specs_mappings_2.append(spec_mappings_2*4*np.pi*(1e20)**2*1e-7/pir2)
    
#%%


def compare_spec_1d(spec_ref, err_ref, specs_comp, range_limit=[0, 0.05], progress_bar=False, power=1):
    k2s = []
    ps = []
    
    def prop(spec, a):
        if range_limit[0] < a < range_limit[1]:
            return a*spec
        else:
            return -spec*(min([abs(a-range_limit[0]), abs(a-range_limit[1])]))
    
    if progress_bar:
        ran = tqdm(range(len(specs_comp)))
    else:
        ran = range(len(specs_comp))
    for n in ran:
        spec_comp = specs_comp[n]
        if np.sum(spec_comp)>0:
            try:
                p, c = curve_fit(prop, spec_comp**power, spec_ref**power, p0 = np.mean(range_limit), sigma=err_ref, absolute_sigma=True)
            except Exception as e:
                print("Exception: ", n, e)
                p = [0]
        else:
            p = [0]
        spec_mod = prop(spec_comp, *p)
        k2 = np.sum((spec_mod**power-spec_ref**power)**2/err_ref**2)
        k2s.append(k2)
        ps.append(p)
    return k2s, ps

def compare_spec_1d(spec_ref, err_ref, specs_comp, range_limit=[0, 0.05], progress_bar=False, power=1):
    k2s = []
    ps = []
    
    def prop(spec, a):
        if range_limit[0] < a < range_limit[1]:
            return a*spec
        else:
            return -spec*(min([abs(a-range_limit[0]), abs(a-range_limit[1])]))
    
    if progress_bar:
        ran = tqdm(range(len(specs_comp)))
    else:
        ran = range(len(specs_comp))
    for n in ran:
        spec_comp = specs_comp[n]
        if np.sum(spec_comp)>0:
            try:
                p, c = curve_fit(prop, spec_comp**power, spec_ref**power, p0 = np.mean(range_limit))
            except Exception as e:
                print("Exception: ", n, e)
                p = [0]
        else:
            p = [0]
        spec_mod = prop(spec_comp, *p)
        k2 = np.sum((spec_mod**power-spec_ref**power)**2)
        k2s.append(k2)
        ps.append(p)
    return k2s, ps

def get_k2_cube(k2s, infos):
    c1s = []
    c2s = []
    c3s = []
    for info in infos:
        c1s.append(info[-1])
        c2s.append(info[-2])
        c3s.append(info[-3])
    c1s = np.sort([float(i) for i in set(c1s)])
    c2s = np.sort([float(i) for i in set(c2s)])
    c3s = np.sort([float(i) for i in set(c3s)])
    k2_cube = np.ones((len(c1s), len(c2s), len(c3s)))*np.max(k2s)
    for n in range(len(k2s)):
        c3, c2, c1 = [float(i) for i in infos[n][-3:]]
        i = np.argmin(abs(c1s-c1))
        j = np.argmin(abs(c2s-c2))
        k = np.argmin(abs(c3s-c3))
        k2_cube[i, j, k] = k2s[n]
    return k2_cube, c1s, c2s, c3s

def plot_results(ks, ps, lam, spec_ref, std_ref, specs_comp, infos_comp, vmin=None, vmax=None, prefix='cloud_cloumap', path='./FIGURES/'):
    
    n = np.argmin(ks)
    p = ps[n]
    spec_mod = specs_comp[n]*p
    
    mask_1 = lam < 1780e-9
    mask_2 = lam > 1950e-9
    
    plt.figure()
    plt.plot(1e6*lam[mask_1], spec_ref[mask_1], c='k')
    plt.plot(1e6*lam[mask_2], spec_ref[mask_2], c='k')
    plt.plot(1e6*lam[mask_1], spec_mod[mask_1], c='r', alpha=0.8)
    plt.plot(1e6*lam[mask_2], spec_mod[mask_2], c='r', alpha=0.8)
    plt.xlabel('Wavelength ($\mu m$)')
    plt.ylabel('Flux ($W.m^{-2}.\mu m^{-1}$)')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    plt.savefig(path+prefix+'_comp1.png')
    plt.savefig(path+prefix+'_comp1.pdf')
    
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10,5))
    # ax[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.subplots_adjust(hspace=0)
    ax[0].plot(1e6*lam[mask_1], spec_ref[mask_1], c='k')
    ax[0].plot(1e6*lam[mask_2], spec_ref[mask_2], c='k')
    ax[1].plot(1e6*lam[mask_1], spec_mod[mask_1], c='r')
    ax[1].plot(1e6*lam[mask_2], spec_mod[mask_2], c='r')
    ax[1].set_xlabel('Wavelength ($\mu m$)')
    ax[1].set_ylabel('Flux ($W.m^{-2}.\mu m^{-1}$)')
    ax[0].set_ylabel('Flux ($W.m^{-2}.\mu m^{-1}$)')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    plt.savefig(path+prefix+'_comp2.png')
    plt.savefig(path+prefix+'_comp2.pdf')
    
def plot_chi2_cloudy(ks, norm=True, k2_norm=None, prefix='cloud', path='./FIGURES/'):
    infos_comp = infos_cloudy
    k2_cube, c1s, c2s, c3s = get_k2_cube(ks, infos_comp)
    if norm:
        if k2_norm is None:
            k2_norm = np.min(k2_cube)
        k2_cube /= k2_norm    
    i, j, k = np.unravel_index(np.argmin(k2_cube), np.shape(k2_cube))
    
    plt.figure()
    plt.plot(10**c1s, np.min(k2_cube,(1,2)))
    plt.semilogx()
    plt.xlabel('Depth (pc)')
    plt.ylabel('Normalized residuals')
    plt.savefig(path+prefix+'_cloudy_k2_depth.png')
    plt.savefig(path+prefix+'_cloudy_k2_depth.pdf')
    plt.figure()
    plt.plot(10**c2s, np.min(k2_cube,(0,2)))
    plt.semilogx()
    plt.xlabel('Density ($cm^{-3}$)')
    plt.ylabel('Normalized residuals')
    plt.savefig(path+prefix+'_cloudy_k2_density.png')
    plt.savefig(path+prefix+'_cloudy_k2_density.pdf')
    plt.figure()
    plt.plot(10**c3s, np.min(k2_cube,(0,1)))
    plt.semilogx()
    plt.xlabel('Temperature (K)')
    plt.ylabel('Normalized residuals')
    plt.savefig(path+prefix+'_cloudy_k2_temp.png')
    plt.savefig(path+prefix+'_cloudy_k2_temp.pdf')
    
    plt.figure()
    plt.imshow(np.min(k2_cube,(0)), aspect='auto', extent = [np.min(c3s), np.max(c3s), np.min(c2s), np.max(c2s)], origin='lower')
    plt.xlabel('Temperature (log(K))')
    plt.ylabel('Density (log($cm^{-3}$))')
    plt.colorbar()
    plt.savefig(path+prefix+'_cloudy_k2s_temp_dens.png')
    plt.savefig(path+prefix+'_cloudy_k2s_temp_dens.pdf')
    plt.figure()
    plt.imshow(np.min(k2_cube,(1)), aspect='auto', extent = [np.min(c3s), np.max(c3s), np.min(c1s), np.max(c1s)], origin='lower')
    plt.xlabel('Temperature (log(K))')
    plt.ylabel('Depth (log(pc))')
    plt.colorbar()
    plt.savefig(path+prefix+'_cloudy_k2s_temp_depth.png')
    plt.savefig(path+prefix+'_cloudy_k2s_temp_depth.pdf')
    plt.figure()
    plt.imshow(np.min(k2_cube,(2)), aspect='auto', extent = [np.min(c2s), np.max(c2s), np.min(c1s), np.max(c1s)], origin='lower')
    plt.xlabel('Density ($log(cm^{-3}$))')
    plt.ylabel('Depth (log(pc))')
    plt.colorbar()
    plt.savefig(path+prefix+'_cloudy_k2s_dens_depth.png')
    plt.savefig(path+prefix+'_cloudy_k2s_dens_depth.pdf')
    
    
def plot_chi2_mappings(ks, norm=True, k2_norm=None, prefix='cloud', path='./FIGURES/'):
    infos_comp = infos_mappings
    k2_cube, c1s, c2s, c3s = get_k2_cube(ks, infos_comp)
    if norm:
        if k2_norm is None:
            k2_norm = np.min(k2_cube)
        k2_cube /= k2_norm    
    i, j, k = np.unravel_index(np.argmin(k2_cube), np.shape(k2_cube))
    
    log = np.log10
    
    plt.figure()
    plt.plot(c1s, np.min(k2_cube,(1,2)))
    plt.xlabel('Velocity ($km.s^{-1}$)')
    plt.ylabel('Normalized residuals')
    plt.savefig(path+prefix+'_mappings_k2_vel.png')
    plt.savefig(path+prefix+'_mappings_k2_vel.pdf')
    
    plt.figure()
    plt.plot(c2s, np.min(k2_cube,(0,2)))
    plt.semilogx()
    plt.xlabel('Magnetic field ($\mu G$))')
    plt.ylabel('Normalized residuals')
    plt.savefig(path+prefix+'_mappings_k2_mag.png')
    plt.savefig(path+prefix+'_mappings_k2_mag.pdf')
    
    plt.figure()
    plt.plot(c3s, np.min(k2_cube,(0,1)))
    plt.semilogx()
    plt.xlabel('Density ($cm^{-3}$)')
    plt.ylabel('Normalized residuals')
    plt.savefig(path+prefix+'_mappings_k2_den.png')
    plt.savefig(path+prefix+'_mappings_k2_den.pdf')
    
    fig, ax = plt.subplots()
    im = ax.imshow(np.min(k2_cube,(0)), aspect='auto', extent = [log(np.min(c3s)), log(np.max(c3s)), 0, 1], origin='lower')
    ax.set_xlabel('Density (log($cm^{-3}$))')
    ax.set_ylabel('Magnetic field ($\mu G$)')
    ax.yaxis.set_ticks(np.linspace(0,1,len(c2s))[::2])
    ax.yaxis.set_ticklabels(c2s[::2])
    fig.colorbar(im)
    plt.savefig(path+prefix+'_mappings_k2s_den_mag.png')
    plt.savefig(path+prefix+'_mappings_k2s_den_mag.pdf')
    fig, ax = plt.subplots()
    im = ax.imshow(np.min(k2_cube,(1)), aspect='auto', extent = [log(np.min(c3s)), log(np.max(c3s)), np.min(c1s), np.max(c1s)], origin='lower')
    ax.set_xlabel('Density (log($cm^{-3}$))')
    ax.set_ylabel('Velocity ($km.s^{-1}$)')
    fig.colorbar(im)
    plt.savefig(path+prefix+'_mappings_k2s_den_vel.png')
    plt.savefig(path+prefix+'_mappings_k2s_den_vel.pdf')
    fig, ax = plt.subplots()
    im = ax.imshow(np.min(k2_cube,(2)), aspect='auto', extent = [0, 1, np.min(c1s), np.max(c1s)], origin='lower')
    ax.set_xlabel('Magnetic field ($\mu G$)')
    ax.xaxis.set_ticks(np.linspace(0,1,len(c2s))[::4])
    ax.xaxis.set_ticklabels(c2s[::4])
    ax.set_ylabel('Velocity ($km.s^{-1}$)')
    fig.colorbar(im)
    plt.savefig(path+prefix+'_mappings_k2s_mag_vel.png')
    plt.savefig(path+prefix+'_mappings_k2s_mag_vel.pdf')
        
#%%
uccloudy1_k2s, uccloudy1_ps = compare_spec_1d(spec1, std1, specs_cloudy_1, range_limit = [0, 10000])
uccloudy2_k2s, uccloudy2_ps = compare_spec_1d(spec2, std2, specs_cloudy_2, range_limit = [0, 10000])
cloudy1_k2s, cloudy1_ps = compare_spec_1d(spec1, std1, specs_cloudy_1, range_limit = [0, 1])
cloudy2_k2s, cloudy2_ps = compare_spec_1d(spec2, std2, specs_cloudy_2, range_limit = [0, 1])
mappings1_k2s, mappings1_ps = compare_spec_1d(spec1, std1, specs_mappings_1, range_limit = [0, 0.05])
mappings2_k2s, mappings2_ps = compare_spec_1d(spec2, std2, specs_mappings_2, range_limit = [0, 0.05])

plot_results(uccloudy1_k2s, uccloudy1_ps, lam1, spec1, std1, specs_cloudy_1, infos_cloudy)
plot_results(uccloudy2_k2s, uccloudy2_ps, lam2, spec2, std2, specs_cloudy_2, infos_cloudy)
plot_results(cloudy1_k2s, cloudy1_ps, lam1, spec1, std1, specs_cloudy_1, infos_cloudy, prefix='cloud1_cloudy')
plot_results(cloudy2_k2s, cloudy2_ps, lam2, spec2, std2, specs_cloudy_2, infos_cloudy, prefix='cloud2_cloudy')
plot_results(mappings1_k2s, mappings1_ps, lam1, spec1, std1, specs_mappings_1, infos_mappings, prefix='cloud1_mappings')
plot_results(mappings2_k2s, mappings2_ps, lam2, spec2, std2, specs_mappings_2, infos_mappings, prefix='cloud2_mappings')

# plot_chi2_cloudy(uccloudy1_k2s)
# plot_chi2_cloudy(uccloudy2_k2s)
plot_chi2_cloudy(cloudy1_k2s, prefix='cloud1')
plot_chi2_cloudy(cloudy2_k2s, prefix='cloud2')
plot_chi2_mappings(mappings1_k2s, prefix='cloud1')
plot_chi2_mappings(mappings2_k2s, prefix='cloud2')
plt.close('all')


#%%


fig, ax = plt.subplots(3, 1, sharex=True, figsize=(11.5,5))
plt.subplots_adjust(hspace=0)
mask_1 = lam0 < 1450e-9
mask_2 = lam0 < 1780e-9
mask_2 *= ~mask_1
mask_3 = lam0 > 1950e-9
ax[0].plot(1e6*lam0[mask_1], spec0[mask_1], c='g')
ax[0].plot(1e6*lam0[mask_2], spec0[mask_2], c='r')
ax[0].plot(1e6*lam0[mask_3], spec0[mask_3], c='b')
ax[0].set_ylim([-0.1*np.max(spec0[mask_1]), 1.1*np.max(spec0[mask_1])])
mask_1 = lam1 < 1450e-9
mask_2 = lam1 < 1780e-9
mask_2 *= ~mask_1
mask_3 = lam1 > 1950e-9
ax[1].plot(1e6*lam1[mask_1], spec1[mask_1], c='g')
ax[1].plot(1e6*lam1[mask_2], spec1[mask_2], c='r')
ax[1].plot(1e6*lam1[mask_3], spec1[mask_3], c='b')
ax[1].set_ylim([-0.1*np.max(spec1[mask_1]), 1.1*np.max(spec1[mask_1])])
mask_1 = lam2 < 1450e-9
mask_2 = lam2 < 1780e-9
mask_2 *= ~mask_1
mask_3 = lam2 > 1950e-9
ax[2].plot(1e6*lam2[mask_1], spec2[mask_1], c='g')
ax[2].plot(1e6*lam2[mask_2], spec2[mask_2], c='r')
ax[2].plot(1e6*lam2[mask_3], spec2[mask_3], c='b')
ax[2].set_ylim([-0.1*np.max(spec2[mask_1]), 1.1*np.max(spec2[mask_1])])
ax[2].set_xlabel('Wavelength ($\mu m$)')
ax[1].set_ylabel('Flux ($W.m^{-2}.\mu m^{-1}$)')
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    

plt.savefig('./FIGURES/specs_nodules.png')
plt.savefig('./FIGURES/specs_nodules.pdf')

#%%

from astropy.io import fits

hdu = fits.open('./PRODUITS/cleaned.fits')
data = hdu[0].data

im_tot = np.sum(data,0)
im_h = np.sum(data[800:870],0)-70*np.median(data[700:970],0)
im_h[17:20,18:21]=0
im_s = np.sum(data[1000:1040],0)-40*np.median(data[900:1140],0)
im_s[17:20,18:21]=0

extent = 0.05*np.array([19, -18, -19, 18])+0.025

fig, ax = plt.subplots()
im = ax.imshow(im_h, vmin=0, vmax=np.max(median_filter(im_h,(2,2))), origin='lower', extent=extent)
ax.scatter(0,0, c='r')
axis = ax.axis()
ax.plot([0.045+0.19*10,0.045-0.19*10], [0.98*10,-0.98*10], c='g')
ax.plot([-0.045+0.19*10,-0.045-0.19*10], [0.98*10,-0.98*10], c='g')
ax.axis(axis)
ax.set_xlabel('Right Ascension (")')
ax.set_ylabel('Declination (")')
cbar = fig.colorbar(im)
cbar.set_label('Flux ($W.m^{-2}.pixel^{-1}$)', rotation=90, labelpad=15)

plt.savefig('./FIGURES/image_h.png')
plt.savefig('./FIGURES/image_h.pdf')


fig, ax = plt.subplots()
ax.imshow(im_s, vmin=0, vmax=np.max(median_filter(im_s,(2,2))), origin='lower', extent=extent)
ax.scatter(0,0, c='r')
axis = ax.axis()
ax.plot([0.045+0.19*10,0.045-0.19*10], [0.98*10,-0.98*10], c='g')
ax.plot([-0.045+0.19*10,-0.045-0.19*10], [0.98*10,-0.98*10], c='g')
ax.axis(axis)
ax.set_xlabel('Right Ascension (")')
ax.set_ylabel('Declination (")')
cbar = fig.colorbar(im)
cbar.set_label('Flux ($W.m^{-2}.pixel^{-1}$)', rotation=90, labelpad=15)

plt.savefig('./FIGURES/image_s.png')
plt.savefig('./FIGURES/image_s.pdf')


plt.close('all')
#%%
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(figsize=(12, 6))
axs = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,2),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )
# fig, axs = plt.subplots(1, 2, figsize=(12,6))
ax0 = axs[0]
ax1 = axs[1]
im = ax0.imshow(im_h, vmin=0, vmax=np.max(median_filter(im_h,(2,2))), origin='lower', extent=extent)
ax0.scatter(0,0, c='r')
axis = ax0.axis()
ax0.plot([0.045+0.19*10,0.045-0.19*10], [0.98*10,-0.98*10], c='g')
ax0.plot([-0.045+0.19*10,-0.045-0.19*10], [0.98*10,-0.98*10], c='g')
ax0.axis(axis)
ax0.set_xlabel('Right Ascension (")')
ax0.set_ylabel('Declination (")')
ax1.imshow(im_s, vmin=0, vmax=np.max(median_filter(im_s,(2,2))), origin='lower', extent=extent)
ax1.scatter(0,0, c='r')
axis = ax1.axis()
ax1.plot([0.045+0.19*10,0.045-0.19*10], [0.98*10,-0.98*10], c='g')
ax1.plot([-0.045+0.19*10,-0.045-0.19*10], [0.98*10,-0.98*10], c='g')
ax1.axis(axis)
ax1.set_xlabel('Right Ascension (")')
cbar = ax1.cax.colorbar(im)
ax1.cax.toggle_label(True)
# cbar = fig.colorbar(im)
cbar.set_label('Flux ($W.m^{-2}.pixel^{-1}$)', rotation=90, labelpad=15)
# plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)

plt.savefig('./FIGURES/images.png')
plt.savefig('./FIGURES/images.pdf')

# t = np.arange(0, 6.28, 0.01)
# ax1.scatter(0.3*np.sin(t), 0.3*np.cos(t))
# ax1.scatter(0.7*np.sin(t), 0.7*np.cos(t))
plt.close('all')

#%%



hdu = fits.open('./PRODUITS/cleaned_HR.fits')
data = hdu[0].data

im_tot = np.sum(data[-200:],0)
im_h = np.sum(data[820:860],0)-40*np.median(data[700:980],0)
im_h[17:20,18:21]=0
im_s = np.sum(data[1000:1040],0)-40*np.median(data[940:1100],0)
# im_s[17:20,18:21]=0

extent = 0.05*np.array([19, -18, -19, 18])+0.025


#%%

specs_cloudy_db = []
for line_cloudy in lines_cloudy:
    spec_cloudy = cp.make_spec(line_cloudy, lam1, 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)
    spt = spec_cloudy*1e-7/pir2
    if np.sum(spt)>np.sum(spec2):
        specs_cloudy_db.append(spt)

specs_mappings_db = []
for line_mappings in lines_mappings:
    spec_mappings = mp.make_spec(line_mappings, lam1, 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)
    spt = spec_mappings*4*np.pi*(1e20)**2*1e-7/pir2    
    if np.sum(spt)>np.sum(spec2):
        specs_mappings_db.append(spt)

#%%

def compare_spec_2d(spec_ref, err_ref, specs_comp_1, specs_comp_2, range_limit=[0, 0.05]):
    k2s = []
    ps = []
    range_limit = np.array(range_limit)
    
    
    for n1 in tqdm(np.arange(len(specs_comp_1))):
        spec_comp_1 = specs_comp_1[n1]
        for n2 in np.arange(len(specs_comp_2)):
            spec_comp_2 = specs_comp_2[n2]
            def prop(_, a, b):
                if (range_limit[0] < a < range_limit[1]) and (range_limit[0] < b < range_limit[1]):
                    return a*spec_comp_1+b*spec_comp_2
                else:
                    bad_1 = min([abs(a-range_limit[0]), abs(a-range_limit[1])])
                    bad_2 = min([abs(b-range_limit[0]), abs(b-range_limit[1])])
                    exp = np.exp(100*max([bad_1, bad_2])/(range_limit[1]-range_limit[0]))
                    return (a*spec_comp_1+b*spec_comp_2)/exp-(spec_comp_1+spec_comp_2)*max([bad_1, bad_2])*exp
            if True or ((np.sum(spec_comp_1)>np.sum(spec1)) and (np.sum(spec_comp_2)>np.sum(spec2))):
                try:
                    p, c = curve_fit(prop, None, spec_ref, p0 = range_limit*0+np.mean(range_limit), sigma=err_ref, absolute_sigma=True, maxfev=100000)
                except Exception as e:
                    print(e)
                    p = [0, 0]
            else:
                p = [0, 0]
            spec_mod = prop(None, *p)
            k2 = np.sum((spec_mod-spec_ref)**2/err_ref**2)
            k2s.append(k2)
            ps.append(p)
    return k2s, ps    
        
specs_cloudy_db1 = []
infos_cloudy_db1 = []
for sp1, info1 in tqdm(zip(specs_cloudy_1, infos_cloudy)):
    if np.sum(sp1)>0:
        for sp2, info2 in zip(specs_cloudy_1, infos_cloudy):
            if (np.sum(sp2)>0) and (float(info2[-4])==float(info1[-4])) and (float(info2[-3])==float(info1[-3])) and (float(info2[-2])==float(info1[-2])):
                for ratio in np.arange(0, 1.01, 0.1):
                    sp = ratio*sp1+(1-ratio)*sp2*np.sum(sp1)/np.sum(sp2)
                    specs_cloudy_db1.append(sp)
                    infos_cloudy_db1.append(np.concatenate([[ratio], info1, info2]))
                    
specs_cloudy_db2 = []
infos_cloudy_db2 = []
for sp1, info1 in tqdm(zip(specs_cloudy_2, infos_cloudy)):
    if np.sum(sp1)>0:
        for sp2, info2 in zip(specs_cloudy_2, infos_cloudy):
            if (np.sum(sp2)>0) and (float(info2[-4])==float(info1[-4])) and (float(info2[-3])==float(info1[-3])) and (float(info2[-2])==float(info1[-2])):
                for ratio in np.arange(0, 1.01, 0.1):
                    sp = ratio*sp1+(1-ratio)*sp2*np.sum(sp1)/np.sum(sp2)
                    specs_cloudy_db2.append(sp)
                    infos_cloudy_db2.append(np.concatenate([[ratio], info1, info2]))
                    

dbcloudy1_k2s, dbcloudy1_ps = compare_spec_1d(spec1, std1, specs_cloudy_db1, range_limit = [0, 10000])
dbcloudy2_k2s, dbcloudy2_ps = compare_spec_1d(spec2, std2, specs_cloudy_db2, range_limit = [0, 10000])
plot_results(dbcloudy1_k2s, dbcloudy1_ps, lam1, spec1, std1, specs_cloudy_db1, infos_cloudy_db1, prefix='dbcloud1_cloudy')
plot_results(dbcloudy2_k2s, dbcloudy2_ps, lam2, spec2, std2, specs_cloudy_db2, infos_cloudy_db2, prefix='dbcloud2_cloudy')

# double1_k2s, double1_ps = compare_spec_2d(spec1, std1, specs_cloudy_1, specs_cloudy_1, range_limit = [0, 100])
# double2_k2s, double2_ps = compare_spec_2d(spec2, std2, specs_cloudy_2, specs_mappings_2, range_limit = [0, 0.05])
