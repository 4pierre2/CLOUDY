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
herr2 = np.loadtxt('./PRODUITS/herr1.dat')*0

jlam = np.loadtxt('./PRODUITS/lambdas_j.dat')
jnod0 = np.loadtxt('./PRODUITS/jnod0.dat')
jnod1 = np.loadtxt('./PRODUITS/jnod1.dat')
jnod2 = np.loadtxt('./PRODUITS/jnod2.dat')
jerr0 = np.loadtxt('./PRODUITS/jerr0.dat')
jerr1 = np.loadtxt('./PRODUITS/jerr1.dat')
jerr2 = np.loadtxt('./PRODUITS/jerr1.dat')*0

klam = np.loadtxt('./PRODUITS/lambdas_k.dat')
knod0 = np.loadtxt('./PRODUITS/knod0.dat')
knod1 = np.loadtxt('./PRODUITS/knod1.dat')
knod2 = np.loadtxt('./PRODUITS/knod2.dat')
kerr0 = np.loadtxt('./PRODUITS/kerr0.dat')
kerr1 = np.loadtxt('./PRODUITS/kerr1.dat')
kerr2 = np.loadtxt('./PRODUITS/kerr1.dat')*0

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
mask1b = ((lambdas_1 > 1.36e-6)*(lambdas_1 < 1.5e-6)) + ((lambdas_1 > 1.78e-6)*(lambdas_1 < 1.95e-6))
# mask1b =  ~mask1
spec_for_fit_1 = spec_for_fit_1

rawconts = []
for k in range(150):
    rawcont_t = minimum_filter(median_filter(nod2, 10+np.random.randint(100)), 10+np.random.randint(100))
    rawconts.append(rawcont_t)
rawcont2 = np.median(rawconts, 0)
spec2 = (nod2-rawcont2)
spec_for_fit_2 = spec2
mask2 = spec_for_fit_2 > -1e6
spec_for_fit_2 = spec_for_fit_2[mask2]



#%%
from tqdm import tqdm

lam = lambdas_1[mask1]
spec = spec_for_fit_1[mask1]

lam_no_mask = lambdas_1[mask1]
spec_no_mask = spec1[mask1]


#%%


from fitter import fitter, convert_to_fluxes
lam = lambdas_2[mask1]
spec = spec_for_fit_2[mask1]*1e6
      
           
lines = [
    [[np.mean(spec), 9.9e-7, 3.04e-9, 0],
    [None, None, None,   0],
    [9.81e-7, 9.98e-7]],
    
    [[np.mean(spec), 1.00494e-6, np.mean(spec), 1.01237e-6, 3.04e-9, 0],
    [None, 1.00494e-6, None, 1.01237e-6, None, 0],
    [9.95e-7, 10.22e-7]],
    
    [[np.mean(spec), 1.032e-6, np.mean(spec), 1.039e-6, 3.04e-9, 0],
    [None, 1.032e-6, None, 1.039e-6, 3.04e-9, 0],
    [1.024e-6, 1.05e-6]],
    
    
    [[np.mean(spec), 1.0744e-6, np.mean(spec), 1.083e-6, np.mean(spec), 1.0938e-6, 3.04e-9, 0],
    [None, 1.0744e-6, None, 1.083e-6, None, 1.0938e-6, None, 0],
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

    [[np.mean(spec), 2.166e-6, 2.36e-9, 0],
    [None, None, 2.36e-9, None],
    [2.146e-6, 2.186e-6]],
    
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

    
#%%


import pyneb as pn


print('############')
print('# Nodule 1 #')
print('############')

params_1, covs_1 = fitter(lambdas_1, spec_for_fit_1*1e6, lines, plot=True)
lams, fls, sts = convert_to_fluxes(lines, params_1, covs_1)
lams, fls_1, sts = convert_to_fluxes(lines, params_1, covs_1)
for i in range(len(fls)):
    print(str(int(lams[i]*1e9))+' : '+str(fls[i])+' +/- '+str(sts[i])+'  ('+str(np.around(100*sts[i]/fls[i]))+'%)')


H1 = pn.RecAtom('H', 1)
pn.atomicData.getAllAvailableFiles('H1')

lams_obs = np.array([lams[7], lams[12], lams[15], lams[16], lams[17], lams[24]])
H_obs = np.array([fls[7], fls[12], fls[15], fls[16], fls[17], fls[24]])
st_obs = np.array([sts[7], sts[12], sts[15], sts[16], sts[17], sts[24]])
lams_obs = np.array([lams[7], lams[12], lams[15], lams[16], lams[24]])
H_obs = np.array([fls[7], fls[12], fls[15], fls[16], fls[24]])
st_obs = np.array([sts[7], sts[12], sts[15], sts[16], sts[24]])
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
print('A_k = '+str(p1))

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
lams, fls_2, sts = convert_to_fluxes(lines, params_2, covs_2)
for i in range(len(fls)):
    print(i, str(int(lams[i]*1e9))+' : '+str(fls[i])+' +/- '+str(sts[i])+'  ('+str(np.around(100*sts[i]/fls[i]))+'%)')

H1 = pn.RecAtom('H', 1)
pn.atomicData.getAllAvailableFiles('H1')

lams_obs = np.array([lams[7], lams[12], lams[15], lams[16], lams[17], lams[24]])
H_obs = np.array([fls[7], fls[12], fls[15], fls[16], fls[17], fls[24]])
st_obs = np.array([sts[7], sts[12], sts[15], sts[16], sts[17], sts[24]])
lams_obs = np.array([lams[7], lams[12], lams[15], lams[16], lams[24]])
H_obs = np.array([fls[7], fls[12], fls[15], fls[16], fls[24]])
st_obs = np.array([sts[7], sts[12], sts[15], sts[16], sts[24]])
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
print('A_k = '+str(p2))


extinction_2 = extinct(lambdas_2, p2[0])

#%%


lam = lambdas_1[~mask1b]
spec = spec_for_fit_1[~mask1b]*extinction_1[~mask1b]
spec_glob = spec.copy()

infos_cloudy, lines_cloudy = cp.parse_file('model.out', './MODEL_SEMI_COARSE_erg_NOT_TOTAL/', emergent=False, lam0=np.min(lam), lam1=np.max(lam))
infos_mappings, lines_mappings = mp.parse_directory('./MAPPINGS/emission_line_ratios/', lam0=np.min(lam), lam1=np.max(lam))

#%%

def spec_lines_comparator(spec, lines, infos_lines=None, progressbar=False, max_flux=1):
    range_lines = range(len(lines))
    if progressbar:
        range_lines = tqdm(range_lines)
    if infos_lines == None:
        infos_lines = ['No info']*len(lines)
    
    ps = []
    perrs = []
    errs = []
    k2s = []
    infos = []
    
    
    for li in range_lines:
        line = lines[li]
        info = infos_lines[li]
        spec_cloudy = cp.make_spec(line, lam, 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)
        fac_clo = 1e-7/pir2
        spec_cloudy *= fac_clo
        
        def rel_prop(_, a):
            if (a<0.05) and (a>0):
                spec_f = a*spec_cloudy
            else:
                spec_f = -abs(a)*spec
            return spec_f
        
        try:
            p, cov = curve_fit(rel_prop, lam, spec, p0 = [max_flux/2], maxfev=10000)
            err = False
            perr = np.sqrt(np.diag(cov))
        except:
            p = [max_flux/2]
            err = True
            perr = np.zeros(np.shape(p))

        spec_mod = rel_prop(lam, *p)
        diff = (spec_mod-spec)
        k2 = np.sum(diff**2)
        
        ps.append(p)
        perrs.append(perr)
        errs.append(err)
        k2s.append(k2)
        infos.append(info)
        
    return ps, perrs, errs, k2s, infos

def parse_infos(infos):
    lists = []
    for info in infos:
        info_splitted = info.split('\t')
        l = len(info_splitted[-1].split(','))
        mini_list = []
        for n in range(l):
            k = 2+n
            mini_list.append(float(info_splitted[-k]))
        lists.append(mini_list)
    return np.array(lists)

def get_k2_cube(k2s, infos):
    infos = parse_infos(infos)
    uniques = []
    shape = []
    ranges = []
    for n in range(len(infos[0])):
        uniques.append(np.unique(infos[:, n]))
        shape.append(len(uniques[-1]))
        ranges.append(range(len(uniques[-1])))
    mesh_infos = np.array(np.meshgrid(*uniques, indexing='ij'))
    mesh_indices = np.meshgrid(*ranges, indexing='ij')
    k2s_cube = np.zeros(np.shape(mesh_infos[0]))
    for indices in np.nditer(mesh_indices):
        info_cube = []
        for m in range(len(uniques)):
            info_cube.append(mesh_infos[m][indices])
        for k in range(len(k2s)):
            k2 = k2s[k]
            info = infos[k]
            good = True
            for j in range(len(info)):
                if info[j] != info_cube[j]:
                    good = False
            if good:
                k2s_cube[indices] = k2
    return k2s_cube, mesh_infos
    
#%%      
        

infos_cloudy_t = []
lines_cloudy_t = []

for n in range(len(infos_cloudy)):
    lines = lines_cloudy[n]
    infos = infos_cloudy[n]
    lines_t = []
    infos_t = []
    for line in lines:        
        if ('Fe' not in line[0]) and ('Fnu' not in line[0]):
            if 'Si' in line[0] and (abs(1-line[2]/1.960e-6) < 0.001):
                line[2] = 1.963e-6
            lines_t.append(line)
    lines_cloudy_t.append(lines_t)
    infos_cloudy_t.append(infos)
    
lines_cloudy = lines_cloudy_t
infos_cloudy = infos_cloudy_t



plt.figure(figsize=(10, 6))
plt.pause(0.01)
k2min = 1e100
resmin = k2min**0.5

infoss_cloudy = []
infoss_mappings = []
khi2ss = []
pss = []
css = []

for ct in tqdm(range(len(infos_cloudy))):
        c = ct
        line_cloudy = lines_cloudy[c]
        info_cloudy = infos_cloudy[c]
        if (len(line_cloudy)!=0) and (float(info_cloudy.replace('\t', ';').split(';')[-2]) < 2):# and (float(info_cloudy.replace('\t', ';').split(';')[-4]) < 6): 
            spec_cloudy = cp.make_spec(line_cloudy, lam, 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)
            fac_clo = 1e-7/pir2
            spec_cloudy *= fac_clo
            
            vol_cloud = 10/(100*pc)**2
            
            def rel_prop(_, a):
                if (a<0.05) and (a>0):
                    spec_f = a*spec_cloudy
                else:
                    spec_f = -abs(a)*spec
                return spec_f
            try:
                p, cov = curve_fit(rel_prop, lam, spec, p0 = [0.025], maxfev=10000)
            except:
                p = [0.025]
            spec_mod = rel_prop(lam, *p)
            diff = (spec_mod-spec)
            k2 = np.sum(diff**2)
            res = np.sum(abs(diff))
            
            khi2ss.append(k2)
            infoss_cloudy.append(info_cloudy)
            infoss_mappings.append(info_cloudy)
            pss.append(p)
            css.append(c)
            
            if k2 < k2min:
                
                print('')
                print('###################')
                print('#### JACKPOT ! ####')
                print('###################')
                print('')
                print('Old Chi 2 (%) : '+str(k2min)+' ('+str(resmin/np.sum(abs(spec)))+')')
                print('New Chi 2 (%) : '+str(k2)+' ('+str(res/np.sum(abs(spec)))+')')
                print('')
                print('Info cloudy : ', info_cloudy)
                print('')
                print('Relative proportions (C, M, offset) : ', p)
                print('')
                
                k2min = k2
                resmin = res
                best_spec_mod = spec_mod
                best_spec_cloudy = spec_cloudy
                best_p = p
                best_c = c
                best_infos_cloudy = info_cloudy
                best_line_cloudy = line_cloudy
                
                spec_mod_p = rel_prop(lambdas_1, *p)
                
                plt.clf()
                plt.scatter(lam, spec, label = 'Data', c='b', marker='o')
                plt.scatter(lam, spec_mod, label = 'Model', c='r', marker='.')
                plt.legend() 
                plt.xlabel('Wavelength (m)')
                plt.ylabel('Flux ($W.m^{-2}.\mu m^{-1}$ ? )')
                plt.pause(0.01)


#%%


lam = lambdas_2[~mask1b]
spec = spec_for_fit_2[~mask1b]*extinction_2[~mask1b]
spec_glob = spec.copy()
# spec = abs(spec)**0.5*np.sign(spec)

infos_cloudy, lines_cloudy = cp.parse_file('model.out', './MODEL_SEMI_COARSE_erg_NOT_TOTAL/', emergent=False, lam0=np.min(lam), lam1=np.max(lam))


infos_cloudy_t = []
lines_cloudy_t = []

for n in range(len(infos_cloudy)):
    lines = lines_cloudy[n]
    infos = infos_cloudy[n]
    lines_t = []
    infos_t = []
    for line in lines:        
        if 'Fe' not in line[0]:
            if 'Si' in line[0] and (abs(1-line[2]/1.960e-6) < 0.001):
                line[2] = 1.963e-6
            lines_t.append(line)
    lines_cloudy_t.append(lines_t)
    infos_cloudy_t.append(infos)
    
lines_cloudy = lines_cloudy_t
infos_cloudy = infos_cloudy_t


        

k2min = 1e100
ratio_max = 0
infoss_cloudy = []
khi2ss = []
sum_spec = np.sum(fls_2)

for kt in tqdm(range(len(infos_cloudy[::]))):
            k = int(kt)
            
            line_cloudy = lines_cloudy[k]
            info_cloudy = infos_cloudy[k]
            
            fac_map = 1e-7/pir2
            
            flux_lines = 0
            for l in line_cloudy:
                flux_lines += l[-2]
            flux_lines *= fac_map
            
            ratio = flux_lines/sum_spec
            
            if ratio > ratio_max:
                ratio_max = ratio
                print(ratio)
                print(info_cloudy)
            

#%%



# lam = lambdas_2[~mask1b]
# spec = spec_for_fit_2[~mask1b]*extinction_2[~mask1b]
# spec_glob = spec.copy()
# # spec = abs(spec)**0.5*np.sign(spec)

# # infos_cloudy, lines_cloudy = cp.parse_file('model.out', './MODEL_COARSE_erg/', emergent=False, lam0=np.min(lam), lam1=np.max(lam))

# infos_cloudy_t = []
# lines_cloudy_t = []

# for n in range(len(infos_cloudy)):
#     lines = lines_cloudy[n]
#     infos = infos_cloudy[n]
#     lines_t = []
#     infos_t = []
#     for line in lines:        
#         if 'Fe' not in line[0]:
#             if 'Si' in line[0] and (abs(1-line[2]/1.960e-6) < 0.001):
#                 line[2] = 1.963e-6
#             lines_t.append(line)
#     lines_cloudy_t.append(lines_t)
#     infos_cloudy_t.append(infos)
    
# lines_cloudy = lines_cloudy_t
# infos_cloudy = infos_cloudy_t


# infos_mappings_t = []
# lines_mappings_t = []

# for n in range(len(infos_mappings)):
#     lines = lines_mappings[n]
#     infos = infos_mappings[n]
#     lines_t = []
#     infos_t = []
#     for line in lines:        
#         if 'Fe' not in line[0]:
#             if 'Si' in line[0] and (abs(1-line[2]/1.960e-6) < 0.001):
#                 line[2] = 1.963e-6
#             if 'Fe' in line[0]:
#                 line[-2] /= 20
#             lines_t.append(line)
#     lines_mappings_t.append(lines_t)
#     infos_mappings_t.append(infos)
    
# lines_mappings = lines_mappings_t
# infos_mappings = infos_mappings_t
        

# plt.figure(figsize=(10, 6))
# plt.pause(0.01)
# k2min = 1e100
# resmin = k2min**0.5

# infoss_cloudy = []
# infoss_mappings = []
# khi2ss = []
# pss = []
# css = []

# sum_spec = np.sum(fls_2)

# for kt in tqdm(range(len(infos_mappings[:1:2]))):
#     for jt in range(len(infos_mappings[::2])):
#         if jt != kt:
#             k = int(2*kt)
#             j = int(2*jt)
            
#             line_mappings_1 = lines_mappings[k]
#             info_mappings_1 = infos_mappings[k]
#             line_mappings_2 = lines_mappings[j]
#             info_mappings_2 = infos_mappings[j]
            
            
#             fac_map = 4*np.pi*(100*70*pc)**2*1e-7/pir2
            
#             flux_lines_1 = 0
#             for l in line_mappings_1:
#                 flux_lines_1 += l[-2]
#             flux_lines_1 * fac_map
#             flux_lines_2 = 0
#             for l in line_mappings_2:
#                 flux_lines_2 += l[-2]
#             flux_lines_2 * fac_map
#             if(flux_lines_1>(0.125*sum_spec)) and (flux_lines_2>(0.125*sum_spec)):# and (flux_lines_1<(1e7*sum_spec)) and (flux_lines_2<(1e7*sum_spec)):
                    
#             # if flux_line > (0.125*1e-16):
    
#                 spec_mappings_1 = -1+0*mp.make_spec(line_mappings_1, lam, 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)*1e-6
#                 spec_mappings_2 = mp.make_spec(line_mappings_2, lam, 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)*1e-6
                
#                 spec_mappings_1 *= fac_map
#                 spec_mappings_2 *= fac_map
                
#                 # if np.sum(spec_mappings_1+spec_mappings_2) > 1e100*(0.25*np.sum(spec)):
                
#                 def rel_prop(_, a, b):
#                     spec = abs(a)*spec_mappings_1+abs(b)*spec_mappings_2
#                     return spec
#                 p0 = [np.sum(spec)/np.sum(spec_mappings_1), 2*np.sum(spec)/np.sum(spec_mappings_2)]
#                 try:
#                     p, c = curve_fit(rel_prop, lam, spec, p0 = p0, maxfev=1000000)
#                 except Exception as e:
#                     print(e)
#                     p = [1, 1]
#                 spec_mod = rel_prop(lam, *p)
#                 diff = (spec_mod-spec)
#                 k2 = np.sum(diff**2)
#                 res = np.sum(abs(diff))
                
#                 khi2ss.append(k2)
#                 infoss_cloudy.append(info_cloudy)
#                 infoss_mappings.append(info_cloudy)
#                 pss.append(p)
#                 css.append(c)
                
#                 if k2 < k2min:
#                     print('')
#                     print('###################')
#                     print('#### JACKPOT ! ####')
#                     print('###################')
#                     print('')
#                     print('Old Chi 2 (%) : '+str(k2min)+' ('+str(resmin/np.sum(abs(spec)))+')')
#                     print('New Chi 2 (%) : '+str(k2)+' ('+str(res/np.sum(abs(spec)))+')')
#                     print('')
#                     print('Info mappings 1 : ', info_mappings_1)
#                     print('Info mappings 2 : ', info_mappings_2)
#                     print('')
#                     print('Parameters (M1, M2): ', p)
#                     print('Parameters (M1, M2) : ', 4*np.pi*70**2*p, r'($pc^2$)')
                    
#                     spec_mod_1 = rel_prop(lam, p[0], 0)
#                     spec_mod_2 = rel_prop(lam, 0, p[1])
                    
#                     print('Relative proportions (M1, M2) : ', np.sum(spec_mod_1)/np.sum(spec), np.sum(spec_mod_2)/np.sum(spec))
                    
#                     pref = info_mappings_1[0].split('/')[-1].split('.tx')[0].split('_sp')[0]
                    
#                     a = pref.split('_n')[0]
#                     b = pref.split('_b')[-1].replace('_', '.')
#                     n = pref.split('_b')[0].split('_n')[1].replace('_', '.')
#                     pref = '_'.join([a, 'n'+n, 'b'+b])
#                     V = best_infos_mappings[-1]
                    
#                     cols_shock_1, cols_precursor_1 = mp.parse_results(pref, V)
#                     cols_shock_1 = float(cols_shock_1[2][1])
#                     cols_precursor_1 = float(cols_precursor_1[2][1])+float(cols_precursor_1[1][1])
                    
                    
#                     pref = info_mappings_2[0].split('/')[-1].split('.tx')[0].split('_sp')[0]
                    
#                     a = pref.split('_n')[0]
#                     b = pref.split('_b')[-1].replace('_', '.')
#                     n = pref.split('_b')[0].split('_n')[1].replace('_', '.')
#                     pref = '_'.join([a, 'n'+n, 'b'+b])
#                     V = best_infos_mappings[-1]
                    
#                     cols_shock_2, cols_precursor_2 = mp.parse_results(pref, V)
#                     cols_shock_2 = float(cols_shock_2[2][1])
#                     cols_precursor_2 = float(cols_precursor_2[2][1])+float(cols_precursor_2[1][1])
                    
#                     print('Shock masses (M1, M2) : ', m_proton/m_sun*(100*pc)**2*4*np.pi*70**2*p[0]*cols_shock_1, m_proton/m_sun*(100*pc)**2*4*np.pi*70**2*p[1]*cols_shock_2)
#                     print(r'Precursor masses (M1, M2) ($M_{sun}$) : ', m_proton/m_sun*(100*pc)**2*4*np.pi*70**2*p[0]*cols_precursor_1, m_proton/m_sun*(100*pc)**2*4*np.pi*70**2*p[1]*cols_precursor_2)
#                     print('')
                    
#                     k2min = k2
#                     resmin = res
#                     best_spec_mod = spec_mod
#                     plt.clf()
#                     plt.scatter(lam, spec, label = 'Data', c='b', marker='o')
#                     plt.scatter(lam, spec_mod, label = 'Model', c='r', marker='.')
#                     plt.legend() 
#                     plt.xlabel('Wavelength (m)')
#                     plt.ylabel('Flux ($W.m^{-2}.\mu m^{-1}$ ? )')
#                     plt.pause(0.01)
 
#%%


# lam = lambdas_2[~mask1b]
# spec = spec_for_fit_2[~mask1b]*extinction_2[~mask1b]
# spec_glob = spec.copy()
# # spec = abs(spec)**0.5*np.sign(spec)

# # infos_cloudy, lines_cloudy = cp.parse_file('model.out', './MODEL_COARSE_erg/', emergent=False, lam0=np.min(lam), lam1=np.max(lam))

# infos_mappings, lines_mappings  = mp.parse_directory('./MAPPINGS/emission_line_ratios/', lam0=np.min(lam), lam1=np.max(lam))


# infos_cloudy_t = []
# lines_cloudy_t = []

# for n in range(len(infos_cloudy)):
#     lines = lines_cloudy[n]
#     infos = infos_cloudy[n]
#     lines_t = []
#     infos_t = []
#     for line in lines:        
#         if 'Fe' not in line[0]:
#             if 'Si' in line[0] and (abs(1-line[2]/1.960e-6) < 0.001):
#                 line[2] = 1.963e-6
#             lines_t.append(line)
#     lines_cloudy_t.append(lines_t)
#     infos_cloudy_t.append(infos)
    
# lines_cloudy = lines_cloudy_t
# infos_cloudy = infos_cloudy_t


# infos_mappings_t = []
# lines_mappings_t = []

# for n in range(len(infos_mappings)):
#     lines = lines_mappings[n]
#     infos = infos_mappings[n]
#     lines_t = []
#     infos_t = []
#     for line in lines:        
#         if 'Fe' not in line[0]:
#             if 'Si' in line[0] and (abs(1-line[2]/1.960e-6) < 0.001):
#                 line[2] = 1.963e-6
#             if 'Fe' in line[0]:
#                 line[-2] /= 20
#             lines_t.append(line)
#     lines_mappings_t.append(lines_t)
#     infos_mappings_t.append(infos)
    
# lines_mappings = lines_mappings_t
# infos_mappings = infos_mappings_t
        

# plt.figure(figsize=(10, 6))
# plt.pause(0.01)
# k2min = 1e100
# resmin = k2min**0.5

# infoss_cloudy = []
# infoss_mappings = []
# khi2ss = []
# pss = []
# css = []

# sum_spec = np.sum(fls_2)

# for kt in tqdm(range(len(infos_mappings[:1:2]))):
#     for jt in range(len(infos_mappings[::2])):
#         if jt != kt:
#             k = int(2*kt)
#             j = int(2*jt)
            
#             line_mappings_1 = lines_mappings[k]
#             info_mappings_1 = infos_mappings[k]
#             line_mappings_2 = lines_mappings[j]
#             info_mappings_2 = infos_mappings[j]
            
            
#             fac_map = 4*np.pi*(100*70*pc)**2*1e-7/pir2
            
#             flux_lines_1 = 0
#             for l in line_mappings_1:
#                 flux_lines_1 += l[-2]
#             flux_lines_1 * fac_map
#             flux_lines_2 = 0
#             for l in line_mappings_2:
#                 flux_lines_2 += l[-2]
#             flux_lines_2 * fac_map
#             if(flux_lines_1>(0.125*sum_spec)) and (flux_lines_2>(0.125*sum_spec)):# and (flux_lines_1<(1e7*sum_spec)) and (flux_lines_2<(1e7*sum_spec)):
                    
#             # if flux_line > (0.125*1e-16):
    
#                 spec_mappings_1 = -1+0*mp.make_spec(line_mappings_1, lam, 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)*1e-6
#                 spec_mappings_2 = mp.make_spec(line_mappings_2, lam, 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)*1e-6
                
#                 spec_mappings_1 *= fac_map
#                 spec_mappings_2 *= fac_map
                
#                 # if np.sum(spec_mappings_1+spec_mappings_2) > 1e100*(0.25*np.sum(spec)):
                
#                 def rel_prop(_, a, b):
#                     spec = abs(a)*spec_mappings_1+abs(b)*spec_mappings_2
#                     return spec
#                 p0 = [np.sum(spec)/np.sum(spec_mappings_1), 2*np.sum(spec)/np.sum(spec_mappings_2)]
#                 try:
#                     p, c = curve_fit(rel_prop, lam, spec, p0 = p0, maxfev=1000000)
#                 except Exception as e:
#                     print(e)
#                     p = [1, 1]
#                 spec_mod = rel_prop(lam, *p)
#                 diff = (spec_mod-spec)
#                 k2 = np.sum(diff**2)
#                 res = np.sum(abs(diff))
                
#                 khi2ss.append(k2)
#                 infoss_cloudy.append(info_cloudy)
#                 infoss_mappings.append(info_cloudy)
#                 pss.append(p)
#                 css.append(c)
                
#                 if k2 < k2min:
#                     print('')
#                     print('###################')
#                     print('#### JACKPOT ! ####')
#                     print('###################')
#                     print('')
#                     print('Old Chi 2 (%) : '+str(k2min)+' ('+str(resmin/np.sum(abs(spec)))+')')
#                     print('New Chi 2 (%) : '+str(k2)+' ('+str(res/np.sum(abs(spec)))+')')
#                     print('')
#                     print('Info mappings 1 : ', info_mappings_1)
#                     print('Info mappings 2 : ', info_mappings_2)
#                     print('')
#                     print('Parameters (M1, M2): ', p)
#                     print('Parameters (M1, M2) : ', 4*np.pi*70**2*p, r'($pc^2$)')
                    
#                     spec_mod_1 = rel_prop(lam, p[0], 0)
#                     spec_mod_2 = rel_prop(lam, 0, p[1])
                    
#                     print('Relative proportions (M1, M2) : ', np.sum(spec_mod_1)/np.sum(spec), np.sum(spec_mod_2)/np.sum(spec))
                    
#                     pref = info_mappings_1[0].split('/')[-1].split('.tx')[0].split('_sp')[0]
                    
#                     a = pref.split('_n')[0]
#                     b = pref.split('_b')[-1].replace('_', '.')
#                     n = pref.split('_b')[0].split('_n')[1].replace('_', '.')
#                     pref = '_'.join([a, 'n'+n, 'b'+b])
#                     V = best_infos_mappings[-1]
                    
#                     cols_shock_1, cols_precursor_1 = mp.parse_results(pref, V)
#                     cols_shock_1 = float(cols_shock_1[2][1])
#                     cols_precursor_1 = float(cols_precursor_1[2][1])+float(cols_precursor_1[1][1])
                    
                    
#                     pref = info_mappings_2[0].split('/')[-1].split('.tx')[0].split('_sp')[0]
                    
#                     a = pref.split('_n')[0]
#                     b = pref.split('_b')[-1].replace('_', '.')
#                     n = pref.split('_b')[0].split('_n')[1].replace('_', '.')
#                     pref = '_'.join([a, 'n'+n, 'b'+b])
#                     V = best_infos_mappings[-1]
                    
#                     cols_shock_2, cols_precursor_2 = mp.parse_results(pref, V)
#                     cols_shock_2 = float(cols_shock_2[2][1])
#                     cols_precursor_2 = float(cols_precursor_2[2][1])+float(cols_precursor_2[1][1])
                    
#                     print('Shock masses (M1, M2) : ', m_proton/m_sun*(100*pc)**2*4*np.pi*70**2*p[0]*cols_shock_1, m_proton/m_sun*(100*pc)**2*4*np.pi*70**2*p[1]*cols_shock_2)
#                     print(r'Precursor masses (M1, M2) ($M_{sun}$) : ', m_proton/m_sun*(100*pc)**2*4*np.pi*70**2*p[0]*cols_precursor_1, m_proton/m_sun*(100*pc)**2*4*np.pi*70**2*p[1]*cols_precursor_2)
#                     print('')
                    
#                     k2min = k2
#                     resmin = res
#                     best_spec_mod = spec_mod
#                     plt.clf()
#                     plt.scatter(lam, spec, label = 'Data', c='b', marker='o')
#                     plt.scatter(lam, spec_mod, label = 'Model', c='r', marker='.')
#                     plt.legend() 
#                     plt.xlabel('Wavelength (m)')
#                     plt.ylabel('Flux ($W.m^{-2}.\mu m^{-1}$ ? )')
#                     plt.pause(0.01)
 
# #%%

# dist = 14.4*1e6*pc
# pir2 = 4*np.pi*dist**2

# line_cloudy_p = lines_cloudy[best_j]
# info_cloudy_p = infos_cloudy[best_j]
# line_mappings_p = lines_mappings[best_k]
# info_mappings_p = infos_mappings[best_k]
# spec_cloudy_p = cp.make_spec(line_cloudy_p, lambdas_1[~mask1b], 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)*1e-7/pir2
# spec_mappings_p = cp.make_spec(line_mappings_p, lambdas_1[~mask1b], 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)*1e-7/pir2

# def abs_prop(_, a, b, c):
#     spec = abs(a)*spec_cloudy_p+abs(b)*spec_mappings_p+c
#     return spec

# maxi = np.max(spec_for_fit_2[~mask1b])
# p, c = curve_fit(abs_prop, lambdas_2[~mask1b], spec_for_fit_2[~mask1b], p0 = [maxi/np.max(spec_cloudy_p), maxi/np.max(spec_mappings_p), 0])

# inf = best_infos_cloudy.replace('\t', ';').split(';')

# surfacic_mass_cloudy = 1  # kg.cm-2

# S_cloudy = p[0]*pir2


# N_cloudy = p[0]*pir2*10**-3  #

# # M_obs_cloudy = N_cloudy*M_one_cloudy

# pref = best_infos_mappings[0].split('/')[-1].split('.tx')[0].split('_sp')[0]

# a = pref.split('_n')[0]
# b = pref.split('_b')[-1].replace('_', '.')
# n = pref.split('_b')[0].split('_n')[1].replace('_', '.')
# pref = '_'.join([a, 'n'+n, 'b'+b])
# V = best_infos_mappings[-1]

# cols_shock, cols_precursor = mp.parse_results(pref, V)


# nh_shock = p[1]
# mh_shock_prec = p[1]*float(cols_shock[1][1])*m_proton/m_sun
# mh_shock_shock = p[1]*float(cols_shock[1][2])*m_proton/m_sun


# p[0]*1e-3
#%%                     

plt.figure(figsize=(10, 6))
plt.pause(0.01)
k2min = 1e100
resmin = k2min**0.5
for k in tqdm(range(len(infos_cloudy))):
    for j in range(len(infos_cloudy)):
        # k = int(6*kt)
        line_cloudy = lines_cloudy[j]
        info_cloudy = infos_cloudy[j]
        if (len(line_cloudy)!=0) and (len(line_cloudy)!=0): 
            line_mappings = lines_cloudy[k]
            info_mappings = infos_cloudy[k]
            spec_cloudy = cp.make_spec(line_cloudy, lam, 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)
            if np.sum(spec_cloudy) != 0:
                fac_clo = np.sum(spec)/2/np.sum(spec_cloudy)
            else:
                spec_cloudy[0] = 1
                fac_clo = np.sum(spec)/2/np.sum(spec_cloudy)
            spec_cloudy *= fac_clo
            spec_mappings = cp.make_spec(line_mappings, lam, 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)
            if np.sum(spec_mappings) != 0:
                fac_map = np.sum(spec)/2/np.sum(spec_mappings)
            else:
                spec_mappings[0] = 1
                fac_map = np.sum(spec)/2/np.sum(spec_mappings)
            spec_mappings *= fac_map
            
            def rel_prop(_, a, b, c):
                return abs(a)*spec_cloudy+abs(b)*spec_mappings+c
            
            try:
                p, c = curve_fit(rel_prop, lam, spec, p0 = [1, 1, np.median(spec)], maxfev=10000)
            except:
                p = [1, 1, np.median(spec)]
            spec_mod = rel_prop(lam, *p)
            diff = (spec_mod-spec)
            # diff[390:420] *= 5
            # diff[1050:1085] *= 5
            k2 = np.sum(diff**2)
            res = np.sum(abs(diff))
            
            if k2 < k2min:
                
                print('')
                print('###################')
                print('#### JACKPOT ! ####')
                print('###################')
                print('')
                print('Old Chi 2 (%) : '+str(k2min)+' ('+str(resmin/np.sum(abs(spec)))+')')
                print('New Chi 2 (%) : '+str(k2)+' ('+str(res/np.sum(abs(spec)))+')')
                print('')
                print('Info cloudy : ', info_cloudy)
                print('Info mappings : ', info_mappings)
                print('')
                print('Relative proportions (C, M, offset) : ', p)
                print('')
                
                k2min = k2
                resmin = res
                best_spec_mod = spec_mod
                best_spec_cloudy = spec_cloudy
                best_spec_mappings = spec_mappings
                best_p = p
                best_j = j
                best_k = k
                best_infos_cloudy = info_cloudy
                best_infos_mappings = info_mappings
                
                
                line_cloudy_p = lines_cloudy[j]
                info_cloudy_p = infos_cloudy[j]
                line_mappings_p = lines_mappings[k]
                info_mappings_p = infos_mappings[k]
                spec_cloudy_p = cp.make_spec(line_cloudy_p, lambdas_1, 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)
                spec_cloudy_p *= fac_clo
                spec_mappings_p = cp.make_spec(line_mappings_p, lambdas_1, 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)
                spec_mappings_p *= fac_map
                
                def rel_prop(_, a, b, c):
                    return abs(a)*spec_cloudy_p+abs(b)*spec_mappings_p+c
                
                
                spec_mod_p = rel_prop(lambdas_1, *p)
                
                plt.clf()
                plt.plot(lambdas_1, spec1, c='b', alpha=0.5)
                # plt.plot(lambdas_1, spec_mod_p, c='r', alpha=0.5)  
                plt.scatter(lam, spec, label = 'Data', c='b', marker='.')
                plt.scatter(lam, best_spec_mod, label = 'Model', c='r', marker='.')
                plt.legend() 
                plt.xlabel('Wavelength (m)')
                plt.ylabel('Flux ($W.m^{-2}.\mu m^{-1}$ ? )')
                plt.pause(0.01)
                
            
        