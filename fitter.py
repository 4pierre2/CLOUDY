#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 23:07:56 2022

@author: pierre
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def flux_to_amp(flux, std):
    return flux/(np.pi*2*std**2)**0.5

def amp_to_flux(amp, std):
    return amp*(np.pi*2*std**2)**0.5

def gauss(x, a, x0, std, c):
    return a*np.exp(-(x-x0)**2/(2*std**2))+c

def double_gauss(x, a0, x0, a1, x1, std, c):
    return gauss(x, a0, x0, std, 0) + gauss(x, a1, x1, std, c)

def triple_gauss(x, a0, x0, a1, x1, a2, x2, std, c):
    return gauss(x, a0, x0, std, 0) + gauss(x, a1, x1, std, 0) + gauss(x, a2, x2, std, c)

def gauss_flux(x, flux, x0, std, c):
    a = flux_to_amp(flux, std)
    return gauss(x, a, x0, std)+c

def double_gauss_flux(x, flux0, x0, flux1, x1, std, c):
    return gauss_flux(x, flux0, x0, std, 0) + gauss_flux(x, flux1, x1, std, c)

def triple_gauss_flux(x, flux0, x0, flux1, x1, flux2, x2, std, c):
    return gauss_flux(x, flux0, x0, std, 0) + gauss_flux(x, flux1, x1, std, 0) + gauss_flux(x, flux2, x2, std, c)

#%%


#%%



#%%



def make_func(p0, c0):
    if len(p0) == 4:
        def func(lam, *args):
            params = []
            n = 0
            for x in range(len(c0)):
                c = c0[x]
                if c is not None:
                    params.append(c)
                else:
                    params.append(args[n])
                    n += 1
            return gauss(lam, *params)
        
    elif len(p0) == 6:
        def func(lam, *args):
            params = []
            n = 0
            for x in range(len(c0)):
                c = c0[x]
                if c is not None:
                    params.append(c)
                else:
                    params.append(args[n])
                    n += 1
            return double_gauss(lam, *params)
        
    elif len(p0) == 8:
        def func(lam, *args):
            params = []
            n = 0
            for x in range(len(c0)):
                c = c0[x]
                if c is not None:
                    params.append(c)
                else:
                    params.append(args[n])
                    n += 1
            return triple_gauss(lam, *params)
        
    return func

def fitter(lambdas, spec, lines, plot=False):
    
    if plot:
        plt.figure()
        plt.plot(lambdas, spec, c='b')
    
    params = []
    covs = []
    
    for line in lines:
        p0 = line[0]
        c0 = line[1]
        l0, l1 = line[2]
        
        x0 = np.argmin(abs(lambdas-l0))
        x1 = np.argmin(abs(lambdas-l1))
        
        l = lambdas[x0:x1]
        sp = spec[x0:x1]
        
        p0s = []
        for j in range(len(c0)):
            if c0[j] == None:
                p0s.append(p0[j])


        func = make_func(p0, c0)
        
        # p0s = p0[c0==None]
        # print(p0s)
        param, cov = curve_fit(func, l, sp, p0=p0s, maxfev=1000000)
        params.append(param)
        covs.append(cov)
        if plot:
            plt.plot(l, func(l, *param), c='r')
        
    return params, covs

#%%

def ampstd_to_flux(amp, std):
    alpha = 1/(2*std**2)
    f = amp*(np.pi/alpha)**0.5
    return f

def ampstd_to_std(amp, std, damp, dstd):
    df = (2*np.pi)**0.5*((damp*std)**2+(amp*dstd)**2)**0.5
    return df

def get_flux_unc(line, param, cov):
    stds = np.sqrt(np.diag(cov))
    p0 = line[0]
    c0 = line[1]
    
    nn = 0
    full_params = []
    full_stds = []
    lams = []
    for cx in range(len(c0)):
        if c0[cx] != None:
            full_stds.append(0)
            full_params.append(c0[cx])
        else:
            full_params.append(param[nn])
            full_stds.append(stds[nn])
            nn += 1
    if len(line[0])==4:
        amp = full_params[0]
        damp = full_stds[0]
        std = full_params[2]
        dstd = full_stds[2]
        amps = [amp]
        damps = [damp]
        stds = [std]
        dstds = [dstd]
        lams = [full_params[1]]
    elif len(line[0])==6:
        amp1 = full_params[0]
        damp1 = full_stds[0]
        amp2 = full_params[2]
        damp2 = full_stds[2]
        std = full_params[-2]
        dstd = full_stds[-2]
        amps = [amp1, amp2]
        damps = [damp1, damp2]
        stds = [std, std]
        dstds = [dstd, dstd]   
        lams = [full_params[1], full_params[3]]
    elif len(line[0])==8:
        amp1 = full_params[0]
        damp1 = full_stds[0]
        amp2 = full_params[2]
        damp2 = full_stds[2]
        amp3 = full_params[4]
        damp3 = full_stds[4]
        std = full_params[-2]
        dstd = full_stds[-2]
        amps = [amp1, amp2, amp3]
        damps = [damp1, damp2, damp3]
        stds = [std, std, std]
        dstds = [dstd, dstd, dstd]
        lams = [full_params[1], full_params[3], full_params[5]]
    else:
        print('Oups')
        print(line)
        
    flux_results = []
    std_results = []
    lam_results = []
    for m in range(len(amps)):
        fl = ampstd_to_flux(amps[m], stds[m])
        st = ampstd_to_std(amps[m], stds[m], damps[m], dstds[m])
        flux_results.append(fl)
        std_results.append(st)
        lam_results.append(lams[m])
        
    return lam_results, flux_results, std_results

def convert_to_fluxes(lines, params, covs):
    fluxes = []
    stds = []
    lams = []
    for k in range(len(lines)):
        lam, flux, std = get_flux_unc(lines[k], params[k], covs[k])
        for z in range(len(flux)):
            lams.append(lam[z])
            fluxes.append(flux[z])
            stds.append(std[z])
    return lams, fluxes, stds
        
#%%
  
# lam = lambdas_2[mask1]
# spec = spec_for_fit_2[mask1]
      
           
# lines = [
#     [[np.mean(spec), 9.9e-7, 3.04e-9, 0],
#     [None, None, None,   0],
#     [9.81e-7, 9.98e-7]],
    
#     [[np.mean(spec), 1.00494e-6, np.mean(spec), 1.01237e-6, 3.04e-9, 0],
#     [None, 1.00494e-6, None, 1.01237e-6, None, 0],
#     [9.95e-7, 10.22e-7]],
    
#     [[np.mean(spec), 1.032e-6, np.mean(spec), 1.039e-6, 3.04e-9, 0],
#     [None, 1.032e-6, None, 1.039e-6, 3.04e-9, 0],
#     [1.024e-6, 1.05e-6]],
    
    
#     [[np.mean(spec), 1.0744e-6, np.mean(spec), 1.083e-6, np.mean(spec), 1.0938e-6, 3.04e-9, 0],
#     [None, 1.0744e-6, None, 1.083e-6, None, 1.0938e-6, None, 0],
#     [1.06e-6, 1.11e-6]],
    
#     [[np.mean(spec), 1.146e-6, np.mean(spec), 1.163e-6, np.mean(spec), 1.189e-6, 3.04e-9, 0],
#     [None, 1.146e-6, None, 1.163e-6, None, 1.189e-6, None, None],
#     [1.135e-6, 1.235e-6]],
    
#     [[np.mean(spec), 1.252e-6, np.mean(spec), 1.281e-6, 3.04e-9, 0],
#     [None, None, None, None, None, None],
#     [1.21e-6, 1.33e-6]],
    
#     [[np.mean(spec), 1.43e-6, 3.04e-9, 0],
#     [None, None, None, None],
#     [1.4e-6, 1.46e-6]],
    
#     [[np.mean(spec), 1.64e-6, np.mean(spec), 1.68e-6, np.mean(spec), 1.74e-6, 3.04e-9, 0],
#     [None, None, None, None, None, None, 3.04e-9, None],
#     [1.6e-6, 1.77e-6]],

#     [[np.mean(spec), 1.875e-6, np.mean(spec), 1.963e-6, 3.04e-9, 0],
#     [None, None, None, None, None, 0],
#     [1.75e-6, 2.0e-6]],
    
#     [[np.mean(spec), 2.033e-6, np.mean(spec), 2.045e-6, np.mean(spec), 2.0582e-6, 2.36e-9, 0],
#     [None, None, None, None, None, None, 2.36e-9, None],
#     [2.02e-6, 2.07e-6]],

#     [[np.mean(spec), 2.166e-6, 2.36e-9, 0],
#     [None, None, 2.36e-9, None],
#     [2.146e-6, 2.186e-6]],
    
#     [[np.mean(spec), 2.12e-6, 2.36e-9, 0],
#     [None, None, None, None],
#     [2.117e-6, 2.14e-6]],
    
#     [[np.mean(spec), 2.166e-6, 2.36e-9, 0],
#     [None, None, 2.36e-9, None],
#     [2.146e-6, 2.186e-6]],
    
#     [[np.mean(spec), 2.32e-6, 2.36e-9, 0],
#     [None, None, None, None],
#     [2.28e-6, 2.34e-6]],

#     [[np.mean(spec), 2.406e-6, np.mean(spec), 2.423e-6, 2.36e-9, 0],
#     [None, None, None, None, 2.36e-9, 0],
#     [2.38e-6, 2.43e-6]]
    
#     ]

# params_1, covs_1 = fitter(lambdas_1, spec_for_fit_1, lines, plot=True)


# lams, fls, sts = convert_to_fluxes(lines, params_1, covs_1)
# for i in range(len(fls)):
#     print(str(int(lams[i]*1e9))+' : '+str(fls[i])+' +/- '+str(sts[i])+'  ('+str(np.around(3*100*sts[i]/fls[i]))+'%)')
    
# print('   ')
# print('   ')
# print('   ')
# params_2, covs_2 = fitter(lambdas_2, spec_for_fit_2, lines, plot=True)

# lams, fls, sts = convert_to_fluxes(lines, params_2, covs_2)
# for i in range(len(fls)):
#     print(str(int(lams[i]*1e9))+' : '+str(fls[i])+' +/- '+str(sts[i])+'  ('+str(np.around(3*100*sts[i]/fls[i]))+'%)')
    
    
#%%


    