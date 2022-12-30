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

import pickle
pc = float(pc.value)
m_proton = float(m_p.value)
m_sun = float(M_sun.value)

dist = 14.4*1e6*pc
pir2 = 4*np.pi*dist**2

#%%




#%%
from tqdm import tqdm

lam = np.loadtxt('/home/pierre/Documents/JWST/lam.txt')*1e-6*0.984
spec = np.loadtxt("/home/pierre/Documents/JWST/spec.txt")


#%%



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
            # if 'Fe' not in line[0]:
                # if 'Si' in line[0] and (abs(1-line[2]/1.960e-6) < 0.002):
            # line[2] = 1.963e-6
            lines_t.append(line)
        clean_lines_t.append(lines_t)
        clean_infos_t.append(infos)
            
    clean_lines = clean_lines_t
    clean_infos = clean_infos_t
    
    return clean_infos, clean_lines

infos_cloudy, lines_cloudy = cp.parse_file('model.out', './MODEL_CONSTANT_TEMP_HD/', emergent=False, lam0=np.min(lam), lam1=np.max(lam))
# infos_cloudy, lines_cloudy = cp.parse_file('model.out', './MODEL_SEMI_COARSE_erg_NOT_TOTAL/', emergent=True, lam0=np.min(lam), lam1=np.max(lam))
# infos_cloudy, lines_cloudy = lines_modificiations(infos_cloudy, lines_cloudy)
infos_mappings, lines_mappings = mp.parse_directory('./MAPPINGS/emission_line_ratios/', lam0=np.min(lam), lam1=np.max(lam))
# infos_mappings, lines_mappings = lines_modificiations(infos_mappings, lines_mappings)



#%%

specs_cloudy = []
for line_cloudy in lines_cloudy:
    spec_cloudy = cp.make_spec(line_cloudy, lam, 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)*4*np.pi*(10**20)**2
    specs_cloudy.append(spec_cloudy*1e-7/pir2)

specs_mappings = []
for line_mappings in lines_mappings:
    spec_mappings = mp.make_spec(line_mappings, lam, 2.72e-9, 1.48e-9, lambda_break = 1.78969e-6)
    specs_mappings.append(spec_mappings*4*np.pi*(1e20)**2*1e-7/pir2)
    
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
# uccloudy_k2s, uccloudy1_ps = compare_spec_1d(spec, spec*0+1, specs_cloudy, range_limit = [0, 1e15])

# #%%
# zs = np.arange(0.98, 0.99, 0.0001)
# bzs = []

# for spec_c in tqdm(specs_cloudy):
#     spec_c = np.array(spec_c, dtype=float)
#     k2 = 1e100
#     for z in zs:
#         lam_t = lam*z
#         spec_interp = np.interp(lam_t, lam, spec_c)
#         spec_interp *= np.max(spec)/np.max(spec_interp)
#         k2s = np.sum(abs(spec_interp-spec))
#         if k2s < k2:
#             k2 = k2s
#             bz = z
#     bzs.append(bz)
        

# #%%
# lam0 = 14.1e-6
# lam1 = 14.4e-6

# for lines in lines_cloudy:
#     for line in lines:
#         lam = line[2]
#         if lam0 < lam < lam1:
#             print(line)

pickle.dump(lines_cloudy, open('/home/pierre/Documents/JWST/lines_cloudy_consT2', 'wb'))
pickle.dump(lines_mappings, open('/home/pierre/Documents/JWST/lines_mappings_consT2', 'wb'))
pickle.dump(infos_cloudy, open('/home/pierre/Documents/JWST/infos_cloudy_consT2', 'wb'))
pickle.dump(infos_mappings, open('/home/pierre/Documents/JWST/infos_mappings_consT2', 'wb'))

"""
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
"""