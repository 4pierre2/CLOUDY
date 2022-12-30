#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 17:15:21 2021

@author: pierre
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import rotate


# K band
F_0 = 4.283e-14*1e4# *1e4 pour passer de W/cm2/um à W/m2/um
bwidth = 0.261
filter_name = './2MASS_filters/K.txt'
filt = np.loadtxt(filter_name).T


# H+K

# NGC 1068 0.1
mag = 8.9
rad_pix=5
coadd_name_100 = './SINFO_REDUCED_DATA/NGC1068-100mas-HK-1_COADD_OBJ_53649.24330859.fits'

spec_star_filename = './SINFO_REDUCED_DATA/Telluric_Standard_STD_STAR_SPECTRA.fits'
typ = 'g2v'


def get_obj(filename):
    hdu = fits.open(filename)
    obj = hdu[0].data
    return np.nan_to_num(obj)
    

def get_info(filename):
    hdu = fits.open(filename)
    prog_id = hdu[0].header['HIERARCH ESO OBS PROG ID'].replace('.','').replace('-','').replace('(','').replace(')','')
    filt = hdu[0].header['HIERARCH ESO INS GRAT1 NAME'].replace(' ','')
    pix_scale = hdu[0].header['HIERARCH ESO INS OPTI1 NAME'].replace(' ','')
    date = hdu[0].header['MJD-OBS']
    return prog_id, date, filt, pix_scale

def get_obj_info(filename):
    hdu = fits.open(filename)
    prog_id = hdu[0].header['HIERARCH ESO OBS PROG ID'].replace('.','').replace('-','').replace('(','').replace(')','')
    obj_name = hdu[0].header['HIERARCH ESO OBS TARG NAME']
    date = hdu[0].header['MJD-OBS']
    return obj_name, date

def load_th_spec(typ, rep='./pickles_atlas/'):
    full = np.loadtxt(rep+'/uk'+typ+'.dat')
    return full[:,0], full[:,1], full[:,2]

#%%

# find_std_stars(coadd_1_name)


spec_star_wl = fits.open(spec_star_filename)[1].data['wavelength']
spec_star_spectrum = fits.open(spec_star_filename)[1].data['counts_bkg']
spec_star_bg = fits.open(spec_star_filename)[1].data['bkg_tot']
spec_star_tot = fits.open(spec_star_filename)[1].data['counts_tot']


w, f, e = load_th_spec(typ, rep = './pickles_atlas')
spec_th = np.interp(spec_star_wl, w/1e4, f)

transmi = np.nan_to_num(spec_star_spectrum/spec_th)
transmi /= np.median(transmi)
transmi[transmi<5e-2]=5e-2


obj = get_obj(coadd_name_100)
ymax, xmax = np.unravel_index(np.argmax(np.median(np.nan_to_num(obj),0)), np.shape(obj[0]))
obj_detransmitted = obj/transmi[:, np.newaxis, np.newaxis]

filter_interp = np.interp(spec_star_wl, filt[0], filt[1])
obj_2MASSed = obj_detransmitted*filter_interp[:, np.newaxis, np.newaxis]

x = np.arange(len(obj_2MASSed[0,0]))
y = np.arange(len(obj_2MASSed[0]))
xx, yy = np.meshgrid(x,y)



mask = ((xx-xmax)**2+(yy-ymax)**2)**0.5<rad_pix
tot_adu = np.sum(np.sum(obj_2MASSed,0)*mask)
tot_flux = F_0*10**(-0.4*mag)*bwidth
adu = tot_flux/tot_adu
dwl = spec_star_wl[2]-spec_star_wl[1]
obj_final = obj_detransmitted*adu

#%%
import sys
# sys.path.append('/home/pierre/Documents/2021/NEW_SPHERE')

from spec_obj import Spec_Obj
from copy import deepcopy
from scipy.ndimage import shift, rotate


pos = np.loadtxt('./SPHERE_REDUCED_DATA/pos')
lam = np.loadtxt('./SPHERE_REDUCED_DATA/lam')
obj = np.loadtxt('./SPHERE_REDUCED_DATA/obj')
obj_std = np.loadtxt('./SPHERE_REDUCED_DATA/obj_std')
cal = np.loadtxt('./SPHERE_REDUCED_DATA/cal')
cal_std = np.loadtxt('./SPHERE_REDUCED_DATA/cal_std')

sphobj = Spec_Obj(obj, obj_std, lam, pos)

sinforate = rotate(obj_final, 12, axes=(2, 1), reshape=False,order=3)
ymax, xmax = np.unravel_index(np.argmax(np.median(np.nan_to_num(sinforate),0)), np.shape(sinforate[0]))

obj_for_slit = deepcopy(sinforate)
for k in range(len(obj_for_slit)):
    im = sinforate[k]
    obj_for_slit[k] = im
    


slit = np.sum(obj_for_slit[90:2078,:,48:50],2)*90/100
err = np.ones(np.shape(slit))*0.05*slit

xm = np.argmax(np.mean(slit,0))-1
pos = (np.arange(np.shape(slit)[1])-xm)*0.0125
pos = (np.arange(np.shape(slit)[1])-xm)*0.05
wl = spec_star_wl[90:2078]*1e3

sinfobj = Spec_Obj(slit, err, wl, pos)


plt.figure()
sinfobj.plot_spec(0.2,0.4)
lsinf, nod_sinf_1, nod_sinf_1_err = sinfobj.make_spec(0.2,0.4)
sphobj.plot_spec(0.2,0.4)
lsph, nod_sph_1, nod_sph_1_err = sphobj.make_spec(0.2,0.4)

lj = lsph[:np.argmin((lsinf[0]-lsph)**2)]
jnod1 = nod_sph_1[:np.argmin((lsinf[0]-lsph)**2)]
jerr1 = nod_sph_1_err[:np.argmin((lsinf[0]-lsph)**2)]

lh = lsph[np.argmin((lsinf[0]-lsph)**2)+1:]
lh_sinf = lsinf[:np.argmin((lsinf-lsph[-1])**2)]
hnod1_sph = nod_sph_1[np.argmin((lsinf[0]-lsph)**2)+1:]
hnod1_sinf = nod_sinf_1[:np.argmin((lsinf-lsph[-1])**2)]
hnod1_sinfinterp = np.interp(lh, lh_sinf, hnod1_sinf)
hnod1 = np.mean([hnod1_sinfinterp, hnod1_sph],0)
herr1 = np.std([hnod1_sinfinterp, hnod1_sph*np.median(hnod1_sinfinterp)/np.median(hnod1_sph)],0)


lk = lsinf[np.argmin((lsinf-lsph[-1])**2):]
knod1 = nod_sinf_1[np.argmin((lsinf-lsph[-1])**2):]
kerr1 = nod_sinf_1_err[np.argmin((lsinf-lsph[-1])**2):]/9


plt.figure()
sinfobj.plot_spec(0.6,0.8)
lsinf, nod_sinf_2, nod_sinf_2_err = sinfobj.make_spec(0.6,0.8)
sphobj.plot_spec(0.6,0.8)
lsph, nod_sph_2, nod_sph_2_err = sphobj.make_spec(0.6,0.8)

lj = lsph[:np.argmin((lsinf[0]-lsph)**2)]
jnod2 = nod_sph_2[:np.argmin((lsinf[0]-lsph)**2)]
jerr2 = nod_sph_2_err[:np.argmin((lsinf[0]-lsph)**2)]

lh = lsph[np.argmin((lsinf[0]-lsph)**2)+1:]
lh_sinf = lsinf[:np.argmin((lsinf-lsph[-1])**2)]
hnod2_sph = nod_sph_2[np.argmin((lsinf[0]-lsph)**2)+1:]
hnod2_sinf = nod_sinf_2[:np.argmin((lsinf-lsph[-1])**2)]
hnod2_sinfinterp = np.interp(lh, lh_sinf, hnod2_sinf)
hnod2 = np.mean([hnod2_sinfinterp, hnod2_sph],0)
herr2 = np.std([hnod2_sinfinterp, hnod2_sph*np.median(hnod2_sinfinterp)/np.median(hnod2_sph)],0)


lk = lsinf[np.argmin((lsinf-lsph[-1])**2):]
knod2 = nod_sinf_2[np.argmin((lsinf-lsph[-1])**2):]
kerr2 = nod_sinf_2_err[np.argmin((lsinf-lsph[-1])**2):]/3


plt.figure()
sinfobj.plot_spec(-0.065,0.115)
lsinf, nod_sinf_0, nod_sinf_0_err = sinfobj.make_spec(-0.065,0.135)
# nod_sinf_0 *= 2.4/0.52
nod_sinf_0 *= 5.28/1.83
sphobj.plot_spec(-0.065,0.135)
lsph, nod_sph_0, nod_sph_0_err = sphobj.make_spec(-0.065,0.135)

lj = lsph[:np.argmin((lsinf[0]-lsph)**2)]
jnod0 = nod_sph_0[:np.argmin((lsinf[0]-lsph)**2)]
jerr0 = nod_sph_0_err[:np.argmin((lsinf[0]-lsph)**2)]

lh = lsph[np.argmin((lsinf[0]-lsph)**2)+1:]
lh_sinf = lsinf[:np.argmin((lsinf-lsph[-1])**2)]
hnod0_sph = nod_sph_0[np.argmin((lsinf[0]-lsph)**2)+1:]
hnod0_sinf = nod_sinf_0[:np.argmin((lsinf-lsph[-1])**2)]
hnod0_sinfinterp = np.interp(lh, lh_sinf, hnod0_sinf)
hnod0 = np.mean([hnod0_sinfinterp, hnod0_sph],0)
herr0 = np.std([hnod0_sinfinterp, hnod0_sph*np.median(hnod0_sinfinterp)/np.median(hnod0_sph)],0)


lk = lsinf[np.argmin((lsinf-lsph[-1])**2):]
knod0 = nod_sinf_0[np.argmin((lsinf-lsph[-1])**2):]
kerr0 = nod_sinf_0_err[np.argmin((lsinf-lsph[-1])**2):]/3

plt.figure()
plt.errorbar(lk, knod1, kerr1)
plt.errorbar(lh, hnod1, herr1)
plt.errorbar(lj, jnod1, jerr1)

plt.figure()
plt.errorbar(lk, knod2, kerr2)
plt.errorbar(lh, hnod2, herr2)
plt.errorbar(lj, jnod2, jerr2)

plt.figure()
plt.errorbar(lk, knod0, kerr0)
plt.errorbar(lh, hnod0, herr0)
plt.errorbar(lj, jnod0, jerr0)

surface = 0.09*0.1

np.savetxt('./PRODUITS/lambdas_k.dat', lk)
np.savetxt('./PRODUITS/lambdas_h.dat', lh)
np.savetxt('./PRODUITS/lambdas_j.dat', lj)

np.savetxt('./PRODUITS/jnod0.dat', jnod0)
np.savetxt('./PRODUITS/hnod0.dat', hnod0)
np.savetxt('./PRODUITS/knod0.dat', knod0)

np.savetxt('./PRODUITS/jnod1.dat', jnod1)
np.savetxt('./PRODUITS/hnod1.dat', hnod1)
np.savetxt('./PRODUITS/knod1.dat', knod1)

np.savetxt('./PRODUITS/jnod2.dat', jnod2)
np.savetxt('./PRODUITS/hnod2.dat', hnod2)
np.savetxt('./PRODUITS/knod2.dat', knod2)

np.savetxt('./PRODUITS/jerr0.dat', jerr0)
np.savetxt('./PRODUITS/herr0.dat', herr0)
np.savetxt('./PRODUITS/kerr0.dat', kerr0)

np.savetxt('./PRODUITS/jerr1.dat', jerr1)
np.savetxt('./PRODUITS/herr1.dat', herr1)
np.savetxt('./PRODUITS/kerr1.dat', kerr1)

np.savetxt('./PRODUITS/jerr2.dat', jerr2)
np.savetxt('./PRODUITS/herr2.dat', herr2)
np.savetxt('./PRODUITS/kerr2.dat', kerr2)

# np.savetxt('./PRODUITS/surface.dat', [surface])

# a, b1, c = sinfobj.make_spec(0.65,0.75)
# a, b2, c = sinfobj.make_spec(0.55,0.65)
# a, b3, c = sinfobj.make_spec(0.75,0.85)

# ad, b4, c = sphobj.make_spec(0.65,0.75)
# ad, b5, c = sphobj.make_spec(0.55,0.65)
# ad, b6, c = sphobj.make_spec(0.75,0.85)

# plt.plot(a, b1-(b2+b3)/2)
# plt.plot(ad, b4-(b5+b6)/2)


# a, b1, c = sinfobj.make_spec(0.25,0.35)
# a, b2, c = sinfobj.make_spec(0.35,0.45)
# a, b3, c = sinfobj.make_spec(0.15,0.25)

# ad, b4, c = sphobj.make_spec(0.25,0.35)
# ad, b5, c = sphobj.make_spec(0.35,0.45)
# ad, b6, c = sphobj.make_spec(0.15,0.25)


# plt.plot(a, b1-(b2+b3)/2)
# plt.plot(ad, b4-(b5+b6)/2)
# plt.plot(a, b1-(b2+b3)/2)
# plt.plot(ad, b4-(b5+b6)/2)

# sinfobj.plot_spec(-0.2,-0.1)
# ngc1068.plot_spec(-0.2,-0.1)

# plt.figure()
# sinfobj.plot_spec(-.5,.5)
# ngc1068.plot_spec(-0.5,0.5)


# import pickle 
# naco = pickle.load(open('/media/pierre/Disque_2/ORDI_1/2018/NACO/REDUCTION_2/PRODUITS/NOT_CAL/K_-103-3/objet.p', 'rb'))
# naco = pickle.load(open('/media/pierre/Disque_2/ORDI_1/2018/NACO/REDUCTION_2/PRODUITS/NOT_CAL/L_12/objet.p', 'rb'))
