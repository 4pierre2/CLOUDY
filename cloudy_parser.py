#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 17:15:19 2022

@author: pierre
"""

#%%

def read_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    return lines

#%%

def is_multigrid(lines):
    n = 0
    for l in lines:
        if 'VARY' in l:
            n += 1
    return n>0, n

def split_grids(lines):
    grids = []
    grid_numbers = []
    test, n = is_multigrid(lines)
    is_first_cp = True
    if not test:
        grids.append(lines)
        return ['grid000000000'], grids
    else:
        grid = []
        for l in lines:
            if 'GRID_DELIMIT' in l:
                if not is_first_cp:
                    grid_numbers.append(l.split('--')[-1].strip())
                    grids.append(grid)
                grid = []
                is_first_cp = False
            else:
                grid.append(l)
    return grid_numbers, grids
        
#%%    
           
def parse_line_line(line):
    name = line[:10].strip()
    wl_str = line[10:20].strip()
    wl = 0
    if wl_str[-1] == 'm':
        wl = float(wl_str[:-1])*1e-6
    elif wl_str[-1] == 'c':
        wl = float(wl_str[:-1])*1e-2
    elif wl_str[-1] == 'A':
        wl = float(wl_str[:-1])*1e-10
    fl = float(line[20:29].strip())
    fl_r = float(line[30:38].strip())
    return [name, wl_str, wl, fl, fl_r]

     
def get_intrinsic_lines(grid):
    go = False
    cp = False
    lines = []
    wls = []
    for l in grid:
        if '     Intrinsic line intensities' in l:
            go = True
            cp = True
        if '     Emergent line intensities' in l:
            go = False
        if go:
            if ('...' not in l) and ('TOTL' not in l) and ('***' not in l) and (not cp) and (len(l)>2):# and ('Ca A' not in l) and ('Ca B' not in l):# and ('TOTL' not in l) 
                lines.append(parse_line_line(l))
                wls.append(lines[-1][2])
        cp = False
    lines = [x for _, x in sorted(zip(wls, lines))]
    return lines
            
def get_emergent_lines(grid):
    go = False
    cp = False
    lines = []
    wls = []
    for l in grid:
        if '     Emergent line intensities' in l:
            go = True
            cp = True
        if "blends..............................." in l:
            go = False
        if go:
            if ('...' not in l) and ('TOTL' not in l) and ('***' not in l) and (not cp) and (len(l)>2) and ('Ca A' not in l) and ('Ca B' not in l) and ('nFnu' not in l) and ('MIR' not in l) and ('PAH' not in l)  and ('IR' not in l):
                lines.append(parse_line_line(l))
                wls.append(lines[-1][2])
        cp = False
    lines = [x for _, x in sorted(zip(wls, lines))]
    return lines

def restrict_wavelength_range(lines, lam0, lam1):
    new_lines = []
    for l in lines:
        wl = l[2]
        if (wl > lam0) and (wl < lam1):
            new_lines.append(l)
    return new_lines


#%%

def parse_file(filename, path='./', lam0 = 0.3e-6, lam1=10e-6, emergent=True):
    
    lines_raw = read_file(path+filename)
    grid_names, grids = split_grids(lines_raw)
    parsed_lines = []
    grids_infos = []
    for grid in grids:        
        if emergent:
            sorted_lines = get_emergent_lines(grid)
        else:
            sorted_lines = get_intrinsic_lines(grid)
        lines = restrict_wavelength_range(sorted_lines, lam0, lam1)
        parsed_lines.append(lines)
    for grid_name in grid_names:
        if len(grid_names)>1:
            grid_info_t = read_file(path+'/'+grid_name+'_model.txt')
            for info in grid_info_t:
                if info[0] != '#':
                    grid_info = info.split('\t')
            grids_infos.append(grid_info[:-1])
    return grids_infos, parsed_lines
            
#%%

import numpy as np

def gauss(x, flux, x0, std):
    a = flux/(np.pi*2*std**2)**0.5
    return a*np.exp(-(x-x0)**2/(2*std**2))


def make_spec(lines, lambdas, std_sph, std_sinfo, lambda_break=1.78969e-6):
    mask_sph = lambdas < lambda_break
    mask_sinfo = lambdas > lambda_break
    spec = np.zeros(np.shape(lambdas))
    for l in lines:
        wl = l[2]
        fl = l[3]
        spec += gauss(lambdas, fl, wl, std_sph)*mask_sph
        spec += gauss(lambdas, fl, wl, std_sinfo)*mask_sinfo
    return spec
        
#%%

# lines_simple = read_file('./MODEL_test_scraper_simple/model.out')
# lines_double = read_file('./MODEL_test_scraper_double/model.out')
# lines_triple = read_file('./MODEL_test_scraper_triple/model.out')
# is_multigrid(lines_simple)
# is_multigrid(lines_double)
# is_multigrid(lines_triple)
# i1, grids_simple = split_grids(lines_simple)
# i2, grids_double = split_grids(lines_double)
# i3, grids_triple = split_grids(lines_triple)
# sorted_lines_simple = get_emergent_lines(grids_simple[0])
# sorted_lines_double = get_emergent_lines(grids_double[0])
# sorted_lines_triple = get_emergent_lines(grids_triple[0])

# lines = parse_file('model.out', './MODEL_SEMICOARSE/', emergent=False)

# hlam = np.loadtxt('./PRODUITS/lambdas_h.dat')
# hnod0 = np.loadtxt('./PRODUITS/hnod0.dat')
# hnod1 = np.loadtxt('./PRODUITS/hnod1.dat')
# hnod2 = np.loadtxt('./PRODUITS/hnod2.dat')
# herr0 = np.loadtxt('./PRODUITS/herr0.dat')
# herr1 = np.loadtxt('./PRODUITS/herr1.dat')
# herr2 = np.loadtxt('./PRODUITS/herr1.dat')*0

# jlam = np.loadtxt('./PRODUITS/lambdas_j.dat')
# jnod0 = np.loadtxt('./PRODUITS/jnod0.dat')
# jnod1 = np.loadtxt('./PRODUITS/jnod1.dat')
# jnod2 = np.loadtxt('./PRODUITS/jnod2.dat')
# jerr0 = np.loadtxt('./PRODUITS/jerr0.dat')
# jerr1 = np.loadtxt('./PRODUITS/jerr1.dat')
# jerr2 = np.loadtxt('./PRODUITS/jerr1.dat')*0

# klam = np.loadtxt('./PRODUITS/lambdas_k.dat')
# knod0 = np.loadtxt('./PRODUITS/knod0.dat')
# knod1 = np.loadtxt('./PRODUITS/knod1.dat')
# knod2 = np.loadtxt('./PRODUITS/knod2.dat')
# kerr0 = np.loadtxt('./PRODUITS/kerr0.dat')
# kerr1 = np.loadtxt('./PRODUITS/kerr1.dat')
# kerr2 = np.loadtxt('./PRODUITS/kerr1.dat')*0

# lambdas = np.concatenate([jlam, hlam, klam])
# nod0 = np.concatenate([jnod0, hnod0, knod0])
# err0 = np.concatenate([jerr0, herr0, kerr0])
# nod1 = np.concatenate([jnod1, hnod1, knod1])
# err1 = np.concatenate([jerr1, herr1, kerr1])
# nod2 = np.concatenate([jnod2, hnod2, knod2])
# err2 = np.concatenate([jerr2, herr2, kerr2])


# from scipy.ndimage import median_filter, minimum_filter, maximum_filter
# from scipy.optimize import curve_fit

# def prop(x, a):
#     return a*x


# lambdas_1 = lambdas*1.0009242
# lambdas_2 = lambdas*231965/232115*1.0009242

# rawconts = []
# for k in range(150):
#     rawcont_t = minimum_filter(median_filter(nod1, 10+np.random.randint(100)), 10+np.random.randint(100))
#     rawconts.append(rawcont_t)
# rawcont1 = np.median(rawconts, 0)
# spec1 = (nod1-rawcont1)
# spec_for_fit_1 = spec1/np.mean(spec1)
# mask1 = spec_for_fit_1 > -0.2
# mask1 = ((lambdas<1600)+(lambdas>1700))*((lambdas<1780)+(lambdas>1950))*((lambdas<1270)+(lambdas>1300))
# spec_for_fit_1 = spec_for_fit_1[mask1]
# # spec_for_fit_1 = np.log(spec_for_fit_1[mask1]+1

# rawconts = []
# for k in range(150):
#     rawcont_t = minimum_filter(median_filter(nod2, 10+np.random.randint(100)), 10+np.random.randint(100))
#     rawconts.append(rawcont_t)
# rawcont2 = np.median(rawconts, 0)
# spec2 = (nod2-rawcont2)
# spec_for_fit_2 = spec2/np.sum(spec2)


# nested_models = []
# nested_models_infos = []

# for N in np.arange(-2, 3.2, 0.25):
#     nested_models.append([])
#     nested_models_infos.append([])
#     for T in np.arange(3,10.2,0.25):
#         D = 0       
#         for k in range(len(lines[0][1:])):
#             model_infos = lines[0][1+k].replace('\t', ';').split(';')
#             model_lines = lines[1][1+k]
#             if (T == float(model_infos[-3])) and (N == float(model_infos[-2])):
#                 nested_models[-1].append(model_lines)
#                 nested_models_infos[-1].append(model_infos)
            
# k2min = 1e100
# k2s = []

# for j in range(len(nested_models)):
#     print(j)
#     models = nested_models[j]
#     models_infos = nested_models_infos[j]
#     for k in range(len(models)):
#         model_1 = models[k]
#         model_info_1 = models_infos[k]
#         spec_mod_1 = make_spec(model_1, lambdas_1*1e-9, 2.72e-9)[mask1]
#         if np.sum(spec_mod_1)>0:
#             spec_mod_1 /= 2*np.mean(spec_mod_1)
#         for l in range(len(models)):
#             model_2 = models[l]
#             model_info_2 = models_infos[l]
#             spec_mod_2 = make_spec(model_2, lambdas_2*1e-9, 2.72e-9)[mask1]
#             if np.sum(spec_mod_2)>0:
#                 spec_mod_2 /= 2*np.mean(spec_mod_2)      
#             def double_prop(_, a, b):
#                 return abs(a)*spec_mod_1+abs(b)*spec_mod_2
#             try:
#                 p, c = curve_fit(double_prop, lambdas[mask1], spec_for_fit_1, p0=[1, 1], maxfev=100000)
#             except:
#                 p = [1, 1]
#             spec_mod = double_prop(lambdas, *p)
#             k2 = np.sum((spec_for_fit_1-spec_mod)**2)
#             k2s.append(k2)
#             if k2 < k2min:
#                 k2min =  k2
#                 best_spec_mod = spec_mod
#                 best_p = p
#                 best_model_1_info = model_info_1
#                 best_model_2_info = model_info_2
        
# for k in range(len(lines[0][1:])):
#     model_info = lines[0][1+k].replace('\t', ';').split(';')
    
#     model_lines = lines[1][1+k]
#     spec = make_spec(model_lines, lambdas_1*1e-9, 2.72e-9)
#     if np.sum(spec)>0:
#         spec /= np.sum(spec)
#     p, c = curve_fit(prop, spec, spec1)
#     spec_mod = prop(spec, *p)
#     k2 = np.sum((spec1-spec_mod)**2)
#     k2s.append(k2)
#     if k2 < k2min:
#         k2min =  k2
#         best_spec_mod = spec_mod
#         best_model_info = model_info